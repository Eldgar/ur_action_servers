#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <std_msgs/msg/float64.hpp>        

#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/static_transform_broadcaster.h>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <moveit/move_group_interface/move_group_interface.h>

#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "rclcpp/utilities.hpp"
#include "ur_action_servers/action/camera_calibrate.hpp"

using namespace std::chrono_literals;
using CameraCalibrate = ur_action_servers::action::CameraCalibrate;
using GoalHandle      = rclcpp_action::ServerGoalHandle<CameraCalibrate>;

class CameraCalibAction : public rclcpp::Node
{
public:
  CameraCalibAction()
  : Node("camera_calibration_server"),
    tf_buffer_(get_clock()),
    tf_listener_(tf_buffer_)
  {
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);

    auto distance_qos = rclcpp::QoS(rclcpp::KeepLast(1))
                          .reliable()          // ensure delivery
                          .transient_local();  // latch last value
    distance_pub_ = create_publisher<std_msgs::msg::Float64>("/d415_distance", distance_qos);

    server_ = rclcpp_action::create_server<CameraCalibrate>(
        this, "camera_calibrate",
        std::bind(&CameraCalibAction::handleGoal,   this, std::placeholders::_1, std::placeholders::_2),
        std::bind(&CameraCalibAction::handleCancel, this, std::placeholders::_1),
        std::bind(&CameraCalibAction::handleAccepted, this, std::placeholders::_1));
  }

  void initialize_move_group()
  {
    move_group_ = std::make_shared<
        moveit::planning_interface::MoveGroupInterface>(
        shared_from_this(), "ur_manipulator");
    move_group_->setPlanningTime(10.0);
    /* Collision environment disabled */
  }

private:
  rclcpp_action::GoalResponse handleGoal(
      const rclcpp_action::GoalUUID &,
      std::shared_ptr<const CameraCalibrate::Goal> goal)
  {
    if (goal->command == "calibrate" || goal->command == "stop")
      return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;

    RCLCPP_WARN(get_logger(), "Unknown command '%s'", goal->command.c_str());
    return rclcpp_action::GoalResponse::REJECT;
  }

  rclcpp_action::CancelResponse handleCancel(std::shared_ptr<GoalHandle> gh)
  {
    // Stop any ongoing trajectory execution and TF broadcasting
    if (move_group_)
      move_group_->stop();
    stopBroadcasting();
    RCLCPP_WARN(get_logger(), "⏹ Calibration goal cancelled – robot motion halted");
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handleAccepted(std::shared_ptr<GoalHandle> gh)
  { std::thread{std::bind(&CameraCalibAction::execute, this, gh)}.detach(); }

  void execute(std::shared_ptr<GoalHandle> gh)
  {
    auto goal   = gh->get_goal();
    auto result = std::make_shared<CameraCalibrate::Result>();

    if (goal->command == "stop") {
      // Stop TF broadcasting and immediately halt arm motion
      stopBroadcasting();
      if (move_group_)
        move_group_->stop();
      result->success = true;
      result->message = "Stopped TF broadcast and halted motion.";
      gh->succeed(result);
      return;
    }

    gh->publish_feedback(makeFeedback("Collecting samples…"));

    if (!runCalibration()) {
      result->success = false;
      result->message = "Calibration failed.";
      gh->abort(result);
      return;
    }

    gh->publish_feedback(makeFeedback("Broadcasting mean TF…"));
    startBroadcasting();

    /* Move arm to standby pose */
    gh->publish_feedback(makeFeedback("Moving to standby pose…"));
    if (!moveToJointPose(standby_pose_)) {
      RCLCPP_WARN(get_logger(), "⚠️ Failed to move to standby pose");
    }

    result->success = true;
    result->message = "Calibration complete; TF broadcasting.";
    gh->succeed(result);
  }

  // ─────────────── Calibration Logic ───────────────
  bool runCalibration()
  {
    constexpr size_t NUM_DETECTIONS = 5;
    constexpr double TIME_PER_POSE = 12.0;
    constexpr const char* TARGET_FRAME = "camera_color_optical_frame";

    tf_samples_.clear();

    for (size_t i = 0; i < joint_poses_.size(); ++i) {
      RCLCPP_INFO(get_logger(), "➡️ Pose %zu / %zu", i + 1, joint_poses_.size());
      rclcpp::sleep_for(500ms);
      if (!moveToJointPose(joint_poses_[i])) {
        RCLCPP_WARN(get_logger(), "❌ Skipping pose %zu - planning failed", i + 1);
        continue;
      }

      std::vector<Eigen::Isometry3d> detections;
      auto start = now();

      while (detections.size() < NUM_DETECTIONS &&
             (now() - start) < rclcpp::Duration::from_seconds(TIME_PER_POSE)) {
        Eigen::Isometry3d tf = computeTF(TARGET_FRAME);
        if (!tf.isApprox(Eigen::Isometry3d::Identity())) {
        rclcpp::sleep_for(400ms);
          detections.push_back(tf);
          RCLCPP_INFO(get_logger(), "✅ [%zu/%zu] detections", detections.size(), NUM_DETECTIONS);
        } else {
          rclcpp::sleep_for(100ms);
        }
      }

      if (detections.empty()) {
        RCLCPP_WARN(get_logger(), "⚠️ No valid detections at pose %zu", i + 1);
        continue;
      }

      Eigen::Isometry3d avg_tf = Eigen::Isometry3d::Identity();
      avg_tf.linear() = detections.front().rotation();   // keep orientation
      tf_samples_.push_back(avg_tf);                     // translation ignored
    }

    if (tf_samples_.empty()) {
      RCLCPP_ERROR(get_logger(), "❌ Calibration failed – no TF samples.");
      return false;
    }

    // Mean rotation
    std::vector<Eigen::Quaterniond> quats;
    for (auto &tf : tf_samples_) quats.emplace_back(tf.rotation());

    Eigen::Quaterniond mean_q = quats.front();
    for (size_t i = 1; i < quats.size(); ++i) {
      if (mean_q.dot(quats[i]) < 0) quats[i].coeffs() *= -1;
      mean_q = mean_q.slerp(1.0 / (i + 1), quats[i]);
    }
    mean_q.normalize();

    // Get current camera position
    Eigen::Vector3d cur_pos;
    {
        Eigen::Isometry3d tf_now = computeTF(TARGET_FRAME);   // base → camera
        if (tf_now.isApprox(Eigen::Isometry3d::Identity())) {
            RCLCPP_ERROR(get_logger(), "No valid TF for current camera position.");
            return false;
        }
        cur_pos = tf_now.translation();
    }

    // Rotation covariance
    Eigen::Matrix3d rot_cov = Eigen::Matrix3d::Zero();
    for (auto &q : quats) {
      Eigen::Quaterniond dq = mean_q.inverse() * q;
      if (dq.w() < 0) dq.coeffs() *= -1;
      Eigen::AngleAxisd aa(dq);
      Eigen::Vector3d rv = aa.angle() * aa.axis();
      rot_cov += rv * rv.transpose();
    }
    rot_cov /= quats.size();
    Eigen::Vector3d rot_stddev = rot_cov.eigenvalues().cwiseSqrt().real();

    RCLCPP_INFO(get_logger(),
      "\n Current Position:      (%.4f, %.4f, %.4f)"
      "\n Mean Orientation:      (x=%.4f, y=%.4f, z=%.4f, w=%.4f)",
      cur_pos.x(), cur_pos.y(), cur_pos.z(),
      mean_q.x(), mean_q.y(), mean_q.z(), mean_q.w());

    // Represent the measured transform T_base_to_color_measured
    Eigen::Isometry3d T_base_to_color_measured = Eigen::Isometry3d::Identity();
    T_base_to_color_measured.translation() = cur_pos;
    T_base_to_color_measured.linear() = mean_q.toRotationMatrix();

    // --- Get the static transform T_link_to_color ---
    Eigen::Isometry3d T_link_to_color;
    try {
        // Use the correct frame names as published in your TF tree
        auto tf_link_to_color_msg = tf_buffer_.lookupTransform(
            "D415_link",                 // Target frame for the static TF
            "D415_color_optical_frame",  // Source frame for the static TF
            tf2::TimePointZero);

        T_link_to_color = tf2::transformToEigen(tf_link_to_color_msg); // Convert geometry_msgs::TransformStamped to Eigen::Isometry3d

    } catch (const tf2::TransformException &ex) {
        RCLCPP_ERROR(get_logger(), "Could not get static transform from D415_link to D415_color_optical_frame: %s", ex.what());
        return false; // Cannot proceed without this transform
    }

    // --- Calculate the desired transform T_base_to_link ---
    Eigen::Isometry3d T_base_to_link_calculated = T_base_to_color_measured * T_link_to_color.inverse();

    // --- Extract results for broadcasting ---
    Eigen::Vector3d final_pos = T_base_to_link_calculated.translation();
    Eigen::Quaterniond final_q(T_base_to_link_calculated.linear());
    final_q.normalize();

    RCLCPP_INFO(get_logger(),
      "\n Measured Color Frame Pos: (%.4f, %.4f, %.4f)"
      "\n Measured Color Frame Quat:(x=%.4f, y=%.4f, z=%.4f, w=%.4f)"
      "\n Static Link->Color Pos: (%.4f, %.4f, %.4f)"
      "\n Static Link->Color Quat: (x=%.4f, y=%.4f, z=%.4f, w=%.4f)"
      "\n Calculated Link Pos: (%.4f, %.4f, %.4f)"
      "\n Calculated Link Quat: (x=%.4f, y=%.4f, z=%.4f, w=%.4f)",
      cur_pos.x(), cur_pos.y(), cur_pos.z(),
      mean_q.x(), mean_q.y(), mean_q.z(), mean_q.w(),
      T_link_to_color.translation().x(), T_link_to_color.translation().y(), T_link_to_color.translation().z(),
      Eigen::Quaterniond(T_link_to_color.linear()).x(), Eigen::Quaterniond(T_link_to_color.linear()).y(), Eigen::Quaterniond(T_link_to_color.linear()).z(), Eigen::Quaterniond(T_link_to_color.linear()).w(),
      final_pos.x(), final_pos.y(), final_pos.z(),
      final_q.x(), final_q.y(), final_q.z(), final_q.w()
      );

    // --- Populate mean_tf_ with the calculated T_base_to_link pose ---
    mean_tf_.header.frame_id = "base_link";
    mean_tf_.child_frame_id  = "D415_link";
    mean_tf_.transform.translation.x = final_pos.x();
    mean_tf_.transform.translation.y = final_pos.y();
    mean_tf_.transform.translation.z = final_pos.z();
    mean_tf_.transform.rotation.x = final_q.x();
    mean_tf_.transform.rotation.y = final_q.y();
    mean_tf_.transform.rotation.z = final_q.z();
    mean_tf_.transform.rotation.w = final_q.w();

    /* ----- constant TF:  D415_link  →  D415_right_ir_frame ----- */
    ir_tf_.header.frame_id  = "D415_link";
    ir_tf_.child_frame_id   = "D415_right_ir_frame";
    ir_tf_.transform.translation.x =  0.0;
    ir_tf_.transform.translation.y = -0.055;
    ir_tf_.transform.translation.z =  0.0;
    ir_tf_.transform.rotation.x = -0.5;
    ir_tf_.transform.rotation.y =  0.5;
    ir_tf_.transform.rotation.z = -0.5;
    ir_tf_.transform.rotation.w =  0.5;

    // Publish the distance from base_link to D415_link
    {
      std_msgs::msg::Float64 msg;
      msg.data = final_pos.norm();
      distance_pub_->publish(msg);
      RCLCPP_INFO(get_logger(),
                  "Latched distance base_link→D415_link = %.4f m", msg.data);
    }

    return true;
  }

  bool moveToJointPose(const std::vector<double>& joints)
  {
    move_group_->setJointValueTarget(joints);
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    rclcpp::sleep_for(500ms);
    if (move_group_->plan(plan) != moveit::core::MoveItErrorCode::SUCCESS)
      return false;
    return move_group_->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS;
  }

  Eigen::Isometry3d computeTF(const std::string& target)
  {
    try {
      auto tf_msg = tf_buffer_.lookupTransform("base_link", target,
                                               tf2::TimePointZero, 100ms);

      Eigen::Isometry3d tf = Eigen::Isometry3d::Identity();
      tf.translation() = Eigen::Vector3d(tf_msg.transform.translation.x,
                                         tf_msg.transform.translation.y,
                                         tf_msg.transform.translation.z);
      tf.linear() = Eigen::Quaterniond(tf_msg.transform.rotation.w,
                                       tf_msg.transform.rotation.x,
                                       tf_msg.transform.rotation.y,
                                       tf_msg.transform.rotation.z).toRotationMatrix();
      return tf;
    } catch (const tf2::TransformException &ex) {
      RCLCPP_DEBUG(get_logger(), "TF lookup failed: %s", ex.what());
      return Eigen::Isometry3d::Identity();
    }
  }

  // ─────────────────── Broadcaster helpers ────────────────────
  void startBroadcasting()
  {
    if (!timer_)
      timer_ = create_wall_timer(33ms,
               std::bind(&CameraCalibAction::broadcastTF, this));
  }

  void stopBroadcasting()
  { if (timer_) timer_->cancel();  timer_.reset(); }

  void broadcastTF()
  {
    if (!timer_) return;

    const auto stamp = now();   

    mean_tf_.header.stamp = now();
    ir_tf_.header.stamp   = stamp;  
    tf_broadcaster_->sendTransform(
      std::vector<geometry_msgs::msg::TransformStamped>{mean_tf_, ir_tf_});
  }

  CameraCalibrate::Feedback::SharedPtr makeFeedback(const std::string &txt)
  {
    auto fb = std::make_shared<CameraCalibrate::Feedback>();
    fb->status = txt;
    return fb;
  }

  // ───────────────────────── Members ──────────────────────────
  std::shared_ptr<tf2_ros::TransformBroadcaster>                      tf_broadcaster_;
  tf2_ros::Buffer                                                     tf_buffer_;
  tf2_ros::TransformListener                                          tf_listener_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface>     move_group_;
  rclcpp::TimerBase::SharedPtr                                        timer_;
  geometry_msgs::msg::TransformStamped                                mean_tf_;
  geometry_msgs::msg::TransformStamped                                ir_tf_; 
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster>                static_broadcaster_;
  geometry_msgs::msg::TransformStamped                                static_color_tf_;
  std::vector<Eigen::Isometry3d>                                      tf_samples_;
  rclcpp_action::Server<CameraCalibrate>::SharedPtr                   server_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr                distance_pub_;

  const std::vector<std::vector<double>> joint_poses_ = {
            // { 2.81891, -0.89820, 1.69067, -1.75335, -0.48133, -2.20762 },
            // { 2.65093, -0.72145, 1.65343, -1.67958, -0.59240, -2.34463 },
            // { 2.72545, -0.68781, 1.48349, -1.81851, -0.70533, -2.08955 },
            // { 3.03425, -0.79504, 1.25535, -1.63884, -0.34689, -1.88130 },
            { 2.76359, -0.87828, 1.58700, -1.83645, -0.71333, -2.09312 },
            { 3.03425, -0.79504, 1.25535, -1.63884, -0.34689, -2.04130 },
            { 3.03425, -0.79504, 1.25535, -1.63884, -0.34689, -2.14130 },
            { 2.96834, -0.16573, 0.67028, -1.27966, -0.18557, -2.19351 },
            { 2.96834, -0.16573, 0.67028, -1.27966, -0.18557, -2.39351 },
            { 2.96834, -0.16573, 0.67028, -1.27966, -0.18557, -2.59351 },
            { 2.89589, -0.52993, 1.15244, -1.58780, -0.39359, -2.19087 },
            { 2.81969, -0.64596, 1.62408, -1.80887, -0.44131, -2.33796 },
            { 2.90300, -0.66354, 1.93476, -2.25093, -0.38970, -2.17547 },
            { 2.96924, -0.27762, 0.96071, -1.80989, -0.35685, -2.01731 },
            { 2.88067, -0.19469, 0.67950, -1.48274, -0.42683, -2.34294 },
            { 2.53973, -0.50475, 1.59799, -1.50125, -0.62416, -2.81514 },
            { 2.76359, -0.87828, 1.58700, -1.83645, -0.71333, -2.09312 }
    };

  /* ───── Pose to move to after calibration completes ───── */
  const std::vector<double> standby_pose_ = {0.0, -1.5708, 0.0, -1.5708, 0.0, -1.5708};
};

// ───────────────────────── Main ─────────────────────────
int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<CameraCalibAction>();
  node->initialize_move_group();
  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}