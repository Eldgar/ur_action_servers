#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/LinearMath/Quaternion.h>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <moveit/move_group_interface/move_group_interface.h>

#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "ur_action_servers/action/camera_calibrate.hpp"

using namespace std::chrono_literals;
using CameraCalibrate = ur_action_servers::action::CameraCalibrate;
using GoalHandle      = rclcpp_action::ServerGoalHandle<CameraCalibrate>;

class CameraCalibAction : public rclcpp::Node
{
public:
  CameraCalibAction()                                  // ← constructor: no shared_from_this()
  : Node("camera_calibration_server"),
    tf_buffer_(get_clock()),
    tf_listener_(tf_buffer_)
  {
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    server_ = rclcpp_action::create_server<CameraCalibrate>(
        this, "camera_calibrate",
        std::bind(&CameraCalibAction::handleGoal,   this, std::placeholders::_1, std::placeholders::_2),
        std::bind(&CameraCalibAction::handleCancel, this, std::placeholders::_1),
        std::bind(&CameraCalibAction::handleAccepted, this, std::placeholders::_1));
  }

  /** Must be called once **after** the node is wrapped in std::shared_ptr. */
  void initialize_move_group()
  {
    move_group_ = std::make_shared<
        moveit::planning_interface::MoveGroupInterface>(
        shared_from_this(), "ur_manipulator");
    move_group_->setPlanningTime(10.0);
  }

private:
  // ───────────────────────── Action plumbing ──────────────────────────
  rclcpp_action::GoalResponse handleGoal(
      const rclcpp_action::GoalUUID &,
      std::shared_ptr<const CameraCalibrate::Goal> goal)
  {
    if (goal->command == "calibrate" || goal->command == "stop")
      return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;

    RCLCPP_WARN(get_logger(), "Unknown command '%s'", goal->command.c_str());
    return rclcpp_action::GoalResponse::REJECT;
  }

  rclcpp_action::CancelResponse handleCancel(std::shared_ptr<GoalHandle>)
  { return rclcpp_action::CancelResponse::ACCEPT; }

  void handleAccepted(std::shared_ptr<GoalHandle> gh)
  { std::thread{std::bind(&CameraCalibAction::execute, this, gh)}.detach(); }

  void execute(std::shared_ptr<GoalHandle> gh)
  {
    auto goal   = gh->get_goal();
    auto result = std::make_shared<CameraCalibrate::Result>();

    if (goal->command == "stop") {
      stopBroadcasting();
      result->success = true;
      result->message = "Stopped TF broadcast.";
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

    result->success = true;
    result->message = "Calibration complete; TF broadcasting.";
    gh->succeed(result);
  }

  // ─────────────── Calibration Logic ───────────────
  bool runCalibration()
  {
    constexpr size_t NUM_DETECTIONS = 5;
    constexpr double TIME_PER_POSE = 10.0;
    constexpr const char* TARGET_FRAME = "camera_link";

    tf_samples_.clear();

    for (size_t i = 0; i < joint_poses_.size(); ++i) {
      RCLCPP_INFO(get_logger(), "➡️ Pose %zu / %zu", i + 1, joint_poses_.size());

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

      // Average translation
      Eigen::Vector3d avg_t = Eigen::Vector3d::Zero();
      for (const auto &t : detections) avg_t += t.translation();
      avg_t /= detections.size();

      Eigen::Isometry3d avg_tf = Eigen::Isometry3d::Identity();
      avg_tf.linear() = detections.front().rotation();
      avg_tf.translation() = avg_t;
      tf_samples_.push_back(avg_tf);
    }

    if (tf_samples_.empty()) {
      RCLCPP_ERROR(get_logger(), "❌ Calibration failed – no TF samples.");
      return false;
    }

    // Mean translation
    Eigen::Vector3d mean_pos = Eigen::Vector3d::Zero();
    for (auto &tf : tf_samples_) mean_pos += tf.translation();
    mean_pos /= tf_samples_.size();

    // Covariance
    Eigen::Matrix3d pos_cov = Eigen::Matrix3d::Zero();
    for (auto &tf : tf_samples_) {
      Eigen::Vector3d d = tf.translation() - mean_pos;
      pos_cov += d * d.transpose();
    }
    pos_cov /= tf_samples_.size();
    Eigen::Vector3d pos_stddev = pos_cov.eigenvalues().cwiseSqrt().real();

    // Mean rotation
    std::vector<Eigen::Quaterniond> quats;
    for (auto &tf : tf_samples_) quats.emplace_back(tf.rotation());

    Eigen::Quaterniond mean_q = quats.front();
    for (size_t i = 1; i < quats.size(); ++i) {
      if (mean_q.dot(quats[i]) < 0) quats[i].coeffs() *= -1;
      mean_q = mean_q.slerp(1.0 / (i + 1), quats[i]);
    }
    mean_q.normalize();

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
      "\n📍 Mean Position:       (%.4f, %.4f, %.4f)"
      "\n📍 Mean Orientation:    (x=%.4f, y=%.4f, z=%.4f, w=%.4f)"
      "\nσ Translation:          (%.4f, %.4f, %.4f)"
      "\nσ Rotation (radians*):  (%.4f, %.4f, %.4f)",
      mean_pos.x(), mean_pos.y(), mean_pos.z(),
      mean_q.x(), mean_q.y(), mean_q.z(), mean_q.w(),
      pos_stddev.x(), pos_stddev.y(), pos_stddev.z(),
      rot_stddev.x(), rot_stddev.y(), rot_stddev.z());

    mean_tf_.header.frame_id = "base_link";
    mean_tf_.child_frame_id  = "camera_mean";
    mean_tf_.transform.translation.x = mean_pos.x();
    mean_tf_.transform.translation.y = mean_pos.y();
    mean_tf_.transform.translation.z = mean_pos.z();
    mean_tf_.transform.rotation.x = mean_q.x();
    mean_tf_.transform.rotation.y = mean_q.y();
    mean_tf_.transform.rotation.z = mean_q.z();
    mean_tf_.transform.rotation.w = mean_q.w();

    return true;
  }

  bool moveToJointPose(const std::vector<double>& joints)
  {
    move_group_->setJointValueTarget(joints);
    moveit::planning_interface::MoveGroupInterface::Plan plan;
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
    mean_tf_.header.stamp = now();
    tf_broadcaster_->sendTransform(mean_tf_);
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
  std::vector<Eigen::Isometry3d>                                      tf_samples_;
  rclcpp_action::Server<CameraCalibrate>::SharedPtr                   server_;

  const std::vector<std::vector<double>> joint_poses_ = {
    {2.81349, -0.76867, 2.42993, -2.67641, 5.68814, -0.72473},
    {2.73414, -0.85652, 1.89450, -1.70130, 5.74765, -1.05458},
    {2.36194, -1.19495, 2.65155, -2.10590, 5.37617, -1.21476},
    {2.89310, -0.45249, 2.59039, -3.26049, 5.72645, -0.59660},
    {3.05772, -0.50430, 1.90387, -2.79052, 5.77800, -0.28618},
    {2.91038, -0.54873, 1.93223, -2.53217, 5.73402, -0.56643},
    {2.38786, -0.69632, 2.26797, -2.23364, 5.39698, -1.19453},
    {2.22103, -0.66438, 2.47734, -2.40532, 5.26166, -1.31442}
  };
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




