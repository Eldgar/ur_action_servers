/*********************************************************************
 * aruco_calibration_server.cpp
 *
 * Action server that drives the UR3e through a series of joint poses,
 * waits for the ArUco-centre pixel, its depth, and base←marker TF,
 * estimates base←D415_link, and broadcasts it.  Uses the generic
 * CameraCalibrate.action:
 *
 *   # Goal
 *   string command          # e.g. "start"
 *   ---
 *   # Result
 *   bool   success
 *   string message
 *   ---
 *   # Feedback
 *   string status
 *********************************************************************/

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <moveit/move_group_interface/move_group_interface.h>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/point32.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <cv_bridge/cv_bridge.h>

#include <Eigen/Dense>
#include <opencv2/imgproc.hpp>
#include <opencv2/aruco.hpp>

#include "ur_action_servers/action/camera_calibrate.hpp"

using namespace std::chrono_literals;

class ArucoCalibrationServer : public rclcpp::Node
{
  /* ──────────────── aliases ──────────────── */
  using CameraCalibrate = ur_action_servers::action::CameraCalibrate;
  using GoalHandle      = rclcpp_action::ServerGoalHandle<CameraCalibrate>;

public:
  ArucoCalibrationServer() :
    Node("aruco_calibration_server"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {


    /* ─────────── MoveIt setup ─────────── */
    // move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
    //     this->shared_from_this(), "ur_manipulator");
    // move_group_->setPlanningTime(10.0);
    // move_group_->setMaxVelocityScalingFactor(0.3);

    /* ─────────── action server ─────────── */
    action_server_ = rclcpp_action::create_server<CameraCalibrate>(
        this,
        "depth_calibrate",
        std::bind(&ArucoCalibrationServer::handle_goal,    this, std::placeholders::_1, std::placeholders::_2),
        std::bind(&ArucoCalibrationServer::handle_cancel,  this, std::placeholders::_1),
        std::bind(&ArucoCalibrationServer::handle_accepted,this, std::placeholders::_1));

    /* ─────────── subscribers ─────────── */
    //  Color image (for ArUco detection)
    color_sub_ = image_transport::create_subscription(
        this, "/D415/color/image_raw",
        std::bind(&ArucoCalibrationServer::colorCb, this, std::placeholders::_1),
        "raw");

    //  Aligned depth image to query depth at detected pixel
    depth_sub_ = image_transport::create_subscription(
        this, "/D415/aligned_depth_to_color/image_raw",
        std::bind(&ArucoCalibrationServer::depthCb, this, std::placeholders::_1),
        "raw");
    debug_pub_ = image_transport::create_publisher(this, "/calib_debug/image");


    ur_tf_sub_ = create_subscription<geometry_msgs::msg::TransformStamped>(
        "/ur_transform", 10,
        std::bind(&ArucoCalibrationServer::urTfCb, this, std::placeholders::_1));

    tf_pub_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    /* ─────────── joint targets ─────────── */
    target_joints_ = {
        {2.56935, -0.28104, 1.72121, -1.69158, -0.54607, -2.89642},
        {2.34848, -0.28942, 1.44769, -1.45035, -0.77670, -2.90084},
        {2.55849, -0.07539, 0.89807, -1.20097, -0.57799, -2.79087},
        {2.55849, -0.07539, 0.89807, -1.20097, -0.57799, -2.79087},
        {2.55849, -0.07539, 0.89807, -1.20097, -0.57799, -2.79087},
        {2.50411, -0.49894, 1.33130, -1.44830, -0.70317, -2.61530},
        {2.33896, -0.49356, 1.67311, -1.70402, -0.84243, -2.74282},
        {2.40193, -0.44400, 1.54049, -1.65163, -0.78845, -2.69828}};
  }

    // Call this once the node is inside a shared_ptr
    void init_move_group()
    {
        move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
            shared_from_this(), "ur_manipulator");
        move_group_->setPlanningTime(10.0);
        move_group_->setMaxVelocityScalingFactor(0.3);
    }

private:
  /* ───────── helper structs & enums ───────── */
  struct Sample
    {
    Eigen::Vector3d p_B_M;   // marker origin in base
    double          range;   // depth along optical axis (metres)
    };
    image_transport::Publisher debug_pub_;  
    enum class Stage { IDLE, MOVING, WAITING } stage_{Stage::IDLE};

  /* ───────── ROS handles ───────── */
  tf2_ros::Buffer     tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_pub_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;

  // image subscribers
  image_transport::Subscriber                                        color_sub_;
  image_transport::Subscriber                                        depth_sub_;
  rclcpp::Subscription<geometry_msgs::msg::TransformStamped>::SharedPtr ur_tf_sub_;

  rclcpp_action::Server<CameraCalibrate>::SharedPtr action_server_;

  /* ───────── runtime state ───────── */
  std::vector<std::array<double,6>> target_joints_;
  size_t                idx_{0};
  geometry_msgs::msg::Point32 last_px_;
  double                 last_depth_{0.0};
  bool                   got_px_{false}, got_depth_{false};
  Eigen::Isometry3d      last_T_B_M_{Eigen::Isometry3d::Identity()};
  std::vector<Sample>    samples_;
  std::mutex             mutex_;              // protects shared state
  std::shared_ptr<GoalHandle> active_goal_;   // current goal handle
  rclcpp::Time            wait_until_;   // deadline for marker/depth at current pose

  /* ════════════════════════════════════════════════════════════════════════
   *                      Action-server callbacks
   * ════════════════════════════════════════════════════════════════════════ */
  rclcpp_action::GoalResponse
  handle_goal(const rclcpp_action::GoalUUID &,
              std::shared_ptr<const CameraCalibrate::Goal> goal)
  {
    if (stage_ != Stage::IDLE) {
      RCLCPP_WARN(get_logger(), "Calibration already running – rejecting new goal");
      return rclcpp_action::GoalResponse::REJECT;
    }
    if (goal->command != "start") {
      RCLCPP_WARN(get_logger(), "Unknown command \"%s\"", goal->command.c_str());
      return rclcpp_action::GoalResponse::REJECT;
    }
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse
  handle_cancel(const std::shared_ptr<GoalHandle> /*goal_handle*/)
  {
    std::lock_guard<std::mutex> lk(mutex_);
    RCLCPP_INFO(get_logger(), "Cancel request received");
    stage_ = Stage::IDLE;
    move_group_->stop();
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_accepted(const std::shared_ptr<GoalHandle> goal_handle)
  {
    /* run the long-running job in a separate thread so the executor can
       continue spinning subscriptions */
    std::thread{&ArucoCalibrationServer::execute, this, goal_handle}.detach();
  }



  /* ───────── main execute routine ───────── */
  void execute(const std::shared_ptr<GoalHandle> goal_handle)
  {
    active_goal_ = goal_handle;
    auto feedback = std::make_shared<CameraCalibrate::Feedback>();

    {
      std::lock_guard<std::mutex> lk(mutex_);
      stage_  = Stage::MOVING;
      idx_    = 0;
      samples_.clear();
      got_px_ = got_depth_ = false;
      feedback->status = "Starting calibration (pose 1)";
      goal_handle->publish_feedback(feedback);
    }

    RCLCPP_INFO(get_logger(), "Starting calibration – moving to pose 1/%zu",
                target_joints_.size());
    moveToPose();

    // Wait until the goal finishes (either success, cancel, or timeout)
    rclcpp::Rate r(10);
    while (rclcpp::ok())
    {
      bool timed_out = false;
      {
        std::lock_guard<std::mutex> lk(mutex_);

        // If calibration has already finished (result() set active_goal_ to nullptr)
        if (stage_ == Stage::IDLE && !active_goal_) {
          return;   // success path finished – exit execute()
        }

        // If client cancelled (goal still valid but cancel requested elsewhere)
        if (stage_ == Stage::IDLE && active_goal_) {
          result(false, "Cancelled by client");
          return;
        }

        // timeout check while waiting for detections
        if (stage_ == Stage::WAITING && now() > wait_until_) {
          timed_out = true;
          stage_ = Stage::MOVING; // prevent re-entering here
        }
      }

      if (timed_out) {
        RCLCPP_WARN(get_logger(), "⏱️  Timeout waiting for ArUco/depth at pose %zu", idx_ + 1);
        publishFeedback("Timeout – skipping pose " + std::to_string(idx_ + 1));

        if (++idx_ < target_joints_.size()) {
          rclcpp::sleep_for(500ms);
          moveToPose();
        } else {
          // finished all poses – evaluate samples
          if (samples_.size() < 3) {
            result(false, "❌ Calibration failed – only " + std::to_string(samples_.size()) + " valid samples");
          } else {
            if (samples_.size() == 3)
              RCLCPP_WARN(get_logger(), "⚠️ Only 3 samples collected – results may be less accurate");
            else
              RCLCPP_INFO(get_logger(), "✅ %zu valid samples collected", samples_.size());
            publishFeedback("Computing TF from " + std::to_string(samples_.size()) + " samples");
            computeTf();
          }
        }
      }
      r.sleep();
    }
    // Node shutting down:
    result(false, "Aborted – node shutting down");
  }

  void result(bool ok, const std::string &msg)
  {
    auto res = std::make_shared<CameraCalibrate::Result>();
    res->success = ok;
    res->message = msg;
    active_goal_->succeed(res);
    stage_ = Stage::IDLE;
    active_goal_.reset();
  }

  /* ───────── subscriber callbacks ───────── */
  void colorCb(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
  {
  std::lock_guard<std::mutex> lk(mutex_);
  if (stage_ != Stage::WAITING || got_px_) return;      // only once/pose

  /* 1.  Convert to OpenCV BGR */
  cv::Mat bgr;
  try {
    bgr = cv_bridge::toCvCopy(msg, "bgr8")->image;
  } catch (const cv_bridge::Exception &e) {
    RCLCPP_WARN(get_logger(), "cv_bridge failed: %s", e.what());
    return;
  }

  /* 2.  Detect the marker (same as before) */
  static const auto dict   = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
  static const auto params = cv::aruco::DetectorParameters::create();

  std::vector<int> ids;
  std::vector<std::vector<cv::Point2f>> corners;
  cv::aruco::detectMarkers(bgr, dict, corners, ids, params);
  if (ids.empty()) return;                              // no marker

  cv::Point2f c(0,0);
  for (auto &pt : corners[0]) c += pt;
  c *= 0.25f;                       // marker centre (pixels)

  last_px_.x = c.x;
  last_px_.y = c.y;
  got_px_    = true;

  /* 3.  Annotate and publish the debug image now (depth added later) */
  cv::Mat anno = bgr.clone();
  cv::circle(anno, c, 5, {0,0,255}, -1);
  cv::putText(anno, "await depth", c + cv::Point2f(5,-5),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, {255,255,255}, 1);

  if (debug_pub_)          
    debug_pub_.publish( cv_bridge::CvImage(msg->header, "bgr8", anno).toImageMsg() );

  publishFeedback("aruco pixel detected");
  }


  void depthCb(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
  {
    std::lock_guard<std::mutex> lk(mutex_);
    if (stage_ != Stage::WAITING || !got_px_) return;
    auto cv_ptr = cv_bridge::toCvShare(msg);
    int u = static_cast<int>(last_px_.x + 0.5);
    int v = static_cast<int>(last_px_.y + 0.5);
    if (u < 0 || v < 0 || u >= cv_ptr->image.cols || v >= cv_ptr->image.rows)
      return;
    uint16_t d_mm = cv_ptr->image.at<uint16_t>(v, u);
    if (d_mm == 0) return;
    last_depth_ = d_mm / 1000.0;
    got_depth_  = true;

      /* ---- DEBUG IMAGE WITH DEPTH LABEL ---- */
  cv::Mat anno;
    if (debug_pub_) {                 // publish only if someone listens
        anno = cv_ptr->image.clone(); 
        cv::circle(anno, {u,v}, 5, {0,0,255}, -1);
        char txt[32]; std::snprintf(txt, sizeof(txt), "%.3f m", last_depth_);
        cv::putText(anno, txt, {u+5,v-5},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, {255,255,255}, 1);
        debug_pub_.publish(cv_bridge::CvImage(msg->header, "bgr8", anno).toImageMsg());
    }
    publishFeedback("depth received");
    maybeFinishPose();
  }

  void urTfCb(const geometry_msgs::msg::TransformStamped::SharedPtr msg)
  {
    if (msg->child_frame_id != "aruco_link_rotated") return;
    std::lock_guard<std::mutex> lk(mutex_);
    last_T_B_M_ = tf2::transformToEigen(*msg);
  }

  /* ───────── movement helpers ───────── */
  void moveToPose()
  {
    stage_    = Stage::MOVING;
    got_px_   = got_depth_ = false;

    std::vector<double> vec(target_joints_[idx_].begin(), target_joints_[idx_].end());
    move_group_->setJointValueTarget(vec);
    moveit::planning_interface::MoveGroupInterface::Plan p;
    if (move_group_->plan(p) != moveit::core::MoveItErrorCode::SUCCESS)
    {
      RCLCPP_ERROR(get_logger(), "Plan failed @ pose %zu", idx_ + 1);
      result(false, "IK/plan failed");
      return;
    }
    move_group_->execute(p);
    stage_ = Stage::WAITING;

    // set 15-second timeout for this pose
    wait_until_ = now() + rclcpp::Duration::from_seconds(15.0);

    publishFeedback("Reached pose " + std::to_string(idx_ + 1) +
                    " – waiting for marker & depth");
  }

  void maybeFinishPose()
    {
    if (!(got_px_ && got_depth_)) return;

    Sample s;
    s.p_B_M = last_T_B_M_.translation();   // marker origin in base
    s.range = last_depth_;                 // d_i  (metres)
    samples_.push_back(s);

    publishFeedback("Pose " + std::to_string(idx_ + 1) + " captured");

    if (++idx_ < target_joints_.size())
    {
        rclcpp::sleep_for(500ms);
        moveToPose();
        return;
    }

    publishFeedback("Collected samples – computing TF");
    computeTf();   // <─ will broadcast aruco_link_rotated → camera
    RCLCPP_INFO(get_logger(),
            "Sample %zu  pixel=(%.1f,%.1f)  depth=%.3f m  "
            "p_B_M=(%.3f,%.3f,%.3f)",
            idx_ + 1,
            last_px_.x, last_px_.y, last_depth_,
            s.p_B_M.x(), s.p_B_M.y(), s.p_B_M.z());


    }


  /* ───────── final computation ───────── */
   void computeTf()
    {
    const size_t N = samples_.size();
    if (N < 3)
    {
        RCLCPP_ERROR(get_logger(), "Need at least 3 samples, got %zu", N);
        result(false, "Too few samples");
        return;
    }

    RCLCPP_INFO(get_logger(), "----- Raw samples -----");
    for (size_t i = 0; i < N; ++i)
        RCLCPP_INFO(get_logger(),
            "  %zu: P_B_M = (%.3f, %.3f, %.3f) m   range = %.3f m",
            i,
            samples_[i].p_B_M.x(),
            samples_[i].p_B_M.y(),
            samples_[i].p_B_M.z(),
            samples_[i].range);
    RCLCPP_INFO(get_logger(), "-----------------------");

    /* ---------------------------------------------------------------
    * 1.  Trilaterate p_B_C  from the (p_B_Mᵢ,  rᵢ) pairs
    * --------------------------------------------------------------- */
    Eigen::MatrixXd A(N - 1, 3);
    Eigen::VectorXd b(N - 1);

    const auto &s0 = samples_[0];
    for (size_t i = 1; i < N; ++i)
    {
        const auto &si = samples_[i];

        A.row(i-1) = 2.0 * (si.p_B_M - s0.p_B_M).transpose();
        b(i-1)     =  s0.range*s0.range - si.range*si.range
                    + si.p_B_M.squaredNorm() - s0.p_B_M.squaredNorm();
    }

    Eigen::Vector3d p_B_C =
        A.colPivHouseholderQr().solve(b);           // camera position in base

    /* ---------------------------------------------------------------
    * 2.  Express the pose in the marker frame of *sample 0*
    * --------------------------------------------------------------- */
    const Eigen::Isometry3d &T_B_M0 = last_T_B_M_;   // still holds last value
    Eigen::Isometry3d T_M_C = Eigen::Isometry3d::Identity();
    T_M_C.translation() = T_B_M0.inverse() * p_B_C;   // p_M0_C

    /* Orientation:  +Z points back to marker origin, build RHS frame */
    Eigen::Vector3d z_cam = -T_M_C.translation().normalized(); // from C→M0
    Eigen::Vector3d x_cam =
        (fabs(z_cam.z()) < 0.9 ? Eigen::Vector3d::UnitZ()
                                : Eigen::Vector3d::UnitY()).cross(z_cam).normalized();
    Eigen::Vector3d y_cam = z_cam.cross(x_cam);
    Eigen::Matrix3d R_M_C;
    R_M_C.col(0) = x_cam;
    R_M_C.col(1) = y_cam;
    R_M_C.col(2) = z_cam;
    T_M_C.linear() = R_M_C;

    /* ---------------------------------------------------------------
    * 3.  Broadcast  aruco_link_rotated → camera
    * --------------------------------------------------------------- */
    geometry_msgs::msg::TransformStamped tf_out =
        tf2::eigenToTransform(T_M_C);
    tf_out.header.stamp    = now();
    tf_out.header.frame_id = "aruco_link_rotated";
    tf_out.child_frame_id  = "calib_camera";
    tf_pub_->sendTransform(tf_out);

    RCLCPP_INFO(get_logger(),
        "Camera in marker frame: (%.3f, %.3f, %.3f) m",
        T_M_C.translation().x(),
        T_M_C.translation().y(),
        T_M_C.translation().z());

    for (size_t i = 0; i < N; ++i)
    {
    double err = (p_B_C - samples_[i].p_B_M).norm() - samples_[i].range;
    RCLCPP_INFO(get_logger(),
        "   res[%zu] = %+7.4f m  (expected %.3f, got %.3f)",
        i, err, samples_[i].range,
        (p_B_C - samples_[i].p_B_M).norm());
    }

    result(true, "Calibration OK (marker frame)");
    }

  /* ───────── feedback helper ───────── */
  void publishFeedback(const std::string &txt)
  {
    if (!active_goal_) return;
    auto fb = std::make_shared<CameraCalibrate::Feedback>();
    fb->status = txt;
    active_goal_->publish_feedback(fb);
    RCLCPP_INFO(get_logger(), "%s", txt.c_str());
  }

  /* sentinel for success path */
  Stage IDLE_DONE = Stage::IDLE;  // not used but avoids compiler warnings
};

/* ────────────────────────── main ────────────────────────── */
int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ArucoCalibrationServer>();
  node->init_move_group();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
