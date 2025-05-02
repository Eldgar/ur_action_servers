/*********************************************************************
 * real_depth_calibration.cpp
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

#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <geometry_msgs/msg/point32.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <cv_bridge/cv_bridge.h>

#include <Eigen/Dense>
#include <opencv2/imgproc.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/highgui.hpp>  // for imshow
#include <opencv2/calib3d.hpp>

#include "ur_action_servers/action/camera_calibrate.hpp"

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <fstream>

using namespace std::chrono_literals;

using CameraCalibrate = ur_action_servers::action::CameraCalibrate;
using GoalHandle = rclcpp_action::ServerGoalHandle<CameraCalibrate>;

class ArucoCalibrationServer : public rclcpp::Node
{
public:
  ArucoCalibrationServer() :
    Node("aruco_calibration_server"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    /* Defer MoveIt group initialisation until after constructor */
    init_timer_ = create_wall_timer(500ms,[this]{
        if (mg_) return;
        mg_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), "ur_manipulator");
        mg_->setPlanningTime(10.0);
        mg_->setMaxVelocityScalingFactor(0.2);
        init_timer_->cancel();
    });

    /* Camera subs */
    color_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
        "/D415/color/image_raw/compressed", rclcpp::SensorDataQoS(),
        std::bind(&ArucoCalibrationServer::colorCb, this, std::placeholders::_1));

    depth_sub_ = image_transport::create_subscription(
        this, "/D415/aligned_depth_to_color/image_raw",
        std::bind(&ArucoCalibrationServer::depthCb,this,std::placeholders::_1), "raw");

    /* Intrinsics */
    K_ = (cv::Mat_<double>(3,3) << 306.805847,0,214.441849,
                                   0,306.642456,124.910301,
                                   0,0,1);
    D_ = cv::Mat::zeros(5,1,CV_64F);
    dict_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    params_ = cv::aruco::DetectorParameters::create();

    /* action server */
    server_ = rclcpp_action::create_server<CameraCalibrate>(
        this,
        "depth_calibrate",
        std::bind(&ArucoCalibrationServer::handleGoal,this,std::placeholders::_1,std::placeholders::_2),
        std::bind(&ArucoCalibrationServer::handleCancel,this,std::placeholders::_1),
        std::bind(&ArucoCalibrationServer::handleAccept,this,std::placeholders::_1));

    /* target joints */
    target_joints_ = {
        {2.56935, -0.28104, 1.72121, -1.69158, -0.54607, -2.89642},
        {2.56935, -0.28104, 1.72121, -1.69158, -0.54607, -2.39642},
        {2.34848, -0.28942, 1.44769, -1.45035, -0.77670, -2.90084},
        {2.34848, -0.28942, 1.44769, -1.45035, -0.77670, -2.60084},
        {2.55849, -0.07539, 0.89807, -1.20097, -0.57799, -2.79087},
        {2.55849, -0.07539, 0.89807, -1.20097, -0.57799, -2.39087},
        {2.50411, -0.49894, 1.33130, -1.44830, -0.70317, -2.61530},
        {2.33896, -0.49356, 1.67311, -1.70402, -0.84243, -2.74282},
        {2.33896, -0.49356, 1.67311, -1.70402, -0.84243, -2.34282},
        {2.40193, -0.44400, 1.54049, -1.65163, -0.78845, -2.69828},
        {2.40193, -0.44400, 1.54049, -1.65163, -0.78845, -2.49828},
    };
    csv_.open("depth_calibration_samples.csv", std::ios::out | std::ios::app);
  }

private:
  /* goal callbacks */
  rclcpp_action::GoalResponse handleGoal(const rclcpp_action::GoalUUID &,
                                         CameraCalibrate::Goal::ConstSharedPtr goal) const
  {
    if (goal->command != "start") {
      RCLCPP_WARN(get_logger(),"Unknown command %s",goal->command.c_str());
      return rclcpp_action::GoalResponse::REJECT;
    }
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handleCancel(const std::shared_ptr<GoalHandle>){
    if (mg_) mg_->stop();
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handleAccept(const std::shared_ptr<GoalHandle> gh){
    std::thread(&ArucoCalibrationServer::execute,this,gh).detach();
  }

  void execute(std::shared_ptr<GoalHandle> gh)
  {
    auto feedback = std::make_shared<CameraCalibrate::Feedback>();
    for(size_t i=0;i<target_joints_.size();++i){
      const auto &pose = target_joints_[i];
      sample_ready_=false; color_ready_=false;

      feedback->status = "moving to pose "+std::to_string(i+1)+"/"+std::to_string(target_joints_.size());
      gh->publish_feedback(feedback);

      RCLCPP_INFO(get_logger(),"Moving to pose %zu", i+1);
      mg_->setJointValueTarget(pose);
      moveit::planning_interface::MoveGroupInterface::Plan p;
      if (mg_->plan(p)!=moveit::core::MoveItErrorCode::SUCCESS || mg_->execute(p)!=moveit::core::MoveItErrorCode::SUCCESS){
        RCLCPP_ERROR(get_logger(),"Motion failed at index %zu", i);
        auto res = std::make_shared<CameraCalibrate::Result>();
        res->success=false; res->message="motion failed";
        gh->abort(res); return;
      }

      RCLCPP_INFO(get_logger(),"Pose reached, settling 1 s");
      rclcpp::sleep_for(1s);

      feedback->status = "waiting for marker";
      gh->publish_feedback(feedback);
      auto deadline = now()+rclcpp::Duration::from_seconds(10.0);
      while(rclcpp::ok() && now()<deadline && !sample_ready_) rclcpp::sleep_for(100ms);
      if(!sample_ready_){
        RCLCPP_WARN(get_logger(),"No marker within 10 s at pose %zu", i);
        continue; // skip sample
      }

      logSample();
      writeSample(pose);
    }

    auto res = std::make_shared<CameraCalibrate::Result>();
    res->success=true; res->message="all samples captured";
    gh->succeed(res);
  }

  /* callbacks */
  void colorCb(const sensor_msgs::msg::CompressedImage::ConstSharedPtr &msg){
    if(sample_ready_||color_ready_) return;
    cv::Mat bgr=cv::imdecode(cv::Mat(msg->data),cv::IMREAD_COLOR);
    if(bgr.empty()) return;
    color_frame_=bgr.clone();
    std::vector<int> ids; std::vector<std::vector<cv::Point2f>> corners;
    cv::aruco::detectMarkers(bgr,dict_,corners,ids,params_);
    if(ids.empty()) return;
    corners_=corners[0];
    px_=cv::Point2f(0,0); for(auto &p:corners_) px_+=p; px_*=0.25f;
    color_ready_=true;
  }
  void depthCb(const sensor_msgs::msg::Image::ConstSharedPtr &msg){
    if(!color_ready_||sample_ready_) return;
    cv::Mat depth16; try{depth16=cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::TYPE_16UC1)->image;}catch(...){return;}
    depth_frame_=depth16.clone();
    int u=int(px_.x+0.5), v=int(px_.y+0.5); if(u<0||v<0||u>=depth16.cols||v>=depth16.rows) return;
    uint16_t d=depth16.at<uint16_t>(v,u); if(d==0) return; z_=d/1000.0;
    std::vector<cv::Point3f> obj={{-0.0225f,0.0225f,0},{0.0225f,0.0225f,0},{0.0225f,-0.0225f,0},{-0.0225f,-0.0225f,0}};
    cv::Vec3d rvec,tvec; 
#ifdef SOLVEPNP_IPPE_SQUARE
    cv::solvePnP(obj,corners_,K_,D_,rvec,tvec,false,cv::SOLVEPNP_IPPE_SQUARE);
#else
    cv::solvePnP(obj,corners_,K_,D_,rvec,tvec,false,cv::SOLVEPNP_ITERATIVE);
#endif
    x_=tvec[0]; y_=tvec[1]; z_solve_=tvec[2]; sample_ready_=true;
  }

  void logSample(){
    RCLCPP_INFO(get_logger(),"==== SAMPLE ====");
    RCLCPP_INFO(get_logger(),"Pixel u=%d  v=%d", int(px_.x+0.5), int(px_.y+0.5));
    RCLCPP_INFO(get_logger(),"Color SolvePnP  x=%.3f  y=%.3f  z=%.3f m", x_, y_, z_solve_);
    RCLCPP_INFO(get_logger(),"Depth camera     z=%.3f m", z_);
    double xp=(px_.x-K_.at<double>(0,2))*z_/K_.at<double>(0,0);
    double yp=(px_.y-K_.at<double>(1,2))*z_/K_.at<double>(1,1);
    RCLCPP_INFO(get_logger(),"Depth projection x=%.3f  y=%.3f m", xp, yp);

    // Calculate percentage errors between ArUco and depth measurements
    double z_error_pct = std::abs(z_solve_ - z_) / z_ * 100.0;
    double x_error_pct = std::abs(x_ - xp) / xp * 100.0;
    double y_error_pct = std::abs(y_ - yp) / yp * 100.0;
    
    RCLCPP_INFO(get_logger(),"Error percentages:");
    RCLCPP_INFO(get_logger(),"  X: %.2f%%", x_error_pct);
    RCLCPP_INFO(get_logger(),"  Y: %.2f%%", y_error_pct);
    RCLCPP_INFO(get_logger(),"  Z: %.2f%%", z_error_pct);

    // base_link <- aruco_link position
    try {
        auto tf = tf_buffer_.lookupTransform("base_link","aruco_link_rotated", tf2::TimePointZero, 50ms);
        RCLCPP_INFO(get_logger(),"aruco_link in base:  x=%.3f  y=%.3f  z=%.3f m",
                    tf.transform.translation.x,
                    tf.transform.translation.y,
                    tf.transform.translation.z);
    } catch (const tf2::TransformException &e) {
        RCLCPP_WARN(get_logger(),"TF lookup failed: %s", e.what());
    }

    /* visualisation */
    if (!color_frame_.empty()) {
        cv::Mat vis = color_frame_.clone();
        cv::circle(vis, px_, 5, {0,0,255}, -1);
        cv::imshow("color_marker", vis);
    }
    if (!depth_frame_.empty()) {
        cv::Mat depth_vis; cv::convertScaleAbs(depth_frame_, depth_vis, 255.0/4000.0);
        cv::applyColorMap(depth_vis, depth_vis, cv::COLORMAP_JET);
        cv::circle(depth_vis, px_, 5, {0,0,255}, -1);
        cv::imshow("depth_marker", depth_vis);
    }
    cv::waitKey(1);
  }

  void writeSample(const std::vector<double>& joint){
      if(!csv_.is_open()){
          std::string path = "depth_calibration_samples.csv";
          csv_.open(path, std::ios::out | std::ios::app);
      }
      if(!csv_header_written_){
          csv_ << "j1,j2,j3,j4,j5,j6,u,v,color_x,color_y,color_z,depth_z,depth_x,depth_y,base_x,base_y,base_z\n";
          csv_header_written_ = true;
      }
      double xp=(px_.x-K_.at<double>(0,2))*z_/K_.at<double>(0,0);
      double yp=(px_.y-K_.at<double>(1,2))*z_/K_.at<double>(1,1);
      double bx=0,by=0,bz=0;
      try{
          auto tf = tf_buffer_.lookupTransform("base_link","aruco_link_rotated", tf2::TimePointZero, 50ms);
          bx=tf.transform.translation.x;
          by=tf.transform.translation.y;
          bz=tf.transform.translation.z;
      }catch(const tf2::TransformException &){/* leave zeros */}

      csv_ << joint[0] << ',' << joint[1] << ',' << joint[2] << ',' << joint[3] << ',' << joint[4] << ',' << joint[5] << ','
           << int(px_.x+0.5) << ',' << int(px_.y+0.5) << ','
           << x_ << ',' << y_ << ',' << z_solve_ << ','
           << z_ << ',' << xp << ',' << yp << ','
           << bx << ',' << by << ',' << bz << '\n';
  }

  /* members */
  rclcpp_action::Server<CameraCalibrate>::SharedPtr server_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> mg_;
  image_transport::Subscriber depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr color_sub_;
  cv::Ptr<cv::aruco::Dictionary> dict_; cv::Ptr<cv::aruco::DetectorParameters> params_;
  cv::Mat K_,D_;
  bool color_ready_=false,sample_ready_=false; cv::Point2f px_; std::vector<cv::Point2f> corners_;
  double z_=0,x_=0,y_=0,z_solve_=0;
  cv::Mat color_frame_, depth_frame_;
  rclcpp::TimerBase::SharedPtr init_timer_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::vector<std::vector<double>> target_joints_;
  std::ofstream csv_; bool csv_header_written_ = false;
};

int main(int argc,char** argv){rclcpp::init(argc,argv);
    rclcpp::spin(std::make_shared<ArucoCalibrationServer>());
    rclcpp::shutdown();
    return 0;
}
