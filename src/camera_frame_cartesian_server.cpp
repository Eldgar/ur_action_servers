/*********************************************************************
 * camera_frame_cartesian_server.cpp
 *
 * Action  : ur_action_servers/action/Rg2RelativeMove
 * Goal    : float64 x y z  (offsets in planning frame, metres)
 * Result  : bool success · string message
 * Feedback: float32 progress   (0‥1)
 *********************************************************************/

#include "ur_action_servers/action/rg2_relative_move.hpp"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_msgs/msg/robot_trajectory.hpp>

#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using Rg2Move    = ur_action_servers::action::Rg2RelativeMove;
using GoalHandle = rclcpp_action::ServerGoalHandle<Rg2Move>;
using namespace std::chrono_literals;
using std::placeholders::_1;
using std::placeholders::_2;

class RG2CartesianServer : public rclcpp::Node
{
public:
  RG2CartesianServer()
  : Node("rg2_cartesian_server")
  {
    init_timer_ = create_wall_timer(1s, std::bind(&RG2CartesianServer::late_init,this));
  }

private:
  /* late-init after shared_ptr exists */
  void late_init()
  {
    init_timer_->cancel();

    try {
      mg_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
               shared_from_this(), "ur_manipulator");
      mg_->setEndEffectorLink("tool0");
      mg_->setMaxVelocityScalingFactor(0.1);
      mg_->setMaxAccelerationScalingFactor(0.1);
      mg_->setPlanningTime(5.0);
    } catch (const std::exception &e) {
      RCLCPP_FATAL(get_logger(),"MoveGroup init failed: %s",e.what());
      rclcpp::shutdown();
      return;
    }

    server_ = rclcpp_action::create_server<Rg2Move>(
      shared_from_this(),"rg2_relative_move",
      std::bind(&RG2CartesianServer::handle_goal,  this,_1,_2),
      std::bind(&RG2CartesianServer::handle_cancel,this,_1),
      std::bind(&RG2CartesianServer::handle_accept,this,_1));

    RCLCPP_INFO(get_logger(),"Rg2 Cartesian server ready");
  }

  /* helpers */
  std::shared_ptr<Rg2Move::Result> make_result(bool ok,const std::string &msg)
  {
    auto r=std::make_shared<Rg2Move::Result>();
    r->success=ok; r->message=msg; return r;
  }

  /* goal/cancel */
  rclcpp_action::GoalResponse handle_goal(
      const rclcpp_action::GoalUUID&,
      Rg2Move::Goal::ConstSharedPtr goal) const
  {
    RCLCPP_INFO(get_logger(),
      "Offsets request  x:%.3f  y:%.3f  z:%.3f  (planning frame)",
      goal->x,goal->y,goal->z);
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handle_cancel(
      const std::shared_ptr<GoalHandle>)
  {
    if (mg_) mg_->stop();
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_accept(const std::shared_ptr<GoalHandle> gh)
  {
    std::thread(&RG2CartesianServer::execute,this,gh).detach();
  }

  /* main execution */
  void execute(const std::shared_ptr<GoalHandle> gh)
  {
    const auto goal=gh->get_goal();
    auto start_pose = mg_->getCurrentPose().pose;

    geometry_msgs::msg::Pose target = start_pose;
    target.position.x += goal->x;
    target.position.y += goal->y;
    target.position.z += goal->z;

    std::vector<geometry_msgs::msg::Pose> waypoints{target};
    moveit_msgs::msg::RobotTrajectory traj;
    double fraction = mg_->computeCartesianPath(
        waypoints, 0.005 /*eef_step*/, 0.0 /*jump_thresh*/, traj, true);

    if (fraction < 0.99) {
      gh->abort(make_result(false,"Cartesian planning failed"));
      return;
    }

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    plan.trajectory_ = traj;

    if (mg_->execute(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
      gh->abort(make_result(false,"Execution failed"));
      return;
    }

    /* simple progress feedback */
    auto fb = std::make_shared<Rg2Move::Feedback>();
    rclcpp::Rate r(20);
    for(int i=0;i<=20 && rclcpp::ok();++i){
      if (gh->is_canceling()){
        gh->canceled(make_result(false,"Canceled")); return;
      }
      fb->progress = static_cast<float>(i)/20.0f;
      gh->publish_feedback(fb);
      r.sleep();
    }

    gh->succeed(make_result(true,"Motion complete"));
  }

  /* members */
  rclcpp::TimerBase::SharedPtr init_timer_;
  moveit::planning_interface::MoveGroupInterfacePtr mg_;
  rclcpp_action::Server<Rg2Move>::SharedPtr server_;
};

/* ------------- main ------------- */
int main(int argc,char** argv)
{
  rclcpp::init(argc,argv);
  rclcpp::spin(std::make_shared<RG2CartesianServer>());
  rclcpp::shutdown();
  return 0;
}




