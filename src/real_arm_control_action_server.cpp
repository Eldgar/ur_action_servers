#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "ur_action_servers/action/arm_control.hpp"

using ArmControl  = ur_action_servers::action::ArmControl;
using GoalHandle  = rclcpp_action::ServerGoalHandle<ArmControl>;

class ArmControlServer : public rclcpp::Node
{
public:
  ArmControlServer()
  : Node("arm_control_server"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_),
    psi_() {}

  void initialize()
  {
    move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
        shared_from_this(), "ur_manipulator");
    move_group_->setPlanningTime(10.0);
    move_group_->setEndEffectorLink("tool0");

    action_server_ = rclcpp_action::create_server<ArmControl>(
        shared_from_this(), "arm_control",
        std::bind(&ArmControlServer::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
        std::bind(&ArmControlServer::handle_cancel, this, std::placeholders::_1),
        std::bind(&ArmControlServer::handle_accepted, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "✅ Arm Control Action Server initialized.");
  }

private:
  rclcpp_action::GoalResponse handle_goal(const rclcpp_action::GoalUUID &,
                                          std::shared_ptr<const ArmControl::Goal> goal)
  {
    RCLCPP_INFO(get_logger(), "🎯 Received goal: '%s'", goal->command.c_str());
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handle_cancel(const std::shared_ptr<GoalHandle>)
  {
    RCLCPP_WARN(get_logger(), "⏹ Goal was cancelled.");
    move_group_->stop();
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_accepted(const std::shared_ptr<GoalHandle> gh)
  {
    std::thread{std::bind(&ArmControlServer::execute, this, gh)}.detach();
  }

  // ────────────────────────────────────────────────────────────────────────────
  // MAIN EXECUTION
  // ────────────────────────────────────────────────────────────────────────────
  void execute(const std::shared_ptr<GoalHandle> gh)
  {
    const auto goal   = gh->get_goal();
    auto       result = std::make_shared<ArmControl::Result>();
    auto       fb     = std::make_shared<ArmControl::Feedback>();

    // Safety: ensure collision objects are present
    if (psi_.getKnownObjectNames().empty())
    {
      result->success = false;
      result->message = "❌ No collision objects present; aborting for safety";
      gh->abort(result);
      RCLCPP_ERROR(get_logger(), "%s", result->message.c_str());
      return;
    }

    // ── DIRECT PICKUP HANDLING ────────────────────────────────────────────────
    if (goal->command.rfind("pickup", 0) == 0)
    {
      // 0) Move through HOME → pick_and_place to establish desired orientation
      const std::vector<double> home_pose        = {0.0, -1.5708, 0.0, -1.5708, 0.0, -1.5708};
      const std::vector<double> pick_place_pose  = {0.00000, -2.02190, -0.92046, -1.57077, 0.00022, -1.57071};

      auto goToJoints = [&](const std::vector<double>& joints)->bool {
        move_group_->setJointValueTarget(joints);
        moveit::planning_interface::MoveGroupInterface::Plan p;
        if (move_group_->plan(p) != moveit::core::MoveItErrorCode::SUCCESS)
          return false;
        return move_group_->execute(p) == moveit::core::MoveItErrorCode::SUCCESS;
      };

      if (!goToJoints(home_pose)) {
        result->success = false;
        result->message = "❌ Failed to move to home pose";
        gh->abort(result);
        return;
      }

      if (!goToJoints(pick_place_pose)) {
        result->success = false;
        result->message = "❌ Failed to move to pick_and_place pose";
        gh->abort(result);
        return;
      }

      // 1) lookup pickup target TF
      int idx = std::stoi(goal->command.substr(6));            // "pickup<N>"
      geometry_msgs::msg::TransformStamped tf;
      try
      {
        tf = tf_buffer_.lookupTransform("base_link",
                                        "pickup_target_" + std::to_string(idx) + "_top",
                                        rclcpp::Time(0), tf2::durationFromSec(1.0));
      }
      catch (const tf2::TransformException &e)
      {
        result->success = false;
        result->message = std::string("TF lookup failed: ") + e.what();
        gh->abort(result);
        return;
      }

      // 2) build pose – replace orientation
      geometry_msgs::msg::Pose target;
      target.position.x = tf.transform.translation.x;
      target.position.y = tf.transform.translation.y;
      target.position.z = tf.transform.translation.z;
      target.orientation.x = 0.7071067;
      target.orientation.y = -0.7071067;
      target.orientation.z = 0.0;
      target.orientation.w = 0.0;

      // 3) Phase A – plan to HOVER pose (5 cm above)
      geometry_msgs::msg::Pose hover = target;
      hover.position.z += 0.07;   // 7 cm above target

      fb->status = "Planning hover";
      gh->publish_feedback(fb);

      move_group_->setPoseTarget(hover, "tool0");
      move_group_->setStartStateToCurrentState();
      moveit::planning_interface::MoveGroupInterface::Plan hover_plan;
      if (move_group_->plan(hover_plan) != moveit::core::MoveItErrorCode::SUCCESS)
      {
        result->success = false;
        result->message = "❌ Planning hover pose failed";
        gh->abort(result);
        return;
      }

      fb->status = "Moving to hover";
      gh->publish_feedback(fb);
      if (move_group_->execute(hover_plan) != moveit::core::MoveItErrorCode::SUCCESS)
      {
        result->success = false;
        result->message = "❌ Hover execution failed";
        gh->abort(result);
        return;
      }

      // 4) Phase B – Cartesian descend 5 cm
      geometry_msgs::msg::Pose mid = target;
      mid.position.z += 0.035;   // 3.5 cm above final target
      std::vector<geometry_msgs::msg::Pose> wps{hover, mid, target};
      moveit_msgs::msg::RobotTrajectory traj;
      double fraction = move_group_->computeCartesianPath(
          wps,
          0.001,   // eef_step 5 mm
          0.0,     // jump_threshold = 0 → reject large joint jumps
          traj,
          true);   // avoid_collisions

      if (fraction < 0.90)
      {
        result->success = false;
        result->message = "❌ Cartesian path fraction " + std::to_string(fraction);
        gh->abort(result);
        return;
      }

      fb->status = "Descending";
      gh->publish_feedback(fb);
      if (move_group_->execute(traj) != moveit::core::MoveItErrorCode::SUCCESS)
      {
        result->success = false;
        result->message = "❌ Cartesian execution failed";
        gh->abort(result);
        return;
      }

      result->success = true;
      result->message = "✅ Picked up target " + std::to_string(idx);
      gh->succeed(result);
      return;
    }

    // ── PRESET JOINT POSITIONS ───────────────────────────────────────────────
    std::vector<double> joints;
    if (goal->command == "go_home")
      joints = {0.0, -1.5708, 0.0, -1.5708, 0.0, -1.5708};
    else if (goal->command == "initial")
      joints = {2.87216, -0.67663, 1.50457, -1.34811, -0.26294, -2.60747};
    else if (goal->command == "out_of_view")
      joints = {2.26245, -0.68643, 1.42895, -0.68210, -0.85862, -1.62380};
    else if (goal->command == "pick_and_place")
      joints = {0.24884, -2.47939, -1.3856, -0.8579, 1.5721, 4.97611};
    else
    {
      result->success = false;
      result->message = "❌ Unknown command: " + goal->command;
      gh->abort(result);
      return;
    }

    fb->status = "Planning";
    gh->publish_feedback(fb);

    move_group_->setJointValueTarget(joints);
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    if (move_group_->plan(plan) != moveit::core::MoveItErrorCode::SUCCESS)
    {
      result->success = false;
      result->message = "❌ Planning failed";
      gh->abort(result);
      return;
    }

    fb->status = "Executing";
    gh->publish_feedback(fb);
    move_group_->execute(plan);

    if (gh->is_canceling())
    {
      result->success = false;
      result->message = "🛑 Goal cancelled";
      gh->canceled(result);
      return;
    }

    fb->status = "Complete";
    gh->publish_feedback(fb);
    result->success = true;
    result->message = "✅ Successfully executed: " + goal->command;
    gh->succeed(result);
  }

  // ────────────────────────────────────────────────────────────────────────────
  rclcpp_action::Server<ArmControl>::SharedPtr                   action_server_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  moveit::planning_interface::PlanningSceneInterface psi_;
  tf2_ros::Buffer             tf_buffer_;
  tf2_ros::TransformListener  tf_listener_;
};

// ────────────────────────────────────────────────────────────────────────────
int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ArmControlServer>();
  node->initialize();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
