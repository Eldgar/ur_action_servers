#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <moveit/move_group_interface/move_group_interface.h>

#include "ur_action_servers/action/arm_control.hpp"

using ArmControl = ur_action_servers::action::ArmControl;
using GoalHandle = rclcpp_action::ServerGoalHandle<ArmControl>;

class ArmControlServer : public rclcpp::Node
{
public:
    ArmControlServer()
        : Node("arm_control_server")
    {}

    void initialize()
    {
        move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), "ur_manipulator");
        move_group_->setPlanningTime(10.0);

        action_server_ = rclcpp_action::create_server<ArmControl>(
            shared_from_this(),
            "arm_control",
            std::bind(&ArmControlServer::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
            std::bind(&ArmControlServer::handle_cancel, this, std::placeholders::_1),
            std::bind(&ArmControlServer::handle_accepted, this, std::placeholders::_1)
        );

        RCLCPP_INFO(this->get_logger(), "✅ Arm Control Action Server initialized.");
    }

private:
    rclcpp_action::Server<ArmControl>::SharedPtr action_server_;
    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;

    rclcpp_action::GoalResponse handle_goal(
        const rclcpp_action::GoalUUID &,
        std::shared_ptr<const ArmControl::Goal> goal)
    {
        RCLCPP_INFO(this->get_logger(), "🎯 Received goal: '%s'", goal->command.c_str());
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }

    rclcpp_action::CancelResponse handle_cancel(const std::shared_ptr<GoalHandle> /*goal_handle*/)
    {
        RCLCPP_WARN(this->get_logger(), "⏹ Goal was cancelled.");
        move_group_->stop();
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    void handle_accepted(const std::shared_ptr<GoalHandle> goal_handle)
    {
        std::thread{std::bind(&ArmControlServer::execute, this, goal_handle)}.detach();
    }

    void execute(const std::shared_ptr<GoalHandle> goal_handle)
    {
        const auto goal = goal_handle->get_goal();
        auto result = std::make_shared<ArmControl::Result>();
        auto feedback = std::make_shared<ArmControl::Feedback>();

        std::vector<double> target_joint_values;

        if (goal->command == "go_home") {
            target_joint_values = {0.0, -1.5708, 0.0, -1.5708, 0.0, -1.5708};
        } else if (goal->command == "initial") {
            target_joint_values = {2.72271, -0.087, 1.740, -1.535, -0.40, -1.69297};
        } else if (goal->command == "out_of_view") {
            target_joint_values = {2.26245, -0.68643, 1.42895, -0.68210, -0.85862, -1.62380};
        } else {
            result->success = false;
            result->message = "❌ Unknown command: " + goal->command;
            goal_handle->abort(result);
            return;
        }

        feedback->status = "Planning";
        goal_handle->publish_feedback(feedback);

        move_group_->setStartStateToCurrentState();
        move_group_->setJointValueTarget(target_joint_values);
        moveit::planning_interface::MoveGroupInterface::Plan plan;

        if (move_group_->plan(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
            result->success = false;
            result->message = "❌ Planning failed.";
            goal_handle->abort(result);
            return;
        }

        feedback->status = "Executing";
        goal_handle->publish_feedback(feedback);

        move_group_->execute(plan);

        if (goal_handle->is_canceling()) {
            result->success = false;
            result->message = "🛑 Goal cancelled.";
            goal_handle->canceled(result);
            return;
        }

        feedback->status = "Complete";
        goal_handle->publish_feedback(feedback);

        result->success = true;
        result->message = "✅ Successfully executed: " + goal->command;
        goal_handle->succeed(result);
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<ArmControlServer>();
    node->initialize();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}


