/*********************************************************************
 * camera_frame_cartesian_server.cpp
 * Action server that accepts Cartesian position OFFSETS expressed in
 * D415_color_optical_frame, plans a Cartesian path
 * to the resulting pose (keeping current orientation),
 * and executes it with basic safety checks.
 * Includes diagnostic logging in handle_accept thread.
 *********************************************************************/

#include "ur_action_servers/action/camera_move.hpp" // Action definition

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <moveit/robot_state/robot_state.h> // Needed for error codes potentially

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp> // Required for tf2::transform
#include <tf2_eigen/tf2_eigen.hpp>               

#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <thread>
#include <vector>

// Namespaces and using declarations
using namespace std::placeholders;
using namespace std::chrono_literals;
using CameraMove = ur_action_servers::action::CameraMove;
using GoalHandle = rclcpp_action::ServerGoalHandle<CameraMove>;

class CameraCartesianServer : public rclcpp::Node
{
public:
    CameraCartesianServer()
        : Node("camera_cartesian_server"),
          tf_buffer_(this->get_clock()),
          tf_listener_(tf_buffer_) // Initialize TF listener with the buffer
    {
        /* One-shot timer: fires after the node is fully managed by shared_ptr */
        // Using 1s delay as per your modification - good for stability
        init_timer_ = create_wall_timer(
            1s, std::bind(&CameraCartesianServer::late_init, this));
        RCLCPP_INFO(get_logger(), "CameraCartesianServer node constructed.");
    }

private:
    /******************* LATE INITIALISATION  *************************/
    void late_init()
    {
        init_timer_->cancel(); // Run only once
        RCLCPP_INFO(get_logger(), "Late initialization started...");

        /* 1. MoveGroupInterface */
        try
        {
            move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
                shared_from_this(), "ur_manipulator");
            move_group_->setEndEffectorLink("tool0");
            move_group_->setMaxVelocityScalingFactor(0.1);     // Adjust as needed
            move_group_->setMaxAccelerationScalingFactor(0.1); // Adjust as needed
            move_group_->setGoalOrientationTolerance(0.35);   // rad (relax to aid IK)
            move_group_->setGoalPositionTolerance(0.005);     // 5 mm
            move_group_->setNumPlanningAttempts(5);
            move_group_->setPlanningTime(5.0);
            RCLCPP_INFO(get_logger(), "MoveGroupInterface initialized for group '%s'.", move_group_->getName().c_str());
        }
        catch (const std::exception &e)
        {
            RCLCPP_FATAL(get_logger(), "Failed to initialize MoveGroupInterface: %s. Shutting down.", e.what());
            // Can't operate without MoveGroup, maybe shutdown?
            if (rclcpp::ok()) { rclcpp::shutdown(); }
            return;
        }

        /* 2. Action server */
        action_server_ = rclcpp_action::create_server<CameraMove>(
            shared_from_this(), // Use existing node patterns
            "camera_move",      // Action name
            // Callbacks using std::bind
            std::bind(&CameraCartesianServer::handle_goal, this, _1, _2),
            std::bind(&CameraCartesianServer::handle_cancel, this, _1),
            std::bind(&CameraCartesianServer::handle_accept, this, _1));

        RCLCPP_INFO(get_logger(), "MoveGroup and action server ready");
    }

    /********************* HELPERS  **********************************/
    // Helper to create result messages easily
    std::shared_ptr<CameraMove::Result> make_result(
        bool success, const std::string &msg) const
    {
        auto res = std::make_shared<CameraMove::Result>();
        res->success = success;
        res->message = msg;
        return res;
    }

    /********************* ACTION CALLBACKS  *************************/
    // Decide whether to accept the goal
    rclcpp_action::GoalResponse handle_goal(
        const rclcpp_action::GoalUUID &,
        CameraMove::Goal::ConstSharedPtr goal) const
    {
        // Validate the requested frame_id
        // Allow empty frame_id to mean the camera frame for backward compatibility? No, enforce.
        if (goal->target_pose.header.frame_id != "wrist_rgbd_camera_depth_optical_frame")
        {
            RCLCPP_WARN(get_logger(),
                        "Rejected goal: frame_id must be 'wrist_rgbd_camera_depth_optical_frame', but was '%s'",
                        goal->target_pose.header.frame_id.c_str());
            return rclcpp_action::GoalResponse::REJECT;
        }
        RCLCPP_INFO(get_logger(), "Goal accepted: Requesting relative move X:%.3f Y:%.3f Z:%.3f in %s",
                    goal->target_pose.pose.position.x, goal->target_pose.pose.position.y, goal->target_pose.pose.position.z,
                    goal->target_pose.header.frame_id.c_str());
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }

    // Handle cancellation requests
    rclcpp_action::CancelResponse handle_cancel(
        const std::shared_ptr<GoalHandle>)
    {
        RCLCPP_INFO(get_logger(), "Goal cancel requested by client");
        if (move_group_)
        {
            RCLCPP_INFO(get_logger(), "Stopping MoveGroup execution...");
            move_group_->stop(); // Stop any current MoveIt execution
        }
        return rclcpp_action::CancelResponse::ACCEPT; // Acknowledge the cancel request
    }

    // Execute the accepted goal
    void handle_accept(const std::shared_ptr<GoalHandle> gh)
    {
        // Execute in a separate thread to avoid blocking the action server
        std::thread([this, gh]() {
            // <<< ADD LOG 1 >>>
            RCLCPP_INFO(get_logger(), "Handle accept thread started.");

            /* Make sure all late-init resources exist */
            if (!move_group_)
            {
                RCLCPP_ERROR(get_logger(), "Thread: MoveGroup not initialised yet!");
                gh->abort(make_result(false, "MoveGroup not initialised yet"));
                return;
            }

            const auto goal = gh->get_goal();
            const std::string &camera_frame = goal->target_pose.header.frame_id;
            // Get planning frame (usually "base_link")
            const std::string base_frame = move_group_->getPlanningFrame();

            // Declare poses needed for calculation scope
            geometry_msgs::msg::PoseStamped current_pose_base;
            geometry_msgs::msg::PoseStamped current_pose_camera;
            geometry_msgs::msg::PoseStamped target_pose_camera;
            geometry_msgs::msg::PoseStamped target_pose_base; // Final target in base frame

            /******** 1. Calculate Target Pose based on Relative Offset ********/
            try
            {
                // <<< ADD LOG 2 >>>
                RCLCPP_INFO(get_logger(), "Thread: Getting current pose...");
                // --- FIX 1 Start: Manually create PoseStamped ---
                // Get current EE pose (returns geometry_msgs::msg::Pose)
                geometry_msgs::msg::Pose current_pose_raw = move_group_->getCurrentPose().pose;
                // Manually create the PoseStamped message
                current_pose_base.header.frame_id = base_frame;
                current_pose_base.header.stamp = this->get_clock()->now(); // Use current time
                current_pose_base.pose = current_pose_raw;
                // --- FIX 1 End ---
                // <<< ADD LOG 3 >>>
                RCLCPP_INFO(get_logger(), "Thread: Current pose obtained.");

                RCLCPP_DEBUG(get_logger(), "Thread: Current EE Pose (base): X:%.3f Y:%.3f Z:%.3f",
                             current_pose_base.pose.position.x, current_pose_base.pose.position.y, current_pose_base.pose.position.z);

                // <<< ADD LOG 4 >>>
                RCLCPP_INFO(get_logger(), "Thread: Transforming current pose to camera frame (%s -> %s)...", base_frame.c_str(), camera_frame.c_str());
                // Ensure TF buffer has transforms - listener started in constructor
                current_pose_camera = tf_buffer_.transform(
                    current_pose_base, camera_frame, tf2::durationFromSec(1.0));
                // <<< ADD LOG 5 >>>
                RCLCPP_INFO(get_logger(), "Thread: Transform 1 successful.");

                RCLCPP_DEBUG(get_logger(), "Thread: Current EE Pose (camera): X:%.3f Y:%.3f Z:%.3f",
                             current_pose_camera.pose.position.x, current_pose_camera.pose.position.y, current_pose_camera.pose.position.z);

                // Calculate target pose in camera frame by applying goal offsets
                target_pose_camera = current_pose_camera; // Start with current pose/orientation
                // Apply position offsets from the goal message
                target_pose_camera.pose.position.x += goal->target_pose.pose.position.x;
                target_pose_camera.pose.position.y += goal->target_pose.pose.position.y;
                target_pose_camera.pose.position.z += goal->target_pose.pose.position.z;
                // IMPORTANT: Keep the current orientation (already copied)
                target_pose_camera.header.stamp = this->get_clock()->now(); // Update timestamp for the new target

                RCLCPP_DEBUG(get_logger(), "Thread: Target EE Pose (camera): X:%.3f Y:%.3f Z:%.3f",
                             target_pose_camera.pose.position.x, target_pose_camera.pose.position.y, target_pose_camera.pose.position.z);

                // <<< ADD LOG 6 >>>
                RCLCPP_INFO(get_logger(), "Thread: Transforming target pose back to base frame (%s -> %s)...", camera_frame.c_str(), base_frame.c_str());
                // Transform the calculated target pose back to the base frame
                target_pose_base = tf_buffer_.transform(
                    target_pose_camera, base_frame, tf2::durationFromSec(1.0));
                // <<< ADD LOG 7 >>>
                RCLCPP_INFO(get_logger(), "Thread: Transform 2 successful.");

                RCLCPP_DEBUG(get_logger(), "Thread: Target EE Pose (base): X:%.3f Y:%.3f Z:%.3f",
                             target_pose_base.pose.position.x, target_pose_base.pose.position.y, target_pose_base.pose.position.z);
            }
            catch (const tf2::TransformException &ex)
            {
                RCLCPP_ERROR(get_logger(), "Thread: TF Error during pose calculation: %s", ex.what());
                gh->abort(make_result(false, std::string("TF Error: ") + ex.what()));
                return;
            }
            catch (const std::exception &ex)
            { // Catch potential errors from MoveIt or elsewhere
                RCLCPP_ERROR(get_logger(), "Thread: Error during pose calculation: %s", ex.what());
                gh->abort(make_result(false, std::string("Error during pose calculation: ") + ex.what()));
                return;
            }

            // <<< ADD LOG 8 >>>
            RCLCPP_INFO(get_logger(), "Thread: Pose calculation section finished.");

            /******** 2. Linear distance clamp (Safety Check) ********/
            // Calculate the straight-line distance between start and calculated target pose
            const double dx = std::hypot(
                std::hypot(target_pose_base.pose.position.x - current_pose_base.pose.position.x,
                           target_pose_base.pose.position.y - current_pose_base.pose.position.y),
                target_pose_base.pose.position.z - current_pose_base.pose.position.z);
            RCLCPP_INFO(get_logger(), "Thread: Calculated move distance: %.3f m", dx);


            // Check if the calculated move distance exceeds the limit specified in the goal
            // Allow bypass if max_linear_step is <= 0.0
            if (goal->max_linear_step > 0.0 && dx > goal->max_linear_step)
            {
                RCLCPP_WARN(get_logger(), "Thread: Requested move distance (%.3f m) exceeds max_linear_step (%.3f m)", dx, goal->max_linear_step);
                gh->abort(make_result(false, "Requested move too large"));
                return;
            }

            /******** 3. Cartesian path planning ********/
            RCLCPP_INFO(get_logger(), "Thread: Starting Cartesian path planning...");
            move_group_->setPoseTarget(target_pose_base.pose, "rg2_gripper_base_link");
            moveit::planning_interface::MoveGroupInterface::Plan plan;
            auto plan_result = move_group_->plan(plan);

            if (plan_result != moveit::core::MoveItErrorCode::SUCCESS) {
                RCLCPP_WARN(get_logger(), "Thread: OMPL planning failed attempting direct Cartesian interpolation fallback");

                std::vector<geometry_msgs::msg::Pose> waypoints{target_pose_base.pose};
                moveit_msgs::msg::RobotTrajectory cart_traj;
                double fraction = move_group_->computeCartesianPath(
                    waypoints, /*eef_step=*/0.02, /*jump_threshold=*/0.0,
                    cart_traj, /*avoid_collisions=*/false);

                if (fraction < 0.95) {
                    RCLCPP_ERROR(get_logger(), "Thread: Cartesian interpolation also failed (fraction %.2f)", fraction);
                    gh->abort(make_result(false, "Planning failed no IK solution"));
                    return;
                }
                plan.trajectory_ = cart_traj;
                RCLCPP_INFO(get_logger(), "Thread: Cartesian fallback succeeded (fraction %.2f)", fraction);
            } else {
                RCLCPP_INFO(get_logger(), "Thread: OMPL plan found");
            }

            /******** 4. Execute trajectory with feedback ********/
            RCLCPP_INFO(get_logger(), "Thread: Executing trajectory...");
            // Send the planned trajectory to MoveIt for execution
            auto exec_result = move_group_->execute(plan);

            // Check execution result
            if (exec_result != moveit::core::MoveItErrorCode::SUCCESS)
            {
                RCLCPP_ERROR(get_logger(), "Thread: Trajectory execution failed with MoveIt error code: %d", exec_result.val);
                gh->abort(make_result(false, "Trajectory execution failed"));
                return;
            }
            RCLCPP_INFO(get_logger(), "Thread: Trajectory execution successful.");


            // Provide simple progress feedback (replace with actual monitoring if needed)
            RCLCPP_INFO(get_logger(), "Thread: Starting feedback loop...");
            rclcpp::Rate rate(20); // Feedback rate
            auto fb = std::make_shared<CameraMove::Feedback>();
            for (int i = 0; i <= 20 && rclcpp::ok(); ++i)
            {
                // --- FIX 2 Start: Use is_canceling() ---
                if (gh->is_canceling()) // Check if cancellation is actively processed
                // --- FIX 2 End ---
                {
                    RCLCPP_INFO(get_logger(), "Thread: Execution canceled during feedback loop.");
                    // Goal state already set by handle_cancel acceptance, just terminate
                    // MoveGroupInterface::stop() called in handle_cancel
                    gh->canceled(make_result(false, "Execution canceled")); // Mark goal as canceled
                    return;
                }
                fb->progress = static_cast<float>(i) / 20.0f;
                gh->publish_feedback(fb);
                rate.sleep();
            }
             RCLCPP_INFO(get_logger(), "Thread: Feedback loop finished.");

            // Check final pose? (Optional, requires getting current pose again)

            gh->succeed(make_result(true, "Goal reached"));
            RCLCPP_INFO(get_logger(), "Thread: Relative move goal succeeded.");

        }).detach(); // Detach the thread to let it run independently
    }

    /************************ MEMBERS  *******************************/
    rclcpp::TimerBase::SharedPtr init_timer_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    moveit::planning_interface::MoveGroupInterfacePtr move_group_; // Use Ptr type
    rclcpp_action::Server<CameraMove>::SharedPtr action_server_;
}; // End class CameraCartesianServer

/****************************** MAIN  ******************************/
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    // Instantiate and spin the node
    auto node = std::make_shared<CameraCartesianServer>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}







