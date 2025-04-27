#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    # Build the MoveIt config for your UR3e
    moveit_cfg = MoveItConfigsBuilder(
        "ur3e",
        package_name="real_starbot_moveit_config"
    ).to_moveit_configs()

    return LaunchDescription([
        Node(
            package="ur_action_servers",    
            executable="camera_cartesian_server",
            name="camera_cartesian_server",
            output="screen",
            parameters=[ # List starts here
                moveit_cfg.robot_description,
                moveit_cfg.robot_description_semantic,
                moveit_cfg.robot_description_kinematics,
                {"use_sim_time": True}, 
            ]
        )
    ])

