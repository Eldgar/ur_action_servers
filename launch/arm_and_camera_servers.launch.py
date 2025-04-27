# launch/real_arm_and_camera_servers.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ────────────── launch-arg (default: false) ──────────────
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time", default_value="false",
        description="Clock source for all nodes (true = /clock topic)")

    use_sim_time = LaunchConfiguration("use_sim_time")

    # ────────────── action-servers ──────────────
    arm_control = Node(
        package="ur_action_servers",
        executable="real_arm_control_action_server",
        name="arm_control_server",
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen")

    camera_calib = Node(
        package="ur_action_servers",
        executable="real_camera_calibration",
        name="camera_calibration_server",
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen")

    camera_cartesian = Node(
        package="ur_action_servers",
        executable="camera_cartesian_server",
        name="camera_cartesian_server",
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen")

    # ────────────── launch description ──────────────
    return LaunchDescription([
        use_sim_time_arg,
        arm_control,
        camera_calib,
        camera_cartesian,
    ])
