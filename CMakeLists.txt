cmake_minimum_required(VERSION 3.8)
project(ur_action_servers)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ─── Dependencies ─────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Messages & Math
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

# TF & Planning
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_common REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)

# CV if used
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)

# ─── Generate Actions ─────────────────────────────────────────
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/CameraMove.action"
  "action/ArmControl.action"
  "action/CameraCalibrate.action"
  "action/Rg2RelativeMove.action"
  DEPENDENCIES builtin_interfaces geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)

# ─── Executables ──────────────────────────────────────────────

add_executable(arm_control_action_server src/arm_control_action_server.cpp)
ament_target_dependencies(arm_control_action_server
  rclcpp rclcpp_action moveit_ros_planning_interface)
rosidl_target_interfaces(arm_control_action_server
  ${PROJECT_NAME} "rosidl_typesupport_cpp")

add_executable(real_arm_control_action_server src/real_arm_control_action_server.cpp)
ament_target_dependencies(real_arm_control_action_server
  rclcpp rclcpp_action moveit_ros_planning_interface)
rosidl_target_interfaces(real_arm_control_action_server
  ${PROJECT_NAME} "rosidl_typesupport_cpp")

add_executable(sim_camera_calibration src/sim_camera_calibration.cpp)
ament_target_dependencies(sim_camera_calibration
  rclcpp rclcpp_action geometry_msgs sensor_msgs
  tf2 tf2_ros tf2_eigen cv_bridge OpenCV
  moveit_core moveit_ros_planning_interface image_transport)
rosidl_target_interfaces(sim_camera_calibration
  ${PROJECT_NAME} "rosidl_typesupport_cpp")

add_executable(real_camera_calibration src/real_camera_calibration.cpp)
ament_target_dependencies(real_camera_calibration
  rclcpp rclcpp_action geometry_msgs sensor_msgs
  tf2 tf2_ros tf2_eigen cv_bridge OpenCV
  moveit_core moveit_ros_planning_interface image_transport)
rosidl_target_interfaces(real_camera_calibration
  ${PROJECT_NAME} "rosidl_typesupport_cpp")


add_executable(real_depth_calibration src/real_depth_calibration.cpp)
ament_target_dependencies(real_depth_calibration
  rclcpp rclcpp_action geometry_msgs sensor_msgs
  tf2 tf2_ros tf2_eigen cv_bridge OpenCV
  moveit_core moveit_ros_planning_interface image_transport)
rosidl_target_interfaces(real_depth_calibration
  ${PROJECT_NAME} "rosidl_typesupport_cpp")

add_executable(camera_cartesian_server src/camera_frame_cartesian_server.cpp)
ament_target_dependencies(camera_cartesian_server
  rclcpp rclcpp_action geometry_msgs sensor_msgs
  tf2 tf2_ros tf2_eigen tf2_geometry_msgs
  moveit_core moveit_ros_planning_interface moveit_msgs
  ament_index_cpp Eigen3)
rosidl_target_interfaces(camera_cartesian_server
  ${PROJECT_NAME} "rosidl_typesupport_cpp")

# add_executable(camera_cartesian_server_sim src/camera_frame_cartesian_server_sim.cpp)
# ament_target_dependencies(camera_cartesian_server_sim
#   rclcpp rclcpp_action geometry_msgs sensor_msgs
#   tf2 tf2_ros tf2_eigen tf2_geometry_msgs
#   moveit_core moveit_ros_planning_interface moveit_msgs
#   ament_index_cpp Eigen3)
# rosidl_target_interfaces(camera_cartesian_server_sim
#   ${PROJECT_NAME} "rosidl_typesupport_cpp")


install(TARGETS
  arm_control_action_server
  real_arm_control_action_server
  sim_camera_calibration
  real_camera_calibration
  camera_cartesian_server
  real_depth_calibration
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

install(DIRECTORY action
  DESTINATION share/${PROJECT_NAME}
)

ament_package()




