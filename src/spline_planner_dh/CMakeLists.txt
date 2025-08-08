cmake_minimum_required(VERSION 3.8)
project(spline_planner_dh)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate custom messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Wpnt.msg"
  "msg/WpntArray.msg"
  "msg/ObstacleWpnt.msg"
  "msg/ObstacleArray.msg"
  "msg/OTWpnt.msg"
  "msg/OTWpntArray.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

# Include directories
include_directories(include)

# Spline Planner Node
add_executable(spline_planner_node
  src/spline_planner_node.cpp
  src/spline_planner.cpp
  src/frenet_converter.cpp
  src/spline_interpolator.cpp
)

ament_target_dependencies(spline_planner_node
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

rosidl_target_interfaces(spline_planner_node ${PROJECT_NAME} "rosidl_typesupport_cpp")

# Global Waypoints Publisher Node
add_executable(global_waypoints_publisher
  src/global_waypoints_publisher.cpp
)

ament_target_dependencies(global_waypoints_publisher
  rclcpp
  std_msgs
  geometry_msgs
  visualization_msgs
)

rosidl_target_interfaces(global_waypoints_publisher ${PROJECT_NAME} "rosidl_typesupport_cpp")

# Install targets
install(TARGETS
  spline_planner_node
  global_waypoints_publisher
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# Install waypoints
install(DIRECTORY waypoints
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
