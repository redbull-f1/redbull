cmake_minimum_required(VERSION 3.8)
project(spline_planner_dh_test)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
find_package(ament_index_cpp REQUIRED)
find_package(yaml-cpp REQUIRED)

# ★ messages are provided by pl_msg
find_package(pl_msg REQUIRED)

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
  rclcpp rclcpp_components
  std_msgs geometry_msgs nav_msgs visualization_msgs
  tf2 tf2_ros tf2_geometry_msgs
  ament_index_cpp
  pl_msg
)

# Global Waypoints Publisher Node
add_executable(global_waypoints_publisher
  src/global_waypoints_publisher.cpp
)
ament_target_dependencies(global_waypoints_publisher
  rclcpp std_msgs geometry_msgs ament_index_cpp
  pl_msg
)

# Waypoint Visualizer Node
add_executable(waypoint_visualizer
  src/waypoint_visualizer.cpp
)
ament_target_dependencies(waypoint_visualizer
  rclcpp std_msgs geometry_msgs visualization_msgs ament_index_cpp
  pl_msg
)
target_link_libraries(waypoint_visualizer yaml-cpp)

# Static Obstacle Publisher Node
add_executable(static_obstacle_publisher
  src/static_obstacle_publisher.cpp
)
ament_target_dependencies(static_obstacle_publisher
  rclcpp std_msgs geometry_msgs
  pl_msg
)
target_link_libraries(static_obstacle_publisher yaml-cpp)

# Install targets
install(TARGETS
  spline_planner_node
  global_waypoints_publisher
  waypoint_visualizer
  static_obstacle_publisher
  DESTINATION lib/${PROJECT_NAME}
)

# Install resources
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY waypoints DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
