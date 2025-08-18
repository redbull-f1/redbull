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
# find_package(yaml-cpp REQUIRED)        # waypoint_visualizer에서 사용 → (이제 사용 안 함)
find_package(f110_msgs REQUIRED)

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
  f110_msgs
)

# Global Waypoints Publisher Node (시각화까지 통합)
add_executable(global_waypoints_publisher
  src/global_waypoints_publisher.cpp
)
ament_target_dependencies(global_waypoints_publisher
  rclcpp std_msgs geometry_msgs visualization_msgs ament_index_cpp
  f110_msgs
)
# target_link_libraries(global_waypoints_publisher yaml-cpp)  # 필요 없음

# ===== 삭제한 노드/타겟 완전 제거 =====
# (waypoint_visualizer, static_obstacle_publisher) 전부 삭제

# Install targets
install(TARGETS
  spline_planner_node
  global_waypoints_publisher
  DESTINATION lib/${PROJECT_NAME}
)

# Install resources (존재할 때만)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/launch)
  install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/waypoints)
  install(DIRECTORY waypoints DESTINATION share/${PROJECT_NAME})
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config)
  install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
