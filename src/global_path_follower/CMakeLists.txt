cmake_minimum_required(VERSION 3.8)
project(global_path_follower)

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
find_package(ackermann_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Wpnt.msg"
  "msg/WpntArray.msg"
  "msg/OTWpnt.msg"
  "msg/OTWpntArray.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

# Include directories
include_directories(include)

# Create shared library
add_library(${PROJECT_NAME}_lib SHARED
  src/global_path_follower.cpp
  src/waypoint_loader.cpp
)

target_include_directories(${PROJECT_NAME}_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(${PROJECT_NAME}_lib
  rclcpp
  rclcpp_components
  std_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  ackermann_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

# Link the generated interfaces
rosidl_target_interfaces(${PROJECT_NAME}_lib ${PROJECT_NAME} "rosidl_typesupport_cpp")

# Register the component
rclcpp_components_register_nodes(${PROJECT_NAME}_lib "global_path_follower::GlobalPathFollower")

# Create executable
add_executable(${PROJECT_NAME}_node src/main.cpp)
target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME}_lib)

ament_target_dependencies(${PROJECT_NAME}_node
  rclcpp
)

# Install
install(TARGETS ${PROJECT_NAME}_lib
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS ${PROJECT_NAME}_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

install(DIRECTORY
  waypoints
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(rclcpp rclcpp_components std_msgs geometry_msgs nav_msgs visualization_msgs ackermann_msgs tf2 tf2_ros tf2_geometry_msgs rosidl_default_runtime)

ament_package()
