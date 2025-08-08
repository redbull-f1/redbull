cmake_minimum_required(VERSION 3.8)
project(lattice_planner_rb)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(ackermann_msgs REQUIRED)

# Generate interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Wpnt.msg"
  "msg/WpntArray.msg"
  "msg/OTWpnt.msg"
  "msg/OTWpntArray.msg"
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

# Include directories
include_directories(include)

# Add executable
add_executable(lattice_planner_node
  src/lattice_planner.cpp
  src/frenet_converter.cpp
  src/cubic_spline.cpp
)

# Add waypoints to path converter executable
add_executable(waypoints_to_path_converter
  src/waypoints_to_path_converter.cpp
)

# Add simple path follower executable
add_executable(simple_path_follower
  src/simple_path_follower.cpp
)

# Dependencies for the executable
ament_target_dependencies(lattice_planner_node
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  ackermann_msgs
)

# Dependencies for waypoints converter
ament_target_dependencies(waypoints_to_path_converter
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  tf2
  tf2_geometry_msgs
)

# Dependencies for simple path follower
ament_target_dependencies(simple_path_follower
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  ackermann_msgs
)

# Link against the generated interfaces
rosidl_target_interfaces(lattice_planner_node ${PROJECT_NAME} "rosidl_typesupport_cpp")

# Install executable
install(TARGETS
  lattice_planner_node
  waypoints_to_path_converter
  simple_path_follower
  DESTINATION lib/${PROJECT_NAME}
)

# Install directories
install(DIRECTORY
  waypoints/
  launch/
  config/
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
