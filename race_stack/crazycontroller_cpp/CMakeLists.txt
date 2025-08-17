cmake_minimum_required(VERSION 3.16)
project(crazycontroller_cpp)

# -------------------- dependencies --------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(f110_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(tf2_ros REQUIRED)

# -------------------- include dirs --------------------
set(PROJECT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# -------------------- libraries --------------------
# steering_lookup (lookup_steer_angle.cpp)
add_library(steering_lookup
  src/utils/steering_lookup/lookup_steer_angle.cpp
)
target_include_directories(steering_lookup PUBLIC
  $<BUILD_INTERFACE:${PROJECT_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(steering_lookup PUBLIC Eigen3::Eigen)
target_compile_features(steering_lookup PUBLIC cxx_std_17)

# parameter_event_handler (header-only, 의존성 연결하지 않음)
add_library(parameter_event_handler INTERFACE)
target_include_directories(parameter_event_handler INTERFACE
  $<BUILD_INTERFACE:${PROJECT_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>)

# map controller (map.cpp)
add_library(map_controller
  src/map.cpp
)
target_include_directories(map_controller PUBLIC
  $<BUILD_INTERFACE:${PROJECT_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(map_controller PUBLIC
  steering_lookup
  Eigen3::Eigen
)
target_compile_features(map_controller PUBLIC cxx_std_17)

# controller_manager (node logic)
add_library(controller_manager
  src/controller_manager.cpp
)
target_include_directories(controller_manager PUBLIC
  $<BUILD_INTERFACE:${PROJECT_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>)

# ✅ plain signature로 통일 (ament 매크로와 혼용 에러 방지)
target_link_libraries(controller_manager
  map_controller
  steering_lookup
  parameter_event_handler
  Eigen3::Eigen
  yaml-cpp
)

# ✅ ROS 의존성은 ament가 관리
ament_target_dependencies(controller_manager
  rclcpp
  rcl_interfaces
  ackermann_msgs
  f110_msgs
  nav_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
)
target_compile_features(controller_manager PUBLIC cxx_std_17)

# -------------------- executable --------------------
add_executable(controller_manager_node
  src/controller_manager_node.cpp
)
target_link_libraries(controller_manager_node controller_manager)

# -------------------- install --------------------
install(DIRECTORY include/ DESTINATION include)

install(TARGETS
  steering_lookup
  parameter_event_handler
  map_controller
  controller_manager
  controller_manager_node
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME})
install(FILES resource/${PROJECT_NAME} DESTINATION share/ament_index/resource_index/packages)
install(FILES package.xml DESTINATION share/${PROJECT_NAME})

# -------------------- ament export --------------------
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(
  rclcpp rcl_interfaces ackermann_msgs f110_msgs nav_msgs
  tf2 tf2_geometry_msgs Eigen3 yaml-cpp
)
ament_export_include_directories(include)

ament_package()
