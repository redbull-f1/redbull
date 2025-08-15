cmake_minimum_required(VERSION 3.8)
project(pl_msg)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# ★ 메시지에서 쓰는 의존들을 반드시 find_package 해줘야 함
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
# 메시지에 직접 쓰면 활성화
# find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/FrenetDebug.msg"
  "msg/FrenetDebugArray.msg"
  "msg/ObstacleArray.msg"
  "msg/ObstacleWpnt.msg"
  "msg/OTWpnt.msg"
  "msg/OTWpntArray.msg"
  "msg/Wpnt.msg"
  "msg/WpntArray.msg"
  DEPENDENCIES std_msgs geometry_msgs   # (+ builtin_interfaces 사용 시 추가)
)

install(DIRECTORY msg DESTINATION share/${PROJECT_NAME})

ament_export_dependencies(rosidl_default_runtime)
ament_package()
