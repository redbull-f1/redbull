cmake_minimum_required(VERSION 3.8)
project(pl_msg)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# 메시지에서 사용하는 의존 패키지
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
# 필요하면 활성화
# find_package(builtin_interfaces REQUIRED)

# *.msg → 인터페이스 코드 생성
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/FrenetDebug.msg"
  "msg/FrenetDebugArray.msg"
  "msg/ObstacleArray.msg"
  "msg/ObstacleWpnt.msg"
  "msg/OTWpnt.msg"
  "msg/OTWpntArray.msg"
  "msg/Wpnt.msg"
  "msg/WpntArray.msg"
  DEPENDENCIES std_msgs geometry_msgs
  # builtin_interfaces 사용 시 위 라인에 추가
)

# 원본 msg도 설치(선택)
install(DIRECTORY msg DESTINATION share/${PROJECT_NAME})

ament_export_dependencies(rosidl_default_runtime)
ament_package()
