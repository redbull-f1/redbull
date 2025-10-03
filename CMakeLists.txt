cmake_minimum_required(VERSION 3.8)
project(redbull)
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(redbull_msgs REQUIRED)

ament_python_install_package(${PROJECT_NAME})

ament_export_dependencies(redbull_msgs)
ament_package()
