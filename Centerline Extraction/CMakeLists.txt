cmake_minimum_required(VERSION 3.10)
project(CenterlineExtraction)

# C++ 표준 설정
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 헤더 파일 디렉토리 설정
include_directories(include)

# OpenCV 찾기
find_package(OpenCV REQUIRED)

# yaml-cpp 찾기
find_package(PkgConfig REQUIRED)
pkg_check_modules(YAMLCPP REQUIRED yaml-cpp)

# 소스 파일들
set(SOURCES
    src/occupancy_grid_loader.cpp
    src/image_processor.cpp
    src/visualization_utils.cpp
    src/main.cpp
)

# 실행 파일 생성
add_executable(centerline_extraction ${SOURCES})

# 라이브러리 링크
target_link_libraries(centerline_extraction 
    ${OpenCV_LIBS} 
    ${YAMLCPP_LIBRARIES}
)

# 컴파일러 플래그 (디버그 정보 포함)
target_compile_options(centerline_extraction PRIVATE 
    ${YAMLCPP_CFLAGS_OTHER}
)

# 헤더 파일 경로 추가
target_include_directories(centerline_extraction PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
    ${YAMLCPP_INCLUDE_DIRS}
)