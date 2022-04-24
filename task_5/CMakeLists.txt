cmake_minimum_required(VERSION 3.16)
project(task_5)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

find_package( OpenCV REQUIRED )
set(SOURCE_FILES main.cpp)

add_executable(task_5 main.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
set(OPENCV_ENABLE_NONFREE ON)
target_link_libraries(task_5 ${OpenCV_LIBS} )