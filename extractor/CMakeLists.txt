cmake_minimum_required(VERSION 3.14)

project(extractor LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(extractor
  main.cpp
)

target_link_libraries(extractor ${OpenCV_LIBS})
