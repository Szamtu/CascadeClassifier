cmake_minimum_required(VERSION 3.5)

project(dnn_speed_test VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

set(PROJECT_SOURCES
        main.cpp
)

add_executable(dnn_speed_test
    ${PROJECT_SOURCES}
)

target_link_libraries(dnn_speed_test
    ${OpenCV_LIBS}
)

