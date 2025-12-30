cmake_minimum_required(VERSION 3.8)
project(percept)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(yaml-cpp REQUIRED)


# due to known regression in librealsense v2.56.4 and newer.
# librealsense tries to connect with fastDDS but mismatches 
# with ROS 2's shipepd fastDDS version 
if(NOT TARGET fastrtps)
    add_library(fastrtps INTERFACE IMPORTED)
endif()
if(NOT TARGET fastcdr)
    add_library(fastcdr INTERFACE IMPORTED)
endif()
find_package(realsense2 REQUIRED)


install(DIRECTORY config
    DESTINATION share/${PROJECT_NAME}
)

add_subdirectory(src)
ament_package()
