cmake_minimum_required(VERSION 3.18)
project(percept_core LANGUAGES CXX CUDA)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(yaml-cpp REQUIRED)

option(SHM_DISABLE "Disable shared memory features" ON)
# option(GPU_PROFILING_DISABLE "Disable all GPU profiling instrumentation" ON)


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

# if(NOT GPU_PROFILING_DISABLE)
# 	add_subdirectory(third_party/nvtx)
# else()
# 	add_compile_definitions(NVTX_DISABLE)
# endif()

install(
	DIRECTORY 
		config/
		launch/
	DESTINATION 
		share/${PROJECT_NAME}
)

add_subdirectory(src)

ament_package()
