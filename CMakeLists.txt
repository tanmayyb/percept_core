cmake_minimum_required(VERSION 3.8)
project(percept)

# Set CUDA architecture policy
cmake_policy(SET CMP0104 NEW)
set(CMAKE_CUDA_ARCHITECTURES 75)  # For RTX 20xx series, adjust if needed

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#   add_compile_options(-Wall -Wextra -Wpedantic)
# endif()

# Setup CUDA/NVCC
set(CMAKE_C_COMPILER "/usr/lib/nvidia-cuda-toolkit/bin/gcc" CACHE PATH "" FORCE)
set(CMAKE_CXX_COMPILER "/usr/lib/nvidia-cuda-toolkit/bin/g++" CACHE PATH "" FORCE)
set(CUDA_DIR "/usr/lib/nvidia-cuda-toolkit")
set(CUDA_NVCC_FLAGS "-arch=sm_75" CACHE STRING "nvcc flags" FORCE)
set (CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
find_package(CUDA REQUIRED)
enable_language(CUDA)
add_definitions(-w -DUSE_CUDA)


# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(visualization_msgs REQUIRED)

# find_package(percept_interfaces REQUIRED
find_package(percept_interfaces)


# Find Python from Conda environment
execute_process(
    COMMAND ${CMAKE_COMMAND} -E env python -c
    "import sys; print(sys.executable)"
    OUTPUT_VARIABLE PYTHON_EXECUTABLE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Set the Python executable
set(PYTHON_EXECUTABLE ${PYTHON_EXECUTABLE})
message(STATUS "Using Python from: ${PYTHON_EXECUTABLE}")

# Set the Python include directory
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c
    "from sysconfig import get_paths as gp; print(gp()['include'])"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
include_directories(${PYTHON_INCLUDE_DIR})

# Set the Python library directory
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c
    "from sysconfig import get_paths as gp; print(gp()['stdlib'])"
    OUTPUT_VARIABLE PYTHON_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
link_directories(${PYTHON_LIB_DIR})

# Install Python modules
ament_python_install_package(${PROJECT_NAME}
  PACKAGE_DIR src
)


# Install Python executables
install(PROGRAMS
  src/real_pipeline.py
  src/utils/static_tf_publisher.py
  src/utils/fake_panda.py
  src/utils/fake_realsense.py
  src/service_tester.py
  src/sim_stream.py
  src/scene_loader.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch and config files
install(DIRECTORY
  launch
  config
  assets
  rviz2
  DESTINATION share/${PROJECT_NAME}
)

# cpp executable
add_executable(fields_computer 
  src/fields_computer.cpp 
  src/kernels/NearestObstacleDistance.cu
  src/kernels/ObstacleHeuristicCircForce.cu
  src/kernels/VelocityHeuristicCircForce.cu
  src/kernels/GoalObstacleHeuristicCircForce.cu
  src/kernels/GoalHeuristicCircForce.cu
  src/kernels/RandomHeuristicCircForce.cu
)

add_executable(fields_computer_cpu 
  src/fields_computer_cpu.cpp
)

target_include_directories(fields_computer 
  PRIVATE include
)
target_include_directories(fields_computer_cpu 
  PRIVATE include
)
ament_target_dependencies(fields_computer 
  rclcpp 
  std_msgs 
  sensor_msgs
  geometry_msgs
  visualization_msgs
  percept_interfaces
)
ament_target_dependencies(fields_computer_cpu 
  rclcpp 
  std_msgs 
  sensor_msgs
  geometry_msgs
  visualization_msgs
  percept_interfaces
)
target_link_libraries(fields_computer 
  ${CUDA_LIBRARIES}
)
install(TARGETS 
  fields_computer 
  fields_computer_cpu
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
