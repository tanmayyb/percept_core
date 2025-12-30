cmake_minimum_required(VERSION 3.18)
project(percept LANGUAGES CXX CUDA)

set (CUDA_VERBOSE_BUILD ON)
set(CMAKE_CUDA_ARCHITECTURES native)

# ---- NVTX3 (GPU profiling features) ----
set(NVTX3_PATH "${CMAKE_CURRENT_SOURCE_DIR}/include/third_party/nvtx/c/include")
find_path(NVTX3_INCLUDE_DIR nvtx3/nvtx3.hpp PATHS ${NVTX3_PATH})
if(NVTX3_INCLUDE_DIR)
  include_directories(${NVTX3_INCLUDE_DIR})
else()
  add_definitions(-DDISABLE_NVTX)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(percept_interfaces REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

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
  src/kernels/ObstacleDistanceCost.cu
  src/kernels/ObstacleHeuristicCircForce.cu
  src/kernels/VelocityHeuristicCircForce.cu
  src/kernels/GoalObstacleHeuristicCircForce.cu
  src/kernels/GoalHeuristicCircForce.cu
  src/kernels/RandomHeuristicCircForce.cu
  src/kernels/ArtificialPotentialField.cu
  src/kernels/NearestNeighbour.cu
  src/kernels/NavigationFunctionForce.cu
)

add_executable(fields_computer_cpu 
  src/fields_computer_cpu.cpp
)

target_include_directories(fields_computer PRIVATE 
  ${CUDA_DIR}/include 
  include
)
target_compile_definitions(fields_computer PRIVATE USE_CUDA)
target_compile_options(fields_computer PRIVATE -w) # suppresses warnings for the code

target_include_directories(fields_computer_cpu 
  PRIVATE include
)
ament_target_dependencies(fields_computer 
  rclcpp 
  std_msgs 
  sensor_msgs
  geometry_msgs
  visualization_msgs
  tf2
  tf2_geometry_msgs
  percept_interfaces
)
ament_target_dependencies(fields_computer_cpu 
  rclcpp 
  std_msgs 
  sensor_msgs
  geometry_msgs
  visualization_msgs
  tf2
  tf2_geometry_msgs
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
