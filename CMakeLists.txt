cmake_minimum_required(VERSION 3.18)


# project(percept LANGUAGES CXX CUDA)
project(percept)
set(PROJECT_ENV_NAME "percept_env")
if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()
set(CMAKE_CXX_STANDARD 20)

# ---- CUDA ----
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CUDA_DIR "/usr/local/cuda-12.8")
set(CUDAToolkit_ROOT "${CUDA_DIR}")
set(CMAKE_CUDA_COMPILER "${CUDA_DIR}/bin/nvcc")
set(CUDA_LIBS_DIR "${CUDA_DIR}/lib64")
include_directories("${CUDA_DIR}/include")
enable_language(CUDA)
set (CUDA_VERBOSE_BUILD ON)

# ---- NVTX3 (GPU profiling features) ----
set(NVTX3_PATH "${CMAKE_CURRENT_SOURCE_DIR}/include/third_party/nvtx/c/include")
find_path(NVTX3_INCLUDE_DIR nvtx3/nvtx3.hpp PATHS ${NVTX3_PATH})
if(NVTX3_INCLUDE_DIR)
  include_directories(${NVTX3_INCLUDE_DIR})
else()
  add_definitions(-DDISABLE_NVTX)
endif()

# ---- Python ----
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CONDA_BASE "$ENV{HOME}/miniforge3/envs")
set(CONDA_ENV "${CONDA_BASE}/${PROJECT_ENV_NAME}")
set(PYTHON_EXECUTABLE "${CONDA_ENV}/bin/python")
set(PYTHON_INCLUDE_DIRS "${CONDA_ENV}/include/python3.10")
set(PYTHON_LIBRARY "${CONDA_ENV}/lib/libpython3.10.so")

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
find_package(percept_interfaces REQUIRED)

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
