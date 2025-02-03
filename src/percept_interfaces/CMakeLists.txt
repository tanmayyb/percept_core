cmake_minimum_required(VERSION 3.8)
project(percept_interfaces)

# Find Python interpreter first
find_package(PythonInterp 3.6 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Explicitly set all Python-related variables
set(Python3_EXECUTABLE /usr/bin/python3)
set(PYTHON_EXECUTABLE /usr/bin/python3)
set(PYTHONINTERP_FOUND TRUE)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Remove this section as it's causing conflicts
# ament_python_install_package(${PROJECT_NAME}
#   PACKAGE_DIR src
# )

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/PosesToVectors.srv"
  "srv/AgentStateToCircForce.srv"
  DEPENDENCIES geometry_msgs std_msgs sensor_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()