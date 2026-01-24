option(GPU_PROFILING_DISABLE "Disable GPU profiling with NVTX" OFF)

find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(percept_interfaces REQUIRED)
find_package(CUDAToolkit REQUIRED)

# gpu
add_executable(vf_engine
	VFEngine.cpp
  ArtificialPotentialField.cu
  VelocityHeuristic.cu
  GoalHeuristic.cu
  # ObstacleHeuristicCircForce.cu
  # GoalObstacleHeuristicCircForce.cu
  # RandomHeuristicCircForce.cu
  # NearestNeighbour.cu
  # NavigationFunctionForce.cu
  MinObstacleDistance.cu
  NearestNeighbor.cu

)

set_target_properties(vf_engine
	PROPERTIES
	CUDA_ARCHITECTURES "native"
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON
	LINKER_LANGUAGE CUDA
)

target_link_libraries(vf_engine
	rclcpp::rclcpp

	CUDA::cudart

	std_msgs::std_msgs__rosidl_typesupport_cpp
	sensor_msgs::sensor_msgs__rosidl_typesupport_cpp
	geometry_msgs::geometry_msgs__rosidl_typesupport_cpp
	percept_interfaces::percept_interfaces__rosidl_typesupport_cpp
	
	tf2::tf2
	tf2_geometry_msgs::tf2_geometry_msgs

)

target_include_directories(vf_engine PUBLIC .)

if(NOT GPU_PROFILING_DISABLE)
  message(STATUS "NVTX Profiling is ENABLED in CMake configuration")
  set(NVTX_INCLUDE_DIR "/usr/local/cuda/include/nvtx3") # use system NVTX
  target_include_directories(vf_engine PRIVATE ${NVTX_INCLUDE_DIR})   
  target_compile_definitions(vf_engine PRIVATE GPU_PROFILING_ENABLE)
else()
  message(STATUS "NVTX Profiling is DISABLED in CMake configuration")
endif()

install(TARGETS vf_engine
	DESTINATION lib/${PROJECT_NAME}
)


# cpu
add_executable(vf_engine_cpu
  VFEngineCPU.cpp
)

target_link_libraries(vf_engine_cpu
	rclcpp::rclcpp
  
	std_msgs::std_msgs__rosidl_typesupport_cpp
	sensor_msgs::sensor_msgs__rosidl_typesupport_cpp
	geometry_msgs::geometry_msgs__rosidl_typesupport_cpp
	percept_interfaces::percept_interfaces__rosidl_typesupport_cpp

	tf2::tf2
	tf2_geometry_msgs::tf2_geometry_msgs

)

target_include_directories(vf_engine_cpu PUBLIC .)

install(TARGETS vf_engine_cpu
	DESTINATION lib/${PROJECT_NAME}
)