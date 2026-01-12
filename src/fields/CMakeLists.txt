find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(percept_interfaces REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(vf_engine
	VFEngine.cpp
  ObstacleDistanceCost.cu
  ObstacleHeuristicCircForce.cu
  VelocityHeuristicCircForce.cu
  GoalObstacleHeuristicCircForce.cu
  GoalHeuristicCircForce.cu
  RandomHeuristicCircForce.cu
  ArtificialPotentialField.cu
  NearestNeighbour.cu
  NavigationFunctionForce.cu
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
	target_include_directories(vf_engine PRIVATE 
		${CMAKE_SOURCE_DIR}/third_party/nvtx/include
	)

	target_link_libraries(vf_engine PRIVATE nvtx_library)
endif()

install(TARGETS vf_engine
	DESTINATION lib/${PROJECT_NAME}
)