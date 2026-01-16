# Perception Node

find_package(sensor_msgs REQUIRED)
find_package(percept_interfaces REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)

option(OPENMP_DISABLE "Disable OpenMP support" OFF)
option(SHM_DISABLE "Disable shared memory features" ON)

if(NOT OPENMP_DISABLE)
    find_package(OpenMP REQUIRED)
endif()

set(OLD_WARN_DEPRECATED ${CMAKE_WARN_DEPRECATED}) # To supress Open3D CMP0072 warning
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)
find_package(Open3D REQUIRED)
set(CMAKE_WARN_DEPRECATED ${OLD_WARN_DEPRECATED} CACHE BOOL "" FORCE)


add_executable(perception_node 
	Perception.cpp 
	Streamer.cpp 
	Pipeline.cpp
)

set_target_properties(perception_node 
	PROPERTIES 
	CXX_STANDARD 20
	CUDA_ARCHITECTURES "native"
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON
	LINKER_LANGUAGE CUDA
)

target_link_libraries(perception_node 
	PUBLIC
    rclcpp::rclcpp      
    tf2_ros::tf2_ros
    tf2_eigen::tf2_eigen
    realsense2::realsense2
    yaml-cpp
    ament_index_cpp::ament_index_cpp
    Open3D::Open3D
    CUDA::cudart
    sensor_msgs::sensor_msgs__rosidl_typesupport_cpp
    percept_interfaces::percept_interfaces__rosidl_typesupport_cpp
)

if(NOT OPENMP_DISABLE)
  target_link_libraries(perception_node PRIVATE OpenMP::OpenMP_CXX)
  target_compile_options(perception_node PUBLIC ${OpenMP_CXX_FLAGS})
else()
  target_compile_definitions(perception_node PUBLIC OPENMP_DISABLE)
endif()

target_compile_options(perception_node PUBLIC "-Og")

if(SHM_DISABLE)
	target_compile_definitions(perception_node PUBLIC SHM_DISABLE)
endif()

target_include_directories(perception_node PUBLIC .)

install(TARGETS perception_node
  DESTINATION lib/${PROJECT_NAME}
)

# PointsFK Node
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(fk_node PointsFK.cpp)

target_link_libraries(fk_node
  rclcpp::rclcpp
  tf2_ros::tf2_ros
  tf2_eigen::tf2_eigen
  Eigen3::Eigen
	sensor_msgs::sensor_msgs__rosidl_typesupport_cpp
	ament_index_cpp::ament_index_cpp
	yaml-cpp
)

if(SHM_DISABLE)
	target_compile_definitions(fk_node PUBLIC SHM_DISABLE)
endif()

target_include_directories(fk_node PUBLIC .)

install(DIRECTORY assets
	DESTINATION share/${PROJECT_NAME}
)

install(TARGETS fk_node
  DESTINATION lib/${PROJECT_NAME}
)

