# add_executable(perception_node Perception.cpp)

# set_target_properties(perception_node PROPERTIES CXX_STANDARD 11)

# target_link_libraries(perception_node 
#     PRIVATE
#     rclcpp::rclcpp      
#     realsense2::realsense2
#     yaml-cpp
#     ament_index_cpp::ament_index_cpp
# )


# install(TARGETS perception_node
#   DESTINATION lib/${PROJECT_NAME}
# )

add_executable(perception_node Perception.cpp Streamer.cpp)

set_target_properties(perception_node PROPERTIES CXX_STANDARD 11)

target_link_libraries(perception_node 
    PUBLIC
    rclcpp::rclcpp      
    realsense2::realsense2
    yaml-cpp
    ament_index_cpp::ament_index_cpp
)

target_include_directories(perception_node PUBLIC .)

install(TARGETS perception_node
  DESTINATION lib/${PROJECT_NAME}
)


