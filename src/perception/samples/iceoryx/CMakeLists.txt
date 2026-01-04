# # No ROS 2
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # Find iceoryx packages
# find_package(iceoryx_posh REQUIRED)
# find_package(iceoryx_hoofs REQUIRED)

# # Shared include directories
# include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# # Publisher Executable
# add_executable(iox_publisher Publisher.cpp)
# target_link_libraries(iox_publisher 
#     iceoryx_posh::iceoryx_posh 
#     iceoryx_hoofs::iceoryx_hoofs
# )

# # Subscriber Executable
# add_executable(iox_subscriber Subscriber.cpp)
# target_link_libraries(iox_subscriber 
#     iceoryx_posh::iceoryx_posh 
#     iceoryx_hoofs::iceoryx_hoofs
# )

# install(
# 	TARGETS 
# 		iox_publisher 
# 		iox_subscriber
#   DESTINATION 
# 		lib/${PROJECT_NAME}
# )

# # With ROS 2
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find_package(ament_cmake REQUIRED)
# find_package(rclcpp REQUIRED)
# find_package(percept_interfaces REQUIRED) 

# add_executable(iox_publisher PublisherROS.cpp)
# target_include_directories(iox_publisher PRIVATE .)

# target_link_libraries(iox_publisher
# 	rclcpp::rclcpp
# 	percept_interfaces::percept_interfaces__rosidl_typesupport_cpp
# )

# add_executable(iox_subscriber SubscriberROS.cpp)
# target_include_directories(iox_subscriber PRIVATE .)

# target_link_libraries(iox_subscriber
# 	rclcpp::rclcpp
# 	percept_interfaces::percept_interfaces__rosidl_typesupport_cpp
# )

# install(TARGETS
# 	iox_publisher
# 	iox_subscriber
# 	DESTINATION lib/${PROJECT_NAME}
# )

# ament_package()



set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Toggle variable: Default to OFF (No ROS)
option(WITH_ROS "Build with ROS 2 support" ON)

if(WITH_ROS)
    # --- ROS 2 Build Logic ---
    find_package(ament_cmake REQUIRED)
    find_package(rclcpp REQUIRED)
    find_package(percept_interfaces REQUIRED)

    # Publisher
    add_executable(iox_publisher PublisherROS.cpp)
    target_link_libraries(iox_publisher
        rclcpp::rclcpp
        percept_interfaces::percept_interfaces__rosidl_typesupport_cpp
    )

    # Subscriber
    # add_executable(iox_subscriber SubscriberROS.cpp)
    # target_link_libraries(iox_subscriber
    #     rclcpp::rclcpp
    #     percept_interfaces::percept_interfaces__rosidl_typesupport_cpp
    # )

    add_executable(iox_subscriber SubscriberROSMailbox.cpp)
    target_link_libraries(iox_subscriber
        rclcpp::rclcpp
        percept_interfaces::percept_interfaces__rosidl_typesupport_cpp
    )

else()
    # --- Standalone iceoryx Build Logic ---
    find_package(iceoryx_posh REQUIRED)
    find_package(iceoryx_hoofs REQUIRED)

    include_directories(${CMAKE_CURRENT_SOURCE_DIR})

    # Publisher
    add_executable(iox_publisher Publisher.cpp)
    target_link_libraries(iox_publisher 
        iceoryx_posh::iceoryx_posh 
        iceoryx_hoofs::iceoryx_hoofs
    )

    # Subscriber
    add_executable(iox_subscriber Subscriber.cpp)
    target_link_libraries(iox_subscriber 
        iceoryx_posh::iceoryx_posh 
        iceoryx_hoofs::iceoryx_hoofs
    )
endif()

# Shared Installation Logic
install(
  TARGETS 
    iox_publisher 
    iox_subscriber
  DESTINATION 
    lib/${PROJECT_NAME}
)