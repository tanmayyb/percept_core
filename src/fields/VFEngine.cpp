#include "VFEngine.hpp"

// std
#include <memory>
#include <thread>
#include <chrono>
#include <shared_mutex>

// CUDA runtime API
#include <cuda_runtime.h>

// msgs
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/point.hpp>

// tf2 includes
#include <tf2/LinearMath/Quaternion.hpp>
#include <tf2/LinearMath/Vector3.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// CUDA kernels
// helpers
#include "ObstacleDistanceCost.h"
// heuristics
#include "ObstacleHeuristicCircForce.h"
#include "VelocityHeuristicCircForce.h"
#include "GoalHeuristicCircForce.h"
#include "GoalObstacleHeuristicCircForce.h"
#include "RandomHeuristicCircForce.h"
#include "ArtificialPotentialField.h"
#include "NearestNeighbour.h"
#include "NavigationFunctionForce.h"

// nvtx
#ifndef NVTX_DISABLE
	#include <nvtx3/nvtx3.hpp>
#endif

FieldsComputer::FieldsComputer() : Node("vf_engine")
{

  // Select the GPU with the most memory (likely the most powerful)
  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (check_cuda_error(err, "getting device count")) {
    if (deviceCount > 0) {
      // Find the device with the most memory
      int selectedDeviceId = 0;
      size_t maxMemory = 0;
      
      for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        err = cudaGetDeviceProperties(&deviceProp, i);
        if (check_cuda_error(err, "getting device properties")) {
          if (deviceProp.totalGlobalMem > maxMemory) {
            maxMemory = deviceProp.totalGlobalMem;
            selectedDeviceId = i;
          }
        }
      }
      
      // Set the selected device
      err = cudaSetDevice(selectedDeviceId);
      if (check_cuda_error(err, "setting device")) {
        RCLCPP_INFO(this->get_logger(), "CUDA Device Properties:");
        RCLCPP_INFO(this->get_logger(), "  Using CUDA device %d of %d devices", selectedDeviceId, deviceCount);
        
        // Get and log device properties
        cudaDeviceProp deviceProp;
        err = cudaGetDeviceProperties(&deviceProp, selectedDeviceId);
        if (check_cuda_error(err, "getting device properties")) {
          RCLCPP_INFO(this->get_logger(), "  Device name: %s", deviceProp.name);
          RCLCPP_INFO(this->get_logger(), "  Compute capability: %d.%d", deviceProp.major, deviceProp.minor);
          RCLCPP_INFO(this->get_logger(), "  Total global memory: %.2f GB", 
                     static_cast<float>(deviceProp.totalGlobalMem) / (1024.0f * 1024.0f * 1024.0f));
        }
      }
      cudaDeviceSynchronize();
    } else {
      RCLCPP_ERROR(this->get_logger(), "No CUDA devices found!");
    }
  }


  this->declare_parameter("agent_radius", 0.050);
  this->get_parameter("agent_radius", agent_radius);

  this->declare_parameter("mass_radius", 0.050);
  this->get_parameter("mass_radius", mass_radius);

  this->declare_parameter("potential_detect_shell_rad", 1.0);
  this->get_parameter("potential_detect_shell_rad", potential_detect_shell_rad);

  this->declare_parameter("navigation_function_K", 1.0);
  this->get_parameter("navigation_function_K", navigation_function_K);

  this->declare_parameter("navigation_function_world_radius", 10.0);
  this->get_parameter("navigation_function_world_radius", navigation_function_world_radius);

  this->declare_parameter("show_netforce_output", false);
  this->get_parameter("show_netforce_output", show_netforce_output);

  this->declare_parameter("show_processing_delay", false);
  this->get_parameter("show_processing_delay", show_processing_delay);

  this->declare_parameter("show_requests", false);
  this->get_parameter("show_requests", show_service_request_received);

  // Heuristic enable/disable parameters.
  this->declare_parameter("disable_nearest_obstacle_distance", false);
  this->get_parameter("disable_nearest_obstacle_distance", disable_nearest_obstacle_distance);

  // Heuristic enable/disable parameters.
  this->declare_parameter("disable_obstacle_heuristic", false);
  this->get_parameter("disable_obstacle_heuristic", disable_obstacle_heuristic);

  this->declare_parameter("disable_velocity_heuristic", false);
  this->get_parameter("disable_velocity_heuristic", disable_velocity_heuristic);

  this->declare_parameter("disable_goal_heuristic", false);
  this->get_parameter("disable_goal_heuristic", disable_goal_heuristic);

  this->declare_parameter("disable_goalobstacle_heuristic", false);
  this->get_parameter("disable_goalobstacle_heuristic", disable_goalobstacle_heuristic);

  this->declare_parameter("disable_random_heuristic", false);
  this->get_parameter("disable_random_heuristic", disable_random_heuristic);

  this->declare_parameter("disable_apf_heuristic", false);
  this->get_parameter("disable_apf_heuristic", disable_apf_heuristic);

  this->declare_parameter("disable_navigation_function_force", false);
  this->get_parameter("disable_navigation_function_force", disable_navigation_function_force);

  RCLCPP_INFO(this->get_logger(), "Parameters:");
  RCLCPP_INFO(this->get_logger(), "  agent_radius: %.2f", agent_radius);
  RCLCPP_INFO(this->get_logger(), "  mass_radius: %.2f", mass_radius);
  RCLCPP_INFO(this->get_logger(), "  potential_detect_shell_rad: %.2f", potential_detect_shell_rad);
  RCLCPP_INFO(this->get_logger(), "  navigation_function_K: %.2f", navigation_function_K);
  RCLCPP_INFO(this->get_logger(), "  navigation_function_world_radius: %.2f", navigation_function_world_radius);
  RCLCPP_INFO(this->get_logger(), "Helper functions:");
  RCLCPP_INFO(this->get_logger(), "  show_processing_delay: %s", show_processing_delay ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "  show_requests: %s", show_service_request_received ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "Services:");
  RCLCPP_INFO(this->get_logger(), "  disable_nearest_obstacle_distance: %s", disable_nearest_obstacle_distance ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "Heuristics:");
  RCLCPP_INFO(this->get_logger(), "  disable_obstacle_heuristic: %s", disable_obstacle_heuristic ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "  disable_velocity_heuristic: %s", disable_velocity_heuristic ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "  disable_goal_heuristic: %s", disable_goal_heuristic ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "  disable_goalobstacle_heuristic: %s", disable_goalobstacle_heuristic ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "  disable_random_heuristic: %s", disable_random_heuristic ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "  disable_apf_heuristic: %s", disable_apf_heuristic ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "  disable_navigation_function_force: %s", disable_navigation_function_force ? "true" : "false");

  // Profiling
  RCLCPP_INFO(this->get_logger(), "Profiling:");
  #ifndef NVTX_DISABLE
  RCLCPP_INFO(this->get_logger(), "  nvtx enabled: %s", "true");
  #else
  RCLCPP_INFO(this->get_logger(), "  nvtx enabled: %s", "false");
  #endif

  
  // Start the queue processor thread
  queue_processor_ = std::thread(&FieldsComputer::process_queue, this);


  // Subscribe to pointcloud messages.
  subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/primitives", 10,
      std::bind(&FieldsComputer::pointcloud_callback, this, std::placeholders::_1));

  // Create service servers for the helper services that are not disabled.
  if (!disable_nearest_obstacle_distance) {
    service_obstacle_distance_cost = this->create_service<percept_interfaces::srv::AgentPoseToMinObstacleDist>(
        "/get_min_obstacle_distance",
        std::bind(&FieldsComputer::handle_obstacle_distance_cost, this,
                  std::placeholders::_1, std::placeholders::_2));
    obstacle_distance_cost::hello_cuda_world();
  }
  // Create service servers for the heuristics that are not disabled.
  if (!disable_obstacle_heuristic) {
    service_obstacle_heuristic = this->create_service<percept_interfaces::srv::AgentStateToCircForce>(
        "/get_obstacle_heuristic_circforce",
        std::bind(&FieldsComputer::handle_obstacle_heuristic, this,
                  std::placeholders::_1, std::placeholders::_2));
    obstacle_heuristic::hello_cuda_world();
  }
  if (!disable_velocity_heuristic) {
    service_velocity_heuristic = this->create_service<percept_interfaces::srv::AgentStateToCircForce>(
        "/get_velocity_heuristic_circforce",
        std::bind(&FieldsComputer::handle_velocity_heuristic, this,
                  std::placeholders::_1, std::placeholders::_2));
    velocity_heuristic::hello_cuda_world();
  }
  if (!disable_goal_heuristic) {
    service_goal_heuristic = this->create_service<percept_interfaces::srv::AgentStateToCircForce>(
        "/get_goal_heuristic_circforce",
        std::bind(&FieldsComputer::handle_goal_heuristic, this,
                  std::placeholders::_1, std::placeholders::_2));
    goal_heuristic::hello_cuda_world();
  }
  if (!disable_goalobstacle_heuristic) {
    service_goalobstacle_heuristic = this->create_service<percept_interfaces::srv::AgentStateToCircForce>(
        "/get_goalobstacle_heuristic_circforce",
        std::bind(&FieldsComputer::handle_goalobstacle_heuristic, this,
                  std::placeholders::_1, std::placeholders::_2));
    goalobstacle_heuristic::hello_cuda_world();
  }
  if (!disable_random_heuristic) {
    service_random_heuristic = this->create_service<percept_interfaces::srv::AgentStateToCircForce>(
        "/get_random_heuristic_circforce",
        std::bind(&FieldsComputer::handle_random_heuristic, this,
                  std::placeholders::_1, std::placeholders::_2));
    random_heuristic::hello_cuda_world();
  }
  if (!disable_apf_heuristic) {
    service_apf_heuristic = this->create_service<percept_interfaces::srv::AgentStateToCircForce>(
        "/get_apf_heuristic_circforce",
        std::bind(&FieldsComputer::handle_apf_heuristic, this,
                  std::placeholders::_1, std::placeholders::_2));
    artificial_potential_field::hello_cuda_world();
  }
  if (!disable_navigation_function_force) {
    service_navigation_function_force = this->create_service<percept_interfaces::srv::AgentStateToCircForce>(
        "/get_navigation_function_circforce",
        std::bind(&FieldsComputer::handle_navigation_function_force, this,
                  std::placeholders::_1, std::placeholders::_2));
    navigation_function::hello_cuda_world();
  }
}


// Destructor
FieldsComputer::~FieldsComputer()
{
  stop_queue();
  if (queue_processor_.joinable()) {
    queue_processor_.join();
  }
  // Reset the shared pointer. Any ongoing service calls (that copied the pointer)
  // will keep the GPU memory alive until they finish.
  std::unique_lock<std::shared_timed_mutex> lock(gpu_points_mutex_);
  gpu_points_buffer_shared_.reset();
  gpu_nn_index_shared_.reset();
}


// Utility function to check CUDA errors.
bool FieldsComputer::check_cuda_error(cudaError_t err, const char* operation)
{
  if (err != cudaSuccess) {
    RCLCPP_ERROR(this->get_logger(), "CUDA %s failed: %s", operation, cudaGetErrorString(err));
    return false;
  }
  return true;
}



// Callback for processing incoming point cloud messages.
void FieldsComputer::pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  mark_start("PointCloud Callback Received", 0x0000FF);
  // Create a copy of the message since we'll process it asynchronously
  auto msg_copy = std::make_shared<sensor_msgs::msg::PointCloud2>(*msg);
  
  enqueue_operation(OperationType::WRITE, [this, msg_copy]() {
#ifndef NVTX_DISABLE
    nvtx3::scoped_range range{"PointCloud Callback Processing"};
#endif
    // Compute number of points
    size_t num_points = msg_copy->width * msg_copy->height;

    // Create iterators for the x, y, and z fields.
    sensor_msgs::PointCloud2Iterator<float> iter_x(*msg_copy, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(*msg_copy, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(*msg_copy, "z");

    // Copy point cloud into a temporary host array.
    std::vector<double3> points_double3(num_points);
    for (size_t i = 0; i < num_points; ++i, ++iter_x, ++iter_y, ++iter_z) {
      points_double3[i] = make_double3(
          static_cast<double>(*iter_x),
          static_cast<double>(*iter_y),
          static_cast<double>(*iter_z));
    }

    // Allocate GPU memory.
    double3* gpu_buffer_ptr = nullptr;
    cudaError_t err = cudaMalloc(&gpu_buffer_ptr, num_points * sizeof(double3));
    if (!check_cuda_error(err, "cudaMalloc")) {
      return;
    }

    // Copy data from host to GPU.
    err = cudaMemcpy(gpu_buffer_ptr, points_double3.data(),
                     num_points * sizeof(double3), cudaMemcpyHostToDevice);
    if (!check_cuda_error(err, "cudaMemcpy")) {
      cudaFree(gpu_buffer_ptr);
      return;
    }

    // Wrap the raw GPU pointer in a shared_ptr with a custom deleter.
    auto new_gpu_buffer = std::shared_ptr<double3>(gpu_buffer_ptr, [](double3* ptr) {
      if (ptr) {
        cudaFree(ptr);
      }
    });

    // Allocate memory for the nearest neighbour index.
    int* gpu_nn_index_ptr = nullptr;
    err = cudaMalloc(&gpu_nn_index_ptr, num_points * sizeof(int));
    if (!check_cuda_error(err, "cudaMalloc for nearest neighbor index")) {
      cudaFree(gpu_buffer_ptr);
      return;
    }

    // Nearest neighbour kernel is not used for now.
    // // Launch the nearest neighbour kernel.
    // nearest_neighbour::launch_kernel(
    //   gpu_buffer_ptr,
    //   num_points,
    //   gpu_nn_index_ptr,
    //   show_processing_delay
    // );

    // Wrap the raw GPU pointer in a shared_ptr with a custom deleter.
    auto new_gpu_nn_index = std::shared_ptr<int>(gpu_nn_index_ptr, [](int* ptr) {
      if (ptr) {
        cudaFree(ptr);
      }
    });

    // Update the GPU buffer with exclusive access
    std::unique_lock<std::shared_timed_mutex> lock(gpu_points_mutex_);
    gpu_points_buffer_shared_ = new_gpu_buffer;
    gpu_num_points_ = num_points;
    gpu_nn_index_shared_ = new_gpu_nn_index;
  });
}


// Extracts agent, velocity, and goal data from the service request.
std::tuple<double3, double3, double3, double, double, double> FieldsComputer::extract_request_data(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request)
{
  double3 agent_position = make_double3(
      request->agent_pose.position.x,
      request->agent_pose.position.y,
      request->agent_pose.position.z);

  tf2::Quaternion q_world_from_agent; // Transform agent velocity from agent frame to world frame
  tf2::fromMsg(request->agent_pose.orientation, q_world_from_agent);
  q_world_from_agent.normalize();
  const tf2::Vector3 v_agent(
    request->agent_velocity.x, 
    request->agent_velocity.y, 
    request->agent_velocity.z);
  const tf2::Vector3 v_world = tf2::quatRotate(q_world_from_agent, v_agent);
  double3 agent_velocity = make_double3(
    v_world.x(), 
    v_world.y(), 
    v_world.z());

  double3 goal_position = make_double3(
      request->target_pose.position.x,
      request->target_pose.position.y,
      request->target_pose.position.z);

  double detect_shell_rad = request->detect_shell_rad;
  double k_force = request->k_force;
  double max_allowable_force = request->max_allowable_force;


  return std::make_tuple(agent_position, agent_velocity, goal_position, detect_shell_rad, k_force, max_allowable_force);
}


// Processes the net force returned by the CUDA kernel and publishes a response.
void FieldsComputer::process_response(const double3& net_force,
                                        const geometry_msgs::msg::Pose& agent_pose,
                                        std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response)
{
  if (show_netforce_output) {
    RCLCPP_INFO(this->get_logger(), "Net force: x=%.10f, y=%.10f, z=%.10f, num_points=%zu",
                net_force.x, net_force.y, net_force.z, gpu_num_points_);
  }

  // Build quaternion from agent's world pose.
  tf2::Quaternion q_world_from_agent;
  tf2::fromMsg(agent_pose.orientation, q_world_from_agent);
  q_world_from_agent.normalize(); 
  
  // Rotate world vector into the agent frame: v_agent = R^T * v_world = quat.inverse() ⊗ v_world ⊗ quat
  const tf2::Vector3 v_world(net_force.x, net_force.y, net_force.z);
  const tf2::Vector3 v_agent = tf2::quatRotate(q_world_from_agent.inverse(), v_world);

  response->circ_force.x = v_agent.x();
  response->circ_force.y = v_agent.y();
  response->circ_force.z = v_agent.z();
  response->not_null = true;
}


// Service handler for the obstacle distance cost.
void FieldsComputer::handle_obstacle_distance_cost(
    const std::shared_ptr<percept_interfaces::srv::AgentPoseToMinObstacleDist::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentPoseToMinObstacleDist::Response> response)
{
  mark_start("ObstacleDistanceCost Request Received", 0x00FF00);
  if (show_service_request_received) {
    RCLCPP_INFO(this->get_logger(), "Obstacle distance cost service request received");
  }
  
  enqueue_operation(OperationType::READ, [this, request, response]() {
    std::shared_lock<std::shared_timed_mutex> lock(gpu_points_mutex_);
    auto gpu_buffer = gpu_points_buffer_shared_;
    if (!gpu_buffer) {
      response->distance = 0.0;
      return;
    }

    double3 agent_position = make_double3(
        request->agent_pose.position.x,
        request->agent_pose.position.y,
        request->agent_pose.position.z);

    double net_potential = obstacle_distance_cost::launch_kernel(
        gpu_buffer.get(),
        gpu_num_points_,
        agent_position,
        agent_radius,
        mass_radius,
        potential_detect_shell_rad,
        show_processing_delay);

    if (show_service_request_received) {
      RCLCPP_INFO(this->get_logger(), "Obstacle distance cost: %f", net_potential);
    }

    response->distance = net_potential;
  });
}


// Add template implementation after the constructor/destructor

template<typename HeuristicFunc>
void FieldsComputer::handle_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response,
    HeuristicFunc kernel_launcher, const std::string& heuristic_name)
{
  enqueue_operation(OperationType::READ, [this, request, response, kernel_launcher, heuristic_name]() {
    std::shared_lock<std::shared_timed_mutex> lock(gpu_points_mutex_);
    auto gpu_buffer = gpu_points_buffer_shared_;
    if (!gpu_buffer) {
      response->not_null = false;
      return;
    }
    using clock = std::chrono::steady_clock;
    auto start_time = clock::now();
    auto [agent_position, agent_velocity, goal_position, detect_shell_rad, k_force, max_allowable_force] = extract_request_data(request);
    double3 net_force;
    if constexpr (std::is_invocable_v<HeuristicFunc, double3*, size_t, int*, double3, double3, double3, double, double, double, double, double, bool>) {
      // GoalObstacleHeuristic, Obstacle Heuristic
      auto gpu_nn_index = gpu_nn_index_shared_;
      net_force = kernel_launcher(
        gpu_buffer.get(),
        gpu_num_points_,
        gpu_nn_index.get(),
        agent_position,
        agent_velocity, 
        goal_position,
        agent_radius,
        mass_radius,
        detect_shell_rad,
        k_force,
        max_allowable_force,
        show_processing_delay);
    }
    else if constexpr( std::is_invocable_v<HeuristicFunc, double3*, size_t, double3, double3, double3, double, double, double, double, double, bool>){
      // Velocity Heuristic, Goal Heuristic, Random Heuristic, APF Heuristic
      net_force = kernel_launcher(
        gpu_buffer.get(),
        gpu_num_points_,
        agent_position,
        agent_velocity, 
        goal_position,
        agent_radius,
        mass_radius,
        detect_shell_rad,
        k_force,
        max_allowable_force,
        show_processing_delay);
    }
    else if constexpr( std::is_invocable_v<HeuristicFunc, double3*, size_t, double3, double3, double, double, double, double, double, bool>){
      // Navigation Function Force
      net_force = kernel_launcher(
        gpu_buffer.get(),
        gpu_num_points_,
        agent_position,
        goal_position,
        detect_shell_rad,
        k_force,
        navigation_function_K,
        navigation_function_world_radius,
        max_allowable_force,
        show_processing_delay);
    }
    else{
      RCLCPP_ERROR(this->get_logger(), "Invalid heuristic function");
      response->not_null = false;
      return;
    }

    process_response(net_force, request->agent_pose, response);
    auto end_time = clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    if (show_processing_delay) {
      RCLCPP_INFO(this->get_logger(), "Heuristic computation time: %ld microseconds", duration.count());
    }
  });
}

void FieldsComputer::mark_start(const std::string& name, unsigned int color_hex) {
#ifndef NVTX_DISABLE
  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = color_hex;
  // eventAttrib.color = 0xFFFF0000; // Red color
  // eventAttrib.color = 0xFFFFC0CB; // Pink color

  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii = name.c_str();

  nvtxMarkEx(&eventAttrib);
#endif
}

// Replace individual handlers with templated versions
void FieldsComputer::handle_obstacle_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response)
{
  mark_start("ObstacleHeuristic Request Received", 0xFFFF0000);
  if (show_service_request_received) {
    RCLCPP_INFO(this->get_logger(), "Obstacle heuristic service request received");
  }
  handle_heuristic(request, response, obstacle_heuristic::launch_kernel, "ObstacleHeuristic");
}

void FieldsComputer::handle_velocity_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response)
{
  mark_start("VelocityHeuristic Request Received", 0xFFFF0000);
  if (show_service_request_received) {
    RCLCPP_INFO(this->get_logger(), "Velocity heuristic service request received");
  }
  handle_heuristic(request, response, velocity_heuristic::launch_kernel, "VelocityHeuristic");
}

void FieldsComputer::handle_goal_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response)
{
  mark_start("GoalHeuristic Request Received", 0xFFFF0000);
  if (show_service_request_received) {
    RCLCPP_INFO(this->get_logger(), "Goal heuristic service request received");
  }
  handle_heuristic(request, response, goal_heuristic::launch_kernel, "GoalHeuristic");
}

void FieldsComputer::handle_goalobstacle_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response)
{
  mark_start("GoalObstacleHeuristic Request Received", 0xFFFF0000);
  if (show_service_request_received) {
    RCLCPP_INFO(this->get_logger(), "Goal obstacle heuristic service request received");
  }
  handle_heuristic(request, response, goalobstacle_heuristic::launch_kernel, "GoalObstacleHeuristic");
}

void FieldsComputer::handle_random_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response)
{
  mark_start("RandomHeuristic Request Received", 0xFFFF0000);
  if (show_service_request_received) {
    RCLCPP_INFO(this->get_logger(), "Random heuristic service request received");
  }
  handle_heuristic(request, response, random_heuristic::launch_kernel, "RandomHeuristic");
}

void FieldsComputer::handle_apf_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response)
{
  mark_start("APFHeuristic Request Received", 0xFFFF0000);
  if (show_service_request_received) {
    RCLCPP_INFO(this->get_logger(), "APF heuristic service request received");
  }
  handle_heuristic(request, response, artificial_potential_field::launch_kernel, "APFHeuristic");
}

void FieldsComputer::handle_navigation_function_force(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response)
{
  mark_start("NavigationFunctionForce Request Received", 0xFFFF0000);
  if (show_service_request_received) {
    RCLCPP_INFO(this->get_logger(), "Navigation function force service request received");
  }
  handle_heuristic(request, response, navigation_function::launch_kernel, "NavigationFunctionForce");
}

// Add new queue processing methods
void FieldsComputer::process_queue()
{
  while (queue_running_) {
    std::shared_ptr<Operation> op;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      queue_cv_.wait(lock, [this] {
        return !operation_queue_.empty() || !queue_running_;
      });

      if (!queue_running_) break;

      op = operation_queue_.front();
      operation_queue_.pop();
    }

    // Execute the operation
    op->task();
    op->completion.set_value();
  }
}

void FieldsComputer::enqueue_operation(OperationType type, std::function<void()> task)
{
  auto op = std::make_shared<Operation>();
  op->type = type;
  op->task = task;

  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    operation_queue_.push(op);
  }
  queue_cv_.notify_one();

  // Wait for completion
  op->completion.get_future().wait();
}

void FieldsComputer::stop_queue()
{
  queue_running_ = false;
  queue_cv_.notify_all();
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FieldsComputer>());
    rclcpp::shutdown();
    return 0;
}