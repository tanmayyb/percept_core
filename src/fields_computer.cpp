#include "percept/fields_computer.hpp"

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
#include <visualization_msgs/msg/marker.hpp>

// CUDA kernels
// helpers
#include "percept/NearestObstacleDistance.h"
// heuristics
#include "percept/ObstacleHeuristicCircForce.h"
#include "percept/VelocityHeuristicCircForce.h"
#include "percept/GoalHeuristicCircForce.h"
#include "percept/GoalObstacleHeuristicCircForce.h"
#include "percept/RandomHeuristicCircForce.h"


FieldsComputer::FieldsComputer() : Node("fields_computer")
{

  this->declare_parameter("k_cf_velocity", 0.0);
  this->get_parameter("k_cf_velocity", k_cf_velocity);

  this->declare_parameter("k_cf_obstacle", 0.0);
  this->get_parameter("k_cf_obstacle", k_cf_obstacle);

  this->declare_parameter("k_cf_goal", 0.0);
  this->get_parameter("k_cf_goal", k_cf_goal);

  this->declare_parameter("k_cf_goalobstacle", 0.0);
  this->get_parameter("k_cf_goalobstacle", k_cf_goalobstacle);

  this->declare_parameter("k_cf_random", 0.0);
  this->get_parameter("k_cf_random", k_cf_random);  

  this->declare_parameter("agent_radius", 0.1);
  this->get_parameter("agent_radius", agent_radius);

  this->declare_parameter("mass_radius", 0.1);
  this->get_parameter("mass_radius", mass_radius);

  this->declare_parameter("max_allowable_force", 0.0);
  this->get_parameter("max_allowable_force", max_allowable_force);

  this->declare_parameter("detect_shell_rad", 0.0);
  this->get_parameter("detect_shell_rad", detect_shell_rad);
  if (detect_shell_rad > 0.0) {
    override_detect_shell_rad = true;
  }

  this->declare_parameter("publish_force_vector", false);
  this->get_parameter("publish_force_vector", publish_force_vector);

  this->declare_parameter("force_viz_scale", 1.0);
  this->get_parameter("force_viz_scale", force_viz_scale_);

  if (publish_force_vector) {
    marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("force_vector", 10);
  }

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

  RCLCPP_INFO(this->get_logger(), "Parameters:");
  RCLCPP_INFO(this->get_logger(), "  k_cf_velocity: %.10f", k_cf_velocity);
  RCLCPP_INFO(this->get_logger(), "  k_cf_obstacle: %.10f", k_cf_obstacle);
  RCLCPP_INFO(this->get_logger(), "  k_cf_goal: %.10f", k_cf_goal);
  RCLCPP_INFO(this->get_logger(), "  k_cf_goalobstacle: %.10f", k_cf_goalobstacle);
  RCLCPP_INFO(this->get_logger(), "  k_cf_random: %.10f", k_cf_random);
  RCLCPP_INFO(this->get_logger(), "  agent_radius: %.2f", agent_radius);
  RCLCPP_INFO(this->get_logger(), "  mass_radius: %.2f", mass_radius);
  RCLCPP_INFO(this->get_logger(), "  max_allowable_force: %.2f", max_allowable_force);
  RCLCPP_INFO(this->get_logger(), "  detect_shell_rad: %.2f", detect_shell_rad);
  RCLCPP_INFO(this->get_logger(), "  publishing force vectors: %s", publish_force_vector ? "true" : "false");
  if (publish_force_vector) {
    RCLCPP_INFO(this->get_logger(), "  force_viz_scale: %.2f", force_viz_scale_);
  }
  RCLCPP_INFO(this->get_logger(), "  show_processing_delay: %s", show_processing_delay ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "  show_requests: %s", show_service_request_received ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "Helper services:");
  RCLCPP_INFO(this->get_logger(), "  disable_nearest_obstacle_distance: %s", disable_nearest_obstacle_distance ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "Heuristics:");
  RCLCPP_INFO(this->get_logger(), "  disable_obstacle_heuristic: %s", disable_obstacle_heuristic ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "  disable_velocity_heuristic: %s", disable_velocity_heuristic ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "  disable_goal_heuristic: %s", disable_goal_heuristic ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "  disable_goalobstacle_heuristic: %s", disable_goalobstacle_heuristic ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "  disable_random_heuristic: %s", disable_random_heuristic ? "true" : "false");

  // Start the queue processor thread
  queue_processor_ = std::thread(&FieldsComputer::process_queue, this);


  // Subscribe to pointcloud messages.
  subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/primitives", 10,
      std::bind(&FieldsComputer::pointcloud_callback, this, std::placeholders::_1));

  // Create service servers for the helper services that are not disabled.
  if (!disable_nearest_obstacle_distance) {
    service_nearest_obstacle_distance = this->create_service<percept_interfaces::srv::AgentPoseToMinObstacleDist>(
        "/get_min_obstacle_distance",
        std::bind(&FieldsComputer::handle_nearest_obstacle_distance, this,
                  std::placeholders::_1, std::placeholders::_2));
    nearest_obstacle_distance::hello_cuda_world();
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
  // Create a copy of the message since we'll process it asynchronously
  auto msg_copy = std::make_shared<sensor_msgs::msg::PointCloud2>(*msg);
  
  enqueue_operation(OperationType::WRITE, [this, msg_copy]() {
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

    // Update the GPU buffer with exclusive access
    std::unique_lock<std::shared_timed_mutex> lock(gpu_points_mutex_);
    gpu_points_buffer_shared_ = new_gpu_buffer;
    gpu_num_points_ = num_points;
  });
}


// Extracts agent, velocity, and goal data from the service request.
std::tuple<double3, double3, double3> FieldsComputer::extract_request_data(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request)
{
  double3 agent_position = make_double3(
      request->agent_pose.position.x,
      request->agent_pose.position.y,
      request->agent_pose.position.z);

  double3 agent_velocity = make_double3(
      request->agent_velocity.x,
      request->agent_velocity.y,
      request->agent_velocity.z);

  double3 goal_position = make_double3(
      request->target_pose.position.x,
      request->target_pose.position.y,
      request->target_pose.position.z);

  if (!override_detect_shell_rad) {
    detect_shell_rad = request->detect_shell_rad;
  }

  return std::make_tuple(agent_position, agent_velocity, goal_position);
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

  response->circ_force.x = net_force.x;
  response->circ_force.y = net_force.y;
  response->circ_force.z = net_force.z;
  response->not_null = true;

  if (publish_force_vector && marker_pub_) {
    force_vector_publisher(net_force, agent_pose, marker_pub_);
  }
}


// Publishes a visualization marker representing the force vector.
void FieldsComputer::force_vector_publisher(const double3& net_force,
                                              const geometry_msgs::msg::Pose& agent_pose,
                                              rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub)
{
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = "world";  // Adjust frame as necessary.
  marker.header.stamp = this->now();
  marker.ns = "force_vectors";
  marker.id = 0;
  marker.type = visualization_msgs::msg::Marker::ARROW;
  marker.action = visualization_msgs::msg::Marker::ADD;

  marker.points.resize(2);
  // Start point at the agent's position.
  marker.points[0].x = agent_pose.position.x;
  marker.points[0].y = agent_pose.position.y;
  marker.points[0].z = agent_pose.position.z;
  // End point is the agent's position plus a scaled force vector.
  marker.points[1].x = agent_pose.position.x + net_force.x * force_viz_scale_;
  marker.points[1].y = agent_pose.position.y + net_force.y * force_viz_scale_;
  marker.points[1].z = agent_pose.position.z + net_force.z * force_viz_scale_;

  marker.scale.x = 0.1;  // Shaft diameter.
  marker.scale.y = 0.2;  // Head diameter.
  marker.scale.z = 0.3;  // Head length.

  marker.color.r = 1.0;
  marker.color.g = 0.0;
  marker.color.b = 0.0;
  marker.color.a = 1.0;

  marker_pub->publish(marker);
}


// Service handler for the nearest obstacle distance.
void FieldsComputer::handle_nearest_obstacle_distance(
    const std::shared_ptr<percept_interfaces::srv::AgentPoseToMinObstacleDist::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentPoseToMinObstacleDist::Response> response)
{
  if (show_service_request_received) {
    RCLCPP_INFO(this->get_logger(), "Nearest obstacle distance service request received");
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

    double min_dist = nearest_obstacle_distance::launch_kernel(
        gpu_buffer.get(),
        gpu_num_points_,
        agent_position,
        agent_radius,
        mass_radius,
        detect_shell_rad,
        show_processing_delay);

    response->distance = min_dist;
  });
}


// Add template implementation after the constructor/destructor

template<typename HeuristicFunc>
void FieldsComputer::handle_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response,
    HeuristicFunc kernel_launcher,
    double k_cf)
{
  enqueue_operation(OperationType::READ, [this, request, response, kernel_launcher, k_cf]() {
    std::shared_lock<std::shared_timed_mutex> lock(gpu_points_mutex_);
    auto gpu_buffer = gpu_points_buffer_shared_;
    if (!gpu_buffer) {
      response->not_null = false;
      return;
    }

    auto [agent_position, agent_velocity, goal_position] = extract_request_data(request);
    double3 net_force = kernel_launcher(
        gpu_buffer.get(),
        gpu_num_points_,
        agent_position,
        agent_velocity, 
        goal_position,
        agent_radius,
        mass_radius,
        detect_shell_rad,
        k_cf,
        max_allowable_force,
        show_processing_delay);

    process_response(net_force, request->agent_pose, response);
  });
}

// Replace individual handlers with templated versions
void FieldsComputer::handle_obstacle_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response)
{
  if (show_service_request_received) {
    RCLCPP_INFO(this->get_logger(), "Obstacle heuristic service request received");
  }
  handle_heuristic(request, response, obstacle_heuristic::launch_kernel, k_cf_obstacle);
}

void FieldsComputer::handle_velocity_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response)
{
  if (show_service_request_received) {
    RCLCPP_INFO(this->get_logger(), "Velocity heuristic service request received");
  }
  handle_heuristic(request, response, velocity_heuristic::launch_kernel, k_cf_velocity);
}

void FieldsComputer::handle_goal_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response)
{
  if (show_service_request_received) {
    RCLCPP_INFO(this->get_logger(), "Goal heuristic service request received");
  }
  handle_heuristic(request, response, goal_heuristic::launch_kernel, k_cf_goal);
}

void FieldsComputer::handle_goalobstacle_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response)
{
  if (show_service_request_received) {
    RCLCPP_INFO(this->get_logger(), "Goal obstacle heuristic service request received");
  }
  handle_heuristic(request, response, goalobstacle_heuristic::launch_kernel, k_cf_goalobstacle);
}

void FieldsComputer::handle_random_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response)
{
  if (show_service_request_received) {
    RCLCPP_INFO(this->get_logger(), "Random heuristic service request received");
  }
  handle_heuristic(request, response, random_heuristic::launch_kernel, k_cf_random);
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