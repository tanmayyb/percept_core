#include "percept/fields_computer_cpu.hpp"

// std
#include <memory>
#include <thread>
#include <chrono>
#include <shared_mutex>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>

// msgs
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <visualization_msgs/msg/marker.hpp>



// Helper function to create a 3D point (replacing make_double3)
inline Point3D make_point3d(double x, double y, double z) {
    return Point3D(x, y, z);
}

// CPU implementations of CUDA kernel functionality namespaces
namespace nearest_obstacle_distance {

    
    double launch_cpu_kernel(
        const Point3D* points,
        size_t num_points,
        const Point3D& agent_position,
        double agent_radius,
        double mass_radius,
        double detect_shell_rad,
        bool show_processing_delay) 
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        double min_dist = std::numeric_limits<double>::max();
        for (size_t i = 0; i < num_points; ++i) {
            double dx = points[i].x - agent_position.x;
            double dy = points[i].y - agent_position.y;
            double dz = points[i].z - agent_position.z;
            
            double dist = std::sqrt(dx*dx + dy*dy + dz*dz) - agent_radius - mass_radius;
            min_dist = std::min(min_dist, dist);
        }
        
        if (show_processing_delay) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            std::cout << "Nearest obstacle distance computation took " << duration.count() << " microseconds" << std::endl;
        }
        
        return min_dist;
    }
}

// CPU implementation for obstacle heuristic
namespace obstacle_heuristic {

    
    Point3D launch_cpu_kernel(
        const Point3D* points,
        size_t num_points,
        const Point3D& agent_position,
        const Point3D& agent_velocity,
        const Point3D& goal_position,
        double agent_radius,
        double mass_radius,
        double detect_shell_rad,
        double k_force,
        double max_allowable_force,
        bool show_processing_delay)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        Point3D net_force(0.0, 0.0, 0.0);
        
        for (size_t i = 0; i < num_points; ++i) {
            double dx = points[i].x - agent_position.x;
            double dy = points[i].y - agent_position.y;
            double dz = points[i].z - agent_position.z;
            
            double distance_sq = dx*dx + dy*dy + dz*dz;
            double distance = std::sqrt(distance_sq);
            
            // Only consider points within detection shell
            if (distance < detect_shell_rad) {
                double safe_distance = agent_radius + mass_radius;
                
                // Apply repulsive force if point is close enough
                if (distance < safe_distance * 2.0) {
                    double force_magnitude = k_force * (1.0 / distance - 1.0 / (safe_distance * 2.0));
                    force_magnitude = std::min(force_magnitude, max_allowable_force);
                    
                    // Normalize direction vector
                    if (distance > 0) {
                        net_force.x -= force_magnitude * dx / distance;
                        net_force.y -= force_magnitude * dy / distance;
                        net_force.z -= force_magnitude * dz / distance;
                    }
                }
            }
        }
        
        if (show_processing_delay) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            std::cout << "Obstacle heuristic computation took " << duration.count() << " microseconds" << std::endl;
        }
        
        return net_force;
    }
}

// CPU implementation for velocity heuristic
namespace velocity_heuristic {

    
    Point3D launch_cpu_kernel(
        const Point3D* points,
        size_t num_points,
        const Point3D& agent_position,
        const Point3D& agent_velocity,
        const Point3D& goal_position,
        double agent_radius,
        double mass_radius,
        double detect_shell_rad,
        double k_force,
        double max_allowable_force,
        bool show_processing_delay)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Simple implementation - align with velocity direction
        Point3D net_force(0.0, 0.0, 0.0);
        
        double vel_mag = std::sqrt(agent_velocity.x*agent_velocity.x + 
                                  agent_velocity.y*agent_velocity.y + 
                                  agent_velocity.z*agent_velocity.z);
        
        if (vel_mag > 0.001) {
            // Normalize and scale velocity vector
            double force_magnitude = std::min(k_force, max_allowable_force);
            net_force.x = force_magnitude * agent_velocity.x / vel_mag;
            net_force.y = force_magnitude * agent_velocity.y / vel_mag;
            net_force.z = force_magnitude * agent_velocity.z / vel_mag;
        }
        
        if (show_processing_delay) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            std::cout << "Velocity heuristic computation took " << duration.count() << " microseconds" << std::endl;
        }
        
        return net_force;
    }
}

// CPU implementation for goal heuristic
namespace goal_heuristic {

    
    Point3D launch_cpu_kernel(
        const Point3D* points,
        size_t num_points,
        const Point3D& agent_position,
        const Point3D& agent_velocity,
        const Point3D& goal_position,
        double agent_radius,
        double mass_radius,
        double detect_shell_rad,
        double k_force,
        double max_allowable_force,
        bool show_processing_delay)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        Point3D net_force(0.0, 0.0, 0.0);
        
        // Calculate vector to goal
        double dx = goal_position.x - agent_position.x;
        double dy = goal_position.y - agent_position.y;
        double dz = goal_position.z - agent_position.z;
        
        double distance = std::sqrt(dx*dx + dy*dy + dz*dz);
        
        if (distance > 0.001) {
            // Normalize and scale force toward goal
            double force_magnitude = std::min(k_force, max_allowable_force);
            net_force.x = force_magnitude * dx / distance;
            net_force.y = force_magnitude * dy / distance;
            net_force.z = force_magnitude * dz / distance;
        }
        
        if (show_processing_delay) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            std::cout << "Goal heuristic computation took " << duration.count() << " microseconds" << std::endl;
        }
        
        return net_force;
    }
}

// CPU implementation for goal-obstacle heuristic
namespace goalobstacle_heuristic {

    
    Point3D launch_cpu_kernel(
        const Point3D* points,
        size_t num_points,
        const Point3D& agent_position,
        const Point3D& agent_velocity,
        const Point3D& goal_position,
        double agent_radius,
        double mass_radius,
        double detect_shell_rad,
        double k_force,
        double max_allowable_force,
        bool show_processing_delay)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Combine goal and obstacle forces
        Point3D goal_force = goal_heuristic::launch_cpu_kernel(
            points, num_points, agent_position, agent_velocity, goal_position,
            agent_radius, mass_radius, detect_shell_rad, k_force * 0.5, max_allowable_force * 0.5, false);
            
        Point3D obstacle_force = obstacle_heuristic::launch_cpu_kernel(
            points, num_points, agent_position, agent_velocity, goal_position,
            agent_radius, mass_radius, detect_shell_rad, k_force * 0.5, max_allowable_force * 0.5, false);
        
        Point3D net_force;
        net_force.x = goal_force.x + obstacle_force.x;
        net_force.y = goal_force.y + obstacle_force.y;
        net_force.z = goal_force.z + obstacle_force.z;
        
        // Limit the max force
        double force_mag = std::sqrt(net_force.x*net_force.x + 
                                    net_force.y*net_force.y + 
                                    net_force.z*net_force.z);
                                    
        if (force_mag > max_allowable_force && force_mag > 0.001) {
            net_force.x = net_force.x * max_allowable_force / force_mag;
            net_force.y = net_force.y * max_allowable_force / force_mag;
            net_force.z = net_force.z * max_allowable_force / force_mag;
        }
        
        if (show_processing_delay) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            std::cout << "Goal-obstacle heuristic computation took " << duration.count() << " microseconds" << std::endl;
        }
        
        return net_force;
    }
}

// CPU implementation for random heuristic
namespace random_heuristic {

    
    Point3D launch_cpu_kernel(
        const Point3D* points,
        size_t num_points,
        const Point3D& agent_position,
        const Point3D& agent_velocity,
        const Point3D& goal_position,
        double agent_radius,
        double mass_radius,
        double detect_shell_rad,
        double k_force,
        double max_allowable_force,
        bool show_processing_delay)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Create random force vector
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        
        Point3D net_force;
        net_force.x = dist(gen);
        net_force.y = dist(gen);
        net_force.z = dist(gen);
        
        // Normalize and scale
        double force_mag = std::sqrt(net_force.x*net_force.x + 
                                    net_force.y*net_force.y + 
                                    net_force.z*net_force.z);
        
        if (force_mag > 0.001) {
            double scale = std::min(k_force, max_allowable_force) / force_mag;
            net_force.x *= scale;
            net_force.y *= scale;
            net_force.z *= scale;
        }
        
        if (show_processing_delay) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            std::cout << "Random heuristic computation took " << duration.count() << " microseconds" << std::endl;
        }
        
        return net_force;
    }
}

// Operation class to track queue tasks
struct Operation {
    enum Type { READ, WRITE } type;
    std::function<void()> task;
    std::promise<void> completion;
};

FieldsComputerCPU::FieldsComputerCPU() : Node("fields_computer_cpu")
{

  this->declare_parameter("agent_radius", 0.050);
  this->get_parameter("agent_radius", agent_radius);

  this->declare_parameter("mass_radius", 0.050);
  this->get_parameter("mass_radius", mass_radius);

  this->declare_parameter("nn_detect_shell_rad", 2.0);
  this->get_parameter("nn_detect_shell_rad", nn_detect_shell_rad);

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
  RCLCPP_INFO(this->get_logger(), "  agent_radius: %.2f", agent_radius);
  RCLCPP_INFO(this->get_logger(), "  mass_radius: %.2f", mass_radius);
  RCLCPP_INFO(this->get_logger(), "  nn_detect_shell_rad: %.2f", nn_detect_shell_rad);
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
  queue_processor_ = std::thread(&FieldsComputerCPU::process_queue, this);


  // Subscribe to pointcloud messages.
  subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/primitives", 10,
      std::bind(&FieldsComputerCPU::pointcloud_callback, this, std::placeholders::_1));

  // Create service servers for the helper services that are not disabled.
  if (!disable_nearest_obstacle_distance) {
    service_nearest_obstacle_distance = this->create_service<percept_interfaces::srv::AgentPoseToMinObstacleDist>(
        "/get_min_obstacle_distance",
        std::bind(&FieldsComputerCPU::handle_nearest_obstacle_distance, this,
                  std::placeholders::_1, std::placeholders::_2));
  }
  // Create service servers for the heuristics that are not disabled.
  if (!disable_obstacle_heuristic) {
    service_obstacle_heuristic = this->create_service<percept_interfaces::srv::AgentStateToCircForce>(
        "/get_obstacle_heuristic_circforce",
        std::bind(&FieldsComputerCPU::handle_obstacle_heuristic, this,
                  std::placeholders::_1, std::placeholders::_2));
  }
  if (!disable_velocity_heuristic) {
    service_velocity_heuristic = this->create_service<percept_interfaces::srv::AgentStateToCircForce>(
        "/get_velocity_heuristic_circforce",
        std::bind(&FieldsComputerCPU::handle_velocity_heuristic, this,
                  std::placeholders::_1, std::placeholders::_2));
  }
  if (!disable_goal_heuristic) {
    service_goal_heuristic = this->create_service<percept_interfaces::srv::AgentStateToCircForce>(
        "/get_goal_heuristic_circforce",
        std::bind(&FieldsComputerCPU::handle_goal_heuristic, this,
                  std::placeholders::_1, std::placeholders::_2));
  }
  if (!disable_goalobstacle_heuristic) {
    service_goalobstacle_heuristic = this->create_service<percept_interfaces::srv::AgentStateToCircForce>(
        "/get_goalobstacle_heuristic_circforce",
        std::bind(&FieldsComputerCPU::handle_goalobstacle_heuristic, this,
                  std::placeholders::_1, std::placeholders::_2));
  }
  if (!disable_random_heuristic) {
    service_random_heuristic = this->create_service<percept_interfaces::srv::AgentStateToCircForce>(
        "/get_random_heuristic_circforce",
        std::bind(&FieldsComputerCPU::handle_random_heuristic, this,
                  std::placeholders::_1, std::placeholders::_2));
  }
}


// Destructor
FieldsComputerCPU::~FieldsComputerCPU()
{
  stop_queue();
  if (queue_processor_.joinable()) {
    queue_processor_.join();
  }
  // Reset the shared pointer. Any ongoing service calls (that copied the pointer)
  // will keep the CPU memory alive until they finish.
  std::unique_lock<std::shared_timed_mutex> lock(points_mutex_);
  points_buffer_shared_.reset();
}


// Callback for processing incoming point cloud messages.
void FieldsComputerCPU::pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
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

    // Create a new CPU buffer to hold the points
    auto new_points_buffer = std::make_shared<std::vector<Point3D>>(num_points);
    
    // Copy point cloud into the CPU buffer
    for (size_t i = 0; i < num_points; ++i, ++iter_x, ++iter_y, ++iter_z) {
      (*new_points_buffer)[i] = make_point3d(
          static_cast<double>(*iter_x),
          static_cast<double>(*iter_y),
          static_cast<double>(*iter_z));
    }

    // Update the points buffer with exclusive access
    std::unique_lock<std::shared_timed_mutex> lock(points_mutex_);
    points_buffer_shared_ = new_points_buffer;
    num_points_ = num_points;
  });
}


// Extracts agent, velocity, and goal data from the service request.
std::tuple<Point3D, Point3D, Point3D, double, double, double> FieldsComputerCPU::extract_request_data(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request)
{
  Point3D agent_position = make_point3d(
      request->agent_pose.position.x,
      request->agent_pose.position.y,
      request->agent_pose.position.z);

  Point3D agent_velocity = make_point3d(
      request->agent_velocity.x,
      request->agent_velocity.y,
      request->agent_velocity.z);

  Point3D goal_position = make_point3d(
      request->target_pose.position.x,
      request->target_pose.position.y,
      request->target_pose.position.z);

  double detect_shell_rad = request->detect_shell_rad;
  double k_force = request->k_force;
  double max_allowable_force = request->max_allowable_force;


  return std::make_tuple(agent_position, agent_velocity, goal_position, detect_shell_rad, k_force, max_allowable_force);
}


// Processes the net force returned by the CPU computation and publishes a response.
void FieldsComputerCPU::process_response(const Point3D& net_force,
                                      const geometry_msgs::msg::Pose& agent_pose,
                                      std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response)
{
  if (show_netforce_output) {
    RCLCPP_INFO(this->get_logger(), "Net force: x=%.10f, y=%.10f, z=%.10f, num_points=%zu",
                net_force.x, net_force.y, net_force.z, num_points_);
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
void FieldsComputerCPU::force_vector_publisher(const Point3D& net_force,
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
void FieldsComputerCPU::handle_nearest_obstacle_distance(
    const std::shared_ptr<percept_interfaces::srv::AgentPoseToMinObstacleDist::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentPoseToMinObstacleDist::Response> response)
{
  if (show_service_request_received) {
    RCLCPP_INFO(this->get_logger(), "Nearest obstacle distance service request received");
  }
  
  enqueue_operation(OperationType::READ, [this, request, response]() {
    std::shared_lock<std::shared_timed_mutex> lock(points_mutex_);
    auto points_buffer = points_buffer_shared_;
    if (!points_buffer || points_buffer->empty()) {
      response->distance = 0.0;
      return;
    }

    Point3D agent_position = make_point3d(
        request->agent_pose.position.x,
        request->agent_pose.position.y,
        request->agent_pose.position.z);

    double min_dist = nearest_obstacle_distance::launch_cpu_kernel(
        points_buffer->data(),
        num_points_,
        agent_position,
        agent_radius,
        mass_radius,
        nn_detect_shell_rad,
        show_processing_delay);

    response->distance = min_dist;
  });
}


template<typename HeuristicFunc>
void FieldsComputerCPU::handle_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response,
    HeuristicFunc kernel_launcher)
{
  enqueue_operation(OperationType::READ, [this, request, response, kernel_launcher]() {
    std::shared_lock<std::shared_timed_mutex> lock(points_mutex_);
    auto points_buffer = points_buffer_shared_;
    if (!points_buffer || points_buffer->empty()) {
      response->not_null = false;
      return;
    }

    auto [agent_position, agent_velocity, goal_position, detect_shell_rad, k_force, max_allowable_force] = extract_request_data(request);
    Point3D net_force = kernel_launcher(
        points_buffer->data(),
        num_points_,
        agent_position,
        agent_velocity, 
        goal_position,
        agent_radius,
        mass_radius,
        detect_shell_rad,
        k_force,
        max_allowable_force,
        show_processing_delay);

    process_response(net_force, request->agent_pose, response);
  });
}

// Replace individual handlers with templated versions
void FieldsComputerCPU::handle_obstacle_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response)
{
  if (show_service_request_received) {
    RCLCPP_INFO(this->get_logger(), "Obstacle heuristic service request received");
  }
  handle_heuristic(request, response, obstacle_heuristic::launch_cpu_kernel);
}

void FieldsComputerCPU::handle_velocity_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response)
{
  if (show_service_request_received) {
    RCLCPP_INFO(this->get_logger(), "Velocity heuristic service request received");
  }
  handle_heuristic(request, response, velocity_heuristic::launch_cpu_kernel);
}

void FieldsComputerCPU::handle_goal_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response)
{
  if (show_service_request_received) {
    RCLCPP_INFO(this->get_logger(), "Goal heuristic service request received");
  }
  handle_heuristic(request, response, goal_heuristic::launch_cpu_kernel);
}

void FieldsComputerCPU::handle_goalobstacle_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response)
{
  if (show_service_request_received) {
    RCLCPP_INFO(this->get_logger(), "Goal obstacle heuristic service request received");
  }
  handle_heuristic(request, response, goalobstacle_heuristic::launch_cpu_kernel);
}

void FieldsComputerCPU::handle_random_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response)
{
  if (show_service_request_received) {
    RCLCPP_INFO(this->get_logger(), "Random heuristic service request received");
  }
  handle_heuristic(request, response, random_heuristic::launch_cpu_kernel);
}

// Queue processing methods
void FieldsComputerCPU::process_queue()
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

void FieldsComputerCPU::enqueue_operation(OperationType type, std::function<void()> task)
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

void FieldsComputerCPU::stop_queue()
{
  queue_running_ = false;
  queue_cv_.notify_all();
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FieldsComputerCPU>());
    rclcpp::shutdown();
    return 0;
}