#include "percept/fields_computer.hpp"

#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <thread>

#include <cuda_runtime.h>
#include "percept/ObstacleHeuristicCircForce.h"
#include "percept/VelocityHeuristicCircForce.h"
#include "percept/GoalHeuristicCircForce.h"
#include "percept/GoalObstacleHeuristicCircForce.h"


struct GPUGuard {
    std::atomic<bool>& flag;
    GPUGuard(std::atomic<bool>& f) : flag(f) {}
    ~GPUGuard() { flag.store(false); }
};


FieldsComputer::FieldsComputer()
    : Node("fields_computer")
{
    this->declare_parameter("k_circular_force", 0.1);
    this->get_parameter("k_circular_force", k_circular_force);

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
        marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
            "force_vector", 10);
    }

    this->declare_parameter("disable_obstacle_heuristic", false);
    this->get_parameter("disable_obstacle_heuristic", disable_obstacle_heuristic);

    this->declare_parameter("disable_velocity_heuristic", false);
    this->get_parameter("disable_velocity_heuristic", disable_velocity_heuristic);

    this->declare_parameter("disable_goal_heuristic", false);
    this->get_parameter("disable_goal_heuristic", disable_goal_heuristic);

    this->declare_parameter("disable_goalobstacle_heuristic", false);
    this->get_parameter("disable_goalobstacle_heuristic", disable_goalobstacle_heuristic);
       
    RCLCPP_INFO(this->get_logger(), "Parameters:");
    RCLCPP_INFO(this->get_logger(), "  k_circular_force: %.2f", k_circular_force);
    RCLCPP_INFO(this->get_logger(), "  agent_radius: %.2f", agent_radius); 
    RCLCPP_INFO(this->get_logger(), "  mass_radius: %.2f", mass_radius);
    RCLCPP_INFO(this->get_logger(), "  max_allowable_force: %.2f", max_allowable_force);
    RCLCPP_INFO(this->get_logger(), "  detect_shell_rad: %.2f", detect_shell_rad);
    RCLCPP_INFO(this->get_logger(), "  publishing force vectors: %s", publish_force_vector ? "true" : "false");
    if (publish_force_vector) {
        RCLCPP_INFO(this->get_logger(), "  force_viz_scale: %.2f", force_viz_scale_);
    }
    // heuristics
    RCLCPP_INFO(this->get_logger(), "Heuristics:");
    RCLCPP_INFO(this->get_logger(), "  disable_obstacle_heuristic: %s", disable_obstacle_heuristic ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "  disable_velocity_heuristic: %s", disable_velocity_heuristic ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "  disable_goal_heuristic: %s", disable_goal_heuristic ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "  disable_goalobstacle_heuristic: %s", disable_goalobstacle_heuristic ? "true" : "false");


    // pointcloud subscription
    subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/primitives", 10,
        std::bind(&FieldsComputer::pointcloud_callback, this, std::placeholders::_1));

    // test kernels
    obstacle_heuristic::hello_cuda_world();
    velocity_heuristic::hello_cuda_world();
    goal_heuristic::hello_cuda_world();
    goalobstacle_heuristic::hello_cuda_world();

    // service_ = this->create_service<percept_interfaces::srv::AgentStateToCircForce>(
    //     "/get_heuristic_circforce",
    //     std::bind(&FieldsComputer::handle_agent_state_to_circ_force, this,
    //              std::placeholders::_1, std::placeholders::_2));

    // heuristics services
    if (!disable_obstacle_heuristic) {
        service_obstacle_heuristic = this->create_service<percept_interfaces::srv::AgentStateToCircForce>(
            "/get_obstacle_heuristic_circforce",
            std::bind(&FieldsComputer::handle_obstacle_heuristic, this,
                     std::placeholders::_1, std::placeholders::_2));
    }

    if (!disable_velocity_heuristic) {
        service_velocity_heuristic = this->create_service<percept_interfaces::srv::AgentStateToCircForce>(
            "/get_velocity_heuristic_circforce",
            std::bind(&FieldsComputer::handle_velocity_heuristic, this,
                     std::placeholders::_1, std::placeholders::_2));
    }

    if (!disable_goal_heuristic) {
        service_goal_heuristic = this->create_service<percept_interfaces::srv::AgentStateToCircForce>(
            "/get_goal_heuristic_circforce",
            std::bind(&FieldsComputer::handle_goal_heuristic, this,
                     std::placeholders::_1, std::placeholders::_2));
    }

    if (!disable_goalobstacle_heuristic) {
        service_goalobstacle_heuristic = this->create_service<percept_interfaces::srv::AgentStateToCircForce>(
            "/get_goalobstacle_heuristic_circforce",
            std::bind(&FieldsComputer::handle_goalobstacle_heuristic, this,
                     std::placeholders::_1, std::placeholders::_2));
    }


}

FieldsComputer::~FieldsComputer()
{
    std::lock_guard<std::mutex> lock(gpu_points_mutex_);
    if (gpu_points_buffer != nullptr) {
        // Wait for any ongoing operations to complete
        while (is_gpu_points_in_use_.load()) {
            std::this_thread::yield();
        }
        cudaFree(gpu_points_buffer);
        gpu_points_buffer = nullptr;
    }
}

bool FieldsComputer::check_cuda_error(cudaError_t err, const char* operation) {
    if (err != cudaSuccess) {
        RCLCPP_ERROR(this->get_logger(), "CUDA %s failed: %s", 
                    operation, cudaGetErrorString(err));
        return false;
    }
    return true;
}

void FieldsComputer::pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    // Get number of points
    size_t num_points = msg->width * msg->height;
    
    // Create iterators for x,y,z fields
    sensor_msgs::PointCloud2Iterator<float> iter_x(*msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(*msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(*msg, "z");

    // Lock the GPU points update
    std::lock_guard<std::mutex> lock(gpu_points_mutex_);
    
    // Free old GPU memory if it exists
    if (gpu_points_buffer != nullptr) {
        bool expected = false;
        if (!is_gpu_points_in_use_.compare_exchange_strong(expected, true)) {
            // If we couldn't set it to true, it means the buffer is in use
            return;
        }
        
        // We successfully claimed the buffer, now we can free it
        cudaFree(gpu_points_buffer);
        is_gpu_points_in_use_.store(false);
    }

    // Create temporary host array and copy points
    std::vector<double3> points_double3(num_points);
    for (size_t i = 0; i < num_points; ++i, ++iter_x, ++iter_y, ++iter_z) {
        points_double3[i] = make_double3(
            static_cast<double>(*iter_x),
            static_cast<double>(*iter_y), 
            static_cast<double>(*iter_z)
        );
    }

    // Allocate GPU memory
    cudaError_t err = cudaMalloc(&gpu_points_buffer, num_points * sizeof(double3));
    if (!check_cuda_error(err, "malloc")) {
        return;
    }

    // Copy to GPU
    err = cudaMemcpy(gpu_points_buffer, points_double3.data(), 
                    num_points * sizeof(double3), cudaMemcpyHostToDevice);
    if (!check_cuda_error(err, "memcpy")) {
        cudaFree(gpu_points_buffer);
        gpu_points_buffer = nullptr;
        return;
    }

    // Update the count of points
    gpu_num_points_ = num_points;
}

void FieldsComputer::handle_obstacle_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response)
{
    std::lock_guard<std::mutex> lock(gpu_points_mutex_);
    if (!validate_request(response)) return;
    
    GPUGuard guard(is_gpu_points_in_use_);
    auto [agent_position, agent_velocity, goal_position] = extract_request_data(request);

    double3 net_force = obstacle_heuristic::launch_kernel(           
        gpu_points_buffer,
        gpu_num_points_,
        agent_position,
        agent_velocity,
        goal_position,
        agent_radius,
        mass_radius,
        detect_shell_rad,
        k_circular_force,
        max_allowable_force,
        false
    );

    process_response(net_force, request->agent_pose, response);
}


void FieldsComputer::handle_velocity_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response)
{
    std::lock_guard<std::mutex> lock(gpu_points_mutex_);
    if (!validate_request(response)) return;
    
    GPUGuard guard(is_gpu_points_in_use_);
    auto [agent_position, agent_velocity, goal_position] = extract_request_data(request);

    double3 net_force = velocity_heuristic::launch_kernel(           
        gpu_points_buffer,
        gpu_num_points_,
        agent_position,
        agent_velocity,
        goal_position,
        agent_radius,
        mass_radius,
        detect_shell_rad,
        k_circular_force,
        max_allowable_force,
        false
    );

    process_response(net_force, request->agent_pose, response);

}


void FieldsComputer::handle_goal_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response)
{
    std::lock_guard<std::mutex> lock(gpu_points_mutex_);
    if (!validate_request(response)) return;
    
    GPUGuard guard(is_gpu_points_in_use_);
    auto [agent_position, agent_velocity, goal_position] = extract_request_data(request);

    double3 net_force = goal_heuristic::launch_kernel(           
        gpu_points_buffer,
        gpu_num_points_,
        agent_position,
        agent_velocity,
        goal_position,
        agent_radius,
        mass_radius,
        detect_shell_rad,
        k_circular_force,
        max_allowable_force,
        false
    );

    process_response(net_force, request->agent_pose, response);
    
}

void FieldsComputer::handle_goalobstacle_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response)
{
    std::lock_guard<std::mutex> lock(gpu_points_mutex_);
    if (!validate_request(response)) return;
    
    GPUGuard guard(is_gpu_points_in_use_);
    auto [agent_position, agent_velocity, goal_position] = extract_request_data(request);

    double3 net_force = goalobstacle_heuristic::launch_kernel(           
        gpu_points_buffer,
        gpu_num_points_,
        agent_position,
        agent_velocity,
        goal_position,
        agent_radius,
        mass_radius,
        detect_shell_rad,
        k_circular_force,
        max_allowable_force,
        false
    );

    process_response(net_force, request->agent_pose, response);
}

bool FieldsComputer::validate_request (std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response) {
    if (k_circular_force == 0.0 || gpu_points_buffer == nullptr) {
        response->not_null = false;
        return false;
    }
    return true;
}


std::tuple<double3, double3, double3> FieldsComputer::extract_request_data(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request) {
    double3 agent_position = make_double3(
        request->agent_pose.position.x,
        request->agent_pose.position.y,
        request->agent_pose.position.z
    );

    double3 agent_velocity = make_double3(
        request->agent_velocity.x,
        request->agent_velocity.y,
        request->agent_velocity.z
    );

    double3 goal_position = make_double3(
        request->target_pose.position.x,
        request->target_pose.position.y,
        request->target_pose.position.z
    );

    if (!override_detect_shell_rad) {
        detect_shell_rad = request->detect_shell_rad;
    }

    return {agent_position, agent_velocity, goal_position};
}

void FieldsComputer::process_response(const double3& net_force, 
    const geometry_msgs::msg::Pose& agent_pose,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response) {
    
    RCLCPP_INFO(this->get_logger(), "Net force: x=%.10f, y=%.10f, z=%.10f, num_points=%d", 
                net_force.x, net_force.y, net_force.z, gpu_num_points_);
    
    response->circ_force.x = net_force.x;
    response->circ_force.y = net_force.y;
    response->circ_force.z = net_force.z;
    response->not_null = true;

    if (publish_force_vector) {
        force_vector_publisher(net_force, agent_pose, marker_pub_);
    }
}



void FieldsComputer::force_vector_publisher(const double3& net_force, const geometry_msgs::msg::Pose& agent_pose, rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub)
{
    // Publish force vector as marker
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "world";  // Adjust frame_id as needed
    marker.header.stamp = this->now();
    marker.ns = "force_vectors";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::ARROW;
    marker.action = visualization_msgs::msg::Marker::ADD;
    
    // Set start point (agent position)
    marker.points.resize(2);
    marker.points[0].x = agent_pose.position.x;
    marker.points[0].y = agent_pose.position.y;
    marker.points[0].z = agent_pose.position.z;
    
    // Set end point (agent position + scaled force)
    marker.points[1].x = agent_pose.position.x + net_force.x * force_viz_scale_;
    marker.points[1].y = agent_pose.position.y + net_force.y * force_viz_scale_;
    marker.points[1].z = agent_pose.position.z + net_force.z * force_viz_scale_;
    
    // Set marker properties
    marker.scale.x = 0.1;  // shaft diameter
    marker.scale.y = 0.2;  // head diameter
    marker.scale.z = 0.3;  // head length
    
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    marker.color.a = 1.0;
    
    marker_pub_->publish(marker);

}


int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FieldsComputer>());
    rclcpp::shutdown();
    return 0;
}