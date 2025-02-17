#include "percept/fields_computer.hpp"

#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <thread>

#include <cuda_runtime.h>
#include "percept/ObstacleHeuristicCircForce.h"
#include "percept/VelocityHeuristicCircForce.h"

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

    // experimental
    this->declare_parameter("force_viz_scale", 1.0);
    this->get_parameter("force_viz_scale", force_viz_scale_);
    
    
    marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
        "force_vector", 10);

    double force_viz_scale_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;

    RCLCPP_INFO(this->get_logger(), "Parameters:");
    RCLCPP_INFO(this->get_logger(), "  k_circular_force: %.2f", k_circular_force);
    RCLCPP_INFO(this->get_logger(), "  agent_radius: %.2f", agent_radius); 
    RCLCPP_INFO(this->get_logger(), "  mass_radius: %.2f", mass_radius);
    RCLCPP_INFO(this->get_logger(), "  max_allowable_force: %.2f", max_allowable_force);
    RCLCPP_INFO(this->get_logger(), "  detect_shell_rad: %.2f", detect_shell_rad);
    RCLCPP_INFO(this->get_logger(), "  force_viz_scale: %.2f", force_viz_scale_);


    subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/primitives", 10,
        std::bind(&FieldsComputer::pointcloud_callback, this, std::placeholders::_1));
    obstacle_heuristic::hello_cuda_world();
    velocity_heuristic::hello_cuda_world();


    service_ = this->create_service<percept_interfaces::srv::AgentStateToCircForce>(
        "/get_heuristic_circforce",
        std::bind(&FieldsComputer::handle_agent_state_to_circ_force, this,
                 std::placeholders::_1, std::placeholders::_2));
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

void FieldsComputer::handle_agent_state_to_circ_force(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response)
{
    std::lock_guard<std::mutex> lock(gpu_points_mutex_);
    

    if (k_circular_force == 0.0) {
        response->circ_force.x = 0.0;
        response->circ_force.y = 0.0;
        response->circ_force.z = 0.0;
        response->not_null = false;
        return;
    }

    if (gpu_points_buffer == nullptr) {
        response->not_null = false;
        return;
    }
    
    bool expected = false;
    if (!is_gpu_points_in_use_.compare_exchange_strong(expected, true)) {
        response->not_null = false;
        return;
    }
    
    // Use RAII to ensure is_gpu_points_in_use_ is always reset
    struct GPUGuard {
        std::atomic<bool>& flag;
        GPUGuard(std::atomic<bool>& f) : flag(f) {}
        ~GPUGuard() { flag.store(false); }
    } guard(is_gpu_points_in_use_);
    
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

    double3 net_force = obstacle_heuristic::launch_kernel(           
        gpu_points_buffer, // on device
        gpu_num_points_,
        agent_position,
        agent_velocity,
        goal_position,
        agent_radius,
        mass_radius,
        detect_shell_rad,
        k_circular_force,  // k_circ
        max_allowable_force,
        false // debug
    );

    RCLCPP_INFO(this->get_logger(), "Net force: x=%.10f, y=%.10f, z=%.10f, num_points=%d", net_force.x, net_force.y, net_force.z, gpu_num_points_);
    
    response->circ_force.x = net_force.x;
    response->circ_force.y = net_force.y;
    response->circ_force.z = net_force.z;
    response->not_null = true;

    // experimental
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
    marker.points[0].x = request->agent_pose.position.x;
    marker.points[0].y = request->agent_pose.position.y;
    marker.points[0].z = request->agent_pose.position.z;
    
    // Set end point (agent position + scaled force)
    marker.points[1].x = request->agent_pose.position.x + net_force.x * force_viz_scale_;
    marker.points[1].y = request->agent_pose.position.y + net_force.y * force_viz_scale_;
    marker.points[1].z = request->agent_pose.position.z + net_force.z * force_viz_scale_;
    
    // Set marker properties
    marker.scale.x = 0.1;  // shaft diameter
    marker.scale.y = 0.2;  // head diameter
    marker.scale.z = 0.3;  // head length
    
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    marker.color.a = 1.0;
    
    marker_pub_->publish(marker);


    return;

}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FieldsComputer>());
    rclcpp::shutdown();
    return 0;
}