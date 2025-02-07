#include "percept/fields_computer.hpp"
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <cuda_runtime.h>
#include <thread>
#include "percept/ObstacleHeuristicCircForce.h"

FieldsComputer::FieldsComputer()
    : Node("fields_computer")
{
    subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/primitives", 10,
        std::bind(&FieldsComputer::pointcloud_callback, this, std::placeholders::_1));
    heuristic_kernel::hello_cuda_world();

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
    
    try {    
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
    
        double3 net_force = heuristic_kernel::launch_ObstacleHeuristic_circForce_kernel(           
            gpu_points_buffer, // on device
            gpu_num_points_,
            agent_position,
            agent_velocity,
            goal_position,
            0.01,  // k_circ
            2.0,  // detect_shell_rad_
            false // debug
        );

        // double3 net_force = make_double3(0.0, 0.0, 0.0);

        response->circ_force.x = net_force.x;
        response->circ_force.y = net_force.y;
        response->circ_force.z = net_force.z;
        response->not_null = true;
    }
    catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error in handle_agent_state_to_circ_force: %s", e.what());
        response->not_null = false;
    }
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FieldsComputer>());
    rclcpp::shutdown();
    return 0;
}