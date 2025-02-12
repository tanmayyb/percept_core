#ifndef FIELDS_COMPUTER_HPP_
#define FIELDS_COMPUTER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <vector_types.h>
#include <mutex>
#include <atomic>
#include "percept_interfaces/srv/agent_state_to_circ_force.hpp"

#include <cuda_runtime.h>
#include <vector_types.h>

class FieldsComputer : public rclcpp::Node
{
public:
    FieldsComputer();
    ~FieldsComputer();

private:
    bool check_cuda_error(cudaError_t err, const char* operation);
    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void handle_agent_state_to_circ_force(
        const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
        std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response);

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Service<percept_interfaces::srv::AgentStateToCircForce>::SharedPtr service_;
    double3* gpu_points_buffer{nullptr};
    size_t gpu_num_points_{0};
    std::mutex gpu_points_mutex_;
    std::atomic<bool> is_gpu_points_in_use_{false};
    double agent_radius{0.0};
    double mass_radius{0.0};
    double k_circular_force{0.0}; 
};

#endif  // FIELDS_COMPUTER_HPP_