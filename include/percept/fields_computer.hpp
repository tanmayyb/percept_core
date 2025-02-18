#ifndef FIELDS_COMPUTER_HPP_
#define FIELDS_COMPUTER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <vector_types.h>
#include <mutex>
#include <atomic>
#include "percept_interfaces/srv/agent_state_to_circ_force.hpp"
#include <visualization_msgs/msg/marker.hpp>

#include <cuda_runtime.h>
#include <vector_types.h>

class FieldsComputer : public rclcpp::Node
{
public:
    FieldsComputer();
    ~FieldsComputer();

private:
    bool check_cuda_error(cudaError_t err, const char* operation);

    // pointcloud buffer
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    // heuristics
    bool disable_obstacle_heuristic{false};
    bool disable_velocity_heuristic{false};
    bool disable_goal_heuristic{false};
    bool disable_goalobstacle_heuristic{false};

    rclcpp::Service<percept_interfaces::srv::AgentStateToCircForce>::SharedPtr service_obstacle_heuristic;
    rclcpp::Service<percept_interfaces::srv::AgentStateToCircForce>::SharedPtr service_velocity_heuristic;
    rclcpp::Service<percept_interfaces::srv::AgentStateToCircForce>::SharedPtr service_goal_heuristic;
    rclcpp::Service<percept_interfaces::srv::AgentStateToCircForce>::SharedPtr service_goalobstacle_heuristic;

    void handle_obstacle_heuristic(
        const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
        std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response);
    void handle_velocity_heuristic(
        const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
        std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response);
    void handle_goal_heuristic(
        const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
        std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response);
    void handle_goalobstacle_heuristic(
        const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
        std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response);

    // helpers
    bool validate_request(std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response);

    void process_response(const double3& net_force, 
    const geometry_msgs::msg::Pose& agent_pose,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response);

    std::tuple<double3, double3, double3> extract_request_data(
        const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request);
    

    // common parameters
    double3* gpu_points_buffer{nullptr};
    size_t gpu_num_points_{0};
    std::mutex gpu_points_mutex_;
    std::atomic<bool> is_gpu_points_in_use_{false};
    double agent_radius{0.0};
    double mass_radius{0.0};
    double k_circular_force{0.0}; 
    double max_allowable_force{0.0};
    double detect_shell_rad{0.0};
    bool override_detect_shell_rad{false};


    // experimental
    double force_viz_scale_{1.0};  // Initialize with default value
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
    bool publish_force_vector{false};
    
    void force_vector_publisher(const double3& net_force, const geometry_msgs::msg::Pose& agent_pose, rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub);

};

#endif  // FIELDS_COMPUTER_HPP_