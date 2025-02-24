#ifndef FIELDS_COMPUTER_HPP_
#define FIELDS_COMPUTER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include "percept_interfaces/srv/agent_state_to_circ_force.hpp"
#include <percept_interfaces/srv/agent_pose_to_min_obstacle_dist.hpp>
#include <visualization_msgs/msg/marker.hpp>

// #include <mutex>
#include <shared_mutex>
#include <atomic>

#include <cuda_runtime.h>
#include <vector_types.h>


class FieldsComputer : public rclcpp::Node
{
public:
  FieldsComputer();
  virtual ~FieldsComputer();

private:

  // GPU buffer and synchronization members using double buffering:
  // Instead of a raw pointer, use a shared_ptr that wraps the GPU memory.
  // The custom deleter (defined in the implementation) will call cudaFree.
  // gpu buffer synchronization
  // double3* gpu_points_buffer{nullptr};
  std::shared_ptr<double3> gpu_points_buffer_shared_;
  std::shared_timed_mutex gpu_points_mutex_;
  size_t gpu_num_points_{0};
  // std::mutex gpu_points_mutex_;
  // std::atomic<bool> is_gpu_points_in_use_{false};

  // common parameters
  double agent_radius{0.0};
  double mass_radius{0.0};
  // double k_circular_force{0.0}; // deprecated

  double k_cf_velocity{0.0};
  double k_cf_obstacle{0.0};
  double k_cf_goal{0.0};
  double k_cf_goalobstacle{0.0};
  double k_cf_random{0.0};

  double detect_shell_rad{0.0};

  double max_allowable_force{0.0};
  bool override_detect_shell_rad{false};
  // helper services parameters
  bool disable_nearest_obstacle_distance{false};
  // heuristics parameters
  bool disable_obstacle_heuristic{false};
  bool disable_velocity_heuristic{false};
  bool disable_goal_heuristic{false};
  bool disable_goalobstacle_heuristic{false};
  bool disable_random_heuristic{false};
  // debug parameters
  bool show_netforce_output{false};
  bool show_processing_delay{false};
  // experimental
  double force_viz_scale_{1.0};
  bool publish_force_vector{false};


  // pointcloud buffer
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;

  // helper services
  rclcpp::Service<percept_interfaces::srv::AgentPoseToMinObstacleDist>::SharedPtr service_nearest_obstacle_distance;

  // heuristic services
  rclcpp::Service<percept_interfaces::srv::AgentStateToCircForce>::SharedPtr service_obstacle_heuristic;
  rclcpp::Service<percept_interfaces::srv::AgentStateToCircForce>::SharedPtr service_velocity_heuristic;
  rclcpp::Service<percept_interfaces::srv::AgentStateToCircForce>::SharedPtr service_goal_heuristic;
  rclcpp::Service<percept_interfaces::srv::AgentStateToCircForce>::SharedPtr service_goalobstacle_heuristic;
  rclcpp::Service<percept_interfaces::srv::AgentStateToCircForce>::SharedPtr service_random_heuristic;

  // experimental
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;

  // helpers
  bool check_cuda_error(cudaError_t err, const char* operation);
  bool waitForGpuBuffer();
  bool validate_request(std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response, double k_cf);
  std::tuple<double3, double3, double3> extract_request_data( const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request);
  void process_response(const double3& net_force, const geometry_msgs::msg::Pose& agent_pose,
  std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response);

  // handlers
  // pointcloud callback
  void pointcloud_callback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  // helper services handlers
  void handle_nearest_obstacle_distance(
    const std::shared_ptr<percept_interfaces::srv::AgentPoseToMinObstacleDist::Request> request, std::shared_ptr<percept_interfaces::srv::AgentPoseToMinObstacleDist::Response> response);
  // heuristics handlers
  void handle_goalobstacle_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request, std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response);
  void handle_velocity_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request, std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response);
  void handle_goal_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request, std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response);
  void handle_random_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request, std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response); 
  void handle_obstacle_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request, std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response);

  template<typename HeuristicFunc>
  void handle_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response,
    HeuristicFunc kernel_launcher,
    double k_cf);

  // experimental
  void force_vector_publisher(const double3& net_force, const geometry_msgs::msg::Pose& agent_pose, rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub);

};

#endif  // FIELDS_COMPUTER_HPP_