#ifndef FIELDS_COMPUTER_HPP_
#define FIELDS_COMPUTER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include "percept_interfaces/srv/agent_state_to_circ_force.hpp"
#include <percept_interfaces/srv/agent_pose_to_min_obstacle_dist.hpp>

// #include <mutex>
#include <shared_mutex>
#include <atomic>

#include <cuda_runtime.h>
#include <vector_types.h>

#include <queue>
#include <condition_variable>
#include <functional>


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
  std::shared_ptr<int> gpu_nn_index_shared_;
  size_t gpu_num_points_{0};
  std::shared_timed_mutex gpu_points_mutex_;
  // std::mutex gpu_points_mutex_;
  // std::atomic<bool> is_gpu_points_in_use_{false};

  // common parameters
  double agent_radius{0.0};
  double mass_radius{0.0};
  double potential_detect_shell_rad{0.0};
  double navigation_function_K{0.0};
  double navigation_function_world_radius{0.0};

  // helper services parameters
  bool disable_nearest_obstacle_distance{false};  
  bool disable_obstacle_heuristic{false};
  bool disable_velocity_heuristic{false};
  bool disable_goal_heuristic{false};
  bool disable_goalobstacle_heuristic{false};
  bool disable_random_heuristic{false};
  bool disable_apf_heuristic{false};
  bool disable_navigation_function_force{false};

  // debug parameters
  bool show_netforce_output{false};
  bool show_processing_delay{false};
  bool show_service_request_received{false};

  // pointcloud buffer
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;

  // helper services
  rclcpp::Service<percept_interfaces::srv::AgentPoseToMinObstacleDist>::SharedPtr service_obstacle_distance_cost;

  // heuristic services
  rclcpp::Service<percept_interfaces::srv::AgentStateToCircForce>::SharedPtr service_obstacle_heuristic;
  rclcpp::Service<percept_interfaces::srv::AgentStateToCircForce>::SharedPtr service_velocity_heuristic;
  rclcpp::Service<percept_interfaces::srv::AgentStateToCircForce>::SharedPtr service_goal_heuristic;
  rclcpp::Service<percept_interfaces::srv::AgentStateToCircForce>::SharedPtr service_goalobstacle_heuristic;
  rclcpp::Service<percept_interfaces::srv::AgentStateToCircForce>::SharedPtr service_random_heuristic;
  rclcpp::Service<percept_interfaces::srv::AgentStateToCircForce>::SharedPtr service_apf_heuristic;
  rclcpp::Service<percept_interfaces::srv::AgentStateToCircForce>::SharedPtr service_navigation_function_force;

  // Operation queue structures
  enum class OperationType {
    WRITE,  // Pointcloud callback
    READ    // Service handlers
  };

  struct Operation {
    OperationType type;
    std::function<void()> task;
    std::promise<void> completion;
  };

  std::queue<std::shared_ptr<Operation>> operation_queue_;
  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::atomic<bool> queue_running_{true};
  std::thread queue_processor_;

  // Queue processing methods
  void process_queue();
  void enqueue_operation(OperationType type, std::function<void()> task);
  void stop_queue();

  // helpers
  bool check_cuda_error(cudaError_t err, const char* operation);
  std::tuple<double3, double3, double3, double, double, double> extract_request_data( const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request);
  void process_response(const double3& net_force, const geometry_msgs::msg::Pose& agent_pose,
  std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response);

  // handlers
  // pointcloud callback
  void pointcloud_callback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  // helper services handlers
  void handle_obstacle_distance_cost(
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
  void handle_apf_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request, std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response);
  void handle_navigation_function_force(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request, std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response);
  
  template<typename HeuristicFunc>
  void handle_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response,
    HeuristicFunc kernel_launcher, const std::string& heuristic_name);

  // nvtx
  void mark_start(const std::string& name, unsigned int color_hex);

};

#endif  // FIELDS_COMPUTER_HPP_