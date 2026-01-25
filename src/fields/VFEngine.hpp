#ifndef VF_ENGINE_HPP_
#define VF_ENGINE_HPP_

// CUDA
#include "cuda_vector_ops.cuh"

// Std
#include <memory>
#include <thread>
#include <chrono>
#include <shared_mutex>
#include <map>
#include <string>
#include <vector>
#include <queue>
#include <condition_variable>
#include <future>
#include <functional>

// ROS 2
#include "rclcpp/rclcpp.hpp"
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/point.hpp>
#include "geometry_msgs/msg/vector3.hpp"

// tf2
#include <tf2/LinearMath/Quaternion.hpp>
#include <tf2/LinearMath/Vector3.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// Interfaces
#include "percept_interfaces/srv/agent_state_to_circ_force.hpp"
#include "percept_interfaces/srv/agent_pose_to_min_obstacle_dist.hpp"


// --- Kernel Wrappers ---
// Artificial Potential Fields
extern "C" double3 artificial_potential_field_kernel(
  double* d_points_x, double* d_points_y, double* d_points_z,
  size_t num_points, double3 agent_position, double3 agent_velocity, double3 goal_position, 
  double agent_radius, double point_radius,
  double detect_shell_rad, double k_force, double max_allowable_force, bool debug);

// Velocity Heuristic
extern "C" double3 velocity_heuristic_kernel(
  double* d_points_x, double* d_points_y, double* d_points_z,
  size_t num_points, double3 agent_position, double3 agent_velocity, double3 goal_position, 
  double agent_radius, double point_radius,
  double detect_shell_rad, double k_circ, double max_allowable_force, bool debug);

// Goal Heuristic
extern "C" double3 goal_heuristic_kernel(
  double* d_points_x, double* d_points_y, double* d_points_z,
  size_t num_points, double3 agent_position, double3 agent_velocity, double3 goal_position, 
  double agent_radius, double point_radius,
  double detect_shell_rad, double k_circ, double max_allowable_force, bool debug);

// Min Obstacle Distance
extern "C" double min_obstacle_distance_kernel(
  double* d_points_x, double* d_points_y, double* d_points_z,
  size_t num_masses, double3 agent_position, bool debug);

// Spatial Hashing NN
extern "C" {
  void build_spatial_index(const double* d_x, const double* d_y, const double* d_z,
                           uint32_t* d_cell_hashes, uint32_t* d_point_indices,
                           uint32_t* d_hash_starts, uint32_t* d_hash_ends,
                           int n, GridConfig config, uint32_t hash_size,
                           cudaStream_t stream);

  void find_nearest_neighbors(const double* d_x, const double* d_y, const double* d_z,
                              const uint32_t* d_sorted_indices,
                              const uint32_t* d_cell_starts, const uint32_t* d_cell_ends,
                              int* d_nearest_idx, int n, GridConfig config, uint32_t hash_size);
}

// --- GPU Snapshot ---
// Contains a snapshot for R/W
struct GpuSnapshot {
  std::shared_ptr<double> x;

  std::shared_ptr<double> y;

  std::shared_ptr<double> z;

  std::shared_ptr<int> nn_indices;

  size_t num_points;

  GpuSnapshot() : num_points(0) {}
};



class FieldsComputer : public rclcpp::Node
{
  public:
    FieldsComputer();
    
    virtual ~FieldsComputer();

  private:  
    cudaStream_t compute_stream_;

    int active_device_id_ = 0;

    // ROS2 params
    double point_radius;
    
    bool show_netforce_output;

    bool show_processing_delay;

    // Atomic Snapshot
    std::shared_ptr<const GpuSnapshot> current_snapshot_;

    // NNS vars
    size_t max_points_;

    uint32_t hash_table_size_;

    GridConfig grid_config_;

    uint32_t *d_hashes_ptr, *d_indices_ptr, *d_starts_ptr, *d_ends_ptr;

    // Callback groups
    rclcpp::CallbackGroup::SharedPtr service_cb_group_;

    rclcpp::CallbackGroup::SharedPtr producer_cb_group_;

    // ROS handles
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;

    rclcpp::Service<percept_interfaces::srv::AgentPoseToMinObstacleDist>::SharedPtr service_min_obstacle_distance;
    
    std::vector<rclcpp::Service<percept_interfaces::srv::AgentStateToCircForce>::SharedPtr> heuristic_services_;

    // Methods
    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    
    template<typename HeuristicFunc>
    void handle_heuristic(
      const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
      std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response,
      HeuristicFunc kernel_launcher, 
      const std::string& name
    );

    void handle_min_obstacle_distance(
      const std::shared_ptr<percept_interfaces::srv::AgentPoseToMinObstacleDist::Request> request,
      std::shared_ptr<percept_interfaces::srv::AgentPoseToMinObstacleDist::Response> response
    );

    std::tuple<double3, double3, double3, double, double, double, double> extract_request_data(
      const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request
    );

    void process_response(
      const double3& net_force,
      const geometry_msgs::msg::Pose& agent_pose,
      std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response
    );

    bool check_cuda_error(cudaError_t err, const char* operation);

    void setupDevice();

    void setupParamsAndServices();

    void allocate_producer_workspace();

    void free_producer_workspace();

};

#endif // VF_ENGINE_HPP_