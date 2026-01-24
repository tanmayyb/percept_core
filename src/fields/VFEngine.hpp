#ifndef VF_ENGINE_HPP_
#define VF_ENGINE_HPP_

// ROS 2
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/vector3.hpp"

// Interfaces
#include "percept_interfaces/srv/agent_state_to_circ_force.hpp"
#include "percept_interfaces/srv/agent_pose_to_min_obstacle_dist.hpp"

// CUDA
#include <vector_types.h>
#include <cuda_runtime.h>


// Standard Library
#include <memory>
#include <string>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <shared_mutex>
#include <thread>
#include <functional>

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



// std
#include <memory>
#include <thread>
#include <chrono>
#include <shared_mutex>
#include <map>

// CUDA runtime API
#include <cuda_runtime.h>

// msgs
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/point.hpp>

// tf2 includes
#include <tf2/LinearMath/Quaternion.hpp>
#include <tf2/LinearMath/Vector3.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>




enum class OperationType 
{ 
  READ, 
 
  WRITE 
};

struct Operation 
{
    OperationType type;

    std::function<void()> task;

    std::promise<void> completion;
};

class FieldsComputer : public rclcpp::Node 
{
  public:
    FieldsComputer();
    
    virtual ~FieldsComputer();

  private:
    
    double point_radius;
    
    bool show_netforce_output;

    bool show_processing_delay;

    bool show_service_request_received;
    
    std::shared_ptr<double> gpu_x_shared_;
    
    std::shared_ptr<double> gpu_y_shared_;
    
    std::shared_ptr<double> gpu_z_shared_;

    std::shared_ptr<int> gpu_nn_index_shared_;
    
    size_t gpu_num_points_ = 0;


    std::shared_timed_mutex gpu_points_mutex_;

    std::thread queue_processor_;

    std::queue<std::shared_ptr<Operation>> operation_queue_;

    std::mutex queue_mutex_;

    std::condition_variable queue_cv_;

    bool queue_running_ = true;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    
    rclcpp::Service<percept_interfaces::srv::AgentPoseToMinObstacleDist>::SharedPtr service_min_obstacle_distance;
    
    std::vector<rclcpp::Service<percept_interfaces::srv::AgentStateToCircForce>::SharedPtr> heuristic_services_;

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
    
    void process_queue();
    
    void enqueue_operation(OperationType type, std::function<void()> task);
    
    void stop_queue();
};

#endif // VF_ENGINE_HPP_