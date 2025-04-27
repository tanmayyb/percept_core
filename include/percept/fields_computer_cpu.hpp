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

// Define a struct to replace CUDA's double3
struct Point3D {
    double x, y, z;    
    Point3D() : x(0.0), y(0.0), z(0.0) {}
    Point3D(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    // Arithmetic operators
    Point3D operator+(const Point3D& other) const {
        return Point3D(x + other.x, y + other.y, z + other.z);
    }

    Point3D operator-(const Point3D& other) const {
        return Point3D(x - other.x, y - other.y, z - other.z);
    }

    Point3D operator*(const double scalar) const {
        return Point3D(x * scalar, y * scalar, z * scalar);
    }

    // Vector operations
    double dot(const Point3D& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    Point3D cross(const Point3D& other) const {
        return Point3D(y * other.z - z * other.y,
                      z * other.x - x * other.z,
                      x * other.y - y * other.x);
    }

    double norm() const {
        return sqrt(x * x + y * y + z * z);
    }

    double squared_distance(const Point3D& other) const {
        double dx = x - other.x;
        double dy = y - other.y;
        double dz = z - other.z;
        return dx * dx + dy * dy + dz * dz;
    }

    Point3D normalized() const {
        double mag = norm();
        if (mag > 0.0) {
            return (*this) * (1.0 / mag);
        }
        return Point3D(0.0, 0.0, 0.0);
    }
};

// Non-member operator for scalar multiplication (allows scalar * Point3D)
inline Point3D operator*(const double scalar, const Point3D& point) {
    return point * scalar;
}


class FieldsComputerCPU : public rclcpp::Node
{
public:
  FieldsComputerCPU();
  virtual ~FieldsComputerCPU();

private:

  // CPU buffer and synchronization members using double buffering:
  // Instead of a raw pointer, use a shared_ptr that wraps the CPU memory.
  // The custom deleter (defined in the implementation) will call free.
  // cpu buffer synchronization
  std::shared_ptr<std::vector<Point3D>> points_buffer_shared_;
  std::shared_ptr<int> nn_index_shared_;
  std::shared_timed_mutex points_mutex_;
  size_t num_points_;

  // common parameters
  double agent_radius{0.0};
  double mass_radius{0.0};
  double potential_detect_shell_rad{0.0};

  // helper services parameters
  bool disable_nearest_obstacle_distance{false};  
  bool disable_obstacle_heuristic{false};
  bool disable_velocity_heuristic{false};
  bool disable_goal_heuristic{false};
  bool disable_goalobstacle_heuristic{false};
  bool disable_random_heuristic{false};
  bool disable_apf_heuristic{false};
  
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
  std::tuple<Point3D, Point3D, Point3D, double, double, double> extract_request_data( const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request);
  void process_response(const Point3D& net_force, const geometry_msgs::msg::Pose& agent_pose,
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

  template<typename HeuristicFunc>
  void handle_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response,
    HeuristicFunc kernel_launcher, const std::string& heuristic_name);

};

#endif  // FIELDS_COMPUTER_CPU_HPP_