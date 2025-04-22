#include "percept/fields_computer_cpu.hpp"

// std
#include <memory>
#include <thread>
#include <chrono>
#include <shared_mutex>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>

// msgs
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <visualization_msgs/msg/marker.hpp>



// Helper function to create a 3D point (replacing make_double3)
inline Point3D make_point3d(double x, double y, double z) {
    return Point3D(x, y, z);
}

// CPU implementations of CUDA kernel functionality namespaces
namespace nearest_obstacle_distance {

    
    double launch_cpu_kernel(
        const Point3D* points,
        size_t num_points,
        const Point3D& agent_position,
        double agent_radius,
        double mass_radius,
        double detect_shell_rad,
        bool show_processing_delay) 
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        double min_dist = std::numeric_limits<double>::max();
        for (size_t i = 0; i < num_points; ++i) {
            double dx = points[i].x - agent_position.x;
            double dy = points[i].y - agent_position.y;
            double dz = points[i].z - agent_position.z;
            
            double dist = std::sqrt(dx*dx + dy*dy + dz*dz) - agent_radius - mass_radius;
            min_dist = std::min(min_dist, dist);
        }
        
        if (show_processing_delay) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            std::cout << "Nearest obstacle distance computation took " << duration.count() << " microseconds" << std::endl;
        }
        
        return min_dist;
    }
}

// CPU implementation for obstacle heuristic
namespace obstacle_heuristic {

    Point3D calculate_rotation_vector(
      int i, 
      const Point3D* points, 
      int num_points, 
      Point3D mass_position, 
      Point3D mass_dist_vec_normalized
    ){
      double nn_distance = 1000.0;
      double nn_mass_dist_k;
      int nn_mass_idx = -1;
      Point3D nn_mass_position;
      Point3D obstacle_vec;
      Point3D current_vec;

      for(int k=0; k<num_points; k++){
        if(k != i){
          nn_mass_dist_k = mass_position.squared_distance(points[k]);
          if(nn_mass_dist_k < nn_distance){
            nn_distance = nn_mass_dist_k;
            nn_mass_idx = k;
          }
        }
      }
  
      nn_mass_position = points[nn_mass_idx];
      obstacle_vec = nn_mass_position - mass_position;
      current_vec = mass_dist_vec_normalized * mass_dist_vec_normalized.dot(obstacle_vec) - obstacle_vec;
      return current_vec.cross(mass_dist_vec_normalized).normalized();     
    }

    Point3D launch_cpu_kernel(
        const Point3D* points,
        size_t num_points,
        const Point3D& agent_position,
        const Point3D& agent_velocity,
        const Point3D& goal_position,
        double agent_radius,
        double mass_radius,
        double detect_shell_rad,
        double k_force,
        double max_allowable_force,
        bool show_processing_delay)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        Point3D net_force(0.0, 0.0, 0.0);
        Point3D goal_vec;
        Point3D mass_position;
        Point3D mass_dist_vec;
        Point3D mass_velocity; 
        Point3D mass_rvel_vec;
        Point3D force_vec;
        Point3D mass_dist_vec_normalized;
        Point3D current_vec;
        Point3D rot_vec;
        Point3D mass_rvel_vec_normalized;
        double dist_to_goal;
        double dist_to_mass;

        for (size_t i = 0; i < num_points; ++i) {
            // implementation of obstacle heuristic circ force
            goal_vec = goal_position - agent_position;
            dist_to_goal = goal_vec.norm();
            mass_position = points[i];
            mass_dist_vec = mass_position - agent_position;
            mass_velocity = {0.0, 0.0, 0.0};
            mass_rvel_vec = agent_velocity - mass_velocity;
            force_vec = {0.0, 0.0, 0.0};
            mass_dist_vec_normalized = mass_dist_vec.normalized();

            // "Skip this obstacle if it's behind us AND we're moving away from it"
            if(mass_dist_vec_normalized.dot(goal_vec.normalized()) < -1e-5 &&
                mass_dist_vec.dot(mass_rvel_vec) < -1e-5)
                {
                  continue; // skip this obstacle
                }

            dist_to_mass = mass_dist_vec.norm() - (agent_radius + mass_radius);
            dist_to_mass = fmax(dist_to_mass, 1e-5); // avoid division by zero
            
            // implement OBSTACLE HEURISTIC
            // calculate rotation vector, current vector, and force vector
            if(dist_to_mass < detect_shell_rad && mass_rvel_vec.norm() > 1e-10){ 

                rot_vec = calculate_rotation_vector(i, points, num_points, mass_position, mass_dist_vec_normalized);
                current_vec = mass_dist_vec_normalized.cross(rot_vec).normalized();

                // calculate force vector
                // force_vec = cross(mass_rvel_vec_normalized, cross(current_vec, mass_rvel_vec_normalized));
                //  A×(B×C) = B(A·C) - C(A·B)
                force_vec = (current_vec * mass_rvel_vec_normalized.dot(mass_rvel_vec_normalized)) - 
                    (mass_rvel_vec_normalized * mass_rvel_vec_normalized.dot(current_vec));
                force_vec = force_vec * (k_force / pow(dist_to_mass, 2));
            }
            net_force = net_force + force_vec;          
        }

        // clamp the force magnitude
        if (max_allowable_force > 0.0) {
          double force_magnitude = net_force.norm();   
          if (force_magnitude > max_allowable_force) {
            double scale = max_allowable_force / force_magnitude;
            net_force = net_force * scale;
          }
        }
        
        if (show_processing_delay) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            std::cout << "Obstacle heuristic computation took " << duration.count() << " microseconds" << std::endl;
        }
        return net_force;
    }
}

// CPU implementation for velocity heuristic
namespace velocity_heuristic {

    Point3D calculate_current_vec(Point3D mass_dist_vec_normalized, Point3D mass_rvel_vec_normalized){
        // Project out the component of velocity parallel to obstacle direction to get perpendicular component
        Point3D current_vec = mass_rvel_vec_normalized - (mass_dist_vec_normalized * mass_rvel_vec_normalized.dot(mass_dist_vec_normalized));
        if (current_vec.norm() < 1e-10) {
            current_vec = Point3D(0.0, 0.0, 1.0); // or make random vector
        }
        current_vec = current_vec.normalized(); // normalize the current vector
        return current_vec;
    }

    Point3D launch_cpu_kernel(
        const Point3D* points,
        size_t num_points,
        const Point3D& agent_position,
        const Point3D& agent_velocity,
        const Point3D& goal_position,
        double agent_radius,
        double mass_radius,
        double detect_shell_rad,
        double k_force,
        double max_allowable_force,
        bool show_processing_delay)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Simple implementation - align with velocity direction
        Point3D net_force(0.0, 0.0, 0.0);
        Point3D goal_vec;
        Point3D mass_position;
        Point3D mass_dist_vec;
        Point3D mass_velocity; 
        Point3D mass_rvel_vec;
        Point3D force_vec;
        Point3D mass_dist_vec_normalized;
        Point3D current_vec;
        Point3D rot_vec;
        Point3D mass_rvel_vec_normalized;
        double dist_to_goal;
        double dist_to_mass;

        for (size_t i = 0; i < num_points; ++i) {
            // implementation of velocity heuristic circ force
            goal_vec = goal_position - agent_position;
            dist_to_goal = goal_vec.norm();
            mass_position = points[i];
            mass_dist_vec = mass_position - agent_position;
            mass_velocity = {0.0, 0.0, 0.0};
            mass_rvel_vec = agent_velocity - mass_velocity;
            force_vec = {0.0, 0.0, 0.0};
            mass_dist_vec_normalized = mass_dist_vec.normalized();

            // implement VELOCITY HEURISTIC
            // calculate rotation vector, current vector, and force vector
            // "Skip this obstacle if it's behind us AND we're moving away from it"
            if(mass_dist_vec_normalized.dot(goal_vec.normalized()) < -1e-5 &&
                mass_dist_vec.dot(mass_rvel_vec) < -1e-5)
                {
                  continue; // skip this obstacle
                }
            dist_to_mass = mass_dist_vec.norm() - (agent_radius + mass_radius);
            dist_to_mass = fmax(dist_to_mass, 1e-5); // avoid division by zero

            // implement VELOCITY HEURISTIC
            // calculate rotation vector, current vector, and force vector
            if(dist_to_mass < detect_shell_rad && mass_rvel_vec.norm() > 1e-10){ 

                // create rotation vector
                rot_vec = {0.0, 0.0, 1.0}; // velocity heuristic does not use rotation vector to calculate current vector or force vector

                // calculate current vector
                mass_rvel_vec_normalized = mass_rvel_vec.normalized();
                current_vec = calculate_current_vec(mass_dist_vec_normalized, mass_rvel_vec_normalized);

                // calculate force vector
                // force_vec = cross(mass_rvel_vec_normalized, cross(current_vec, mass_rvel_vec_normalized));
                //  A×(B×C) = B(A·C) - C(A·B)
                force_vec = (current_vec * mass_rvel_vec_normalized.dot(mass_rvel_vec_normalized)) - 
                    (mass_rvel_vec_normalized * mass_rvel_vec_normalized.dot(current_vec));
                force_vec = force_vec * (k_force / pow(dist_to_mass, 2));
            }
            net_force = net_force + force_vec;            
        }
        
        // clamp the force magnitude
        if (max_allowable_force > 0.0) {
          double force_magnitude = net_force.norm();   
          if (force_magnitude > max_allowable_force) {
            double scale = max_allowable_force / force_magnitude;
            net_force = net_force * scale;
          }
        }

        if (show_processing_delay) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            std::cout << "Velocity heuristic computation took " << duration.count() << " microseconds" << std::endl;
        }
        
        return net_force;
    }
}

// CPU implementation for goal heuristic
namespace goal_heuristic {

    Point3D calculate_current_vec(Point3D mass_dist_vec_normalized, Point3D goal_vec){
        
    // Project goal vector onto the plane perpendicular to mass_dist_vec using vector rejection.
    // This creates a vector that points towards the goal while being perpendicular to the obstacle direction.
    Point3D current_vec = goal_vec - mass_dist_vec_normalized * mass_dist_vec_normalized.dot(goal_vec);

    if (current_vec.norm() < 1e-10) {
        current_vec = Point3D(0.0, 0.0, 1.0); // or make random vector
    }
    current_vec = current_vec.normalized(); // normalize the current vector
    return current_vec;
}
    
    Point3D launch_cpu_kernel(
        const Point3D* points,
        size_t num_points,
        const Point3D& agent_position,
        const Point3D& agent_velocity,
        const Point3D& goal_position,
        double agent_radius,
        double mass_radius,
        double detect_shell_rad,
        double k_force,
        double max_allowable_force,
        bool show_processing_delay)
    {
        auto start_time = std::chrono::high_resolution_clock::now();


        // Simple implementation - align with velocity direction
        Point3D net_force(0.0, 0.0, 0.0);
        Point3D goal_vec;
        Point3D mass_position;
        Point3D mass_dist_vec;
        Point3D mass_velocity; 
        Point3D mass_rvel_vec;
        Point3D force_vec;
        Point3D mass_dist_vec_normalized;
        Point3D current_vec;
        Point3D rot_vec;
        Point3D mass_rvel_vec_normalized;
        double dist_to_goal;
        double dist_to_mass;

        for (size_t i = 0; i < num_points; ++i) {

            // implementation of goal heuristic circ force
            goal_vec = goal_position - agent_position;
            dist_to_goal = goal_vec.norm();
            mass_position = points[i];
            mass_velocity = {0.0, 0.0, 0.0};
            mass_dist_vec = mass_position - agent_position;
            mass_rvel_vec = agent_velocity - mass_velocity;
            force_vec = {0.0, 0.0, 0.0};
            mass_dist_vec_normalized = mass_dist_vec.normalized();

            // implement GOAL HEURISTIC
            // calculate rotation vector, current vector, and force vector
            if(dist_to_mass < detect_shell_rad && mass_rvel_vec.norm() > 1e-10){ 

                // create rotation vector
                rot_vec = {0.0, 0.0, 1.0}; // velocity heuristic does not use rotation vector to calculate current vector or force vector

                // calculate current vector
                mass_rvel_vec_normalized = mass_rvel_vec.normalized();
                current_vec = calculate_current_vec(mass_dist_vec_normalized, mass_rvel_vec_normalized);

                // calculate force vector
                // force_vec = cross(mass_rvel_vec_normalized, cross(current_vec, mass_rvel_vec_normalized));
                //  A×(B×C) = B(A·C) - C(A·B)
                force_vec = (current_vec * mass_rvel_vec_normalized.dot(mass_rvel_vec_normalized)) - 
                    (mass_rvel_vec_normalized * mass_rvel_vec_normalized.dot(current_vec));
                force_vec = force_vec * (k_force / pow(dist_to_mass, 2));
            }
            net_force = net_force + force_vec;                 
        }
        
        // clamp the force magnitude
        if (max_allowable_force > 0.0) {
          double force_magnitude = net_force.norm();   
          if (force_magnitude > max_allowable_force) {
            double scale = max_allowable_force / force_magnitude;
            net_force = net_force * scale;
          }
        }

        if (show_processing_delay) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            std::cout << "Goal heuristic computation took " << duration.count() << " microseconds" << std::endl;
        }
        
        return net_force;
    }
}

// CPU implementation for goal-obstacle heuristic
namespace goalobstacle_heuristic {

    Point3D calculate_rotation_vector(
      int i, 
      const Point3D* points, 
      int num_points, 
      Point3D mass_position, 
      Point3D mass_dist_vec_normalized, 
      Point3D goal_vec
    ){
      double nn_distance = 1000.0;
      double nn_mass_dist_k;
      int nn_mass_idx = -1;
      Point3D nn_mass_position;
      Point3D obstacle_vec;
      Point3D obstacle_current_vec;
      Point3D current_vec;
      Point3D goal_current_vec;

      for(int k=0; k<num_points; k++){
        if(k != i){
          nn_mass_dist_k = mass_position.squared_distance(points[k]);
          if(nn_mass_dist_k < nn_distance){
            nn_distance = nn_mass_dist_k;
            nn_mass_idx = k;
          }
        }
      }
   
      nn_mass_position = points[nn_mass_idx];
      obstacle_vec = nn_mass_position - mass_position;

      // Current vector is perpendicular to obstacle surface normal and shows in opposite direction of obstacle_vec
      obstacle_current_vec = mass_dist_vec_normalized * mass_dist_vec_normalized.dot(obstacle_vec) - obstacle_vec;  

      // Project goal vector onto plane perpendicular to mass_dist_vec by removing its parallel component
      // This gives the component of the goal direction that is tangent to the obstacle surface
      goal_current_vec = goal_vec - mass_dist_vec_normalized * mass_dist_vec_normalized.dot(goal_vec);

      current_vec = goal_current_vec.normalized() + obstacle_current_vec.normalized();
      if (current_vec.norm() < 1e-10){
        current_vec = Point3D(0.0, 0.0, 1.0); // or make random vector
      }
      current_vec = current_vec.normalized(); // normalize the current vector
      return current_vec.cross(mass_dist_vec_normalized).normalized();     
    }
    
    Point3D launch_cpu_kernel(
        const Point3D* points,
        size_t num_points,
        const Point3D& agent_position,
        const Point3D& agent_velocity,
        const Point3D& goal_position,
        double agent_radius,
        double mass_radius,
        double detect_shell_rad,
        double k_force,
        double max_allowable_force,
        bool show_processing_delay)
    {
        auto start_time = std::chrono::high_resolution_clock::now(); 

        Point3D net_force(0.0, 0.0, 0.0);
        Point3D goal_vec;
        Point3D mass_position;
        Point3D mass_dist_vec;
        Point3D mass_velocity; 
        Point3D mass_rvel_vec;
        Point3D force_vec;
        Point3D mass_dist_vec_normalized;
        Point3D current_vec;
        Point3D rot_vec;
        Point3D mass_rvel_vec_normalized;
        double dist_to_goal;
        double dist_to_mass;
        double nn_distance;
        double nn_mass_dist_k;
        int nn_mass_idx;


        for (size_t i = 0; i < num_points; ++i) {

          goal_vec = goal_position - agent_position;
          dist_to_goal = goal_vec.norm();
          mass_position = points[i];
          mass_dist_vec = mass_position - agent_position;
          mass_velocity = {0.0, 0.0, 0.0};
          mass_rvel_vec = agent_velocity - mass_velocity;
          force_vec = {0.0, 0.0, 0.0};
          mass_dist_vec_normalized = mass_dist_vec.normalized();
          

          if (mass_dist_vec_normalized.dot(goal_vec.normalized()) < -1e-5 && mass_dist_vec.dot(mass_rvel_vec) < -1e-5){
            continue;
          }

          dist_to_mass = mass_dist_vec.norm() - (agent_radius + mass_radius);
          dist_to_mass = fmax(dist_to_mass, 1e-5); // avoid division by zero
          
          if (dist_to_mass < detect_shell_rad && mass_rvel_vec.norm() > 1e-10){
            rot_vec = calculate_rotation_vector(i, points, num_points, mass_position, mass_dist_vec_normalized, goal_vec);
            
            mass_rvel_vec_normalized = mass_rvel_vec.normalized();
            current_vec = mass_dist_vec_normalized.cross(rot_vec).normalized();

            // calculate force vector
            // force_vec = cross(mass_rvel_vec_normalized, cross(current_vec, mass_rvel_vec_normalized));
            //  A×(B×C) = B(A·C) - C(A·B)
            force_vec = current_vec * mass_rvel_vec_normalized.dot(mass_rvel_vec_normalized) - 
                mass_rvel_vec_normalized * mass_rvel_vec_normalized.dot(current_vec);
            force_vec = force_vec * (k_force / pow(dist_to_mass, 2));
          }
          net_force = net_force + force_vec;          
        }

        // clamp the force magnitude
        if (max_allowable_force > 0.0) {
          double force_magnitude = net_force.norm();   
          if (force_magnitude > max_allowable_force) {
            double scale = max_allowable_force / force_magnitude;
            net_force = net_force * scale;
          }
        }

        if (show_processing_delay) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            std::cout << "Goal-obstacle heuristic computation took " << duration.count() << " microseconds" << std::endl;
        }
        return net_force;
    }
}

// CPU implementation for random heuristic
namespace random_heuristic {

    Point3D make_random_vector(){
      static std::random_device rd;
      static std::mt19937 gen(rd());
      std::uniform_real_distribution<double> dist(-1.0, 1.0);
      return Point3D(dist(gen), dist(gen), dist(gen));
    }
    
    Point3D launch_cpu_kernel(
        const Point3D* points,
        size_t num_points,
        const Point3D& agent_position,
        const Point3D& agent_velocity,
        const Point3D& goal_position,
        double agent_radius,
        double mass_radius,
        double detect_shell_rad,
        double k_force,
        double max_allowable_force,
        bool show_processing_delay)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Create random force vector
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        
        Point3D net_force(0.0, 0.0, 0.0);
        Point3D goal_vec;
        Point3D mass_position;
        Point3D mass_dist_vec;
        Point3D mass_velocity; 
        Point3D mass_rvel_vec;
        Point3D force_vec;
        Point3D mass_dist_vec_normalized;
        Point3D current_vec;
        Point3D rot_vec;
        Point3D mass_rvel_vec_normalized;
        double dist_to_goal;
        double dist_to_mass;

        for (size_t i = 0; i < num_points; ++i) {
          goal_vec = goal_position - agent_position;
          dist_to_goal = goal_vec.norm();
          mass_position = points[i];
          mass_dist_vec = mass_position - agent_position;
          mass_velocity = {0.0, 0.0, 0.0};
          mass_rvel_vec = agent_velocity - mass_velocity;
          force_vec = {0.0, 0.0, 0.0};
          mass_dist_vec_normalized = mass_dist_vec.normalized();
          

          if (mass_dist_vec_normalized.dot(goal_vec.normalized()) < -1e-5 && mass_dist_vec.dot(mass_rvel_vec) < -1e-5){
            continue;
          }

          dist_to_mass = mass_dist_vec.norm() - (agent_radius + mass_radius);
          dist_to_mass = fmax(dist_to_mass, 1e-5); // avoid division by zero
          
          if (dist_to_mass < detect_shell_rad && mass_rvel_vec.norm() > 1e-10){
            rot_vec = goal_vec.normalized().cross(make_random_vector());
            
            mass_rvel_vec_normalized = mass_rvel_vec.normalized();
            current_vec = mass_dist_vec_normalized.cross(rot_vec).normalized();

            // calculate force vector
            // force_vec = cross(mass_rvel_vec_normalized, cross(current_vec, mass_rvel_vec_normalized));
            //  A×(B×C) = B(A·C) - C(A·B)
            force_vec = current_vec * mass_rvel_vec_normalized.dot(mass_rvel_vec_normalized) - 
                mass_rvel_vec_normalized * mass_rvel_vec_normalized.dot(current_vec);
            force_vec = force_vec * (k_force / pow(dist_to_mass, 2));
          }
          net_force = net_force + force_vec;          
        }

        // clamp the force magnitude
        if (max_allowable_force > 0.0) {
          double force_magnitude = net_force.norm();   
          if (force_magnitude > max_allowable_force) {
            double scale = max_allowable_force / force_magnitude;
            net_force = net_force * scale;
          }
        }

        if (show_processing_delay) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            std::cout << "Random heuristic computation took " << duration.count() << " microseconds" << std::endl;
        }
        
        return net_force;
    }
}

// Operation class to track queue tasks
struct Operation {
    enum Type { READ, WRITE } type;
    std::function<void()> task;
    std::promise<void> completion;
};

FieldsComputerCPU::FieldsComputerCPU() : Node("fields_computer_cpu")
{

  this->declare_parameter("agent_radius", 0.050);
  this->get_parameter("agent_radius", agent_radius);

  this->declare_parameter("mass_radius", 0.050);
  this->get_parameter("mass_radius", mass_radius);

  this->declare_parameter("potential_detect_shell_rad", 1.0);
  this->get_parameter("potential_detect_shell_rad", potential_detect_shell_rad);

  this->declare_parameter("show_netforce_output", false);
  this->get_parameter("show_netforce_output", show_netforce_output);

  this->declare_parameter("show_processing_delay", false);
  this->get_parameter("show_processing_delay", show_processing_delay);

  this->declare_parameter("show_requests", false);
  this->get_parameter("show_requests", show_service_request_received);

  // Heuristic enable/disable parameters.
  this->declare_parameter("disable_nearest_obstacle_distance", false);
  this->get_parameter("disable_nearest_obstacle_distance", disable_nearest_obstacle_distance);

  // Heuristic enable/disable parameters.
  this->declare_parameter("disable_obstacle_heuristic", false);
  this->get_parameter("disable_obstacle_heuristic", disable_obstacle_heuristic);

  this->declare_parameter("disable_velocity_heuristic", false);
  this->get_parameter("disable_velocity_heuristic", disable_velocity_heuristic);

  this->declare_parameter("disable_goal_heuristic", false);
  this->get_parameter("disable_goal_heuristic", disable_goal_heuristic);

  this->declare_parameter("disable_goalobstacle_heuristic", false);
  this->get_parameter("disable_goalobstacle_heuristic", disable_goalobstacle_heuristic);

  this->declare_parameter("disable_random_heuristic", false);
  this->get_parameter("disable_random_heuristic", disable_random_heuristic);

  RCLCPP_INFO(this->get_logger(), "Parameters:");
  RCLCPP_INFO(this->get_logger(), "  agent_radius: %.2f", agent_radius);
  RCLCPP_INFO(this->get_logger(), "  mass_radius: %.2f", mass_radius);
  RCLCPP_INFO(this->get_logger(), "  potential_detect_shell_rad: %.2f", potential_detect_shell_rad);
  RCLCPP_INFO(this->get_logger(), "  show_processing_delay: %s", show_processing_delay ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "  show_requests: %s", show_service_request_received ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "  use_cpu: true");
  RCLCPP_INFO(this->get_logger(), "Helper services:");
  RCLCPP_INFO(this->get_logger(), "  disable_nearest_obstacle_distance: %s", disable_nearest_obstacle_distance ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "Heuristics:");
  RCLCPP_INFO(this->get_logger(), "  disable_obstacle_heuristic: %s", disable_obstacle_heuristic ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "  disable_velocity_heuristic: %s", disable_velocity_heuristic ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "  disable_goal_heuristic: %s", disable_goal_heuristic ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "  disable_goalobstacle_heuristic: %s", disable_goalobstacle_heuristic ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "  disable_random_heuristic: %s", disable_random_heuristic ? "true" : "false");

  // Start the queue processor thread
  queue_processor_ = std::thread(&FieldsComputerCPU::process_queue, this);


  // Subscribe to pointcloud messages.
  subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/primitives", 10,
      std::bind(&FieldsComputerCPU::pointcloud_callback, this, std::placeholders::_1));

  // Create service servers for the helper services that are not disabled.
  if (!disable_nearest_obstacle_distance) {
    service_nearest_obstacle_distance = this->create_service<percept_interfaces::srv::AgentPoseToMinObstacleDist>(
        "/get_min_obstacle_distance",
        std::bind(&FieldsComputerCPU::handle_nearest_obstacle_distance, this,
                  std::placeholders::_1, std::placeholders::_2));
  }
  // Create service servers for the heuristics that are not disabled.
  if (!disable_obstacle_heuristic) {
    service_obstacle_heuristic = this->create_service<percept_interfaces::srv::AgentStateToCircForce>(
        "/get_obstacle_heuristic_circforce",
        std::bind(&FieldsComputerCPU::handle_obstacle_heuristic, this,
                  std::placeholders::_1, std::placeholders::_2));
  }
  if (!disable_velocity_heuristic) {
    service_velocity_heuristic = this->create_service<percept_interfaces::srv::AgentStateToCircForce>(
        "/get_velocity_heuristic_circforce",
        std::bind(&FieldsComputerCPU::handle_velocity_heuristic, this,
                  std::placeholders::_1, std::placeholders::_2));
  }
  if (!disable_goal_heuristic) {
    service_goal_heuristic = this->create_service<percept_interfaces::srv::AgentStateToCircForce>(
        "/get_goal_heuristic_circforce",
        std::bind(&FieldsComputerCPU::handle_goal_heuristic, this,
                  std::placeholders::_1, std::placeholders::_2));
  }
  if (!disable_goalobstacle_heuristic) {
    service_goalobstacle_heuristic = this->create_service<percept_interfaces::srv::AgentStateToCircForce>(
        "/get_goalobstacle_heuristic_circforce",
        std::bind(&FieldsComputerCPU::handle_goalobstacle_heuristic, this,
                  std::placeholders::_1, std::placeholders::_2));
  }
  if (!disable_random_heuristic) {
    service_random_heuristic = this->create_service<percept_interfaces::srv::AgentStateToCircForce>(
        "/get_random_heuristic_circforce",
        std::bind(&FieldsComputerCPU::handle_random_heuristic, this,
                  std::placeholders::_1, std::placeholders::_2));
  }
}


// Destructor
FieldsComputerCPU::~FieldsComputerCPU()
{
  stop_queue();
  if (queue_processor_.joinable()) {
    queue_processor_.join();
  }
  // Reset the shared pointer. Any ongoing service calls (that copied the pointer)
  // will keep the CPU memory alive until they finish.
  std::unique_lock<std::shared_timed_mutex> lock(points_mutex_);
  points_buffer_shared_.reset();
}


// Callback for processing incoming point cloud messages.
void FieldsComputerCPU::pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  // Create a copy of the message since we'll process it asynchronously
  auto msg_copy = std::make_shared<sensor_msgs::msg::PointCloud2>(*msg);
  
  enqueue_operation(OperationType::WRITE, [this, msg_copy]() {
    // Compute number of points
    size_t num_points = msg_copy->width * msg_copy->height;

    // Create iterators for the x, y, and z fields.
    sensor_msgs::PointCloud2Iterator<float> iter_x(*msg_copy, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(*msg_copy, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(*msg_copy, "z");

    // Create a new CPU buffer to hold the points
    auto new_points_buffer = std::make_shared<std::vector<Point3D>>(num_points);
    
    // Copy point cloud into the CPU buffer
    for (size_t i = 0; i < num_points; ++i, ++iter_x, ++iter_y, ++iter_z) {
      (*new_points_buffer)[i] = make_point3d(
          static_cast<double>(*iter_x),
          static_cast<double>(*iter_y),
          static_cast<double>(*iter_z));
    }

    // Update the points buffer with exclusive access
    std::unique_lock<std::shared_timed_mutex> lock(points_mutex_);
    points_buffer_shared_ = new_points_buffer;
    num_points_ = num_points;
  });
}


// Extracts agent, velocity, and goal data from the service request.
std::tuple<Point3D, Point3D, Point3D, double, double, double> FieldsComputerCPU::extract_request_data(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request)
{
  Point3D agent_position = make_point3d(
      request->agent_pose.position.x,
      request->agent_pose.position.y,
      request->agent_pose.position.z);

  Point3D agent_velocity = make_point3d(
      request->agent_velocity.x,
      request->agent_velocity.y,
      request->agent_velocity.z);

  Point3D goal_position = make_point3d(
      request->target_pose.position.x,
      request->target_pose.position.y,
      request->target_pose.position.z);

  double detect_shell_rad = request->detect_shell_rad;
  double k_force = request->k_force;
  double max_allowable_force = request->max_allowable_force;


  return std::make_tuple(agent_position, agent_velocity, goal_position, detect_shell_rad, k_force, max_allowable_force);
}


// Processes the net force returned by the CPU computation and publishes a response.
void FieldsComputerCPU::process_response(const Point3D& net_force,
                                      const geometry_msgs::msg::Pose& agent_pose,
                                      std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response)
{
  if (show_netforce_output) {
    RCLCPP_INFO(this->get_logger(), "Net force: x=%.10f, y=%.10f, z=%.10f, num_points=%zu",
                net_force.x, net_force.y, net_force.z, num_points_);
  }

  response->circ_force.x = net_force.x;
  response->circ_force.y = net_force.y;
  response->circ_force.z = net_force.z;
  response->not_null = true;

}



// Service handler for the nearest obstacle distance.
void FieldsComputerCPU::handle_nearest_obstacle_distance(
    const std::shared_ptr<percept_interfaces::srv::AgentPoseToMinObstacleDist::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentPoseToMinObstacleDist::Response> response)
{
  if (show_service_request_received) {
    RCLCPP_INFO(this->get_logger(), "Nearest obstacle distance service request received");
  }
  
  enqueue_operation(OperationType::READ, [this, request, response]() {
    std::shared_lock<std::shared_timed_mutex> lock(points_mutex_);
    auto points_buffer = points_buffer_shared_;
    if (!points_buffer || points_buffer->empty()) {
      response->distance = 0.0;
      return;
    }

    Point3D agent_position = make_point3d(
        request->agent_pose.position.x,
        request->agent_pose.position.y,
        request->agent_pose.position.z);

    double min_dist = nearest_obstacle_distance::launch_cpu_kernel(
        points_buffer->data(),
        num_points_,
        agent_position,
        agent_radius,
        mass_radius,
        potential_detect_shell_rad,
        show_processing_delay);

    response->distance = min_dist;
  });
}


template<typename HeuristicFunc>
void FieldsComputerCPU::handle_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response,
    HeuristicFunc kernel_launcher)
{
  enqueue_operation(OperationType::READ, [this, request, response, kernel_launcher]() {
    std::shared_lock<std::shared_timed_mutex> lock(points_mutex_);
    auto points_buffer = points_buffer_shared_;
    if (!points_buffer || points_buffer->empty()) {
      response->not_null = false;
      return;
    }

    auto [agent_position, agent_velocity, goal_position, detect_shell_rad, k_force, max_allowable_force] = extract_request_data(request);
    Point3D net_force = kernel_launcher(
        points_buffer->data(),
        num_points_,
        agent_position,
        agent_velocity, 
        goal_position,
        agent_radius,
        mass_radius,
        detect_shell_rad,
        k_force,
        max_allowable_force,
        show_processing_delay);

    process_response(net_force, request->agent_pose, response);
  });
}

// Replace individual handlers with templated versions
void FieldsComputerCPU::handle_obstacle_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response)
{
  if (show_service_request_received) {
    RCLCPP_INFO(this->get_logger(), "Obstacle heuristic service request received");
  }
  handle_heuristic(request, response, obstacle_heuristic::launch_cpu_kernel);
}

void FieldsComputerCPU::handle_velocity_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response)
{
  if (show_service_request_received) {
    RCLCPP_INFO(this->get_logger(), "Velocity heuristic service request received");
  }
  handle_heuristic(request, response, velocity_heuristic::launch_cpu_kernel);
}

void FieldsComputerCPU::handle_goal_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response)
{
  if (show_service_request_received) {
    RCLCPP_INFO(this->get_logger(), "Goal heuristic service request received");
  }
  handle_heuristic(request, response, goal_heuristic::launch_cpu_kernel);
}

void FieldsComputerCPU::handle_goalobstacle_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response)
{
  if (show_service_request_received) {
    RCLCPP_INFO(this->get_logger(), "Goal obstacle heuristic service request received");
  }
  handle_heuristic(request, response, goalobstacle_heuristic::launch_cpu_kernel);
}

void FieldsComputerCPU::handle_random_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response)
{
  if (show_service_request_received) {
    RCLCPP_INFO(this->get_logger(), "Random heuristic service request received");
  }
  handle_heuristic(request, response, random_heuristic::launch_cpu_kernel);
}

// Queue processing methods
void FieldsComputerCPU::process_queue()
{
  while (queue_running_) {
    std::shared_ptr<Operation> op;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      queue_cv_.wait(lock, [this] {
        return !operation_queue_.empty() || !queue_running_;
      });

      if (!queue_running_) break;

      op = operation_queue_.front();
      operation_queue_.pop();
    }

    // Execute the operation
    op->task();
    op->completion.set_value();
  }
}

void FieldsComputerCPU::enqueue_operation(OperationType type, std::function<void()> task)
{
  auto op = std::make_shared<Operation>();
  op->type = type;
  op->task = task;

  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    operation_queue_.push(op);
  }
  queue_cv_.notify_one();

  // Wait for completion
  op->completion.get_future().wait();
}

void FieldsComputerCPU::stop_queue()
{
  queue_running_ = false;
  queue_cv_.notify_all();
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FieldsComputerCPU>());
    rclcpp::shutdown();
    return 0;
}