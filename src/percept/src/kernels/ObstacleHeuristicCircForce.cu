// the usual
#include <iostream>
#include <vector>

// necessary evils
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>

// include the header
#include "percept/ObstacleHeuristicCircForce.h"

// time keeper
#include <chrono>


#define threads 256
//  NEED EDGE CONDITION HANDLER 
//  i.e. when known num_obstacles < 256


namespace heuristic_kernel{

__device__ inline double3 operator+(const double3& a, const double3& b) {
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline double3 operator-(const double3& a, const double3& b) {
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline double3 operator*(const double3& a, const double scalar) {
    return make_double3(a.x * scalar, a.y * scalar, a.z * scalar);
}

__device__ inline double norm(const double3 &v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

__device__ inline double norm_reciprocal(const double3 &v) {
    double mag2 = v.x * v.x + v.y * v.y + v.z * v.z;
    return mag2 > 0.0 ? 1.0 / sqrt(mag2) : 0.0;
}

__device__ inline double fma(double a, double b, double c) {
    return __fma_rn(a, b, c); // computes a * b + c in one instruction
}

__device__ inline double dot(const double3 &a, const double3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline double3 cross(const double3 &a, const double3 &b) {
    return make_double3(a.y * b.z - a.z * b.y,
                        a.z * b.x - a.x * b.z,
                        a.x * b.y - a.y * b.x);
}

__device__ inline double3 normalize(const double3 &v) {
    double mag = sqrt(dot(v, v));
    if (mag > 0.0) {
        return v * (1.0 / mag);
    } else {
        return make_double3(0.0, 0.0, 0.0);
    }
}


// // pmaf functions
// __device__ void calculateCurrForce(
//   double* curr_force,
//   double* rot_vec,
//   double* obstacle_pos_vec, 
//   double* agent_pos_vec, 
//   double* agent_vel_vec, 
//   double* goal_pos_vec, 
//   double* rel_vel, 
//   double k_circ,
//   double dist_obs
// ){

//   double rel_vel_normalized[3];
//   double rel_vel_norm; 

//   // double vel_norm = rel_vel.norm();
//   norm(rel_vel_norm, rel_vel);

//   if(rel_vel_norm!=0.0){
//     // calculate currentVector

//     double cfagent_to_obs[3], current_vec[3], crossproduct1[3], crossproduct2[3];
//     double cfagent_to_obs_normalized[3], current_vec_normalized[3];
//     double scalar1;

//     //   Eigen::Vector3d normalized_vel = rel_vel / vel_norm;
//     normalize_vector(rel_vel_normalized, rel_vel);

//     // Eigen::Vector3d cfagent_to_obs{obstacles[obstacle_id].getPosition() - agent_pos};  
//     subtract_vectors(cfagent_to_obs, obstacle_pos_vec, agent_pos_vec);

//     // cfagent_to_obs.normalize();
//     normalize_vector(cfagent_to_obs_normalized, cfagent_to_obs);

//     // Eigen::Vector3d current{cfagent_to_obs.cross(field_rotation_vecs.at(obstacle_id))};
//     cross_vectors(current_vec, cfagent_to_obs_normalized, rot_vec);

//     // current.normalize();
//     normalize_vector(current_vec_normalized, current_vec);

//     // curr_force = (k_circ / pow(dist_obs, 2)) * rel_vel_normalized.cross(current.cross(rel_vel_normalized));
//     scalar1 = k_circ / pow(dist_obs,2);

//     cross_vectors(crossproduct1, current_vec_normalized, rel_vel_normalized);
//     cross_vectors(crossproduct2, rel_vel_normalized, crossproduct1);
//     scale_vector(curr_force, crossproduct2, scalar1);
//   }

// }



// __device__ void calculateRotationVector(
//   double* rot_vec_result,
//   int &closest_obstacle_it, 
//   int num_obstacles, 
//   ghostplanner::cfplanner::Obstacle *obstacles, 
//   int obstacle_id,
//   double* agent_pos,
//   double* goal_pos,
//   double* goal_vec
// ){

//   double dist_vec[3], obstacle_pos_vec[3], active_obstacle_pos_vec[3], dist_obs;
//   double min_dist_obs = 100.0;

//   for(int i=0; i<num_obstacles; i++){
//     if (i != obstacle_id) {
//       // double dist_obs{(obstacles[obstacle_id].getPosition() - obstacles[i].getPosition()).norm()};
//       get_obstacle_position_vector(active_obstacle_pos_vec, obstacles[obstacle_id]);
//       get_obstacle_position_vector(obstacle_pos_vec, obstacles[i]);
//       subtract_vectors(dist_vec, active_obstacle_pos_vec, obstacle_pos_vec);
//       norm(dist_obs, dist_vec);

//       if(min_dist_obs > dist_obs){
//         min_dist_obs = dist_obs;
//         closest_obstacle_it = i;
//       }
//     }
//   }

//   // printf("closest_obstacle_it: %d\n", closest_obstacle_it);

//   double obstacle_vec[3], cfagent_to_obs[3], cfagent_to_obs_normalized[3]; 
//   double cfagent_to_obs_scaled[3], dot_product1, dot_product2, current_norm;
//   double obst_current[3], goal_current[3], current_vec[3], rot_vec[3];
//   double obst_current_normalized[3], goal_current_normalized[3], current_normalized[3], rot_vec_normalized[3];

//   // Vector from active obstacle to the obstacle which is closest to the active obstacle
//   // Eigen::Vector3d obstacle_vec = obstacles[closest_obstacle_it].getPosition() - obstacles[obstacle_id].getPosition();
//   get_obstacle_position_vector(obstacle_pos_vec, obstacles[closest_obstacle_it]);
//   get_obstacle_position_vector(active_obstacle_pos_vec, obstacles[obstacle_id]);
//   subtract_vectors(obstacle_vec, obstacle_pos_vec, active_obstacle_pos_vec);

//   // Eigen::Vector3d cfagent_to_obs{obstacles[obstacle_id].getPosition() - agent_pos};
//   subtract_vectors(cfagent_to_obs, active_obstacle_pos_vec, agent_pos);

//   // cfagent_to_obs.normalize();
//   normalize_vector(cfagent_to_obs_normalized, cfagent_to_obs);

//   // Current vector is perpendicular to obstacle surface normal and shows in opposite direction of obstacle_vec
//   // Eigen::Vector3d obst_current{ (cfagent_to_obs * obstacle_vec.dot(cfagent_to_obs)) - obstacle_vec};
//   dot_vectors(dot_product1, obstacle_vec, cfagent_to_obs_normalized);
//   scale_vector(cfagent_to_obs_scaled, cfagent_to_obs_normalized, dot_product1);
//   subtract_vectors(obst_current, cfagent_to_obs_scaled, obstacle_vec);

//   // passed by kernel so we ingore: Eigen::Vector3d goal_vec{goal_pos - agent_pos};
//   // Eigen::Vector3d goal_current{goal_vec - cfagent_to_obs * (cfagent_to_obs.dot(goal_vec))};
//   dot_vectors(dot_product2, cfagent_to_obs_normalized, goal_vec);
//   scale_vector(cfagent_to_obs_scaled, cfagent_to_obs_normalized, dot_product2); // reusing cfagent_to_obs_scaled
//   subtract_vectors(goal_current, goal_vec, cfagent_to_obs_scaled);

//   // Eigen::Vector3d current{goal_current.normalized() +
//   //                         obst_current.normalized()};
//   normalize_vector(goal_current_normalized, goal_current);
//   normalize_vector(obst_current_normalized, obst_current);
//   add_vectors(current_vec, goal_current_normalized, obst_current_normalized);

//   // printf("%f\t%f\t%f\n", current_vec[0],current_vec[1],current_vec[2]);

//   // check norm
//   norm(current_norm, current_vec);
//   if (current_norm < 1e-10) {
//     current_vec[0] = 0.0;
//     current_vec[1] = 0.0;
//     current_vec[2] = 1.0;
//   }
//   normalize_vector(current_normalized, current_vec);

//   // get rotation vector
//   // Eigen::Vector3d rot_vec{current.cross(cfagent_to_obs)};
//   cross_vectors(rot_vec, current_normalized, cfagent_to_obs_normalized);

//   // rot_vec.normalize();
//   normalize_vector(rot_vec_normalized, rot_vec);

//   // return rot_vec_normalized;
//   copy_vector(rot_vec_result, rot_vec_normalized);
// }



__global__ void ObstacleHeuristic_circForce_kernel(
    double3* d_net_force,
    double3* d_masses,
    size_t num_masses,
    double3 agent_position,
    double3 agent_velocity,
    double k_circ,
    double detect_shell_rad_
){
    extern __shared__ double3 sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    sdata[tid] = make_double3(0.0, 0.0, 0.0);

    // Each thread computes a force if within bound

    if (i>=num_masses){
        return;
    }


    // TODO: Compute your force calculation here
    // For now, just using a placeholder calculation
    double3 force = make_double3(0.0, 0.0, 0.0);
    sdata[tid] = force;





    // Perform reduction in shared memory
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid+s<num_masses) {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 of each block adds the block's sum to the global sum using atomics
    if (tid == 0) {
        atomicAdd(&(d_net_force->x), sdata[0].x);
        atomicAdd(&(d_net_force->y), sdata[0].y);
        atomicAdd(&(d_net_force->z), sdata[0].z);
    }

}

double3 launch_ObstacleHeuristic_circForce_kernel(
    double3* d_masses,
    size_t num_masses,
    double3 agent_position,
    double3 agent_velocity,
    double k_circ, 
    double detect_shell_rad_,
    bool debug
){
    // Copy agent data to device
    double3 d_agent_position = agent_position;  
    double3 d_agent_velocity = agent_velocity;

    // Allocate device memory for net force
    double3* d_net_force;
    cudaMalloc(&d_net_force, sizeof(double3));
    cudaMemset(d_net_force, 0, sizeof(double3));

    // Ceiling division
    int num_blocks = (num_masses + threads - 1) / threads;  
    size_t shared_mem_size = threads * sizeof(double3);
   
    ObstacleHeuristic_circForce_kernel<<<num_blocks, threads, shared_mem_size>>>(
        d_net_force, d_masses, num_masses,
        d_agent_position, d_agent_velocity,
        k_circ, detect_shell_rad_
    );

    // Copy result back to host
    double3 net_force;
    cudaMemcpy(&net_force, d_net_force, sizeof(double3), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_net_force);
    // Don't free d_masses here as it was allocated elsewhere

    return net_force;
}







// best function ever
__host__  void hello_cuda_world(){
  std::cout<<"Hello CUDA World!"<<std::endl;
}


}