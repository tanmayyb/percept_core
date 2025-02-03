// the usual
#include <iostream>
#include <vector>

// necessary evils
#include <cuda.h>
#include <cuda_runtime.h>

// include the header
#include "percept/ObstacleHeuristicCircForce.h"

// time keeper
#include <chrono>


#define threads 256
//  NEED EDGE CONDITION HANDLER 
//  i.e. when known num_obstacles < 256


namespace cuda_kernel{
    
// helper functions
// __host__ __device__ void init_vector(double* result_vec){
//   result_vec[0] = 0.0;
//   result_vec[1] = 0.0;
//   result_vec[2] = 0.0;
// }

// __host__ __device__ void copy_vector(double* result_vec, double* ref_vec){
//   result_vec[0] = ref_vec[0];
//   result_vec[1] = ref_vec[1];
//   result_vec[2] = ref_vec[2];
// }

// __host__ __device__ void place_vector_in_array_at_index(double* array, double* vec, int index){
//   array[index+0] = vec[0];
//   array[index+1] = vec[1];
//   array[index+2] = vec[2];
// }

// __host__ __device__ void add_vectors(double* result_vec, double* vec1, double* vec2){
//   result_vec[0] = vec1[0] + vec2[0];
//   result_vec[1] = vec1[1] + vec2[1];
//   result_vec[2] = vec1[2] + vec2[2];
// }

// __host__ __device__ void subtract_vectors(double* result_vec, double* vec1, double* vec2){
//   result_vec[0] = vec1[0] - vec2[0];
//   result_vec[1] = vec1[1] - vec2[1];
//   result_vec[2] = vec1[2] - vec2[2];
// }

// __host__ __device__ void dot_vectors(double &result, double* vec1, double *vec2){
//   double product[3];
//   product[0] = vec1[0] * vec2[0];
//   product[1] = vec1[1] * vec2[1];
//   product[2] = vec1[2] * vec2[2];
//   result = product[0] + product[1] + product[2];
// }

// __host__ __device__ void cross_vectors(double* result_vec, double* vec1, double* vec2) {
//   result_vec[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1];
//   result_vec[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2];
//   result_vec[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0];
// }

// __host__ __device__ void normalize_vector(double* result_vec, double* orig_vec){
//   double orig_vec_mag = sqrt(orig_vec[0]*orig_vec[0] + orig_vec[1]*orig_vec[1] + orig_vec[2]*orig_vec[2]); 
//   if (orig_vec_mag == 0.f){
//     result_vec[0] = 0.0;
//     result_vec[1] = 0.0;
//     result_vec[2] = 0.0;
//   }
//   else{
//     result_vec[0] = orig_vec[0]/orig_vec_mag;
//     result_vec[1] = orig_vec[1]/orig_vec_mag;
//     result_vec[2] = orig_vec[2]/orig_vec_mag;
//   }
// }

// __host__ __device__ void scale_vector(double* result_vec, double* orig_vec, double scalar){
//   result_vec[0] = orig_vec[0]*scalar;
//   result_vec[1] = orig_vec[1]*scalar;
//   result_vec[2] = orig_vec[2]*scalar;
// }

// __host__ __device__ void norm(double &result, double* vec){
//   result = sqrt(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]);
// }


// // pmaf helper functions
// __host__ __device__ void get_obstacle_position_vector(double* result_vector, ghostplanner::cfplanner::Obstacle &obstacle){
//   result_vector[0] = obstacle.getPosX();
//   result_vector[1] = obstacle.getPosY();
//   result_vector[2] = obstacle.getPosZ();
// }

// __host__ __device__ void get_obstacle_velocity_vector(double* result_vector, ghostplanner::cfplanner::Obstacle &obstacle){
//   result_vector[0] = obstacle.getVelX();
//   result_vector[1] = obstacle.getVelY();
//   result_vector[2] = obstacle.getVelZ();
// }

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


// // fancy kernel that does everything
// __global__ void circForce_kernel(
//   int num_obstacles,
//   double* force,
//   ghostplanner::cfplanner::Obstacle *obstacles,
//   double* goal_pos_vec,
//   double* goal_vec,
//   double* agent_pos_vec,
//   double* agent_vel_vec,
//   int* active_obstacles,
//   double min_obs_dist_,
//   double detect_shell_rad_,
//   double k_circ
// ){
//   int i = blockIdx.x * blockDim.x + threadIdx.x;   // i refers to obstacle being computed
//   if(i >= num_obstacles) return; 

//   double robot_obstacle_vec[3], rel_vel[3], obstacle_pos_vec[3], obstacle_vel_vec[3];

//   // get robot_obstacle_vec
//   get_obstacle_position_vector(obstacle_pos_vec, obstacles[i]);
//   subtract_vectors(robot_obstacle_vec, obstacle_pos_vec, agent_pos_vec);

//   // get rel_vel
//   get_obstacle_velocity_vector(obstacle_vel_vec, obstacles[i]);
//   subtract_vectors(rel_vel, obstacle_vel_vec, agent_vel_vec);

//   // if (robot_obstacle_vec.normalized().dot(goal_vec.normalized()) < -0.01 && robot_obstacle_vec.dot(rel_vel) < -0.01) {continue;}
//   double dot_product1, dot_product2, robot_obstacle_vec_normalized[3], goal_vec_normalized[3];
//   normalize_vector(robot_obstacle_vec_normalized, robot_obstacle_vec);
//   normalize_vector(goal_vec_normalized, goal_vec);
//   dot_vectors(dot_product1, robot_obstacle_vec_normalized, goal_vec);
//   dot_vectors(dot_product2, robot_obstacle_vec, rel_vel);
//   if (dot_product1 < -0.01 && dot_product2 < -0.01){ // compute condition
//     return;
//   }

//   // double dist_obs{robot_obstacle_vec.norm() - (rad_ + obstacles.at(i).getRadius())};
//   double norm1;
//   norm(norm1, robot_obstacle_vec);
//   // const double rad_ = 0.5; // what is this for?
//   const double rad_ = 0.0; // what is this for?
//   double dist_obs = norm1 - rad_ + obstacles[i].getRad();

//   // get dist_obs and check if more than min_obs_dist_
//   dist_obs = max(dist_obs, 1e-5);
//   if (dist_obs<min_obs_dist_)
//     min_obs_dist_ = dist_obs;

//   double curr_force[3];
//   init_vector(curr_force);

//   if(dist_obs < detect_shell_rad_){
//     // calculate rotation vector (Goal Obstacle Heuristic)
//     double rot_vec[3];
//     int closest_obstacle_it;
//     calculateRotationVector(
//       rot_vec,
//       closest_obstacle_it,
//       num_obstacles, 
//       obstacles, 
//       i,
//       agent_pos_vec,
//       goal_pos_vec,
//       goal_vec
//     );
//     atomicAdd(active_obstacles,1);

//     // calculate current force
//     calculateCurrForce(
//       curr_force,
//       rot_vec,
//       obstacle_pos_vec, 
//       agent_pos_vec, 
//       agent_vel_vec, 
//       goal_pos_vec, 
//       rel_vel, 
//       k_circ,
//       dist_obs
//     );

//     // printf("%f\t%f\t%f\n", curr_force[0], curr_force[1], curr_force[2]);

//   }

//   // force_ += curr_force;
//   place_vector_in_array_at_index(force, curr_force, i);
// }


// void launch_GoalObstacleHeuristic_circForce_kernel(
//     const std::vector<ghostplanner::cfplanner::Obstacle> &obstacles, 
//     int n_obstacles,
//     double k_circ, 
//     double detect_shell_rad_,
//     double* goalPosition,
//     double* agentPosition,
//     double* agentVelocity,
//     double* net_force,
//     bool debug
// ){
  
//   auto chrono_start = std::chrono::high_resolution_clock::now();
  
//   // const double collision_rad_ = 0.5; 
//   const double min_obs_dist_ = detect_shell_rad_;
//   int *active_obstacles = new int[1];
//   active_obstacles[0] = 0;
  
//   double force_vec[3];
//   init_vector(force_vec);

//   // std::vector<bool> known_obstacles_(n_obstacles, false);
//   std::vector<double*> field_rotation_vecs_(n_obstacles*3*sizeof(double));

//   // helper variables
//   int obstacle_data_size = n_obstacles * sizeof(ghostplanner::cfplanner::Obstacle);
//   int sizeof_vector3d = 3*sizeof(double);

//   // host variables
//   double* h_force = new double[n_obstacles*3];

//   // device data
//   ghostplanner::cfplanner::Obstacle *d_obstacles;
//   double* d_goalPosition;
//   double* d_agentPosition;
//   double* d_agentVelocity;
//   double* d_goal_vec;
//   int* d_active_obstacles;
//   double* d_force;

//   // preliminary calculations 
//   // Note: can be moved inside kernel but with time cost
//   double goal_vec[3];
//   subtract_vectors(goal_vec, goalPosition, agentPosition);



//   // alloc memory on device
//   cudaMalloc((void**)&d_obstacles, obstacle_data_size);
//   cudaMalloc((void**)&d_goalPosition, sizeof_vector3d);
//   cudaMalloc((void**)&d_agentPosition, sizeof_vector3d);
//   cudaMalloc((void**)&d_agentVelocity, sizeof_vector3d);
//   cudaMalloc((void**)&d_goal_vec, sizeof_vector3d);
//   cudaMalloc((void**)&d_active_obstacles, 1*sizeof(int));
//   cudaMalloc((void**)&d_force, n_obstacles*sizeof_vector3d);

      
//   // move memory to device
//   cudaMemcpy(d_obstacles, (obstacles).data(), obstacle_data_size, cudaMemcpyHostToDevice);
//   cudaMemcpy(d_goalPosition, goalPosition, sizeof_vector3d, cudaMemcpyHostToDevice);
//   cudaMemcpy(d_agentPosition, agentPosition, sizeof_vector3d, cudaMemcpyHostToDevice);
//   cudaMemcpy(d_agentVelocity, agentVelocity, sizeof_vector3d, cudaMemcpyHostToDevice);
//   cudaMemcpy(d_goal_vec, goal_vec, sizeof_vector3d, cudaMemcpyHostToDevice);
//   cudaMemcpy(d_active_obstacles, active_obstacles, 1*sizeof(int), cudaMemcpyHostToDevice);

//   // run kernel
//   int blocks = n_obstacles/threads + 1;
//   circForce_kernel<<<blocks, threads>>>(
//     n_obstacles,
//     d_force,
//     d_obstacles,
//     d_goalPosition,
//     d_goal_vec,
//     d_agentPosition,
//     d_agentVelocity,
//     d_active_obstacles,
//     min_obs_dist_,
//     detect_shell_rad_,
//     k_circ
//   );

//   // synchronize
//   cudaDeviceSynchronize();

//   // transfer memory back
//   cudaMemcpy(active_obstacles, d_active_obstacles, 1*sizeof(int), cudaMemcpyDeviceToHost);
//   cudaMemcpy(h_force, d_force, n_obstacles*sizeof_vector3d, cudaMemcpyDeviceToHost);

//   // final calc:
//   for(int i=0; i<n_obstacles; i++){
//     force_vec[0] += h_force[i+0];
//     force_vec[1] += h_force[i+1];
//     force_vec[2] += h_force[i+2];
//   }

//   // cleanup
//   cudaFree(d_obstacles);
//   cudaFree(d_goalPosition);
//   cudaFree(d_agentPosition);
//   cudaFree(d_agentVelocity);
//   cudaFree(d_goal_vec);
//   cudaFree(d_active_obstacles);
//   cudaFree(d_force);

//   // prints
//   if(debug){
//     auto chrono_stop = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> duration = chrono_stop - chrono_start;
//     std::cout<<"\t"<<"[ num_obstacles: "<<n_obstacles<<",\tdetect_shell_rad_: "<<detect_shell_rad_<<",\tactive_obstacles: "<<*active_obstacles<<",\tduration (s): "<<duration.count();
//     std::cout<<",\tforce: ["<<force_vec[0]<<", "<< force_vec[1]<<", "<< force_vec[2];
//     std::cout<<" ],"<<std::endl;
//   }

//   net_force[0] = force_vec[0];
//   net_force[1] = force_vec[1];
//   net_force[2] = force_vec[2];
// }




// best function ever
__host__  void hello_cuda_world(){
  std::cout<<"Hello CUDA World!"<<std::endl;
}


}