// the usual
#include <iostream>
#include <vector>

// necessary evils
#include <cuda.h>
#include <cuda_runtime.h>

// include the header
#include "cf.h"

// time keeper
#include <chrono>


#define threads 256
//  NEED EDGE CONDITION HANDLER 
//  i.e. when known num_obstacles < 256




// helper functions
__host__ __device__ void subtract_vectors(double* result, double* vec1, double* vec2){
  result[0] = vec1[0] - vec2[0];
  result[1] = vec1[1] - vec2[1];
  result[2] = vec1[2] - vec2[2];
}

__host__ __device__ void dot_vectors(double &result, double* vec1, double *vec2){
  double product[3];
  product[0] = vec1[0] * vec2[0];
  product[1] = vec1[1] * vec2[1];
  product[2] = vec1[2] * vec2[2];
  result = product[0] + product[1] + product[2];
}

__host__ __device__ void normalize_vector(double* result_vector, double* orig_vector){
  double orig_vector_mag = sqrt(orig_vector[0]*orig_vector[0] + orig_vector[1]*orig_vector[1] + orig_vector[2]*orig_vector[2]); 
  if (orig_vector_mag == 0.f){
    result_vector[0] = 0.0;
    result_vector[1] = 0.0;
    result_vector[2] = 0.0;
  }
  else{
    result_vector[0] = orig_vector[0]/orig_vector_mag;
    result_vector[1] = orig_vector[1]/orig_vector_mag;
    result_vector[2] = orig_vector[2]/orig_vector_mag;
  }
}

__host__ __device__ void norm(double &result, double* vec){
  result = sqrt(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]);
}




// fancy kernel that does everything
__global__ void circForce_kernel(
  int num_obstacles,
  Obstacle *obstacles,
  double* goalPosition,
  double* goal_vec,
  double* agentPosition,
  double* agentVelocity,
  double min_obs_dist_,
  double detect_shell_rad_
){
  int i = blockIdx.x * blockDim.x + threadIdx.x;   // i refers to obstacle being computed
  if(i >= num_obstacles) return; 

  double robot_obstacle_vec[3], rel_vel[3];

  // get robot_obstacle_vec
  robot_obstacle_vec[0] = obstacles[i].getPosX() - agentPosition[0];
  robot_obstacle_vec[1] = obstacles[i].getPosY() - agentPosition[1];
  robot_obstacle_vec[2] = obstacles[i].getPosZ() - agentPosition[2];

  // get rel_vel
  rel_vel[0] = obstacles[i].getVelX() - agentVelocity[0];
  rel_vel[1] = obstacles[i].getVelY() - agentVelocity[1];
  rel_vel[2] = obstacles[i].getVelZ() - agentVelocity[2];


  // if (robot_obstacle_vec.normalized().dot(goal_vec.normalized()) < -0.01 && robot_obstacle_vec.dot(rel_vel) < -0.01) {continue;}
  double dot_product1, dot_product2, robot_obstacle_vec_normalized[3], goal_vec_normalized[3];
  normalize_vector(robot_obstacle_vec_normalized, robot_obstacle_vec);
  normalize_vector(goal_vec_normalized, goal_vec);
  dot_vectors(dot_product1, robot_obstacle_vec_normalized, goal_vec);
  dot_vectors(dot_product2, robot_obstacle_vec, rel_vel);
  if (dot_product1 < -0.01 && dot_product2 < -0.01){ // compute condition
    return;
  }

  // double dist_obs{robot_obstacle_vec.norm() - (rad_ + obstacles.at(i).getRadius())};
  double norm1;
  const double rad_ = 0.5; // what is this for?
  norm(norm1, robot_obstacle_vec);
  double dist_obs = norm1 - rad_ + obstacles[i].getRad();

  dist_obs = max(dist_obs, 1e-5);
  if (dist_obs<min_obs_dist_)
    min_obs_dist_ = dist_obs;

  // Eigen::Vector3d curr_force{0.0, 0.0, 0.0};
  // Eigen::Vector3d current;
  if(dist_obs < detect_shell_rad_){

    // // calculate rotation vector
    // if (!known_obstacles_.at(i)) {
    //   field_rotation_vecs_.at(i) = calculateRotationVector(
    //       getLatestPosition(), 
    //       g_pos_, 
    //       obstacles, i
    //     );
    //   known_obstacles_.at(i) = true;
    //   active_obstacles++;
    // }

    // double vel_norm = rel_vel.norm();
    // if (vel_norm != 0) {
    //   Eigen::Vector3d normalized_vel = rel_vel / vel_norm;
    //   current = currentVector(
    //     getLatestPosition(), rel_vel, getGoalPosition(),
    //     obstacles, i, field_rotation_vecs_);
    //   curr_force = (k_circ / pow(dist_obs, 2)) *
    //     normalized_vel.cross(current.cross(normalized_vel));
    // }
  }

  // force_ += curr_force;

} 


void launch_circForce_kernel(
    std::vector<Obstacle> *obstacles, 
    int n_obstacles,
    double k_circ, 
    double detect_shell_rad_,
    double* goalPosition,
    double* agentPosition,
    double* agentVelocity,
    double* net_force
){
    auto chrono_start = std::chrono::high_resolution_clock::now();

    const double collision_rad_ = 0.5; 
    const int active_obstacles = 0;
    const double min_obs_dist_ = detect_shell_rad_;

    std::vector<bool> known_obstacles_(n_obstacles, false);
    std::vector<double*> field_rotation_vecs_(n_obstacles*3*sizeof(double));

    // helper variables
    int obstacle_data_size = n_obstacles * sizeof(Obstacle);
    int sizeof_vector3d = 3*sizeof(double);

    // device data
    Obstacle *d_obstacles;
    double* d_goalPosition;
    double* d_agentPosition;
    double* d_agentVelocity;
    double* d_goal_vec;

    // preliminary calculations 
    // Note: can be moved inside kernel but with time cost
    double goal_vec[3];
    subtract_vectors(goal_vec, goalPosition, agentPosition);



    // alloc memory on device
    cudaMalloc((void**)&d_obstacles, obstacle_data_size);
    cudaMalloc((void**)&d_goalPosition, sizeof_vector3d);
    cudaMalloc((void**)&d_agentPosition, sizeof_vector3d);
    cudaMalloc((void**)&d_agentVelocity, sizeof_vector3d);
    cudaMalloc((void**)&d_goal_vec, sizeof_vector3d);
        
    // move memory to device
    cudaMemcpy(d_obstacles, (*obstacles).data(), obstacle_data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_goalPosition, goalPosition, sizeof_vector3d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_agentPosition, agentPosition, sizeof_vector3d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_agentVelocity, agentVelocity, sizeof_vector3d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_goal_vec, goal_vec, sizeof_vector3d, cudaMemcpyHostToDevice);


    // run kernel
    int blocks = n_obstacles/threads + 1;
    circForce_kernel<<<blocks, threads>>>(
      n_obstacles,
      d_obstacles,
      d_goalPosition,
      d_goal_vec,
      d_agentPosition,
      d_agentVelocity,
      min_obs_dist_,
      detect_shell_rad_
    );

    // synchronize
    cudaDeviceSynchronize();


    // cleanup

    // prints
    auto chrono_stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = chrono_stop - chrono_start;
    std::cout<<"\t"<<"[ detect_shell_rad_: "<<detect_shell_rad_<<", active_obstacles: "<<active_obstacles<<", duration: "<<duration.count()<<" ],"<<std::endl;

}




// best function ever
__host__  void hello_world(){
    std::cout<<"Hello World!"<<std::endl;
}