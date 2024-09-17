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

__host__ __device__ void add_vectors(double* result, double* vec1, double* vec2){
  result[0] = vec1[0] + vec2[0];
  result[1] = vec1[1] + vec2[1];
  result[2] = vec1[2] + vec2[2];
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

__host__ __device__ void scale_vector(double* result_vector, double* orig_vector, double scalar){
  result_vector[0] = orig_vector[0]*scalar;
  result_vector[1] = orig_vector[1]*scalar;
  result_vector[2] = orig_vector[2]*scalar;
}


__host__ __device__ void norm(double &result, double* vec){
  result = sqrt(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]);
}

// pmaf helper functions
__host__ __device__ void get_obstacle_position_vector(double* result_vector, Obstacle &obstacle){
  result_vector[0] = obstacle.getPosX();
  result_vector[1] = obstacle.getPosY();
  result_vector[2] = obstacle.getPosZ();
}

__host__ __device__ void get_obstacle_velocity_vector(double* result_vector, Obstacle &obstacle){
  result_vector[0] = obstacle.getVelX();
  result_vector[1] = obstacle.getVelY();
  result_vector[2] = obstacle.getVelZ();
}

// pmaf functions
__device__ void calculateRotationVector(
  int &closest_obstacle_it, 
  int num_obstacles, 
  Obstacle *obstacles, 
  int obstacle_id,
  double* agent_pos,
  double* goal_pos,
  double* goal_vec

){

  double dist_vec[3], obstacle_pos_vec[3], active_obstacle_pos_vec[3], dist_obs;
  double min_dist_obs = 100.0;

  for(int i=0; i<num_obstacles; i++){
    if (i != obstacle_id) {
      // double dist_obs{(obstacles[obstacle_id].getPosition() - obstacles[i].getPosition()).norm()};
      get_obstacle_position_vector(active_obstacle_pos_vec, obstacles[obstacle_id]);
      get_obstacle_position_vector(obstacle_pos_vec, obstacles[i]);
      subtract_vectors(dist_vec, active_obstacle_pos_vec, obstacle_pos_vec);
      norm(dist_obs, dist_vec);

      if(min_dist_obs > dist_obs){
        min_dist_obs = dist_obs;
        closest_obstacle_it = i;
      }
    }
  }

  // printf("closest_obstacle_it: %d\n", closest_obstacle_it);

  double obstacle_vec[3], cfagent_to_obs[3], cfagent_to_obs_normalized[3]; 
  double cfagent_to_obs_scaled[3], dot_product1, dot_product2;
  double obst_current[3], goal_current[3], current[3];
  double obst_current_normalized[3], goal_current_normalized[3];

  // Vector from active obstacle to the obstacle which is closest to the active obstacle
  // Eigen::Vector3d obstacle_vec = obstacles[closest_obstacle_it].getPosition() - obstacles[obstacle_id].getPosition();
  get_obstacle_position_vector(obstacle_pos_vec, obstacles[closest_obstacle_it]);
  get_obstacle_position_vector(active_obstacle_pos_vec, obstacles[obstacle_id]);
  subtract_vectors(obstacle_vec, obstacle_pos_vec, active_obstacle_pos_vec);

  // Eigen::Vector3d cfagent_to_obs{obstacles[obstacle_id].getPosition() - agent_pos};
  subtract_vectors(cfagent_to_obs, active_obstacle_pos_vec, agent_pos);

  // cfagent_to_obs.normalize();
  normalize_vector(cfagent_to_obs_normalized, cfagent_to_obs);

  // Current vector is perpendicular to obstacle surface normal and shows in opposite direction of obstacle_vec
  // Eigen::Vector3d obst_current{ (cfagent_to_obs * obstacle_vec.dot(cfagent_to_obs)) - obstacle_vec};
  dot_vectors(dot_product1, obstacle_vec, cfagent_to_obs_normalized);
  scale_vector(cfagent_to_obs_scaled, cfagent_to_obs_normalized, dot_product1);
  subtract_vectors(obst_current, cfagent_to_obs_scaled, obstacle_vec);

  // passed by kernel so we ingore: Eigen::Vector3d goal_vec{goal_pos - agent_pos};
  // Eigen::Vector3d goal_current{goal_vec - cfagent_to_obs * (cfagent_to_obs.dot(goal_vec))};
  
  // cfagent_to_obs_normalized
  dot_vectors(dot_product2, cfagent_to_obs_normalized, goal_vec);
  scale_vector(cfagent_to_obs_scaled, cfagent_to_obs_normalized, dot_product2); // reusing cfagent_to_obs_scaled
  subtract_vectors(goal_current, goal_vec, cfagent_to_obs_scaled);

  // Eigen::Vector3d current{goal_current.normalized() +
  //                         obst_current.normalized()};
  normalize_vector(goal_current_normalized, goal_current);
  normalize_vector(obst_current_normalized, obst_current);
  add_vectors(current, goal_current_normalized, obst_current_normalized);

  // printf("%f\t%f\t%f\n", current[0],current[1],current[2]);

  if (current.norm() < 1e-10) {
    current << 0.0, 0.0, 1.0;
    // current = makeRandomVector();
  }
  current.normalize();
  Eigen::Vector3d rot_vec{current.cross(cfagent_to_obs)};
  rot_vec.normalize();
  return rot_vec;

}


// fancy kernel that does everything
__global__ void circForce_kernel(
  int num_obstacles,
  Obstacle *obstacles,
  double* goalPosition,
  double* goal_vec,
  double* agentPosition,
  double* agentVelocity,
  int* active_obstacles,
  double min_obs_dist_,
  double detect_shell_rad_
){
  int i = blockIdx.x * blockDim.x + threadIdx.x;   // i refers to obstacle being computed
  if(i >= num_obstacles) return; 

  double robot_obstacle_vec[3], rel_vel[3], obstacle_pos_vec[3], obstacle_vel_vec[3];

  // get robot_obstacle_vec
  get_obstacle_position_vector(obstacle_pos_vec, obstacles[i]);
  subtract_vectors(robot_obstacle_vec, obstacle_pos_vec, agentPosition);

  // get rel_vel
  get_obstacle_velocity_vector(obstacle_vel_vec, obstacles[i]);
  subtract_vectors(rel_vel, obstacle_vel_vec, agentVelocity);

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
  norm(norm1, robot_obstacle_vec);
  // const double rad_ = 0.5; // what is this for?
  const double rad_ = 0.0; // what is this for?
  double dist_obs = norm1 - rad_ + obstacles[i].getRad();

  // get dist_obs and check if more than min_obs_dist_
  dist_obs = max(dist_obs, 1e-5);
  if (dist_obs<min_obs_dist_)
    min_obs_dist_ = dist_obs;

  // Eigen::Vector3d curr_force{0.0, 0.0, 0.0};
  // Eigen::Vector3d current;

  if(dist_obs < detect_shell_rad_){

    // calculate rotation vector (Goal Obstacle Heuristic)
    int closest_obstacle_it;
    calculateRotationVector(
      closest_obstacle_it,
      num_obstacles, 
      obstacles, 
      i,
      agentPosition,
      goalPosition,
      goal_vec
    );
    atomicAdd(active_obstacles,1);

    // // calculate current vector
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
    const double min_obs_dist_ = detect_shell_rad_;
    int *active_obstacles = new int[1];
    active_obstacles[0] = 0;

    // std::vector<bool> known_obstacles_(n_obstacles, false);
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
    int* d_active_obstacles;

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
    cudaMalloc((void**)&d_active_obstacles, 1*sizeof(int));

        
    // move memory to device
    cudaMemcpy(d_obstacles, (*obstacles).data(), obstacle_data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_goalPosition, goalPosition, sizeof_vector3d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_agentPosition, agentPosition, sizeof_vector3d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_agentVelocity, agentVelocity, sizeof_vector3d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_goal_vec, goal_vec, sizeof_vector3d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_active_obstacles, active_obstacles, 1*sizeof(int), cudaMemcpyHostToDevice);

    // run kernel
    int blocks = n_obstacles/threads + 1;
    circForce_kernel<<<blocks, threads>>>(
      n_obstacles,
      d_obstacles,
      d_goalPosition,
      d_goal_vec,
      d_agentPosition,
      d_agentVelocity,
      d_active_obstacles,
      min_obs_dist_,
      detect_shell_rad_
    );

    // synchronize
    cudaDeviceSynchronize();

    // transfer memory back
    cudaMemcpy(active_obstacles, d_active_obstacles, 1*sizeof(int), cudaMemcpyDeviceToHost);


    // cleanup

    // prints
    auto chrono_stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = chrono_stop - chrono_start;
    std::cout<<"\t"<<"[ num_obstacles: "<<n_obstacles<<",\tdetect_shell_rad_: "<<detect_shell_rad_<<",\tactive_obstacles: "<<*active_obstacles<<",\tduration: "<<duration.count()<<" ],"<<std::endl;

}




// best function ever
__host__  void hello_world(){
    std::cout<<"Hello World!"<<std::endl;
}