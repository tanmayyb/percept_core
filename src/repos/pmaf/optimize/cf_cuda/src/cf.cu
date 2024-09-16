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






// fancy kernel that does everything
__global__ void circForce_kernel(
  int num_obstacles,
  Obstacle *obstacles,
  double* goalPosition,
  double* goal_vec,
  double* agentPosition,
  double* agentVelocity
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


  // if (robot_obstacle_vec.normalized().dot(goal_vec.normalized()) < -0.01 &&
  //   robot_obstacle_vec.dot(rel_vel) < -0.01) {
  //   continue;
  // }

  // l2 normalization on robot_obstacle_vec

  // dot robot_obstacle_vec with goal_vec



  // prints
  // printf("%f %f %f\n", goalPosition[0],goalPosition[1], goalPosition[2]);

  // printf("%f %f %f\n", 
  //   obstacles[i].getPosX(),
  //   obstacles[i].getPosY(),
  //   obstacles[i].getPosZ());



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
    goal_vec[0] = goalPosition[0] - agentPosition[0];
    goal_vec[1] = goalPosition[1] - agentPosition[1];
    goal_vec[2] = goalPosition[2] - agentPosition[2];



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
      d_agentVelocity
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