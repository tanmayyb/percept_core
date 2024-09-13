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
//  i.e. when num_obstacles < 256



// fancy kernel that does everything
__global__ void circForce_kernel(){


} 

void launch_circForce_kernel(
    std::vector<Obstacle> *obstacles, 
    int n_obstacles,
    double k_circ, 
    double detect_shell_rad_,
    double* goalPosition,
    double* agentPosition,
    double* agentVelocity,
    double* force
){
    auto chrono_start = std::chrono::high_resolution_clock::now();

    double collision_rad_ = 0.5; 
    int active_obstacles = 0;
    double min_obs_dist_ = detect_shell_rad_;

    std::vector<bool> known_obstacles_(n_obstacles, false);
    std::vector<double*> field_rotation_vecs_(n_obstacles*3*sizeof(double));
    

    // alloc memory on device
    

    // move memory to device

    // run kernel
    int blocks = n_obstacles/threads;
    circForce_kernel<<<blocks, threads>>>();

    // synchronize
    cudaDeviceSynchronize();


    // cleanup

    auto chrono_stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = chrono_stop - chrono_start;
    std::cout<<"\t"<<"[ "<<detect_shell_rad_<<", "<<active_obstacles<<", "<<duration.count()<<" ],"<<std::endl;

}




// best function ever
__host__  void hello_world(){
    std::cout<<"Hello World!"<<std::endl;
}