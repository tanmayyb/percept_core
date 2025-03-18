// Artificial Potential Field
// credit: https://github.com/ShuiXinYun/Path_Plan/blob/master/APF_CPP/APF.h
// Khatib (1986) : https://khatib.stanford.edu/publications/pdfs/Khatib_1986_IJRR.pdf

// the usual
#include <iostream>
#include <vector>

// necessary evils
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>

// include the header
#include "percept/ArtificialPotentialField.h"
#include "percept/cuda_vector_ops.cuh"

// time keeper
#include <chrono>
#include <iomanip>

#define threads 1024

namespace artificial_potential_field{
using namespace cuda_vector_ops;


__global__ void kernel(
    double3* d_net_force,
    double3* d_masses,
    size_t num_masses,
    double3 agent_position,
    double3 agent_velocity,
    double3 goal_position,
    double agent_radius,
    double mass_radius,
    double detect_shell_rad,
    double k_force
){
    extern __shared__ double3 sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    sdata[tid] = make_double3(0.0, 0.0, 0.0); // set as zero

    // Each thread computes a force if within bound
    if (i >= num_masses) {
        return;
    }

    double3 goal_vec;
    double dist_to_goal;
    double3 mass_position;
    double3 mass_dist_vec;
    double3 mass_velocity; 
    double3 mass_rvel_vec;
    double3 force_vec;
    double3 mass_dist_vec_normalized;
    double dist_to_mass;
    double3 mass_rvel_vec_normalized;

    // implementation of obstacle heuristic circ force
    goal_vec = goal_position - agent_position;
    dist_to_goal = norm(goal_vec);
    mass_position = d_masses[i];
    mass_dist_vec = mass_position - agent_position;
    mass_velocity = make_double3(0.0, 0.0, 0.0);
    mass_rvel_vec = agent_velocity - mass_velocity;
    force_vec = make_double3(0.0, 0.0, 0.0); // set default as zero
    mass_dist_vec_normalized = normalized(mass_dist_vec);


    dist_to_mass = norm(mass_dist_vec) - (agent_radius + mass_radius);
    dist_to_mass = fmax(dist_to_mass, 1e-5); // avoid division by zero

    // implement ARTIFICIAL POTENTIAL FIELD
    if(dist_to_mass <= detect_shell_rad){ // && norm(mass_rvel_vec) > 1e-10){ 

        // calculate force vector
        force_vec = mass_dist_vec_normalized*(-1.0)*(1/dist_to_mass - 1/detect_shell_rad);
        // Khatib (1986) : https://khatib.stanford.edu/publications/pdfs/Khatib_1986_IJRR.pdf
        // Volpe and Khosla (1990): https://www.ri.cmu.edu/pub_files/pub1/volpe_r_1990_1/volpe_r_1990_1.pdf

        force_vec = force_vec * (k_force / pow(dist_to_mass, 1));      
    }




    sdata[tid] = force_vec;

reduction:
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




__host__ double3 launch_kernel(
    double3* d_masses,
    size_t num_masses,
    double3 agent_position,
    double3 agent_velocity,
    double3 goal_position,
    double agent_radius,
    double mass_radius,
    double detect_shell_rad,
    double k_force, 
    double max_allowable_force,
    bool debug
){
    // Start timing if debug is enabled
    auto start_time = std::chrono::high_resolution_clock::now();

    // Allocate device memory for net force
    double3* d_net_force;
    cudaError_t err = cudaMalloc(&d_net_force, sizeof(double3));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory: %s\n", cudaGetErrorString(err));
        return make_double3(0.0, 0.0, 0.0);  // or handle error appropriately
    }
    // set memory to zero
    err = cudaMemset(d_net_force, 0, sizeof(double3));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set device memory: %s\n", cudaGetErrorString(err));
        cudaFree(d_net_force);
        return make_double3(0.0, 0.0, 0.0);
    }

    int num_blocks = (num_masses + threads - 1) / threads; // ceiling division
    size_t shared_mem_size = threads * sizeof(double3);
    kernel<<<num_blocks, threads, shared_mem_size>>>(
        d_net_force, d_masses, num_masses,
        agent_position, agent_velocity, goal_position,
        agent_radius, mass_radius, detect_shell_rad,
        k_force
    ); // CUDA kernels automatically copy value-type parameters to the device when called
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch kernel: %s\n", cudaGetErrorString(err));
        cudaFree(d_net_force);
        return make_double3(0.0, 0.0, 0.0);
    }

    // Add synchronization check after kernel launch
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to synchronize: %s\n", cudaGetErrorString(err));
        cudaFree(d_net_force);
        return make_double3(0.0, 0.0, 0.0);
    }

    // Copy result back to host
    double3 net_force;
    err = cudaMemcpy(&net_force, d_net_force, sizeof(double3), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy result from device: %s\n", cudaGetErrorString(err));
        cudaFree(d_net_force);
        return make_double3(0.0, 0.0, 0.0);
    }
    
    // Free device memory
    cudaFree(d_net_force);
    // Don't free d_masses here as it was allocated elsewhere

    // cap the force magnitude
    if(max_allowable_force > 0.0){
        double force_magnitude = norm(net_force);   
        if (force_magnitude > max_allowable_force) {
            double scale = max_allowable_force / force_magnitude;
            net_force = net_force * scale;
        }
    }

    // Print timing information if debug is enabled
    if (debug) {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        std::cout << std::left << std::setw(45) << "ArtificialPotentialField"
                  << "kernel execution time: " 
                  << std::fixed << std::setprecision(9) 
                  << elapsed.count() << " seconds" << std::endl;
    }

    return net_force;
}


// best function ever
__host__  void hello_cuda_world(){
  std::cout<<"Hello CUDA World! -From Artificial Potential Field Kernel <3"<<std::endl;
}


}