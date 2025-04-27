// // the usual
// #include <iostream>
// #include <vector>

// // necessary evils
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <vector_types.h>

// // include the header

// // time keeper
// #include <chrono>
// #include <iomanip>

#include <cstdio>
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>

// necessary goods
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>

#include "percept/cuda_vector_ops.cuh"
#include "percept/ObstacleDistanceCost.h"


#define threads 1024


namespace obstacle_distance_cost{
using namespace cuda_vector_ops;

// Kernel to compute the minimum effective distance
__global__ void kernel(
    double* d_net_potential,
    const double3* d_masses,
    size_t num_masses,
    double3 agent_position,
    double agent_radius,
    double mass_radius,
    double detect_shell_rad
){
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // Initialize shared memory with a zero value
    sdata[tid] = 0.0;

    if (i >= num_masses) {
        return;
    }

    double3 mass_position = d_masses[i];
    double3 dist_vec = mass_position - agent_position;
    double dist = norm(dist_vec);

    // Truncated Quadratic (CHOMP)
    if (dist <= detect_shell_rad){
        sdata[tid] = (1.0 / 2.0*detect_shell_rad) * (dist-detect_shell_rad)* (dist-detect_shell_rad);
    }

    // Perform reduction in shared memory, add up all the potentials
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid+s<num_masses) {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 atomically updates the global minimum distance
    if (tid == 0) {
        atomicAdd(d_net_potential, sdata[0]);
    }
}


__host__ double launch_kernel(
    double3* d_masses,
    size_t num_masses,
    double3 agent_position,
    double agent_radius,
    double mass_radius,
    double detect_shell_rad,
    bool debug
){
    auto start_time = std::chrono::high_resolution_clock::now();

    double* d_net_potential;
    cudaError_t err = cudaMalloc(&d_net_potential, sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory: %s\n", cudaGetErrorString(err));
        return 0.0;
    }

    // Initialize d_net_potential with a zero value on the device
    double init_val = 0.0;
    err = cudaMemcpy(d_net_potential, &init_val, sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to initialize device memory: %s\n", cudaGetErrorString(err));
        cudaFree(d_net_potential);
        return 0.0;
    }

    // Define number of threads per block (example value)
    // int threads = 256;
    int num_blocks = (num_masses + threads - 1) / threads;
    size_t shared_mem_size = threads * sizeof(double);

    kernel<<<num_blocks, threads, shared_mem_size>>>(
        d_net_potential,
        d_masses,
        num_masses,
        agent_position,
        agent_radius,
        mass_radius,
        detect_shell_rad
    );

    // Check for any kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_net_potential);
        return 0.0;
    }

    // Copy the result back to host memory
    double host_net_potential;
    err = cudaMemcpy(&host_net_potential, d_net_potential, sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy result from device: %s\n", cudaGetErrorString(err));
        cudaFree(d_net_potential);
        return 0.0;
    }
    cudaFree(d_net_potential);


    if (debug) {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        std::cout << std::left << std::setw(45) << "ObstacleDistanceCost"
                  << "kernel execution time: " 
                  << std::fixed << std::setprecision(9) 
                  << elapsed.count() << " seconds" << std::endl;
    }

    return host_net_potential;
}


// best function ever
__host__  void hello_cuda_world(){
  std::cout<<"Hello CUDA World! -From Obstacle Distance Cost Kernel <3"<<std::endl;
}


}