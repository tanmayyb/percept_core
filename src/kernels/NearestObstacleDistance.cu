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
#include "percept/NearestObstacleDistance.h"



#define threads 256


namespace nearest_obstacle_distance{
using namespace cuda_vector_ops;


// Device helper: atomicMin for double using atomicCAS
__device__ double atomicMinDouble(double* addr, double value) {
    unsigned long long int* addr_as_ull = (unsigned long long int*) addr;
    unsigned long long int old = *addr_as_ull, assumed;
    do {
        assumed = old;
        double old_val = __longlong_as_double(assumed);
        double new_val = fmin(old_val, value);
        unsigned long long int new_ull = __double_as_longlong(new_val);
        old = atomicCAS(addr_as_ull, assumed, new_ull);
    } while (assumed != old);
    return __longlong_as_double(old);
}


// Kernel to compute the minimum effective distance
__global__ void kernel(
    double* d_distance,
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

    // Initialize shared memory with a very large value
    sdata[tid] = 1.0e20;

    // Each thread computes its effective distance if in bounds
    if (i < num_masses) {
        double3 mass_position = d_masses[i];
        double3 dist_vec = mass_position - agent_position;
        double dist = sqrt(norm(dist_vec));
        // Compute effective distance (adjusting for the agent and mass radii)
        double effective_distance = dist - (agent_radius + mass_radius);
        if (effective_distance < 0.0)
            effective_distance = 0.0;
        // Optionally, one can check if the obstacle is within a detection shell
        // if(effective_distance <= detect_shell_rad) { ... }
        sdata[tid] = effective_distance;
    }

    __syncthreads();

    // Perform reduction to find the minimum distance in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmin(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Thread 0 atomically updates the global minimum distance
    if (tid == 0) {
        atomicMinDouble(d_distance, sdata[0]);
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

    double* d_distance;
    cudaError_t err = cudaMalloc(&d_distance, sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory: %s\n", cudaGetErrorString(err));
        return 0.0;
    }

    // Initialize d_distance with a very large value on the device
    double init_val = 1.0e20;
    err = cudaMemcpy(d_distance, &init_val, sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to initialize device memory: %s\n", cudaGetErrorString(err));
        cudaFree(d_distance);
        return 0.0;
    }

    // Define number of threads per block (example value)
    // int threads = 256;
    int num_blocks = (num_masses + threads - 1) / threads;
    size_t shared_mem_size = threads * sizeof(double);

    kernel<<<num_blocks, threads, shared_mem_size>>>(
        d_distance,
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
        cudaFree(d_distance);
        return 0.0;
    }

    // Copy the result back to host memory
    double host_distance;
    err = cudaMemcpy(&host_distance, d_distance, sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy result from device: %s\n", cudaGetErrorString(err));
        cudaFree(d_distance);
        return 0.0;
    }
    cudaFree(d_distance);

    if (debug) {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        std::cout << std::left << std::setw(45) << "NearestObstacleDistance"
                  << "kernel execution time: " 
                  << std::fixed << std::setprecision(9) 
                  << elapsed.count() << " seconds" << std::endl;
    }

    return host_distance;
}


// best function ever
__host__  void hello_cuda_world(){
  std::cout<<"Hello CUDA World! -From Nearest Obstacle Distance Kernel <3"<<std::endl;
}


}