// the usual
#include <iostream>
#include <vector>

// necessary evils
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>

// include the header
#include "percept/NearestNeighbour.h"
#include "percept/cuda_vector_ops.cuh"

// time keeper
#include <chrono>
#include <iomanip>

#define threads 1024

namespace nearest_neighbour{
using namespace cuda_vector_ops;

__global__ void kernel(
    double3* d_masses,
    size_t num_masses,
    int* nn_index
){

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    if(i >= num_masses){
        return;
    }

    int nn_idx = -1;
    double min_dist = 1e10;
    double3 mass_position;
    double3 mass_dist_vec;
    double dist_to_mass;

    for(int j = 0; j < num_masses; j++){
        if(i == j){
            continue;
        }

        mass_position = d_masses[j];
        mass_dist_vec = mass_position  - d_masses[i];
        dist_to_mass = norm(mass_dist_vec);

        if(dist_to_mass < min_dist){
            min_dist = dist_to_mass;
            nn_idx = j;
        }
    }

    nn_index[i] = nn_idx;
}

__host__ void launch_kernel(
    double3* d_masses,
    size_t num_masses,
    int* nn_index,
    bool debug
){

    // Start timing if debug is enabled
    auto start_time = std::chrono::high_resolution_clock::now();

    // set memory to zero
    cudaError_t err = cudaMemset(nn_index, 0, sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set device memory: %s\n", cudaGetErrorString(err));
        cudaFree(nn_index);
        return;
    }

    int num_blocks = (num_masses + threads - 1) / threads; // ceiling division
    kernel<<<num_blocks, threads>>>(
        d_masses, num_masses, nn_index
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch kernel: %s\n", cudaGetErrorString(err));
        return;
    }

    // Add synchronization check after kernel launch
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to synchronize: %s\n", cudaGetErrorString(err));
        return;
    }

    // Print timing information if debug is enabled
    if (debug) {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        std::cout << std::left << std::setw(45) << "NearestNeighbour"
                  << "kernel execution time: " 
                  << std::fixed << std::setprecision(9) 
                  << elapsed.count() << " seconds" << std::endl;
    }
    return;
}


}