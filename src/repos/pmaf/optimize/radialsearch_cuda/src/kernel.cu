#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>


__global__ void radial_search_kernel(
    float* points, 
    float* target, 
    float obstacle_rad,
    float detect_shell_rad, 
    int num_points, 
    int* output, 
    int* output_count
){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    // printf("%d %d\n", idx, num_points);
    if (idx >= num_points) return;

    // printf("radial_search.cu: ln20\n");
    // get dx, dy, dz
    float dx = points[idx * 3] - target[0];
    float dy = points[idx * 3 + 1] - target[1];
    float dz = points[idx * 3 + 2] - target[2];

    // printf("radial_search.cu: ln26\n");

    float distance = sqrtf(dx * dx + dy * dy + dz * dz) - obstacle_rad;
    // printf("%f\n", &distance);
    if (distance <= detect_shell_rad) {
        int insert_idx = atomicAdd(output_count, 1);
        output[insert_idx] = idx;
    }
}


void launch_radial_search_kernel(
    float* coords,
    float* target,
    float obstacle_rad,
    float detect_shell_rad,
    int num_points,
    int* output,
    int* output_count
){

    float* coords_;
    float* target_;
    int* output_;
    int* output_count_;

    // Allocate memory on the device (GPU)
    cudaMalloc(&coords_, num_points * 3 * sizeof(float));
    cudaMalloc(&target_, 3 * sizeof(float));
    cudaMalloc(&output_, num_points * 3 * sizeof(int));
    cudaMalloc(&output_count_, 1 * sizeof(int));


    // Copy the host array to the device
    cudaMemcpy(coords_, coords, num_points * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(target_, target, 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(output_, output, num_points * 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(output_count_, output_count, 1 * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    int threads = 256;
    int blocks = (num_points + threads - 1) / threads;        
    radial_search_kernel<<<blocks, threads>>>(
        coords_,
        target_,
        obstacle_rad,
        detect_shell_rad,
        num_points,
        output_,
        output_count_
    );
    cudaDeviceSynchronize();

    cudaMemcpy(output, output_, num_points * 3 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_count, output_count_, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    
}

