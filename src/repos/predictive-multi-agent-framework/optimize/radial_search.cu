#include <iostream>
#include <cuda_runtime.h>


__global__ void radial_search_kernel(
    const float* points, 
    const float* target, 
    float obstacle_rad,
    float detect_shell_rad, 
    int num_points, 
    int* output, 
    int* output_count
){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    
    // get dx, dy, dz
    float dx = points[idx * 3] - target[0];
    float dy = points[idx * 3 + 1] - target[1];
    float dz = points[idx * 3 + 2] - target[2];

    float distance = sqrtf(dx * dx + dy * dy + dz * dz) - obstacle_rad;
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
    int threads = 256;
    int blocks = (num_points + threads - 1) / threads;        
    radial_search_kernel<<<blocks, threads>>>(
        coords,
        target,
        obstacle_rad,
        detect_shell_rad,
        num_points,
        output,
        output_count
    );
    cudaDeviceSynchronize();

}




// torch::Tensor radial_search_cuda(
//     torch::Tensor points, 
//     torch::Tensor target, 
//     float obstacle_rad, 
//     float detect_shell_rad, 
//     int num_points
// ) {
//     CHECK_INPUT(points);

//     auto output = torch::zeros({num_points}, torch::dtype(torch::kInt32).device(torch::kCUDA));
//     auto output_count = torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));

//     int threads = 256;
//     int blocks = (num_points + threads - 1) / threads;        
//     radial_search_kernel<<<blocks, threads>>>(
//         points.data_ptr<float>(),
//         target.data_ptr<float>(),
//         obstacle_rad, 
//         detect_shell_rad,
//         num_points,
//         output.data_ptr<int>(),
//         output_count.data_ptr<int>()
//     );
//     C10_CUDA_KERNEL_LAUNCH_CHECK();
//     cudaDeviceSynchronize();

//     int result_size = output_count.item<int>();
//     return output.slice(0, 0, result_size);
// }
