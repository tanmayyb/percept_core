// Kernel to calculate distance to the closest obstacle

#include "cuda_vector_ops.cuh"
#include <cstdio>
#include <cmath>

#define threads 1024

namespace min_obstacle_distance 
{
  using namespace cuda_vector_ops;

  __device__ static double atomicMinDouble(double* address, double val) 
  {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
  
    unsigned long long int old = *address_as_ull, assumed;
  
    do 
    {
      assumed = old;
    
      old = atomicCAS(address_as_ull, assumed,
        __double_as_longlong(fmin(val, __longlong_as_double(assumed))));
    
    } while (assumed != old);
    
    return __longlong_as_double(old);
}

  __global__ void kernel(
    double* d_min_distance,
    const double* __restrict__ d_points_x,
    const double* __restrict__ d_points_y,
    const double* __restrict__ d_points_z,
    size_t num_points,
    double3 agent_position)
  {
    extern __shared__ double sdata[];

    int tid = threadIdx.x;

    int i = blockIdx.x * blockDim.x + tid;

    sdata[tid] = 1.7976931348623157e+308; 

    if (i < num_points) 
    {
      double3 mass_position = make_double3(d_points_x[i], d_points_y[i], d_points_z[i]);

      double3 dist_vec = mass_position - agent_position;

      sdata[tid] = norm(dist_vec);
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) 
    {

      if (tid < s) 
      {
        if (sdata[tid + s] < sdata[tid]) 
        {
          sdata[tid] = sdata[tid + s];
        }

      }

      __syncthreads();
    }

    if (tid == 0) 
    {
      atomicMinDouble(d_min_distance, sdata[0]);
    }
  }

  __host__ double launch_kernel(
    double* d_points_x, double* d_points_y, double* d_points_z,
    size_t num_points, double3 agent_position, bool debug
  ){
    double* d_min_distance;

    cudaMalloc(&d_min_distance, sizeof(double));

    double init_val = 1.7976931348623157e+308;

    cudaMemcpy(d_min_distance, &init_val, sizeof(double), cudaMemcpyHostToDevice);

    int num_blocks = (num_points + threads - 1) / threads;

    size_t shared_mem_size = threads * sizeof(double);

    kernel<<<num_blocks, threads, shared_mem_size>>>(
      d_min_distance, d_points_x, d_points_y, d_points_z, num_points, agent_position
    );

    cudaDeviceSynchronize();

    double host_net_potential;

    cudaMemcpy(&host_net_potential, d_min_distance, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_min_distance);

    return host_net_potential;
  }

} // namespace min_obstacle_distance

extern "C" double min_obstacle_distance_kernel(
  double* d_points_x, double* d_points_y, double* d_points_z,
  size_t num_points, double3 agent_position, bool debug) 
{
  return min_obstacle_distance::launch_kernel(
    d_points_x, d_points_y, d_points_z, num_points, agent_position, debug
  );
}