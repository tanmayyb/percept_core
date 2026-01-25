#include "cuda_vector_ops.cuh"

#define threads 1024

namespace random_heuristic 
{
  using namespace cuda_vector_ops;

  __global__ void kernel(
    double3* d_net_force,
    const double* __restrict__ d_points_x,
    const double* __restrict__ d_points_y,
    const double* __restrict__ d_points_z,
    size_t num_points,
    double3 agent_position,
    double3 agent_velocity,
    double3 goal_position,
    double agent_radius,
    double point_radius,
    double detect_shell_rad,
    double k_force,
    unsigned long long seed
  ) 
  {
    extern __shared__ double3 sdata[];

    int tid = threadIdx.x;
    
    int i = blockIdx.x * blockDim.x + tid;

    sdata[tid] = make_double3(0.0, 0.0, 0.0);

    if (i >= num_points) return;

    double3 point_position = make_double3(d_points_x[i], d_points_y[i], d_points_z[i]);
  
    double3 point_dist_vec = point_position - agent_position;
  
    double3 goal_vec = goal_position - agent_position;
  
    double3 point_rvel_vec = agent_velocity; // Assumes static points
  
    double3 force_vec = make_double3(0.0, 0.0, 0.0);

    double3 point_dist_vec_normalized = normalized(point_dist_vec);
  
    double3 goal_vec_normalized = normalized(goal_vec);

    if (!(dot(point_dist_vec_normalized, goal_vec_normalized) < -1e-5 && dot(point_dist_vec, point_rvel_vec) < -1e-5)) 
    {
      double dist_to_point = norm(point_dist_vec) - (agent_radius + point_radius);

      dist_to_point = fmax(dist_to_point, 1e-5);

      if (dist_to_point < detect_shell_rad && norm(point_rvel_vec) > 1e-10) 
      {
        // Random heuristic specific logic
        double3 random_vec = make_random_vector(i, seed);

        double3 rot_vec = cross(goal_vec_normalized, random_vec);

        double3 point_rvel_vec_normalized = normalized(point_rvel_vec);

        double3 current_vec = normalized(
          cross(point_dist_vec_normalized, rot_vec)
        );

        force_vec = (current_vec * dot(point_rvel_vec_normalized, point_rvel_vec_normalized)) - 
                    (point_rvel_vec_normalized * dot(point_rvel_vec_normalized, current_vec));

        force_vec = force_vec * (k_force / (dist_to_point * dist_to_point));
      }

      sdata[tid] = force_vec;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) 
    {
      if (tid < s) 
      {
        sdata[tid].x += sdata[tid + s].x;

        sdata[tid].y += sdata[tid + s].y;

        sdata[tid].z += sdata[tid + s].z;
      }

      __syncthreads();
    }

    if (tid == 0) 
    {
      atomicAdd(&(d_net_force->x), sdata[0].x);

      atomicAdd(&(d_net_force->y), sdata[0].y);

      atomicAdd(&(d_net_force->z), sdata[0].z);
    }
  }

  __host__ double3 launch_kernel(
    double* d_points_x, double* d_points_y, double* d_points_z,
    size_t num_points,
    double3 agent_position, double3 agent_velocity, double3 goal_position,
    double agent_radius, double point_radius,
    double detect_shell_rad, double k_force,
    double max_allowable_force, bool debug, cudaStream_t stream)
  {

    double3* d_net_force;
  
    cudaMallocAsync(&d_net_force, sizeof(double3), stream);
  
    cudaMemsetAsync(d_net_force, 0, sizeof(double3), stream);

    int num_blocks = (num_points + threads - 1) / threads;
  
    size_t shared_mem_size = threads * sizeof(double3);

    kernel<<<num_blocks, threads, shared_mem_size, stream>>>(
      d_net_force, d_points_x, d_points_y, d_points_z,
      num_points, agent_position, agent_velocity, goal_position,
      agent_radius, point_radius, detect_shell_rad, k_force, 42
    );

    double3 net_force;
  
    cudaMemcpyAsync(&net_force, d_net_force, sizeof(double3), cudaMemcpyDeviceToHost, stream);
  
    cudaStreamSynchronize(stream);
  
    cudaFreeAsync(d_net_force, stream);

    if (max_allowable_force > 0.0) 
    {
      double force_magnitude = norm(net_force);

      if (force_magnitude > max_allowable_force) 
      {
        net_force = net_force * (max_allowable_force / force_magnitude);
      }
    }

    return net_force;
  }

} // namespace random_heuristic

extern "C" double3 random_heuristic_kernel(
  double* d_points_x, double* d_points_y, double* d_points_z,
  size_t num_points, double3 agent_position, double3 agent_velocity, double3 goal_position,
  double agent_radius, double point_radius,
  double detect_shell_rad, double k_force,
  double max_allowable_force, bool debug, cudaStream_t stream) 
{
  return random_heuristic::launch_kernel(
    d_points_x, d_points_y, d_points_z,
    num_points, agent_position, agent_velocity, goal_position,
    agent_radius, point_radius,
    detect_shell_rad, k_force, max_allowable_force, debug, stream
  );
}