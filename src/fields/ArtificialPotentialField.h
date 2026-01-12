#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <vector_types.h>

#include "cuda_vector_ops.cuh"

namespace artificial_potential_field{

__host__ double3 launch_kernel(
    double3* gpu_points_, 
    size_t gpu_num_points_,
    double3 agent_position,
    double3 agent_velocity,
    double3 goal_position,
    double agent_radius,
    double mass_radius,
    double detect_shell_rad,
    double k_force,
    double max_allowable_force,
    bool debug
);

__host__ void hello_cuda_world();

}