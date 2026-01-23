#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <vector_types.h>


#include "cuda_vector_ops.cuh"


namespace obstacle_distance_cost{

__host__ double launch_kernel(
    double3* d_masses,
    size_t num_masses,
    double3 agent_position,
    // double agent_radius,
    // double mass_radius,
    // double detect_shell_rad,
    bool debug
);

__host__ void hello_cuda_world();

}