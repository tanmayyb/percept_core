#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <vector_types.h>

namespace heuristic_kernel{

double3 launch_ObstacleHeuristic_circForce_kernel(
    double3* gpu_points_, 
    size_t gpu_num_points_,
    double3 agent_position,
    double3 agent_velocity,
    double k_circ, 
    double detect_shell_rad_,
    bool debug
);

__host__ void hello_cuda_world();

}