#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <vector_types.h>

namespace heuristic_kernel{

const double MAX_ALLOWABLE_FORCE = 200.0; // Define maximum allowed force magnitude

double3 launch_ObstacleHeuristic_circForce_kernel(
    double3* gpu_points_, 
    size_t gpu_num_points_,
    double3 agent_position,
    double3 agent_velocity,
    double3 goal_position,
    double agent_radius,
    double mass_radius,
    double detect_shell_rad,
    double k_circ, 
    bool debug
);

__host__ void hello_cuda_world();

}