#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>

namespace navigation_function {

// Main kernel launch function that computes the navigation function force
__host__ double3 launch_kernel(
    double3* d_obstacles,
    double* d_obstacle_radii,
    size_t num_obstacles,
    double3 agent_position,
    double3 goal_position,
    double3 world_center,
    double world_radius,
    int K,
    float eps,
    double max_allowable_force,
    bool debug = false
);

// Utility function for testing
__host__ void hello_cuda_world();

} // namespace navigation_function 