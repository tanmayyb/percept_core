#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>

namespace navigation_function {

// Main kernel launch function that computes the navigation function force
__host__ double3 launch_kernel(
	double3* d_masses,
	size_t num_masses,
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

// Utility function for testing
__host__ void hello_cuda_world();

} // namespace navigation_function 