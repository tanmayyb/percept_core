// Navigation Function Force
// Credit: https://github.com/leengh/navigation-function
// Rimon-Koditschek (1990) : https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/r/rimon_koditschek_potential.pdf

// the usual
#include <iostream>
#include <vector>

// necessary evils
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>

// include the header
#include "NavigationFunctionForce.h"
#include "cuda_vector_ops.cuh"

// time keeper
#include <chrono>
#include <iomanip>

#define threads 1024
// #define num_grid_points 6

// Add these constants for numerical stability
#define THRESHOLD_POS_X 700.0  // log(DBL_MAX) is approx 709.78
#define THRESHOLD_NEG_X -700.0 // log(DBL_MIN) is approx -708.39

namespace navigation_function {
using namespace cuda_vector_ops;


__host__ __forceinline__ double3 calculate_grad_L_gamma(const double3& agent_position, const double3& goal_position){
	const double3 dist = agent_position - goal_position;
	const double dist_norm = norm(dist);
	return (dist*2.0)/(dist_norm*dist_norm);
}

__global__ void accumulate_grad_L_beta(double3* d_grad_beta, const double3* d_masses, size_t num_masses, const double3& agent_position, const double3& goal_position, const double detect_shell_rad){
	extern __shared__ double3 sdata[];

	const unsigned int tid = threadIdx.x;
	const unsigned int idx = blockIdx.x * blockDim.x + tid;
	if (idx >= num_masses) return;
	sdata[tid] = make_double3(0.0, 0.0, 0.0);

	const double3 dist_vec = agent_position - d_masses[idx];
	const double dist_norm = norm(dist_vec);
	const double3 grad_L_beta = (dist_vec*2.0)/(fmax(dist_norm*dist_norm - detect_shell_rad*detect_shell_rad, 1.0e-5));

reduction:
	__syncthreads();
	// Reduction in shared memory
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] = sdata[tid] + sdata[tid + s];
		}
		__syncthreads();
	}

	// Thread 0 of each block adds the block's sum to the global sum using atomics
	if (tid == 0) {
		atomicAdd(&(d_grad_beta->x), sdata[0].x);
		atomicAdd(&(d_grad_beta->y), sdata[0].y);
		atomicAdd(&(d_grad_beta->z), sdata[0].z);
	}
}

__host__ __forceinline__ double3 calculate_grad_phi(const double3& d_grad_beta, const double3& agent_position, const double3& goal_position, const double& K){

	const double3 X = calculate_grad_L_gamma(agent_position, goal_position)*K - d_grad_beta;
	double3 grad_phi  = make_double3(0.0, 0.0, 0.0);

	double3 X_clamped;
	X_clamped.x = exp(fmin(fmax(X.x, THRESHOLD_NEG_X), THRESHOLD_POS_X));
	X_clamped.y = exp(fmin(fmax(X.y, THRESHOLD_NEG_X), THRESHOLD_POS_X));
	X_clamped.z = exp(fmin(fmax(X.z, THRESHOLD_NEG_X), THRESHOLD_POS_X));

	grad_phi.x = (X_clamped.x/(1.0+X_clamped.x))/K;
	grad_phi.y = (X_clamped.y/(1.0+X_clamped.y))/K;
	grad_phi.z = (X_clamped.z/(1.0+X_clamped.z))/K;

	return grad_phi;
}


__host__ double3 launch_kernel(
	double3* d_masses,
	size_t num_masses,
	double3 agent_position,
	double3 goal_position,
	double detect_shell_rad,
	double k_force, 
	double K,
	double world_radius,
	double max_allowable_force,
	bool debug
) {

	cudaError_t err;
	// Start timing if debug is enabled
	auto start_time = std::chrono::high_resolution_clock::now();


	// Allocate device memory for d_grad_beta
	double3* d_grad_beta; // the gradients are accumulated to this vector
	err = cudaMalloc(&d_grad_beta, sizeof(double3));
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to set device memory: %s\n", cudaGetErrorString(err));
		cudaFree(d_grad_beta);
		return make_double3(0.0, 0.0, 0.0);
	}

	// set memory to zero
	err = cudaMemset(d_grad_beta, 0.0, sizeof(double3));
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to set device memory: %s\n", cudaGetErrorString(err));
		cudaFree(d_grad_beta);
		return make_double3(0.0, 0.0, 0.0);
	}

	// Launch the accumulate_grad_L_beta kernel
	int num_blocks = (num_masses + threads - 1) / threads; // ceiling division
	size_t shared_mem_size = threads * sizeof(double3);
	accumulate_grad_L_beta<<<num_blocks, threads, shared_mem_size>>>(d_grad_beta, d_masses, num_masses, agent_position, goal_position, detect_shell_rad);

	// Add synchronization check after kernel launch
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to synchronize: %s\n", cudaGetErrorString(err));
		cudaFree(d_grad_beta);
		return make_double3(0.0, 0.0, 0.0);
	}

	// copy the grad_beta to the host
	double3 grad_beta;
	err = cudaMemcpy(&grad_beta, d_grad_beta, sizeof(double3), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy device memory to host: %s\n", cudaGetErrorString(err));
		cudaFree(d_grad_beta);
		return make_double3(0.0, 0.0, 0.0);
	}

	cudaFree(d_grad_beta);

	// calculate the net force from the potential field
	double3 net_force;
	net_force = calculate_grad_phi(grad_beta, agent_position, goal_position, K)*-k_force;

	// cap the force magnitude
	if(max_allowable_force > 0.0){
		double force_magnitude = norm(net_force);   
		if (force_magnitude > max_allowable_force) {
			double scale = max_allowable_force / force_magnitude;
			net_force = net_force * scale;
		}
	}

	// Print timing information if debug is enabled
	if (debug) {
		auto end_time = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = end_time - start_time;
		std::cout << std::left << std::setw(45) << "NavigationFunctionForce"
				  << "kernel execution time: " 
				  << std::fixed << std::setprecision(9) 
				  << elapsed.count() << " seconds" << std::endl;
	}

	// // print the net force
	// std::cout << "net_force: " << std::left << std::setw(10) << net_force.x << ", " << std::left << std::setw(10) << net_force.y << ", " << std::left << std::setw(10) << net_force.z << std::endl;

	return net_force;
}

// best function ever
__host__ void hello_cuda_world() {
	std::cout << "Hello CUDA World! -From Navigation Function Force Kernel <3" << std::endl;
}

}
