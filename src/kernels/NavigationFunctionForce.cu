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
#include "percept/NavigationFunctionForce.h"
#include "percept/cuda_vector_ops.cuh"

// time keeper
#include <chrono>
#include <iomanip>

#define threads 1024
#define num_grid_points 6

// ---	Navigation Function Params	---
#define K 3.0	// K is a shaping parameter for the navigation function
#define eps 0.05 // 5 cm

namespace navigation_function {
using namespace cuda_vector_ops;


__device__ __forceinline__ double calculate_gamma(const double& dist_to_goal) {
	return pow(dist_to_goal, 2.0 * K);
}

__global__ void accumulate_log_beta(
	double* d_log_beta,
	const double3* d_masses,
	size_t num_masses,
	const double3* d_grid_points,
	double detect_shell_rad
){
	extern __shared__ double sdata[];
	int tid = threadIdx.x;
	const unsigned int mass_idx = blockIdx.x * blockDim.x + tid;
	const unsigned int gp_idx   = blockIdx.y;
	sdata[tid] = 0.0; // set as zero

	if (gp_idx >= num_grid_points || mass_idx >= num_masses) return;

	const double3 gp	= d_grid_points[gp_idx];
	const double3 mass_pos = d_masses[mass_idx];

	double dist = norm(mass_pos - gp);
	dist = fmax(dist, 1e-5);
	double term = dist * dist - detect_shell_rad * detect_shell_rad;         // (‖p-oᵢ‖²−rᵢ²)
	term = fmax(term, 1e-12);
	double log_term = log(term);

	sdata[tid] = log_term;

reduction:
	__syncthreads();
	// Reduction in shared memory
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	// Thread 0 of each block adds the block's sum to the global sum using atomics
	if (tid == 0) {
		atomicAdd(&d_log_beta[gp_idx], sdata[0]);
	}

}

__global__ void compute_phi(
	double* d_unav,
	const double* d_log_beta,
	const double3* d_grid_points,
	double3 goal_position
){
	const unsigned int gp_idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (gp_idx >= num_grid_points) return;

	const double dist_to_goal = norm(goal_position - d_grid_points[gp_idx]);
	const double gamma = calculate_gamma(dist_to_goal);
	const double beta = exp(d_log_beta[gp_idx]);
	const double alpha = gamma / beta;
	const double phi = (alpha < 0.0) ? 1.0 : pow(alpha / (1.0 + alpha), 1.0 / K);
	d_unav[gp_idx] = phi;
}


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
) {
	// Start timing if debug is enabled
	auto start_time = std::chrono::high_resolution_clock::now();

	// Allocate host memory for grid points and set them
	double3 grid_points[num_grid_points];
	grid_points[0] = make_double3(agent_position.x - eps, agent_position.y, agent_position.z);
	grid_points[1] = make_double3(agent_position.x + eps, agent_position.y, agent_position.z);
	grid_points[2] = make_double3(agent_position.x, agent_position.y - eps, agent_position.z);
	grid_points[3] = make_double3(agent_position.x, agent_position.y + eps, agent_position.z);
	grid_points[4] = make_double3(agent_position.x, agent_position.y, agent_position.z - eps);
	grid_points[5] = make_double3(agent_position.x, agent_position.y, agent_position.z + eps);

	// Allocate device memory for grid positions
	double3* d_grid_points;
	cudaError_t err = cudaMalloc(&d_grid_points, sizeof(double3) * num_grid_points);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device memory for grid_points: %s\n", cudaGetErrorString(err));
		cudaFree(d_grid_points);
		return make_double3(0.0, 0.0, 0.0);
	}


	// Allocate device memory for log_beta
	double* d_log_beta;
	err = cudaMalloc(&d_log_beta, sizeof(double)*num_grid_points);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to set device memory: %s\n", cudaGetErrorString(err));
		cudaFree(d_grid_points);
		cudaFree(d_log_beta);
		return make_double3(0.0, 0.0, 0.0);
	}

	err = cudaMemset(d_log_beta, 0, sizeof(double) * num_grid_points);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to set d_log_beta to zero: %s\n", cudaGetErrorString(err));
		cudaFree(d_grid_points);
		cudaFree(d_log_beta);
		return make_double3(0.0, 0.0, 0.0);
	}

	// Allocate device memory for unav
	double* d_unav;
	err = cudaMalloc(&d_unav, sizeof(double)*num_grid_points);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to set device memory: %s\n", cudaGetErrorString(err));
		cudaFree(d_grid_points);
		cudaFree(d_log_beta);
		cudaFree(d_unav);
		return make_double3(0.0, 0.0, 0.0);
	}

	err = cudaMemset(d_unav, 0, sizeof(double) * num_grid_points);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to set d_unav to zero: %s\n", cudaGetErrorString(err));
		cudaFree(d_grid_points);
		cudaFree(d_log_beta);
		cudaFree(d_unav);
		return make_double3(0.0, 0.0, 0.0);
	}

	// Copy grid_positions to device
	err = cudaMemcpy(d_grid_points, grid_points, sizeof(double3) * num_grid_points, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy grid_points to device: %s\n", cudaGetErrorString(err));
		cudaFree(d_grid_points);
		cudaFree(d_log_beta);
		cudaFree(d_unav);
		return make_double3(0.0, 0.0, 0.0);
	}

	// Launch the accumulate_log_beta kernel
	int num_blocks = (num_masses + threads - 1) / threads;
	dim3 blockDimAcc(threads);
	dim3 gridDimAcc(num_blocks, num_grid_points);
	size_t shared_mem_size = threads * sizeof(double);
	accumulate_log_beta<<<gridDimAcc, blockDimAcc, shared_mem_size>>>(
		d_log_beta, d_masses, num_masses, d_grid_points, detect_shell_rad
	);

	// Check for kernel launch errors
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to get last error: %s\n", cudaGetErrorString(err));
		cudaFree(d_grid_points);
		cudaFree(d_log_beta);
		cudaFree(d_unav);
		return make_double3(0.0, 0.0, 0.0);
	}

	// Add synchronization check after kernel launch
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to synchronize: %s\n", cudaGetErrorString(err));
		cudaFree(d_grid_points);
		cudaFree(d_log_beta);
		cudaFree(d_unav);
		return make_double3(0.0, 0.0, 0.0);
	}

	// Launch the compute_phi kernel
	dim3 blockDimPhi(num_grid_points);
	compute_phi<<<1, blockDimPhi>>>(d_unav, d_log_beta, d_grid_points, goal_position);

	// Check for kernel launch errors
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to get last error: %s\n", cudaGetErrorString(err));
		cudaFree(d_grid_points);
		cudaFree(d_log_beta);
		cudaFree(d_unav);
		return make_double3(0.0, 0.0, 0.0);
	}

	// Add synchronization check after kernel launch
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to synchronize: %s\n", cudaGetErrorString(err));
		cudaFree(d_grid_points);
		cudaFree(d_log_beta);
		cudaFree(d_unav);
		return make_double3(0.0, 0.0, 0.0);
	}

	// Copy the result from the device to the host
	double unav[6];
	err = cudaMemcpy(unav, d_unav, sizeof(double)*num_grid_points, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy result from device: %s\n", cudaGetErrorString(err));
		cudaFree(d_grid_points);
		cudaFree(d_log_beta);
		cudaFree(d_unav);
		return make_double3(0.0, 0.0, 0.0);
	}

	cudaFree(d_grid_points);
	cudaFree(d_log_beta);
	cudaFree(d_unav);

	// calculate the net force from the potential field
	double3 net_force;
	net_force.x = -k_force*(unav[1] - unav[0])/(2.0*eps);
	net_force.y = -k_force*(unav[3] - unav[2])/(2.0*eps);
	net_force.z = -k_force*(unav[5] - unav[4])/(2.0*eps);

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

	return net_force;
}

// best function ever
__host__ void hello_cuda_world() {
	std::cout << "Hello CUDA World! -From Navigation Function Force Kernel <3" << std::endl;
}

}
