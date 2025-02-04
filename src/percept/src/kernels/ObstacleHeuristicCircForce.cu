// the usual
#include <iostream>
#include <vector>

// necessary evils
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>

// include the header
#include "percept/ObstacleHeuristicCircForce.h"

// time keeper
#include <chrono>


#define threads 256
//  NEED EDGE CONDITION HANDLER 
//  i.e. when known num_obstacles < 256


namespace heuristic_kernel{

__device__ inline double3 operator+(const double3& a, const double3& b) {
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline double3 operator-(const double3& a, const double3& b) {
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline double3 operator*(const double3& a, const double scalar) {
    return make_double3(a.x * scalar, a.y * scalar, a.z * scalar);
}

__device__ inline double norm(const double3 &v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

__device__ inline double norm_reciprocal(const double3 &v) {
    double mag2 = v.x * v.x + v.y * v.y + v.z * v.z;
    return mag2 > 0.0 ? 1.0 / sqrt(mag2) : 0.0;
}

__device__ inline double squared_distance(const double3 a, const double3 b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    double dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

__device__ inline double fma(double a, double b, double c) {
    return __fma_rn(a, b, c); // computes a * b + c in one instruction
}

__device__ inline double dot(const double3 &a, const double3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline double3 cross(const double3 &a, const double3 &b) {
    return make_double3(a.y * b.z - a.z * b.y,
                        a.z * b.x - a.x * b.z,
                        a.x * b.y - a.y * b.x);
}

__device__ inline double3 normalize(const double3 &v) {
    double mag = sqrt(dot(v, v));
    if (mag > 0.0) {
        return v * (1.0 / mag);
    } else {
        return make_double3(0.0, 0.0, 0.0);
    }
}


__global__ void ObstacleHeuristic_circForce_kernel(
    double3* d_net_force,
    double3* d_masses,
    size_t num_masses,
    double3 agent_position,
    double3 agent_velocity,
    double3 goal_position,
    double k_circ,
    double detect_shell_rad_
){
    extern __shared__ double3 sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    sdata[tid] = make_double3(0.0, 0.0, 0.0); // set as zero

    // Each thread computes a force if within bound
    if (i >= num_masses) {
        return;
    }

    double3 goal_vec;
    double dist_to_goal;
    double3 mass_position;
    double3 mass_dist_vec;
    double3 mass_velocity;
    double3 mass_rvel_vec;
    double3 force_vec;
    double3 mass_dist_vec_normalized;
    double dist_to_mass;
    double agent_radius = 0.0;
    double mass_radius = 0.1;
    double3 current_vec;
    double nn_distance = 1000.0;
    double nn_mass_dist_k;
    int nn_mass_idx = -1;
    double3 nn_mass_position;
    double3 obstacle_vec;
    double3 rot_vec;
    double3 mass_rvel_vec_normalized;

    // implementation of obstacle heuristic circ force
    goal_vec = goal_position - agent_position;
    dist_to_goal = norm(goal_vec);
    mass_position = d_masses[i];
    mass_dist_vec = mass_position - agent_position;
    mass_velocity = make_double3(0.0, 0.0, 0.0);
    mass_rvel_vec = agent_velocity - mass_velocity;
    force_vec = make_double3(0.0, 0.0, 0.0); // set default as zero
    mass_dist_vec_normalized = normalize(mass_dist_vec);


    // "Skip this obstacle if it's behind us AND we're moving away from it"
    if(dot(mass_dist_vec_normalized, normalize(goal_vec)) < -0.01 &&
        dot(mass_dist_vec, mass_rvel_vec) < -0.01)
        {
            goto reduction; //sdata already set to zero
        }

    dist_to_mass = norm(mass_dist_vec) - (agent_radius + mass_radius);
    dist_to_mass = fmax(dist_to_mass, 1e-5); // avoid division by zero

    // implement OBSTACLE HEURISTIC
    // calculate rotation vector, current vector, and force vector
    if(dist_to_mass < detect_shell_rad_ && norm(mass_rvel_vec) > 1e-10){ 
            

        // find nearest neighbor using brute force :/
        for(int k=0; k<num_masses; k++){
        // find ways to optimize this
        // need to use kdtrees/FLANN or some other method to find nearest neighbor
            if(k != i){
                nn_mass_dist_k = squared_distance(d_masses[k], mass_position);
                if(nn_mass_dist_k < nn_distance){ // update nearest neighbor
                    nn_distance = nn_mass_dist_k;
                    nn_mass_idx = k;
                }
            }
        
        }

        // calculate rotation vector
        nn_mass_position = d_masses[nn_mass_idx];
        obstacle_vec = nn_mass_position - mass_position;
        current_vec = mass_dist_vec_normalized * dot(mass_dist_vec_normalized, obstacle_vec) - obstacle_vec;     
        rot_vec = normalize(cross(current_vec, mass_dist_vec_normalized));

        // calculate current vector
        current_vec = normalize(cross(mass_dist_vec_normalized, rot_vec)); // same variable name, different context
        mass_rvel_vec_normalized = normalize(mass_rvel_vec);

        // calculate force vector
        // force_vec = cross(mass_rvel_vec_normalized, cross(current_vec, mass_rvel_vec_normalized));
        //  A×(B×C) = B(A·C) - C(A·B)
        force_vec = (current_vec * dot(mass_rvel_vec_normalized, mass_rvel_vec_normalized)) - 
            (mass_rvel_vec_normalized * dot(mass_rvel_vec_normalized, current_vec));

        force_vec = force_vec * (k_circ / pow(dist_to_mass, 2));      
    }

    sdata[tid] = force_vec;

reduction:
    // Perform reduction in shared memory
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid+s<num_masses) {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 of each block adds the block's sum to the global sum using atomics
    if (tid == 0) {
        atomicAdd(&(d_net_force->x), sdata[0].x);
        atomicAdd(&(d_net_force->y), sdata[0].y);
        atomicAdd(&(d_net_force->z), sdata[0].z);
    }

}


__host__ double3 launch_ObstacleHeuristic_circForce_kernel(
    double3* d_masses,
    size_t num_masses,
    double3 agent_position,
    double3 agent_velocity,
    double3 goal_position,
    double k_circ, 
    double detect_shell_rad_,
    bool debug
){
    // Allocate device memory for net force
    double3* d_net_force;
    cudaError_t err = cudaMalloc(&d_net_force, sizeof(double3));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory: %s\n", cudaGetErrorString(err));
        return make_double3(0.0, 0.0, 0.0);  // or handle error appropriately
    }
    // set memory to zero
    err = cudaMemset(d_net_force, 0, sizeof(double3));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set device memory: %s\n", cudaGetErrorString(err));
        cudaFree(d_net_force);
        return make_double3(0.0, 0.0, 0.0);
    }

    int num_blocks = (num_masses + threads - 1) / threads; // ceiling division
    size_t shared_mem_size = threads * sizeof(double3);
    ObstacleHeuristic_circForce_kernel<<<num_blocks, threads, shared_mem_size>>>(
        d_net_force, d_masses, num_masses,
        agent_position, agent_velocity, goal_position,
        k_circ, detect_shell_rad_
    ); // CUDA kernels automatically copy value-type parameters to the device when called
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch kernel: %s\n", cudaGetErrorString(err));
        cudaFree(d_net_force);
        return make_double3(0.0, 0.0, 0.0);
    }

    // Copy result back to host
    double3 net_force;
    err = cudaMemcpy(&net_force, d_net_force, sizeof(double3), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy result from device: %s\n", cudaGetErrorString(err));
        cudaFree(d_net_force);
        return make_double3(0.0, 0.0, 0.0);
    }
    
    // Free device memory
    cudaFree(d_net_force);
    // Don't free d_masses here as it was allocated elsewhere

    return net_force;
}


// best function ever
__host__  void hello_cuda_world(){
  std::cout<<"Hello CUDA World!"<<std::endl;
}


}