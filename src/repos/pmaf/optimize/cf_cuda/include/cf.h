#include <cuda.h>
#include <cuda_runtime.h>

#pragma once
#include <vector>
#include "obstacles.hpp"




void launch_circForce_kernel(
    std::vector<Obstacle> *obstacles, 
    int n_obstacles,
    double k_circ, 
    double detect_shell_rad_,
    double* goalPosition,
    double* agentPosition,
    double* agentVelocity,
    double* force
);

__host__ void hello_world();