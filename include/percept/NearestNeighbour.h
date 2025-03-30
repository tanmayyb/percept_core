#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <vector_types.h>


#include "percept/cuda_vector_ops.cuh"

namespace nearest_neighbour{

__host__ void launch_kernel(
    double3* d_masses,
    size_t num_masses,
    int* nn_index,
    bool debug
);

}