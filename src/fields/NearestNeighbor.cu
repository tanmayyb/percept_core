#include "cuda_vector_ops.cuh"
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>


// Spatial Hash Function
__device__ inline uint32_t computeHash(int ix, int iy, int iz, uint32_t hash_size) {
    return ((uint32_t)(ix * 73856093) ^ (uint32_t)(iy * 19349663) ^ (uint32_t)(iz * 83492791)) % hash_size;
}

__global__ void computeHashKernel(
    const double* x, const double* y, const double* z, 
    uint32_t* cell_hashes, uint32_t* point_indices, 
    int n, GridConfig config, uint32_t hash_size) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int ix = (int)floor((x[i] - config.min_boundary.x) / config.cell_size);
    int iy = (int)floor((y[i] - config.min_boundary.y) / config.cell_size);
    int iz = (int)floor((z[i] - config.min_boundary.z) / config.cell_size);

    cell_hashes[i] = computeHash(ix, iy, iz, hash_size);
    point_indices[i] = i; 
}

// buildCellIndicesKernel remains similar but operates on hash_size
__global__ void buildCellIndicesKernel(
    const uint32_t* sorted_hashes, 
    uint32_t* hash_starts, 
    uint32_t* hash_ends, 
    int n) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint32_t current_hash = sorted_hashes[i];
    if (i == 0) hash_starts[current_hash] = 0;
    else {
        if (current_hash != sorted_hashes[i - 1]) {
            hash_ends[sorted_hashes[i - 1]] = i;
            hash_starts[current_hash] = i;
        }
    }
    if (i == n - 1) hash_ends[current_hash] = n;
}

__global__ void findNearestNeighborKernel(
    const double* x, const double* y, const double* z,
    const uint32_t* sorted_indices,
    const uint32_t* hash_starts, 
    const uint32_t* hash_ends,
    int* nearest_idx, 
    int n, GridConfig config, uint32_t hash_size) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double px = x[i], py = y[i], pz = z[i];
    double min_dist_sq = 1e18;
    int best_idx = -1;

    int cx = (int)floor((px - config.min_boundary.x) / config.cell_size);
    int cy = (int)floor((py - config.min_boundary.y) / config.cell_size);
    int cz = (int)floor((pz - config.min_boundary.z) / config.cell_size);

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                uint32_t h = computeHash(cx + dx, cy + dy, cz + dz, hash_size);
                
                uint32_t start = hash_starts[h];
                if (start == 0xFFFFFFFF) continue;
                uint32_t end = hash_ends[h];

                for (uint32_t k = start; k < end; ++k) {
                    int j = sorted_indices[k];
                    if (i == j) continue;
                    
                    double d2 = pow(px - x[j], 2) + pow(py - y[j], 2) + pow(pz - z[j], 2);
                    if (d2 < min_dist_sq) {
                        min_dist_sq = d2;
                        best_idx = j;
                    }
                }
            }
        }
    }
    nearest_idx[i] = best_idx;
}

extern "C" void build_spatial_index(
    const double* d_x, const double* d_y, const double* d_z,
    uint32_t* d_cell_hashes, uint32_t* d_point_indices,
    uint32_t* d_hash_starts, uint32_t* d_hash_ends,
    int n, GridConfig config, uint32_t hash_size,
    cudaStream_t stream
  ) 
{ 
    if (n <= 0) return;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    computeHashKernel<<<blocks, threads, 0, stream>>>(d_x, d_y, d_z, d_cell_hashes, d_point_indices, n, config, hash_size);
    
    thrust::device_ptr<uint32_t> t_hashes(d_cell_hashes);
    thrust::device_ptr<uint32_t> t_indices(d_point_indices);
    thrust::sort_by_key(thrust::cuda::par.on(stream), t_hashes, t_hashes + n, t_indices);

    cudaMemset(d_hash_starts, 0xFF, hash_size * sizeof(uint32_t));
    cudaMemset(d_hash_ends, 0, hash_size * sizeof(uint32_t));

    buildCellIndicesKernel<<<blocks, threads, 0, stream>>>(d_cell_hashes, d_hash_starts, d_hash_ends, n);
}

extern "C" void find_nearest_neighbors(
    const double* d_x, const double* d_y, const double* d_z,
    const uint32_t* d_sorted_indices,
    const uint32_t* d_cell_starts, const uint32_t* d_cell_ends,
    int* d_nearest_idx, int n, GridConfig config, uint32_t hash_size) 
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    findNearestNeighborKernel<<<blocks, threads>>>(
        d_x, d_y, d_z, d_sorted_indices, d_cell_starts, d_cell_ends, 
        d_nearest_idx, n, config, hash_size
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel failed: %s\n", cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize();
}