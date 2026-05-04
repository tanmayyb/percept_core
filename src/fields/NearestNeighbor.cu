#include "cuda_vector_ops.cuh"
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>


// Spatial Hash Function
__device__ inline uint32_t computeHash(int ix, int iy, int iz, uint32_t hash_size) 
{
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

// NEW: Reorders position arrays to match the sorted hash order.
// This enables coalesced memory access during the search phase.
__global__ void reorderDataKernel(
  const double* in_x, const double* in_y, const double* in_z,
  double* out_x, double* out_y, double* out_z,
  const uint32_t* sorted_indices,
  int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (i >= n) return;

  // The index 'i' here represents the position in the SORTED list.
  // 'src_idx' is the original particle index.
  int src_idx = sorted_indices[i];

  out_x[i] = in_x[src_idx];
  
  out_y[i] = in_y[src_idx];
  
  out_z[i] = in_z[src_idx];
}

__global__ void buildCellIndicesKernel(
  const uint32_t* sorted_hashes, 
  uint32_t* hash_starts, 
  uint32_t* hash_ends, 
  int n) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    uint32_t current_hash = sorted_hashes[i];

    if (i == 0) 
    {
      hash_starts[current_hash] = 0;
    }
    else 
    {
      uint32_t prev_hash = sorted_hashes[i - 1];
    
      if (current_hash != prev_hash) 
      {
        hash_ends[prev_hash] = i;
      
        hash_starts[current_hash] = i;
      }
    }
    if (i == n - 1) hash_ends[current_hash] = n;
}


__global__ void findNearestNeighborKernel(
  const double* x_sorted, const double* y_sorted, const double* z_sorted, // Sorted inputs
  const uint32_t* sorted_indices, // Needed only to return original ID
  const uint32_t* hash_starts, 
  const uint32_t* hash_ends,
  int* nearest_idx, 
  int n, GridConfig config, uint32_t hash_size) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= n) return;

  // Optimization: Read local position from sorted array. 
  // This read is fully coalesced.
  double px = x_sorted[i];

  double py = y_sorted[i];

  double pz = z_sorted[i];

  double min_dist_sq = 1e18;

  int best_original_idx = -1;

  // Recompute grid cell for the sorted particle
  int cx = (int)floor((px - config.min_boundary.x) / config.cell_size);

  int cy = (int)floor((py - config.min_boundary.y) / config.cell_size);

  int cz = (int)floor((pz - config.min_boundary.z) / config.cell_size);

  // Unroll loops manually or rely on compiler. 
  // Flattened loop reduces integer overhead slightly, but nested is readable.
  for (int dz = -1; dz <= 1; ++dz) 
  {
    for (int dy = -1; dy <= 1; ++dy) 
    {
      for (int dx = -1; dx <= 1; ++dx) 
      {
        uint32_t h = computeHash(cx + dx, cy + dy, cz + dz, hash_size);
        
        uint32_t start = hash_starts[h];

        if (start == 0xFFFFFFFF) continue;
        
        uint32_t end = hash_ends[h];

        for (uint32_t k = start; k < end; ++k) 
        {
          if (i == k) continue; // i and k are both indices in the sorted array

          // OPTIMIZATION: Read neighbor position from sorted arrays directly.
          // This replaces the indirect read: x[sorted_indices[k]]
          double nx = x_sorted[k];

          double ny = y_sorted[k];

          double nz = z_sorted[k];

          // OPTIMIZATION: No pow(), manual multiply
          double dx_v = px - nx;

          double dy_v = py - ny;

          double dz_v = pz - nz;

          double d2 = dx_v*dx_v + dy_v*dy_v + dz_v*dz_v;

          if (d2 < min_dist_sq) 
          {
            min_dist_sq = d2;

            // Retrieve the ORIGINAL particle ID only when a new best is found
            best_original_idx = sorted_indices[k]; 
          }
        }
      }
    }
  }
    
  // Write result to the slot corresponding to the SORTED particle 'i'.
  // If the user expects nearest_idx aligned to ORIGINAL order, this needs a scatter.
  // Assuming output buffer is aligned to sorted order for simulation steps:
  nearest_idx[i] = best_original_idx; 
  
  // NOTE: If output must be in original order, add:
  // int original_i = sorted_indices[i];
  // nearest_idx[original_i] = best_original_idx;
  // But this causes uncoalesced writes. Better to keep data sorted.
}

extern "C" void build_spatial_index(
  const double* d_x, const double* d_y, const double* d_z,
  double* d_sorted_x, double* d_sorted_y, double* d_sorted_z,
  uint32_t* d_cell_hashes, uint32_t* d_point_indices,
  uint32_t* d_hash_starts, uint32_t* d_hash_ends,
  int n, GridConfig config, uint32_t hash_size,
  cudaStream_t stream
  ) 
{ 
  if (n <= 0) return;
  
  int threads = 256;
  
  int blocks = (n + threads - 1) / threads;

  // 1. Compute Hashes
  computeHashKernel<<<blocks, threads, 0, stream>>>(d_x, d_y, d_z, d_cell_hashes, d_point_indices, n, config, hash_size);
  
  // 2. Sort Indices by Hash
  thrust::device_ptr<uint32_t> t_hashes(d_cell_hashes);

  thrust::device_ptr<uint32_t> t_indices(d_point_indices);
  
  thrust::sort_by_key(thrust::cuda::par.on(stream), t_hashes, t_hashes + n, t_indices);

  // 3. NEW: Reorder Position Arrays
  reorderDataKernel<<<blocks, threads, 0, stream>>>(
    d_x, d_y, d_z, 
    d_sorted_x, d_sorted_y, d_sorted_z, 
    d_point_indices, n
  );

  // 4. Build Cell Map
  cudaMemsetAsync(d_hash_starts, 0xFF, hash_size * sizeof(uint32_t), stream);
  
  cudaMemsetAsync(d_hash_ends, 0, hash_size * sizeof(uint32_t), stream);
  
  buildCellIndicesKernel<<<blocks, threads, 0, stream>>>(d_cell_hashes, d_hash_starts, d_hash_ends, n);
}

extern "C" void find_nearest_neighbors(
  const double* d_sorted_x, const double* d_sorted_y, const double* d_sorted_z, // Uses sorted inputs
  const uint32_t* d_sorted_indices,
  const uint32_t* d_cell_starts, const uint32_t* d_cell_ends,
  int* d_nearest_idx, int n, GridConfig config, uint32_t hash_size, 
  cudaStream_t stream) 
{
  int threads = 256; // 1024 can increase register pressure; 256 is usually safer for heavy kernels

  int blocks = (n + threads - 1) / threads;

  findNearestNeighborKernel<<<blocks, threads, 0, stream>>>(
    d_sorted_x, d_sorted_y, d_sorted_z, 
    d_sorted_indices, d_cell_starts, d_cell_ends, 
    d_nearest_idx, n, config, hash_size
  );

  cudaStreamSynchronize(stream);
}