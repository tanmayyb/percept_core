#pragma once
#include <vector>
#include <vector_types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <random>
#include <iostream>
#include <iomanip>

namespace cuda_vector_ops
{
  __host__ __device__ inline double3 operator+(const double3& a, const double3& b) 
  {
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
  }

  __host__ __device__ inline double3 operator-(const double3& a, const double3& b) 
  {
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
  }

  __host__ __device__ inline double3 operator*(const double3& a, const double scalar) 
  {
    return make_double3(a.x * scalar, a.y * scalar, a.z * scalar);
  }

  __host__ __device__ inline double3 operator/(const double3& a, const double scalar) 
  {
    return make_double3(a.x / scalar, a.y / scalar, a.z / scalar);
  }

  __host__ __device__ inline double norm(const double3 &v) 
  {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
  }

  __host__ __device__ inline double squared_norm(const double3 &v) 
  {
    return v.x * v.x + v.y * v.y + v.z * v.z;
  }

  __host__ __device__ inline double norm_reciprocal(const double3 &v) 
  {
    double mag2 = v.x * v.x + v.y * v.y + v.z * v.z;
  
    return mag2 > 0.0 ? 1.0 / sqrt(mag2) : 0.0;
  }

  __host__ __device__ inline double squared_distance(const double3 a, const double3 b) 
  {
    double dx = a.x - b.x;
    
    double dy = a.y - b.y;
    
    double dz = a.z - b.z;
    
    return dx * dx + dy * dy + dz * dz;
  }

  __device__ inline double fma(double a, double b, double c) 
  {
    // __fma_rn not working :()
    // return __fma_rn(a, b, c); // computes a * b + c in one instruction
    return a * b + c;
  }

  __host__ __device__ inline double dot(const double3 &a, const double3 &b) 
  {
    return a.x * b.x + a.y * b.y + a.z * b.z;
  }

  __host__ __device__ inline double3 cross(const double3 &a, const double3 &b) 
  {
    return make_double3(a.y * b.z - a.z * b.y,
                        a.z * b.x - a.x * b.z,
                        a.x * b.y - a.y * b.x);
  }

  __host__ __device__ inline double3 normalized(const double3 &v) 
  {
    double mag = norm(v);
  
    if (mag > 0.0) 
    {
      return v * (1.0 / mag);
    } 
    else 
    {
      return make_double3(0.0, 0.0, 0.0);
    }
  }

  __device__ inline double3 make_random_vector(int index, unsigned long long seed) 
  {
    auto hash = [](unsigned int x) -> unsigned int 
    {
      x = ((x >> 16) ^ x) * 0x45d9f3b;
      
      x = ((x >> 16) ^ x) * 0x45d9f3b;
      
      x = (x >> 16) ^ x;
      
      return x;
    };

    double3 ret;

    unsigned int r1 = hash(index + seed);
    
    unsigned int r2 = hash(r1 + seed);
    
    unsigned int r3 = hash(r2 + seed);
    
    ret.x = (double)r1 / 4294967295.0 * 2.0 - 1.0;
    
    ret.y = (double)r2 / 4294967295.0 * 2.0 - 1.0;
    
    ret.z = (double)r3 / 4294967295.0 * 2.0 - 1.0;
    
    return ret;
  }

} //namespace cuda_vector_ops


// --- Nearest Neighbor ---
struct GridConfig 
{
  double3 min_boundary;
 
  double cell_size;
};


// --- NVTX profiling options ---
enum class NVTXColor : uint32_t 
{
  Red     = 0xFFFF0000,
  
  Green   = 0xFF00FF00,
  
  Blue    = 0xFF0000FF,
  
  Yellow  = 0xFFFFFF00,
  
  Magenta = 0xFFFF00FF,
  
  Cyan    = 0xFF00FFFF,
  
  White   = 0xFFFFFFFF,
  
  Orange  = 0xFFFFA500
};

#ifdef GPU_PROFILING_ENABLE
  #include <nvToolsExt.h>
#endif

inline void push_nvtx_range(const char* name, NVTXColor color) 
{
  #ifdef GPU_PROFILING_ENABLE
    nvtxEventAttributes_t eventAttrib = {0};
    
    eventAttrib.version = NVTX_VERSION;
    
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    
    eventAttrib.color = static_cast<uint32_t>(color);
    
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    
    eventAttrib.message.ascii = name;
    
    nvtxRangePushEx(&eventAttrib);

  #endif
}

inline void pop_nvtx_range() 
{
  #ifdef GPU_PROFILING_ENABLE
    nvtxRangePop();
  #endif
}