#pragma once
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>
#include <random>

namespace cuda_vector_ops{

__host__ __device__ inline double3 operator+(const double3& a, const double3& b) {
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline double3 operator-(const double3& a, const double3& b) {
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline double3 operator*(const double3& a, const double scalar) {
    return make_double3(a.x * scalar, a.y * scalar, a.z * scalar);
}

__host__ __device__ inline double norm(const double3 &v) {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

__host__ __device__ inline double squared_norm(const double3 &v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

__host__ __device__ inline double norm_reciprocal(const double3 &v) {
    double mag2 = v.x * v.x + v.y * v.y + v.z * v.z;
    return mag2 > 0.0 ? 1.0 / sqrt(mag2) : 0.0;
}

__host__ __device__ inline double squared_distance(const double3 a, const double3 b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    double dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

__device__ inline double fma(double a, double b, double c) {
    // __fma_rn not working :()
    // return __fma_rn(a, b, c); // computes a * b + c in one instruction
    return a * b + c;
}

__host__ __device__ inline double dot(const double3 &a, const double3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline double3 cross(const double3 &a, const double3 &b) {
    return make_double3(a.y * b.z - a.z * b.y,
                        a.z * b.x - a.x * b.z,
                        a.x * b.y - a.y * b.x);
}

__host__ __device__ inline double3 normalized(const double3 &v) {
    double mag = norm(v);
    if (mag > 0.0) {
        return v * (1.0 / mag);
    } else {
        return make_double3(0.0, 0.0, 0.0);
    }
}

__host__ __device__ inline double3 make_random_vector() {
    double3 ret;
    #ifdef __CUDA_ARCH__
        // GPU path using curand
        curandState state;
        curand_init(clock64(), threadIdx.x, 0, &state);
        ret.x = 2.0 * curand_uniform_double(&state) - 1.0; // Scale to [-1,1]
        ret.y = 2.0 * curand_uniform_double(&state) - 1.0;
        ret.z = 2.0 * curand_uniform_double(&state) - 1.0;
    #else
        // CPU path using std random
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-1.0, 1.0);
        ret.x = dis(gen);
        ret.y = dis(gen);
        ret.z = dis(gen);
    #endif
    return ret;
}

}