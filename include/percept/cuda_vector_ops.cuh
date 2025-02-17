#pragma once
#include <cuda_runtime.h>
#include <vector_types.h>


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
    return __fma_rn(a, b, c); // computes a * b + c in one instruction
}

__host__ __device__ inline double dot(const double3 &a, const double3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline double3 cross(const double3 &a, const double3 &b) {
    return make_double3(a.y * b.z - a.z * b.y,
                        a.z * b.x - a.x * b.z,
                        a.x * b.y - a.y * b.x);
}

__host__ __device__ inline double3 normalize(const double3 &v) {
    double mag = sqrt(dot(v, v));
    if (mag > 0.0) {
        return v * (1.0 / mag);
    } else {
        return make_double3(0.0, 0.0, 0.0);
    }
}

}