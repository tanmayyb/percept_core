// pointcloud.hpp
#pragma once
#include <cstdint>

struct Point {
    float x, y, z;
    uint32_t intensity;
};

struct PointCloud {
    static constexpr uint32_t MAX_POINTS = 100000;
    uint32_t width{0};
    uint32_t height{0};
    Point points[MAX_POINTS]; // Fixed-size allocation in shared memory
};

// struct PointCloud {
//     static constexpr uint32_t MAX_POINTS = 1000000;
// 		uint32_t num_points{0};
// 		float x[MAX_POINTS];
// 		float y[MAX_POINTS];
// 		float z[MAX_POINTS];
// };