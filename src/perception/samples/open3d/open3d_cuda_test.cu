#include <iostream>
#include <open3d/Open3D.h>
#include <open3d/core/Device.h>
#include <open3d/core/CUDAUtils.h> /// Required for specific CUDA availability checks

int main(int argc, char** argv) {
    // Check if CUDA is available via the core library
    if (open3d::core::cuda::IsAvailable()) {
        std::cout << "Open3D CUDA initialized." << std::endl;
        
        // Attempt to create a device context
        open3d::core::Device device("cuda:0");
        std::cout << "Successfully created device: " << device.ToString() << std::endl;
    } else {
        std::cerr << "Open3D was not built with CUDA support or no GPU found." << std::endl;
        return 1;
    }
    return 0;
}