// publisher.cpp
#include "PointCloud.hpp"
#include <iceoryx_posh/popo/publisher.hpp>
#include <iceoryx_posh/runtime/posh_runtime.hpp>
#include <iostream>

int main() {
    iox::runtime::PoshRuntime::initRuntime("iox-cloud-publisher");

    iox::popo::Publisher<PointCloud> publisher({"Sensor", "LiDAR", "PointCloud"});

		size_t num_points = 1000;

    while (true) {
        // Loan memory from shared pool
        publisher.loan()
            .and_then([&](auto& sample) {
                // Populate sample->points directly (Zero-Copy)
                sample->width = num_points;
                sample->height = 1;
                for (uint32_t i = 0; i < num_points; ++i) {
                    sample->points[i] = {static_cast<float>(i), 0.0f, 0.0f, 100};
                }
                sample.publish();
            })
            .or_else([](auto& error) {
                std::cerr << "Failed to loan sample: " << error << std::endl;
            });

        std::this_thread::sleep_for(std::chrono::milliseconds(100));

			if (num_points>1){
				--num_points;
			}
			else{
				num_points = 1000;
			}
    }
    return 0;
}
