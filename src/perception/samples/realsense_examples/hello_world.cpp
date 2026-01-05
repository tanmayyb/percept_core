#include <librealsense2/rs.hpp>
#include <iostream>

int main(int argc, char * argv[]) try
{
    // Create a Pipeline - standard SDK top-level API
    rs2::pipeline p;

    // Start streaming with default configuration
    p.start();

    while (true)
    {
        // Block program until frames arrive
        rs2::frameset frames = p.wait_for_frames();

        // Retrieve depth frame
        rs2::depth_frame depth = frames.get_depth_frame();

        // Get dimensions
        auto width = depth.get_width();
        auto height = depth.get_height();

        // Query distance to center pixel
        float dist_to_center = depth.get_distance(width / 2, height / 2);

        // Print distance (\r keeps the output on one line)
        std::cout << "The camera is facing an object " << dist_to_center << " meters away \r" << std::flush;
    }

    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}


/* Include the librealsense CPP header files */
#include <librealsense2/rs.hpp>
#include <librealsense2/rs_advanced_mode.hpp>

#include <iostream>

using namespace std;
using namespace rs2;


int main(int argc, char** argv) try
{
		// Obtain a list of devices currently present on the system
		context ctx;
		auto devices = ctx.query_devices();
		size_t device_count = devices.size();
		if (!device_count)
		{
				cout <<"No device detected. Is it plugged in?\n";
				return EXIT_SUCCESS;
		}

		// Get the first connected device
		auto dev = devices[0];

		// Check if current device supports advanced mode
		if (dev.is<rs400::advanced_mode>())
		{
				// Get the advanced mode functionality
				auto advanced_mode_dev = dev.as<rs400::advanced_mode>();
				const int max_val_mode = 2; // 0 - Get maximum values
				// Get the maximum values of depth controls
				auto depth_control_group = advanced_mode_dev.get_depth_control(max_val_mode);

				// Apply the depth controls maximum values
				advanced_mode_dev.set_depth_control(depth_control_group);
		}
		else
		{
				cout << "Current device doesn't support advanced-mode!\n";
				return EXIT_FAILURE;
		}

		return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
		cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << endl;
		return EXIT_FAILURE;
}
catch (const exception & e)
{
		cerr << e.what() << endl;
		return EXIT_FAILURE;
}