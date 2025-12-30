#pragma once

#include <vector>
#include <librealsense2/rs.hpp>

namespace perception
{
	struct PointCloud
	{
		std::vector<float> data;

		size_t n_points = 0;

		void init(size_t n)
		{
			n_points = n;
		
			data.assign(n*3, 0.0f);
		}

		float* x_ptr() {return data.data();}
		
		float* y_ptr() {return data.data() + n_points;}
		
		float* z_ptr() {return data.data() + (n_points*2);}
	};

	// struct PointCloud
	// {
	// 	std::vector<float> x;

	// 	std::vector<float> y;

	// 	std::vector<float> z;

	// 	void resize(size_t size) {
	// 		x.resize(size);
		
	// 		y.resize(size);
		
	// 		z.resize(size);
	// 	}
	// }

	struct PointCloudData {
    std::vector<rs2::vertex> vertices;
	};

}