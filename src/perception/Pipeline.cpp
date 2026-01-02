#include "Pipeline.hpp"
#include <iostream>
#include <vector>
#include "open3d/Open3D.h"

#include <cuda_runtime.h>

namespace perception
{
	Pipeline::Pipeline()
	{
		device_ = open3d::core::Device("cuda:0");
		if(!open3d::core::cuda::IsAvailable())
		{
			device_ = open3d::core::Device("CPU:0");
		}

		auto cuda_devices = open3d::core::Device::GetAvailableCUDADevices();

		for (const auto& dev : cuda_devices) {
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, dev.GetID());
			std::cout << dev.ToString() << " -> " << prop.name << std::endl;
		}

		// open3d::t::geometry::PointCloud pcd(device_);
		
		// pcd.SetPointPositions(points_tensor);

		// int64_t num_points = pcd.GetPointPositions().GetShape(0);
	}

	void Pipeline::setupConfigs(size_t batch_size, size_t n_points, const std::vector<CameraConfig>& cameras)
	{
		batch_size_ = batch_size;
		
		n_points_ = n_points;

		transforms_.clear();

		transforms_.reserve(cameras.size());
		
		for (const auto& camera : cameras)
		{
			transforms_.emplace_back(
				open3d::core::eigen_converter::EigenMatrixToTensor(camera.transform).To(device_)
			);
		}
	}


	void Pipeline::run()
	{
		// for all pipes

		// create pointclouds from buffer

		// transform the pointclouds, use device tensors for the coordinate transforms

		// merge into single pointcloud

		// do robot filtering

		// voxel downsample

		// D2H to get pointset in CPU

		// apply SoA

		// write to mailbox
	}

	void Pipeline::readMailbox()
	{
		std::vector<float>& buffer = mailbox_ptr_->consume();

		pc_buffer_ = open3d::core::Tensor(
			buffer.data(),
			{static_cast<int64_t>(batch_size_ * n_points_), 3},
			open3d::core::Float32,
			device_
		);

		// std::cout << "Shape: "  << points_tensor.GetShape().ToString() << "\n";

		// std::cout << "Device: " << points_tensor.GetDevice().ToString() << "\n";

		// std::cout << "Type: "   << points_tensor.GetDtype().ToString() << "\n";
	}


	Pipeline::~Pipeline(){}


}