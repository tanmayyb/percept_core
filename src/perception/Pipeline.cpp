#include "Pipeline.hpp"
#include "Perception.hpp"

#include <iostream>
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

		transforms_.reserve(batch_size);
		
		for (const auto& camera : cameras)
		{
			transforms_.emplace_back(
				open3d::core::eigen_converter::EigenMatrixToTensor(camera.transform).To(device_)
			);
		}

		if (partial_pcds_.size() != batch_size_)
		{
			partial_pcds_.assign(
				batch_size_, 
				open3d::t::geometry::PointCloud(device_)
			);
		}

		shape_ = {static_cast<int64_t>(batch_size_ * n_points_), 3};

		min_bound_ = open3d::core::Tensor::Init<float>({-1.0f, -1.0f, -1.0f}, device_);
		
		max_bound_ = open3d::core::Tensor::Init<float>({1.0f, 1.0f, 1.0f}, device_);

		bbox_ = open3d::t::geometry::AxisAlignedBoundingBox(min_bound_, max_bound_);
	}


	void Pipeline::run()
	{
		auto start = std::chrono::high_resolution_clock::now();

		auto end = std::chrono::high_resolution_clock::now();
	
		std::chrono::duration<double, std::milli> elapsed;

		int64_t total_points;

		while(running_.load())
		{
			start = std::chrono::high_resolution_clock::now();

			readMailbox();

			for(size_t i=0; i<batch_size_; ++i)
			{
				open3d::core::Tensor points = pc_buffer_.Slice(
					0, i * n_points_, (i + 1) * n_points_
				);
				
				partial_pcds_[i].SetPointPositions(points);

				partial_pcds_[i].Transform(transforms_[i]);	
			}
			
			open3d::t::geometry::PointCloud accumulation = partial_pcds_[0];

			for (size_t i = 1; i < batch_size_; ++i)
			{
				accumulation = accumulation.Append(partial_pcds_[i]);
			}

			open3d::core::Tensor inside_indices = bbox_.GetPointIndicesWithinBoundingBox(accumulation.GetPointPositions());

			// total_points = accumulation.GetPointPositions().GetLength();

			// open3d::core::Tensor mask = open3d::core::Tensor::Zeros({total_points}, open3d::core::Bool, device_);

			// mask.SetItem({open3d::core::TensorKey::IndexTensor(inside_indices)}, 
			// 	open3d::core::Tensor::Ones({inside_indices.GetLength()}, open3d::core::Bool, device_));

			// open3d::core::Tensor outside_indices = mask.LogicalNot().NonZero().Flatten();

			// accumulation = accumulation.SelectByIndex(outside_indices);

			accumulation = accumulation.SelectByIndex(inside_indices);

			total_points = accumulation.GetPointPositions().GetLength();

			// merged_pcd_ = std::make_unique<open3d::t::geometry::PointCloud>(std::move(accumulation));

			// do robot filtering

			// voxel downsample

			// D2H to get pointset in CPU

			// apply SoA

			// write to mailbox

			// running_ = false; // added for debugging


			owner_->publishPointclouds(accumulation, static_cast<size_t>(total_points));

			std::this_thread::sleep_for(std::chrono::milliseconds(15));

			// std::cout << "Points: " << accumulation.GetPointPositions().GetLength() << std::endl;

			// owner_->test(n_points_);

			end = std::chrono::high_resolution_clock::now();
    
			elapsed = end - start;
			
			std::cout << "Duration: " << elapsed.count() << " ms" << std::endl;
		}

	}

	void Pipeline::startPipeline()
	{
		running_ = true;

		thread_ = std::thread(&Pipeline::run, this);
	}

	void Pipeline::stopPipeline()
	{
		running_ = false;

		if (thread_.joinable())
		{
			thread_.join();
		}
	}

	void Pipeline::readMailbox()
	{
		open3d::core::cuda::Synchronize(device_);

		std::vector<float>& buffer = mailbox_ptr_->consume();

		// std::cout<<"Consumer Array Size: "<<buffer.size()<<std::endl;

		open3d::core::Tensor cpu_tensor(
			static_cast<void*>(buffer.data()),
			open3d::core::Float32,
			shape_,
			strides_,
			open3d::core::Device("CPU:0")
		);

		pc_buffer_ = cpu_tensor.To(device_);

		// std::cout << "Buffer Shape: " << pc_buffer_.GetShape().ToString() << std::endl;
	}


	Pipeline::~Pipeline()
	{
		stopPipeline();

		pc_buffer_ = open3d::core::Tensor();

		partial_pcds_.clear();

		transforms_.clear();
	}


}