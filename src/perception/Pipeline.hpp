#pragma once

#include <iostream>
#include <atomic>
#include <thread>
#include <memory>
#include <vector>

#include <open3d/Open3D.h>
#include <open3d/core/Device.h>
#include <open3d/core/CUDAUtils.h> /// Required for specific CUDA availability checks

#include "Configs.hpp"
#include "Mailbox.hpp"

namespace perception
{

	class PerceptionNode;

	class Pipeline
	{
		public:
			Pipeline();
	
			virtual ~Pipeline();

		protected:
		private:
			PerceptionNode* owner_ = nullptr;
			
			open3d::core::Device device_;
			
			open3d::core::Tensor pc_buffer_;

			std::vector<open3d::core::Tensor> transforms_;

			std::vector<open3d::t::geometry::PointCloud> partial_pcds_;

			std::unique_ptr<open3d::t::geometry::PointCloud> merged_pcd_;

			open3d::core::Tensor min_bound_;
			
			open3d::core::Tensor max_bound_;
			
			open3d::t::geometry::AxisAlignedBoundingBox bbox_;

			std::atomic_bool running_{false};

			size_t batch_size_ = 0;

			size_t n_points_ = 0;

			open3d::core::SizeVector strides_ = {3, 1};

			open3d::core::SizeVector shape_;

			Mailbox<float>* mailbox_ptr_ = nullptr;			

			std::thread thread_;

		public:
			void setupConfigs(size_t batch_size, size_t n_points, const std::vector<CameraConfig>& cameras);

			void setOwner(PerceptionNode* node){
				owner_ = node;
			}
			
			void setMailbox(Mailbox<float>* mb){
				mailbox_ptr_ = mb;
			}

			void readMailbox();

			void startPipeline();

			void stopPipeline();

			void run();
	};

}