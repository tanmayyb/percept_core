#pragma once

#include <iostream>
#include <open3d/Open3D.h>
#include <open3d/core/Device.h>
#include <open3d/core/CUDAUtils.h> /// Required for specific CUDA availability checks

#include <atomic>
#include <thread>

#include "Mailbox.hpp"
#include "Configs.hpp"

namespace perception
{
	class Pipeline
	{
		public:
			Pipeline();
	
			virtual ~Pipeline();

		protected:
		private:
			open3d::core::Device device_;
			
			std::atomic_bool running_{false};

			Mailbox<float>* mailbox_ptr_ = nullptr;			

			std::thread thread_;

			open3d::core::Tensor pc_buffer_;

			// std::vector<open3d::core::Tensor> pc_;

			size_t batch_size_;

			size_t n_points_;

			std::vector<open3d::core::Tensor> transforms_;

		public:
			void setupConfigs(size_t batch_size, size_t n_points, const std::vector<CameraConfig>& cameras);
	
			void setMailbox(Mailbox<float>* mb){
				mailbox_ptr_ = mb;
			}

			void readMailbox();


			void run();
	};

}