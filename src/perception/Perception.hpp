#pragma once

#include <rclcpp/rclcpp.hpp>
#include <memory>

#include "Streamer.hpp"
#include "Mailbox.hpp"
#include "Pointcloud.hpp"
#include "Pipeline.hpp"


namespace perception
{

	class PerceptionNode: public rclcpp::Node
	{
		public:
			PerceptionNode();
			virtual ~PerceptionNode();

			Streamer streamer;

			Pipeline pipeline;

			void stopThreads();

		protected:
		private:
			size_t batch_size_;

			size_t n_points_;

			std::unique_ptr<Mailbox<float>> mailbox_;
		};


} // namespace perception