#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/rclcpp.hpp>
#include <memory>

#include "Streamer.hpp"
#include "Mailbox.hpp"
#include "Pointcloud.hpp"
#include "Pipeline.hpp"


#ifdef SHM_DISABLE
	#include "sensor_msgs/msg/point_cloud2.hpp"
	#include "sensor_msgs/point_cloud2_iterator.hpp"
#else
	// #include "percept_interfaces/msg/pointcloud.hpp"
	#include "percept_interfaces/msg/pointcloud1_m.hpp"
#endif

namespace perception
{

	class PerceptionNode: public rclcpp::Node
	{
		public:
			PerceptionNode();
			virtual ~PerceptionNode();

			Streamer streamer;

			Pipeline pipeline;

			void test(size_t i);

			// void publishPointclouds(const open3d::core::Tensor& cuda_points, size_t num_points);
			void publishPointclouds(const open3d::t::geometry::PointCloud& pcd, size_t num_points);

		protected:
		private:
			size_t batch_size_;

			size_t n_points_;

			std::unique_ptr<Mailbox<float>> mailbox_;

			void setupMailboxes(size_t batch_size, size_t n_points);

			void startThreads();

			void stopThreads();

			#ifdef SHM_DISABLE
				rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
			#else
				rclcpp::Publisher<percept_interfaces::msg::Pointcloud1M>::SharedPtr publisher_;
			#endif

		};


} // namespace perception