#pragma once

#include <rclcpp/rclcpp.hpp>
#include <memory>

#include "Streamer.hpp"
#include "Mailbox.hpp"
#include "Pointcloud.hpp"
#include "Pipeline.hpp"

#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/transform_broadcaster.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

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

			void publishPointclouds(const open3d::t::geometry::PointCloud& pcd, size_t num_points);

			void publishTransform(const Eigen::Matrix4d transform, size_t cam_id);

			void storeRobotBody(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

		protected:
		private:
			size_t batch_size_;

			size_t n_points_;

			size_t robot_filter_size_;

			std::unique_ptr<Mailbox<float>> mailbox_;

			std::unique_ptr<Mailbox<float>> robot_filter_mailbox_;

			std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

			void setupMailboxes(size_t batch_size, size_t n_points, size_t robot_body_size);

			void startThreads();

			void stopThreads();

			#ifdef SHM_DISABLE
				rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
			#else
				rclcpp::Publisher<percept_interfaces::msg::Pointcloud1M>::SharedPtr publisher_;
			#endif

			rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscriber_;
		};


} // namespace perception