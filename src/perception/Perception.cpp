#include "Perception.hpp"
#include "Streamer.hpp"
#include "Mailbox.hpp"
#include "Pointcloud.hpp"
// #include "Pipeline.hpp"

#include <thread>
#include <ament_index_cpp/get_package_share_directory.hpp>

// this program/node manages:
// 1. Producer/ingestor/CPU framesets
// 2. Pipeline that takes frameset and loads into GPU to do processes
// 3. Exposing output of 2. to ROS 2 topics for consumption
// 4. (future) interfacing with FK node for zero copy GPU pointclouds for robot filtering


namespace perception
{
  PerceptionNode::PerceptionNode() : rclcpp::Node ("perception_node")
  {
    // allocate memory
		batch_size_ = streamer.getBatchSize();

		n_points_ = streamer.getPCSize();

		pipeline.setupConfigs(
			batch_size_,
			n_points_,
			streamer.getCameraConfigs()
		);

		pipeline.setOwner(this);

		#ifdef SHM_DISABLE
			std::cout<<"SHM Disabled"<<std::endl;
			publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
				"pointcloud", 10
			);
		#else
			std::cout<<"SHM Enabled"<<std::endl;
			publisher_ = this->create_publisher<percept_interfaces::msg::Pointcloud1M>(
				"pointcloud", 10
			);
		#endif

		setupMailboxes(batch_size_, n_points_);

		startThreads();
  }

  // PerceptionNode::~PerceptionNode() = default;

	PerceptionNode::~PerceptionNode() {
    stopThreads();
	}

	void PerceptionNode::setupMailboxes(size_t batch_size, size_t n_points)
	{
		mailbox_ = std::make_unique<Mailbox<float>>(batch_size_, n_points_);

		std::cout<<"Mailbox setup with batch size="<<batch_size_<<" and "<<n_points_<<" points per batch element"<<std::endl;

		streamer.setMailbox(mailbox_.get());

		pipeline.setMailbox(mailbox_.get());
	}

	void PerceptionNode::startThreads()
	{
		streamer.startStreams();

		std::this_thread::sleep_for(std::chrono::seconds(1));

		pipeline.startPipeline();
	}


	void PerceptionNode::stopThreads()
	{
		streamer.stopStreams();		
		
		pipeline.stopPipeline();
	}

	void PerceptionNode::test(size_t i)
	{
		std::cout<<"test: "<<i<<std::endl;
	}

	// void PerceptionNode::publishPointclouds(const open3d::core::Tensor& cuda_points, size_t num_points)

	void PerceptionNode::publishPointclouds(const open3d::t::geometry::PointCloud& pcd, size_t num_points)
	{
		// open3d::core::Tensor cpu_points = cuda_points.To(open3d::core::Device("CPU:0"), /*copy=*/false);

		open3d::core::Tensor cpu_points = pcd.GetPointPositions().To(open3d::core::Device("CPU:0"));

		const float* src = cpu_points.GetDataPtr<float>();

		#ifdef SHM_DISABLE
			// std::cout << "points size" << num_points << std::endl;

			auto msg = sensor_msgs::msg::PointCloud2();

			msg.header.stamp = this->now();

			msg.header.frame_id = "panda_link0";

			sensor_msgs::PointCloud2Modifier modifier(msg);

			modifier.setPointCloud2FieldsByString(1, "xyz");

			modifier.resize(num_points);

			sensor_msgs::PointCloud2Iterator<float> iter_x(msg, "x");

			sensor_msgs::PointCloud2Iterator<float> iter_y(msg, "y");

			sensor_msgs::PointCloud2Iterator<float> iter_z(msg, "z");

			for (size_t i = 0; i < num_points; ++i)
			{
				*iter_x = src[i * 3];

				*iter_y = src[i * 3 + 1];

				*iter_z = src[i * 3 + 2];

				++iter_x; ++iter_y; ++iter_z;
			}

			publisher_->publish(std::move(msg));

		#else
			auto loaned_msg = publisher_->borrow_loaned_message();

			if (loaned_msg.is_valid()) {
				auto& msg = loaned_msg.get();

				msg.num_points = num_points;

				for (size_t i = 0; i < num_points; ++i)
				{
					msg.x[i] = src[i * 3];
					
					msg.y[i] = src[i * 3 + 1];
					
					msg.z[i] = src[i * 3 + 2];
				}

				publisher_->publish(std::move(loaned_msg));
			} 
			else 
			{
				RCLCPP_ERROR(this->get_logger(), "Failed to loan message memory");
			}
		#endif
	}
}




int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<perception::PerceptionNode>());
  rclcpp::shutdown();
  return 0;
}