#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include "Perception.hpp"
#include "Streamer.hpp"
#include "Mailbox.hpp"
#include "Pointcloud.hpp"
// #include "Pipeline.hpp"

#include <thread>

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

		mailbox_ = std::make_unique<Mailbox<float>>(batch_size_, n_points_);

		std::cout<<"Mailbox setup with batch size="<<batch_size_<<" and "<<n_points_<<" points per batch element"<<std::endl;

		streamer.setMailbox(mailbox_.get());

		pipeline.setMailbox(mailbox_.get());

		streamer.startStreams();

		std::this_thread::sleep_for(std::chrono::seconds(1));

		pipeline.readMailbox();

		// ...

  }

  // PerceptionNode::~PerceptionNode() = default;

	PerceptionNode::~PerceptionNode() {
    stopThreads();
	}


	void PerceptionNode::stopThreads()
	{
		streamer.stopStreams();		
	}
}




int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<perception::PerceptionNode>());
  rclcpp::shutdown();
  return 0;
}