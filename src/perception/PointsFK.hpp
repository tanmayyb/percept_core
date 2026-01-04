#ifndef POINT_CLOUD_FK_PUBLISHER_HPP_
#define POINT_CLOUD_FK_PUBLISHER_HPP_

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
// #include "tf2_ros/transform_listener.h"
// #include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.hpp"
#include "tf2_ros/buffer.hpp"
#include "tf2_eigen/tf2_eigen.hpp"
#include <Eigen/Geometry>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <yaml-cpp/yaml.h>

class PointCloudFKPublisher : public rclcpp::Node
{
	private:
		void loadSpheresFromYaml(const std::string& file_path);

		void update_and_publish();

		std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

		std::unique_ptr<tf2_ros::Buffer> tf_buffer_;

		rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;

		rclcpp::TimerBase::SharedPtr timer_;

		std::string pkg_share_dir_;

		std::string file_path_;

		std::vector<std::string> link_names_;

		std::map<std::string, std::vector<Eigen::Vector3d>> local_pointsets_;

	public:
		PointCloudFKPublisher();

};

#endif  // POINT_CLOUD_FK_PUBLISHER_HPP_