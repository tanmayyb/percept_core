#include "PointsFK.hpp"


PointCloudFKPublisher::PointCloudFKPublisher() : rclcpp::Node ("perception_node")
{
	tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
	
	tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

	publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("robot_filter", 10);

	link_names_ = {
		"panda_link0", "panda_link1", "panda_link2", "panda_link3", 
		"panda_link4", "panda_link5", "panda_link6", "panda_link7", 
		"panda_hand", "panda_leftfinger", "panda_rightfinger"
	};

	pkg_share_dir_ = ament_index_cpp::get_package_share_directory("percept_core");

	file_path_ = pkg_share_dir_ + "/assets/sphere_database-37.yaml";

	loadSpheresFromYaml(file_path_);

	timer_ = this->create_wall_timer(
		std::chrono::milliseconds(50), 
		std::bind(&PointCloudFKPublisher::update_and_publish, this)
	);
}
	

void PointCloudFKPublisher::loadSpheresFromYaml(const std::string& file_path)
{
	YAML::Node config = YAML::LoadFile(file_path);

	for(YAML::const_iterator it = config.begin(); it != config.end(); ++it)
	{
		const std::string full_key = it->first.as<std::string>();

		size_t delimiter_pos = full_key.find("::");

		if (delimiter_pos == std::string::npos) continue;

		std::string link_name = full_key.substr(0, delimiter_pos);

		std::vector<Eigen::Vector3d> points;

		const YAML::Node& mesh_node = it->second;

		if (mesh_node["8"] && mesh_node["8"]["0"] && mesh_node["8"]["0"]["spheres"])
		{
			const YAML::Node& spheres = mesh_node["8"]["0"]["spheres"];

			for (const auto& sphere : spheres)
			{
				if (sphere["origin"] && sphere["origin"].IsSequence() && sphere["origin"].size() == 3)
				{
					const YAML::Node& origin = sphere["origin"];

					points.emplace_back(
						origin[0].as<double>(),
						origin[1].as<double>(),
						origin[2].as<double>()
					);
				}
			}

			if (!points.empty())
			{
				local_pointsets_[link_name] = std::move(points);
			}
		}
	}
}

void PointCloudFKPublisher::update_and_publish()
{
	auto cloud_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();

	cloud_msg->header.stamp = this->get_clock()->now();

	cloud_msg->header.frame_id = "panda_link0";

	size_t total_points = 0;

	for (const auto& pair : local_pointsets_) total_points += pair.second.size();

	sensor_msgs::PointCloud2Modifier modifier(*cloud_msg);
	
	modifier.setPointCloud2FieldsByString(1, "xyz");

	modifier.resize(total_points);

	sensor_msgs::PointCloud2Iterator<float> iter_x(*cloud_msg, "x");

	sensor_msgs::PointCloud2Iterator<float> iter_y(*cloud_msg, "y");

	sensor_msgs::PointCloud2Iterator<float> iter_z(*cloud_msg, "z");

	for (const auto& link : link_names_)
	{
		try
		{
			if (!tf_buffer_->canTransform("panda_link0", link, tf2::TimePointZero))
			{
				for(size_t i = 0; i< local_pointsets_[link].size(); ++i)
				{
					++iter_x; ++iter_y; ++iter_z;
				}
				
				continue;
			}

			geometry_msgs::msg::TransformStamped tf_stamped = tf_buffer_->lookupTransform(
				"panda_link0", link, tf2::TimePointZero
			);

			Eigen::Isometry3d T = tf2::transformToEigen(tf_stamped);

			for (const auto& p_link : local_pointsets_[link])
			{
				Eigen::Vector3d p_world = T * p_link;

				*iter_x = static_cast<float>(p_world.x());

				*iter_y = static_cast<float>(p_world.y());

				*iter_z = static_cast<float>(p_world.z());

				++iter_x; ++iter_y; ++iter_z;
			}
		}
		catch(const tf2::TransformException& ex)
		{
			for(size_t i = 0; i < local_pointsets_[link].size(); ++i)
			{
				++iter_x; ++iter_y; ++iter_z;
			}

			RCLCPP_WARN(this->get_logger(), "TF failure: %s", ex.what());				
		}
	}

	publisher_->publish(*cloud_msg);
}

int main(int argc, char** argv)
{
	rclcpp::init(argc, argv);

	rclcpp::spin(std::make_shared<PointCloudFKPublisher>());

	rclcpp::shutdown();

	return 0;
}

// class PointCloudFKPublisher : public rclcpp::Node
// {
// 	public:
// 		PointCloudFKPublisher() : Node("pointcloud_fk_publisher")
// 		{
// 			tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
			
// 			tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

// 			publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("transformed_cloud", 10);

// 			link_names_ = {
// 				"panda_link0", "panda_link1", "panda_link2", "panda_link3", 
// 				"panda_link4", "panda_link5", "panda_link6", "panda_link7", 
// 				"panda_hand", "panda_leftfinger", "panda_rightfinger"
// 			};

// 			pkg_share_dir_ = ament_index_cpp::get_package_share_directory("percept_core");

// 			file_path_ = pkg_share_dir_ + "/assets/sphere_database-11.yaml";

// 			timer_ = this->create_wall_timer(
// 				std::chrono::milliseconds(50), 
// 				std::bind(&PointCloudFKPublisher::update_and_publish, this)
// 			);
// 		}
	
// 	private:

// 		void loadSpheresFromYaml(const std::string& file_path)
// 		{
// 			YAML::Node config = YAML::LoadFile(file_path);

// 			for(YAML::const_iterator it = config.begin(); it != config.end(); ++it)
// 			{
// 				const std::string full_key = it->first.as<std::string>();

// 				size_t delimiter_pos = full_key.find("::");

// 				if (delimiter_pos == std::string::npos) continue;

// 				std::string link_name = full_key.substr(0, delimiter_pos);

// 				std::vector<Eigen::Vector3d> points;

// 				const YAML::Node& mesh_node = it->second;

// 				if (mesh_node["8"] && mesh_node["8"]["0"] && mesh_node["8"]["0"]["spheres"])
// 				{
// 					const YAML::Node& spheres = mesh_node["8"]["0"]["spheres"];

// 					for (const auto& sphere : spheres)
// 					{
// 						if (sphere["origin"] && sphere["origin"].IsSequence() && sphere["origin"].size() == 3)
// 						{
// 							const YAML::Node& origin = sphere["origin"];

// 							points.emplace_back(
// 								origin[0].as<double>(),
// 								origin[1].as<double>(),
// 								origin[2].as<double>()
// 							);
// 						}
// 					}

// 					if (!points.empty())
// 					{
// 						local_pointsets_[link_name] = std::move(points);
// 					}
// 				}
// 			}
// 		}

// 		void update_and_publish()
// 		{
// 			auto cloud_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();

// 			cloud_msg->header.stamp = this->get_clock()->now();

// 			cloud_msg->header.frame_id = "panda_link0";

// 			size_t total_points = 0;

// 			for (const auto& pair : local_pointsets_) total_points += pair.second.size();

// 			sensor_msgs::PointCloud2Modifier modifier(*cloud_msg);
			
// 			modifier.setPointCloud2FieldsByString(1, "xyz");

// 			modifier.resize(total_points);

// 			sensor_msgs::PointCloud2Iterator<float> iter_x(*cloud_msg, "x");

// 			sensor_msgs::PointCloud2Iterator<float> iter_y(*cloud_msg, "y");

// 			sensor_msgs::PointCloud2Iterator<float> iter_z(*cloud_msg, "z");

// 			for (const auto& link : link_names_)
// 			{
// 				try
// 				{
// 					if (!tf_buffer_->canTransform("panda_link0", link, tf2::TimePointZero))
// 					{
// 						for(size_t i = 0; i< local_pointsets_[link].size(); ++i)
// 						{
// 							++iter_x; ++iter_y; ++iter_z;
// 						}
						
// 						continue;
// 					}

// 					geometry_msgs::msg::TransformStamped tf_stamped = tf_buffer_->lookupTransform(
// 						"panda_link0", link, tf2::TimePointZero
// 					);

// 					Eigen::Isometry3d T = tf2::transformToEigen(tf_stamped);

// 					for (const auto& p_link : local_pointsets_[link])
// 					{
// 						Eigen::Vector3d p_world = T * p_link;

// 						*iter_x = static_cast<float>(p_world.x());

// 						*iter_y = static_cast<float>(p_world.y());

// 						*iter_z = static_cast<float>(p_world.z());

// 						++iter_x; ++iter_y; ++iter_z;
// 					}
// 				}
// 				catch(const tf2::TransformException& ex)
// 				{
// 					for(size_t i = 0; i < local_pointsets_[link].size(); ++i)
// 					{
// 						++iter_x; ++iter_y; ++iter_z;
// 					}

// 					RCLCPP_WARN(this->get_logger(), "TF failure: %s", ex.what());				
// 				}
// 			}

// 			publisher_->publish(*cloud_msg);
// 		}

// 		std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
		
// 		std::unique_ptr<tf2_ros::Buffer> tf_buffer_;

// 		rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;

// 		rclcpp::TimerBase::SharedPtr timer_;

// 		std::vector<std::string> link_names_;

// 		std::map<std::string, std::vector<Eigen::Vector3d>> local_pointsets_;
			
// };

