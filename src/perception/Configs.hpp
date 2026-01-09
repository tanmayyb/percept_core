#pragma once

#include <string>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace perception{

	struct PoseConfig 
	{
    double x, y, z;

		double roll, pitch, yaw;
	};

	struct CameraConfig 
	{
		size_t id;

		std::string nickname;

		std::string serial_no;

		PoseConfig pose;

		Eigen::Matrix4d transform;
	};


	struct DepthProfile 
	{
		size_t width;

		size_t height;

		size_t fps;
	};

	struct TemporalFilter
	{
		bool is_enabled;

		float smooth_alpha;

		float smooth_delta;

		int holes_fill;
	};

	struct StreamConfig 
	{
		DepthProfile depth_profile;

		TemporalFilter temporal_filter_config;
	};

	struct PipelineConfig
	{
		struct SceneBound
		{
			std::vector<float> min;

			std::vector<float> max;
		} scene_bound;

		float robot_filter_radius;

		float voxel_size;
	};
}

namespace YAML{
	template<>
	struct convert<perception::PipelineConfig>
	{
		static bool decode(const Node& node, perception::PipelineConfig& rhs)
		{
			if (!node.IsMap()) return false;

			rhs.scene_bound.min = node["scene_bound"]["min"].as<std::vector<float>>();

			rhs.scene_bound.max = node["scene_bound"]["max"].as<std::vector<float>>();

			rhs.robot_filter_radius = node["robot_filter_radius"].as<float>();

			rhs.voxel_size = node["voxel_size"].as<float>();

			return true;
		}
	};

	template<>
	struct convert<perception::TemporalFilter>
	{
		static bool decode(const Node& node, perception::TemporalFilter& rhs)
		{
			if (!node.IsMap()) return false;

			rhs.is_enabled = node["enable"].as<bool>();

			rhs.smooth_alpha = node["alpha"].as<float>();

			rhs.smooth_delta = node["delta"].as<float>();

			rhs.holes_fill = node["hfill"].as<int>();

			return true;
		}
	};

	template<>
	struct convert<perception::DepthProfile>
	{
		static bool decode(const Node& node, perception::DepthProfile& rhs)
		{
			if (!node.IsMap()) return false;
			
			rhs.width = node["w"].as<size_t>();

			rhs.height = node["h"].as<size_t>();

			rhs.fps = node["f"].as<size_t>();
			
			return true;
		}
	};

	template<>
	struct convert<perception::StreamConfig>
	{
		static bool decode(const Node& node, perception::StreamConfig& rhs)
		{
			if (!node.IsMap()) return false;

			rhs.depth_profile = node["depth_profile"].as<perception::DepthProfile>();

			rhs.temporal_filter_config = node["temporal_filter"].as<perception::TemporalFilter>();

			return true;
		}
	};

	template<>
	struct convert<perception::PoseConfig>
	{
		static bool decode(const Node& node, perception::PoseConfig& rhs)
		{
			if(!node["position"] || !node["orientation"]) return false;

			rhs.x = node["position"]["x"].as<double>();

			rhs.y = node["position"]["y"].as<double>();

			rhs.z = node["position"]["z"].as<double>();

			rhs.roll = node["orientation"]["roll"].as<double>();
			
			rhs.pitch = node["orientation"]["pitch"].as<double>();

			rhs.yaw = node["orientation"]["yaw"].as<double>();

			return true;
		}
	};

	template<>
	struct convert<perception::CameraConfig>
	{
		static bool decode(const Node& node, perception::CameraConfig& rhs)
		{
			if (!node["nickname"] || !node["serial_no"] || !node["pose"]) return false;
			
			rhs.nickname = node["nickname"].as<std::string>();

			rhs.serial_no = node["serial_no"].as<std::string>();

			rhs.pose = node["pose"].as<perception::PoseConfig>();

			Eigen::AngleAxisd rollAngle(rhs.pose.roll, Eigen::Vector3d::UnitX());

			Eigen::AngleAxisd pitchAngle(rhs.pose.pitch, Eigen::Vector3d::UnitY());

			Eigen::AngleAxisd yawAngle(rhs.pose.yaw, Eigen::Vector3d::UnitZ());

			Eigen::Quaterniond q = yawAngle * pitchAngle * rollAngle;

			Eigen::Translation3d translation(rhs.pose.x, rhs.pose.y, rhs.pose.z);

			rhs.transform = (translation * q).matrix();

			return true;
		}
	};

}