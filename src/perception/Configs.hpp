#pragma once

#include <string>
#include <yaml-cpp/yaml.h>

namespace perception{

	struct PoseConfig {
    double x, y, z;

		double roll, pitch, yaw;
	};

	struct CameraConfig {
		size_t id;

		std::string nickname;

		std::string serial_no;

		PoseConfig pose;
	};

	struct RealsenseConfig {

	};

}

namespace YAML{
	template<>
	struct convert<perception::PoseConfig>{
		static bool decode(const Node& node, perception::PoseConfig& rhs){
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
	struct convert<perception::CameraConfig>{
		static bool decode(const Node& node, perception::CameraConfig& rhs){
			if (!node["nickname"] || !node["serial_no"] || !node["pose"]) return false;
			
			rhs.nickname = node["nickname"].as<std::string>();

			rhs.serial_no = node["serial_no"].as<std::string>();

			rhs.pose = node["pose"].as<perception::PoseConfig>();
			
			return true;
		}
	};

}