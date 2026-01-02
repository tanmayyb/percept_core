#pragma once

#include <librealsense2/rs.hpp>
#include <librealsense2/rs_advanced_mode.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <yaml-cpp/yaml.h>

#include <atomic>
#include <thread>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>


#include "Configs.hpp"
#include "Mailbox.hpp"

namespace perception
{

	class filter_options
	{
	public:
		filter_options(const std::string name, rs2::filter& filter);

		filter_options(filter_options&& other);

		std::string filter_name;                                   

		rs2::filter& filter;                                       

		std::atomic_bool is_enabled;                               
	};

	class Streamer
	{
		public:
			Streamer();

			virtual ~Streamer();

		protected:
		private:
			std::string pkg_share_dir_;

			rs2::context ctx_;

			std::vector<std::string> serials_;

			std::vector<rs2::pipeline>	pipelines_;

			std::atomic_bool running_{false};

			size_t batch_size_ = 0;

			size_t n_points_ = 0;

			Mailbox<float>* mailbox_ptr_ = nullptr;

			std::thread thread_;

			// void dumpSoAtoCSV(const std::vector<float>& soa_array, 
      //              size_t n_pipes, 
      //              size_t n_points, 
      //              const std::string& filename);

			void setupFilters();


			//  rs2 filters
			rs2::disparity_transform depth_to_disparity_{true};

			rs2::temporal_filter temp_filter_;

			rs2::disparity_transform disparity_to_depth_{false};
			
		public:
			std::vector<CameraConfig> cameras;

	    std::vector<filter_options> filters;

			void loadConfigs();

			// std::vector<RealsenseConfig> rs_settings;

			// void setup_static_camera_configs();

			// void setup_rs_settings_configs();

			// void setup_perception_pipeline_configs();

			void setupPipelines();

			size_t getBatchSize(){
				return batch_size_;
			}

			size_t getPCSize(){
				return n_points_;
			}

			void setMailbox(Mailbox<float>* mb){
				mailbox_ptr_ = mb;
			}

			void startStreams();

			void stopStreams();

			void run();

			const std::vector<CameraConfig>& getCameraConfigs() const {
				return cameras;
			}
	};

}