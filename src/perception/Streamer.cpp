#include "Streamer.hpp"
#include "Perception.hpp"
#include "Pointcloud.hpp"

namespace perception
{

	Streamer::Streamer()
	{
		pkg_share_dir_ = ament_index_cpp::get_package_share_directory("percept");

		loadConfigs();
	}

	void Streamer::loadConfigs()
	{
		// load stream and camera configs
		YAML::Node root = YAML::LoadFile(pkg_share_dir_ + "/config/static_cameras_setup.yaml");

		// load stream configs
		stream_config = root["stream"].as<perception::StreamConfig>();

		// load camera configs
		auto sensors_node = root["sensors"];

		for (auto it = sensors_node.begin(); it != sensors_node.end(); ++it)
		{
			perception::CameraConfig camera = it->second.as<perception::CameraConfig>();

			camera.id = it->first.as<size_t>();

			cameras.push_back(camera);
		}
	}

	void Streamer::setupPipelines()
	{
		for(const auto& cam:cameras)
		{
			// setup pipe
			rs2::pipeline pipe;

			rs2::config cfg;

			cfg.enable_device(cam.serial_no);

			cfg.enable_stream(
				RS2_STREAM_DEPTH, 
				stream_config.depth_profile.width, 
				stream_config.depth_profile.height, 
				RS2_FORMAT_Z16, 
				stream_config.depth_profile.fps
			);

			pipe.start(cfg);

			pipelines_.emplace_back(pipe);

			std::cout<<"pipeline "<<cam.serial_no<<" started"<<std::endl;

			// setup filters
			std::vector<filter_options> cam_filters;

			auto depth2disparity = std::make_shared<rs2::disparity_transform>(true);

			cam_filters.emplace_back("Depth2Disparity", std::move(depth2disparity));

			if (stream_config.spatial_filter_config.is_enabled)
			{
				auto spatial = std::make_shared<rs2::spatial_filter>();

				spatial->set_option(
					RS2_OPTION_FILTER_SMOOTH_ALPHA, stream_config.spatial_filter_config.smooth_alpha
				);

				spatial->set_option(
					RS2_OPTION_FILTER_SMOOTH_DELTA, stream_config.spatial_filter_config.smooth_delta
				);

				spatial->set_option(
					RS2_OPTION_HOLES_FILL, stream_config.spatial_filter_config.hole_fill
				);

				spatial->set_option(
					RS2_OPTION_FILTER_MAGNITUDE, stream_config.spatial_filter_config.magnitude
				);

				cam_filters.emplace_back("Spatial", std::move(spatial));
			}

			if (stream_config.temporal_filter_config.is_enabled)
			{
				auto temporal = std::make_shared<rs2::temporal_filter>();

				temporal->set_option(
					RS2_OPTION_FILTER_SMOOTH_ALPHA, stream_config.temporal_filter_config.smooth_alpha
				);
				
				temporal->set_option(
					RS2_OPTION_FILTER_SMOOTH_DELTA, stream_config.temporal_filter_config.smooth_delta
				);
				
				temporal->set_option(
					RS2_OPTION_HOLES_FILL, stream_config.temporal_filter_config.persistence_control
				);

				cam_filters.emplace_back("Temporal", std::move(temporal));
			}
			
			auto disparity2depth = std::make_shared<rs2::disparity_transform>(false);	

			cam_filters.emplace_back("Disparity2Depth", std::move(disparity2depth));

			pipeline_filters_.push_back(std::move(cam_filters));

			// publish camera trasforms
			owner_->publishTransform(cam.transform, cam.id);
		}

		batch_size_ = pipelines_.size();

		n_points_ = stream_config.depth_profile.width*stream_config.depth_profile.height;
	
	}


	void Streamer::run()
	{
		size_t n_pipes = static_cast<int>(pipelines_.size());
		
		size_t frame_size = 3*n_points_;

		std::vector<PointCloudData> results(n_pipes);

		auto start = std::chrono::high_resolution_clock::now();

		auto end = std::chrono::high_resolution_clock::now();
	
		std::chrono::duration<double, std::milli> elapsed;

		while(running_.load())
		{
			start = std::chrono::high_resolution_clock::now();
			// get frames in parallel
			#pragma omp parallel for
			for (int i=0; i<n_pipes; ++i)
			{
				rs2::pointcloud pc;
				
				auto pipe = pipelines_.at(i);

				rs2::frameset frames = pipe.wait_for_frames();

				rs2::frame depth_frame = frames.get_depth_frame();
				
				for (auto& filter_opt : pipeline_filters_[i])
				{
					depth_frame = filter_opt.filter->process(depth_frame);
				}

				rs2::points points  = pc.calculate(depth_frame);

				const rs2::vertex* ptr = points.get_vertices();

				const size_t n = points.size();

				results[i].vertices.assign(ptr, ptr+n);
			}

			std::vector<float>& target_buffer = mailbox_ptr_->get_producer_buffer();

			#pragma omp parallel for
			for (int j = 0; j < n_pipes; j++) 
			{
				const size_t offset = j * frame_size;

				for (size_t i = 0; i < n_points_; i++)
				{
					const size_t base = i * 3 + offset;

					target_buffer[base]     = results[j].vertices[i].x;

					target_buffer[base + 1] = results[j].vertices[i].y;

					target_buffer[base + 2] = results[j].vertices[i].z;
				}
			}

			mailbox_ptr_->commit();


			// dumpSoAtoCSV(soa_array, n_pipes, n_points_, "pointcloud_dump.csv");

			// running_ = false; // added for debugging

			end = std::chrono::high_resolution_clock::now();
    
			elapsed = end - start;
			
			// std::cout << "Duration: " << elapsed.count() << " ms" << std::endl;
		}

	}

	// Streamer::~Streamer() = default;

	Streamer::~Streamer() {
    stopStreams();
	}

	void Streamer::startStreams()
	{
		running_ = true;

		thread_ = std::thread(&Streamer::run, this);
	}

	void Streamer::stopStreams()
	{
		running_ = false;

		if (thread_.joinable())
		{
			thread_.join();
		}
	}



	
	// void Streamer::dumpSoAtoCSV(const std::vector<float>& soa_array, 
	// 									size_t n_pipes, 
	// 									size_t n_points, 
	// 									const std::string& filename)
	// {
	// 		std::ofstream outfile(filename);
	// 		size_t frame_size = 3 * n_points;

	// 		// Headers
	// 		outfile << "Camera,Point_Index,X,Y,Z\n";

	// 		for (size_t j = 0; j < n_pipes; ++j) {
	// 				for (size_t i = 0; i < n_points; ++i) {
	// 						outfile << j << "," 
	// 										<< i << ","
	// 										<< soa_array[i + j * frame_size] << ","
	// 										<< soa_array[i + n_points + j * frame_size] << ","
	// 										<< soa_array[i + 2 * n_points + j * frame_size] << "\n";
	// 				}
	// 		}

	// 		outfile.close();
	// }


}