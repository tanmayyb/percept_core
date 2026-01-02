#include "Streamer.hpp"
#include "Pointcloud.hpp"

namespace perception
{

	filter_options::filter_options(const std::string name, rs2::filter& flt) : filter_name(name), filter(flt), is_enabled(true){}

	filter_options::filter_options(filter_options&& other) : filter_name(std::move(other.filter_name)), filter(other.filter), is_enabled(other.is_enabled.load()){}

	Streamer::Streamer()
	{
		pkg_share_dir_ = ament_index_cpp::get_package_share_directory("percept");

		loadConfigs();
		
		setupPipelines();
	}

	void Streamer::loadConfigs()
	{
		// load camera configs
		YAML::Node root = YAML::LoadFile(pkg_share_dir_ + "/config/static_cameras_setup.yaml");

		for (auto it = root.begin(); it !=root.end(); ++it)
		{
			perception::CameraConfig camera = it->second.as<perception::CameraConfig>();

			camera.id = it->first.as<size_t>();

			cameras.push_back(camera);
		}

    root = YAML::LoadFile(pkg_share_dir_ + "/config/rs_settings.yaml");

		
	}

	void Streamer::setupFilters()
	{
  	temp_filter_.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.0f);

		temp_filter_.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, 100.0f);

		temp_filter_.set_option(RS2_OPTION_HOLES_FILL, 8);

		filters.emplace_back("Depth2Disparity", depth_to_disparity_);

		filters.emplace_back("Temporal", temp_filter_);

		filters.emplace_back("Disparity2Depth", disparity_to_depth_);
	}


	void Streamer::setupPipelines()
	{
		for(const auto& cam:cameras)
		{
			rs2::pipeline pipe;

			rs2::config cfg;

			cfg.enable_device(cam.serial_no);

			cfg.enable_stream(RS2_STREAM_DEPTH, 848, 480, RS2_FORMAT_Z16, 30);

			pipe.start(cfg);

			pipelines_.emplace_back(pipe);

			std::cout<<"pipeline "<<cam.serial_no<<" started"<<std::endl;

		}

		batch_size_ = pipelines_.size();

		n_points_ = 848*480;

		setupFilters();
	
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
				
				for (auto&& filter : filters)
				{
					depth_frame = filter.filter.process(depth_frame);
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