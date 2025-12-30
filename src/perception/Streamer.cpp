#include <librealsense2/rs.hpp>
#include <librealsense2/rs_advanced_mode.hpp>
#include "Streamer.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <iostream>
#include <iomanip>

#include <fstream>

namespace perception
{
	Streamer::Streamer()
	{
		pkg_share_dir_ = ament_index_cpp::get_package_share_directory("percept");

		loadConfigs();
		
		setupPipelines();

		// run();
	}

	Streamer::~Streamer() = default;

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
	
	}

	// void Streamer::startStreams()
	// {

	// 	rs2::pointcloud pc;
	// 	auto pipe = pipelines_.at(0);
	// 	rs2::frameset frames = pipe.wait_for_frames();
	// 	rs2::depth_frame depth = frames.get_depth_frame();

	// 	// 2. Generate points
	// 	rs2::points points = pc.calculate(depth);
	// 	const rs2::vertex* vertices = points.get_vertices();
	// 	const size_t n = points.size();

	// 	std::cout<<"received "<<n<<" points"<<std::endl;
		
	// 	// 3. Allocate 1D SoA Array
	// 	std::vector<float> soa_array(3 * n * 2);

	// 	// 4. Map AoS vertices to SoA format
	// 	for (size_t i = 0; i < n; ++i) {
	// 			soa_array[i]         = vertices[i].x; // X block
	// 			soa_array[i + n]     = vertices[i].y; // Y block
	// 			soa_array[i + 2 * n] = vertices[i].z; // Z block
	// 	}

	// 	mailbox_ptr_->produce(soa_array);	
	// }


	void Streamer::startStreams()
	{
		size_t n_pipes = static_cast<int>(pipelines_.size());
		
		size_t frame_size = 3*n_points_;

		std::vector<PointCloudData> results(n_pipes);

		std::vector<float> soa_array(frame_size*n_pipes); // XYZ*Points*cameras

		// get frames in parallel
		#pragma omp parallel for
		for (int i=0; i<n_pipes; ++i)
		{
			rs2::pointcloud pc;
			
			auto pipe = pipelines_.at(i);

			rs2::frameset frames = pipe.wait_for_frames();

			rs2::depth_frame depth = frames.get_depth_frame();

			rs2::points points  = pc.calculate(depth);

			const rs2::vertex* ptr = points.get_vertices();

			const size_t n = points.size();

			results[i].vertices.assign(ptr, ptr+n);
		}

		for(size_t i=0;i<n_points_;i++)
		{
			#pragma omp parallel for
			for(int j=0;j<n_pipes;j++)
			{
				soa_array[i + j * frame_size]         				= results[j].vertices[i].x; // X block

				soa_array[i + n_points_ + j * frame_size]     = results[j].vertices[i].y; // Y block

				soa_array[i + 2 * n_points_ + j * frame_size] = results[j].vertices[i].z; // Z block
			}
		}

		mailbox_ptr_->produce(soa_array);	

		// dumpSoAtoCSV(soa_array, n_pipes, n_points_, "pointcloud_dump.csv");
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