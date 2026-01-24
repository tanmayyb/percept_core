#include "VFEngine.hpp"


FieldsComputer::FieldsComputer() : Node("vf_engine")
{
  setupDevice();

  setupParamsAndServices();

  // Queue Processor
  queue_processor_ = std::thread(&FieldsComputer::process_queue, this);

  // Pointcloud and NNS arrays
  max_points_ = 1000000;

  hash_table_size_ = 2000003; // prime_number > 2* max_points_

  grid_config_.cell_size = 0.01; // 1 cm

  grid_config_.min_boundary = make_double3(-1000.0, -1000.0, -1000.0); // 1km x 1km x 1km

  bool success = true;

  success &= check_cuda_error(cudaMalloc(&d_x_ptr, max_points_ * sizeof(double)), "Alloc X");

  success &= check_cuda_error(cudaMalloc(&d_y_ptr, max_points_ * sizeof(double)), "Alloc Y");

  success &= check_cuda_error(cudaMalloc(&d_z_ptr, max_points_ * sizeof(double)), "Alloc Z");

  success &= check_cuda_error(cudaMalloc(&d_hashes_ptr, max_points_ * sizeof(uint32_t)), "Alloc Hashes");

  success &= check_cuda_error(cudaMalloc(&d_indices_ptr, max_points_ * sizeof(uint32_t)), "Alloc Indices");

  success &= check_cuda_error(cudaMalloc(&d_nn_ptr, max_points_ * sizeof(int)), "Alloc NN");

  success &= check_cuda_error(cudaMalloc(&d_starts_ptr, hash_table_size_ * sizeof(uint32_t)), "Alloc Starts");

  success &= check_cuda_error(cudaMalloc(&d_ends_ptr, hash_table_size_ * sizeof(uint32_t)), "Alloc Ends");

  if (!success) throw std::runtime_error("Failed to allocate GPU memory");
  
  subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    "/pointcloud", 10, std::bind(&FieldsComputer::pointcloud_callback, this, std::placeholders::_1));

  // Profiling
  #ifdef GPU_PROFILING_ENABLE
    RCLCPP_INFO(this->get_logger(), "GPU PROFILING ENABLED");
  #endif
}


FieldsComputer::~FieldsComputer()
{
  stop_queue();

  if (queue_processor_.joinable()) queue_processor_.join();

  std::unique_lock<std::shared_timed_mutex> lock(gpu_points_mutex_);

  gpu_x_shared_.reset();

  gpu_y_shared_.reset();

  gpu_z_shared_.reset();

  gpu_nn_indices_shared_.reset();

  cudaStreamDestroy(compute_stream_);
}


template<typename HeuristicFunc>
void FieldsComputer::handle_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response,
    HeuristicFunc kernel_launcher, 
    const std::string& name)
{
  enqueue_operation(OperationType::READ, [this, request, response, kernel_launcher]() 
  {
    std::shared_lock<std::shared_timed_mutex> lock(gpu_points_mutex_);
  
    if (!gpu_x_shared_) { response->not_null = false; return; }

    auto [pos, vel, goal, agent_radius, shell, k, max_f] = extract_request_data(request);

  
    double3 res;

    if constexpr (
      std::is_invocable_v<HeuristicFunc, 
                          double*, double*, double*, 
                          size_t, uint32_t*, 
                          double3, double3, double3, double, double, 
                          double, double, double, bool>)
    {
      res = kernel_launcher(
        gpu_x_shared_.get(), gpu_y_shared_.get(), gpu_z_shared_.get(), 
        gpu_num_points_, gpu_nn_indices_shared_.get(), 
        pos, vel, goal, agent_radius, point_radius, 
        shell, k, max_f, show_processing_delay
      );
    }
    else
    {
      res = kernel_launcher(
        gpu_x_shared_.get(), gpu_y_shared_.get(), gpu_z_shared_.get(), 
        gpu_num_points_, pos, vel, goal, agent_radius, point_radius, 
        shell, k, max_f, show_processing_delay
      );
    }


    process_response(res, request->agent_pose, response);
  });
}


void FieldsComputer::handle_min_obstacle_distance(
    const std::shared_ptr<percept_interfaces::srv::AgentPoseToMinObstacleDist::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentPoseToMinObstacleDist::Response> response)
{
  enqueue_operation(OperationType::READ, [this, request, response]() 
  {
    std::shared_lock<std::shared_timed_mutex> lock(gpu_points_mutex_);
    if (!gpu_x_shared_) { response->distance = 0.0; return; }

    double3 agent_pos = make_double3(
      request->agent_pose.position.x, 
      request->agent_pose.position.y, 
      request->agent_pose.position.z
    );

    response->distance = min_obstacle_distance_kernel(
      gpu_x_shared_.get(), gpu_y_shared_.get(), gpu_z_shared_.get(), 
      gpu_num_points_, agent_pos, show_processing_delay);
  });
}


void FieldsComputer::pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  auto msg_copy = std::make_shared<sensor_msgs::msg::PointCloud2>(*msg);

  enqueue_operation(OperationType::WRITE, [this, msg_copy]() 
  {
    size_t total_points = msg_copy->width * msg_copy->height;

    size_t n = std::min(total_points, max_points_);

    if (n<=0)
    {
      RCLCPP_ERROR(this->get_logger(), "total_points: %d, max_points: %d, n: %d", total_points, max_points_, gpu_num_points_);
      return;
    }

    // load pointcloud to host memory
    cudaSetDevice(active_device_id_);

    push_nvtx_range("Pointcloud Received", NVTXColor::Red);

    auto start = std::chrono::steady_clock::now();

    std::vector<double> host_x(n), host_y(n), host_z(n);

    sensor_msgs::PointCloud2Iterator<float> 
      it_x(*msg_copy, "x"), it_y(*msg_copy, "y"), it_z(*msg_copy, "z");

    for (size_t i = 0; i < n; ++i, ++it_x, ++it_y, ++it_z)
    {
      host_x[i] = static_cast<double>(*it_x);

      host_y[i] = static_cast<double>(*it_y);

      host_z[i] = static_cast<double>(*it_z);
    }

    double *d_x, *d_y, *d_z;
    int *d_nn;
    bool success = true;

    success &= check_cuda_error(cudaMalloc(&d_x, n * sizeof(double)), "Malloc X");

    success &= check_cuda_error(cudaMalloc(&d_y, n * sizeof(double)), "Malloc Y");

    success &= check_cuda_error(cudaMalloc(&d_z, n * sizeof(double)), "Malloc Z");

    success &= check_cuda_error(cudaMalloc(&d_nn, n * sizeof(int)), "Malloc NN");

    // copy data to device memory and shared pointers and perform NNS
    if (success)
    {
      cudaMemcpy(d_x, host_x.data(), n * sizeof(double), cudaMemcpyHostToDevice);

      cudaMemcpy(d_y, host_y.data(), n * sizeof(double), cudaMemcpyHostToDevice);

      cudaMemcpy(d_z, host_z.data(), n * sizeof(double), cudaMemcpyHostToDevice);

      build_spatial_index(d_x, d_y, d_z, d_hashes_ptr, d_indices_ptr, 
                          d_starts_ptr, d_ends_ptr, n, grid_config_, hash_table_size_, compute_stream_);


      find_nearest_neighbors(d_x, d_y, d_z, d_indices_ptr, 
                              d_starts_ptr, d_ends_ptr, d_nn, n, grid_config_, hash_table_size_);

      std::unique_lock<std::shared_timed_mutex> lock(gpu_points_mutex_);
      
      gpu_x_shared_ = std::shared_ptr<double>(d_x, [](double* p){ cudaFree(p); });
      
      gpu_y_shared_ = std::shared_ptr<double>(d_y, [](double* p){ cudaFree(p); });
      
      gpu_z_shared_ = std::shared_ptr<double>(d_z, [](double* p){ cudaFree(p); });
      
      gpu_nn_indices_shared_ = std::shared_ptr<int>(d_nn, [](int* p){ cudaFree(p); });
      
      gpu_num_points_ = n;

      auto end = std::chrono::steady_clock::now();

      RCLCPP_INFO(this->get_logger(), "Pointcloud Processing Took %.4f ms", std::chrono::duration<double, std::milli>(end - start).count());


      pop_nvtx_range();
    }
  });
}


std::tuple<double3, double3, double3, 
           double, double, double, double> FieldsComputer::extract_request_data(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request)
{
  double3 pos = make_double3(
    request->agent_pose.position.x, 
    request->agent_pose.position.y, 
    request->agent_pose.position.z
  );

  double3 goal = make_double3(
    request->target_pose.position.x, 
    request->target_pose.position.y, 
    request->target_pose.position.z
  );

  double agent_radius = request->agent_radius;

  tf2::Quaternion q;

  tf2::fromMsg(request->agent_pose.orientation, q);

  tf2::Vector3 v_world = tf2::quatRotate(q.normalized(), tf2::Vector3(request->agent_velocity.x, request->agent_velocity.y, request->agent_velocity.z));

  double3 vel = make_double3(v_world.x(), v_world.y(), v_world.z());

  return {pos, vel, goal, agent_radius, request->detect_shell_rad, request->k_force, request->max_allowable_force};
}


void FieldsComputer::process_response(
  const double3& net_force, const geometry_msgs::msg::Pose& pose,
  std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> res)
{
  tf2::Quaternion q;

  tf2::fromMsg(pose.orientation, q);

  tf2::Vector3 v_agent = tf2::quatRotate(q.normalized().inverse(), tf2::Vector3(net_force.x, net_force.y, net_force.z));

  res->circ_force.x = v_agent.x(); res->circ_force.y = v_agent.y(); res->circ_force.z = v_agent.z();

  res->not_null = true;
}


void FieldsComputer::process_queue()
{
  cudaSetDevice(active_device_id_);

  while (queue_running_) 
  {
    std::shared_ptr<Operation> op;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);

      queue_cv_.wait(lock, [this] { return !operation_queue_.empty() || !queue_running_; });

      if (!queue_running_) break;

      op = operation_queue_.front();

      operation_queue_.pop();
    }

    try 
    {
      op->task();
    } 
    catch (const std::exception& e) 
    {
      RCLCPP_ERROR(this->get_logger(), "Standard error in worker thread: %s", e.what());
    } 
    catch (...) {
      RCLCPP_ERROR(this->get_logger(), "Unknown error in worker thread");
    }

    op->completion.set_value();
  }
}


void FieldsComputer::enqueue_operation(OperationType type, std::function<void()> task)
{
  auto op = std::make_shared<Operation>();
 
  op->type = type; op->task = task;

  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
 
    operation_queue_.push(op);
  }
 
  queue_cv_.notify_one();
 
  op->completion.get_future().wait();
}

void FieldsComputer::stop_queue() 
{ 
  queue_running_ = false; 

  queue_cv_.notify_all(); 
}


void FieldsComputer::setupParamsAndServices()
{

  std::map<std::string, double*> double_params = {
    {"point_radius", &point_radius},
  };

  for (auto const& [name, ptr] : double_params) 
  {
    this->declare_parameter(name, 1.0);
    
    this->get_parameter(name, *ptr);
  }

  this->declare_parameter("show_netforce_output", false);

  this->get_parameter("show_netforce_output", show_netforce_output);


  this->declare_parameter("show_processing_delay", false);
  
  this->get_parameter("show_processing_delay", show_processing_delay);

  // -- load minimum obstacle distance service ---
  bool disable_dist = this->declare_parameter("disable_min_obstacle_distance", false);
  
  if (!disable_dist) 
    service_min_obstacle_distance = this->create_service<percept_interfaces::srv::AgentPoseToMinObstacleDist>(
      "/get_min_obstacle_distance", 
      std::bind(&FieldsComputer::handle_min_obstacle_distance, this, std::placeholders::_1, std::placeholders::_2));

  // --- load all heuristics ---
  struct HeuristicEntry 
  {
    std::string name;

    std::string topic;

    std::string disable_param;

    std::function<void(
      const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request>,
      std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response>)> callback;
  };

  std::vector<HeuristicEntry> entries = {

    {"APF", "/get_apf_heuristic_circforce", "disable_apf_heuristic", 
      [this](auto req, auto res) { handle_heuristic(req, res, artificial_potential_field_kernel, "APF"); }},

    {"Velocity", "/get_velocity_heuristic_circforce", "disable_velocity_heuristic", 
      [this](auto req, auto res) { handle_heuristic(req, res, velocity_heuristic_kernel, "Velocity"); }},

    {"Goal", "/get_goal_heuristic_circforce", "disable_goal_heuristic", 
      [this](auto req, auto res) { handle_heuristic(req, res, goal_heuristic_kernel, "Goal"); }},

    // {"Obstacle", "/get_obstacle_heuristic_circforce", "disable_obstacle_heuristic", 
    //   [this](auto req, auto res) { handle_heuristic(req, res, obstacle_heuristic::launch_kernel, "Obstacle"); }},

    // {"GoalObstacle", "/get_goalobstacle_heuristic_circforce", "disable_goalobstacle_heuristic", 
    //   [this](auto req, auto res) { handle_heuristic(req, res, goalobstacle_heuristic::launch_kernel, "GoalObstacle"); }},

    // {"Random", "/get_random_heuristic_circforce", "disable_random_heuristic", 
    //   [this](auto req, auto res) { handle_heuristic(req, res, random_heuristic::launch_kernel, "Random"); }},
  };

  for (const auto& entry : entries) 
  {
    bool disabled = this->declare_parameter(entry.disable_param, false);

    this->get_parameter(entry.disable_param, disabled);

    if (!disabled) 
    {
      heuristic_services_.push_back(
        this->create_service<percept_interfaces::srv::AgentStateToCircForce>(entry.topic, entry.callback)
      );
    }

  }
  
}

void FieldsComputer::setupDevice()
{
  int deviceCount = 0;

  if (cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0) 
  {
    int selectedDeviceId = 0;
  
    size_t maxMemory = 0;
  
    for (int i = 0; i < deviceCount; i++) 
    {
      cudaDeviceProp prop;
    
      if (cudaGetDeviceProperties(&prop, i) == cudaSuccess && prop.totalGlobalMem > maxMemory) 
      {
        maxMemory = prop.totalGlobalMem;
      
        selectedDeviceId = i;
      }
    }

    active_device_id_ = selectedDeviceId;

    cudaSetDevice(selectedDeviceId);

    cudaStreamCreate(&compute_stream_);

    RCLCPP_INFO(this->get_logger(), "Using CUDA device %d (%.2f GB)", 
                selectedDeviceId, static_cast<float>(maxMemory) / 1073741824.0f);
  }

}

bool FieldsComputer::check_cuda_error(cudaError_t err, const char* op) 
{
  if (err != cudaSuccess) {

    RCLCPP_ERROR(this->get_logger(), "CUDA %s failed: %s", op, cudaGetErrorString(err));

    return false;
  }

  return true;
}


int main(int argc, char **argv) 
{
  rclcpp::init(argc, argv);

  rclcpp::spin(std::make_shared<FieldsComputer>());

  rclcpp::shutdown();

  return 0;
}