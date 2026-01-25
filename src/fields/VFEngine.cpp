#include "VFEngine.hpp"


FieldsComputer::FieldsComputer() : Node("vf_engine")
{
  setupDevice();

  service_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

  setupParamsAndServices();

  current_snapshot_ = std::make_shared<GpuSnapshot>();

  // NNS params
  max_points_ = 1000000;

  hash_table_size_ = 2000003; 

  grid_config_.cell_size = 0.01; 

  grid_config_.min_boundary = make_double3(-1000.0, -1000.0, -1000.0);

  // Producer Configs
  allocate_producer_workspace();

  auto sub_opt = rclcpp::SubscriptionOptions();

  sub_opt.callback_group = producer_cb_group_;
  
  subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    "/pointcloud", 10, 
    std::bind(&FieldsComputer::pointcloud_callback, this, std::placeholders::_1),
    sub_opt);

  #ifdef GPU_PROFILING_ENABLE
    RCLCPP_INFO(this->get_logger(), "GPU PROFILING ENABLED");
  #endif
}


FieldsComputer::~FieldsComputer()
{
  // Cleanup Workspace
  free_producer_workspace();

  cudaStreamDestroy(compute_stream_);
}


void FieldsComputer::allocate_producer_workspace() {
  
  cudaSetDevice(active_device_id_);

  bool success = true;

  success &= check_cuda_error(cudaMalloc(&d_hashes_ptr, max_points_ * sizeof(uint32_t)), "Alloc Hashes");

  success &= check_cuda_error(cudaMalloc(&d_indices_ptr, max_points_ * sizeof(uint32_t)), "Alloc Indices");

  success &= check_cuda_error(cudaMalloc(&d_starts_ptr, hash_table_size_ * sizeof(uint32_t)), "Alloc Starts");

  success &= check_cuda_error(cudaMalloc(&d_ends_ptr, hash_table_size_ * sizeof(uint32_t)), "Alloc Ends");
  
  if (!success) throw std::runtime_error("Failed to allocate GPU workspace");
}


void FieldsComputer::free_producer_workspace() {
  cudaFree(d_hashes_ptr);

  cudaFree(d_indices_ptr);

  cudaFree(d_starts_ptr);

  cudaFree(d_ends_ptr);
}


template<typename HeuristicFunc>
void FieldsComputer::handle_heuristic(
    const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> response,
    HeuristicFunc kernel_launcher, 
    const std::string& name)
{
  auto snap = std::atomic_load(&current_snapshot_);

  if (!snap || !snap->x || snap->num_points == 0) 
  { 
    response->not_null = false; 

    return; 
  }

  auto [pos, vel, goal, agent_radius, 
        shell, k, max_f] = extract_request_data(request);

  double3 res;

  if constexpr (
    std::is_invocable_v<HeuristicFunc, 
                        double*, double*, double*, 
                        size_t, uint32_t*, 
                        double3, double3, double3, double, double, 
                        double, double, double, bool>)
  {
    res = kernel_launcher(
      snap->x.get(), snap->y.get(), snap->z.get(), 
      snap->num_points, snap->nn_indices.get(), 
      pos, vel, goal, agent_radius, point_radius, 
      shell, k, max_f, show_processing_delay
    );
  }
  else
  {
    res = kernel_launcher(
      snap->x.get(), snap->y.get(), snap->z.get(), 
      snap->num_points, pos, vel, goal, agent_radius, point_radius, 
      shell, k, max_f, show_processing_delay
    );
  }

  process_response(res, request->agent_pose, response);
}


void FieldsComputer::handle_min_obstacle_distance(
    const std::shared_ptr<percept_interfaces::srv::AgentPoseToMinObstacleDist::Request> request,
    std::shared_ptr<percept_interfaces::srv::AgentPoseToMinObstacleDist::Response> response)
{
  auto snap = std::atomic_load(&current_snapshot_);

  if (!snap || !snap->x || snap->num_points == 0) { 
    response->distance = 0.0; 

    return; 
  }
  
  double3 agent_pos = make_double3(
    request->agent_pose.position.x, 
    request->agent_pose.position.y, 
    request->agent_pose.position.z
  );

  response->distance = min_obstacle_distance_kernel(
    snap->x.get(), snap->y.get(), snap->z.get(), 
    snap->num_points, agent_pos, show_processing_delay
  );
}


void FieldsComputer::pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  push_nvtx_range("Pointcloud Received", NVTXColor::Red);

  size_t total_points = msg->width * msg->height;

  size_t n = std::min(total_points, max_points_);

  if (n <= 0) return;

  cudaSetDevice(active_device_id_);

  std::vector<double> host_x(n), host_y(n), host_z(n);

  sensor_msgs::PointCloud2Iterator<float> it_x(*msg, "x"), it_y(*msg, "y"), it_z(*msg, "z");

  for (size_t i = 0; i < n; ++i, ++it_x, ++it_y, ++it_z) 
  {
    host_x[i] = static_cast<double>(*it_x);

    host_y[i] = static_cast<double>(*it_y);

    host_z[i] = static_cast<double>(*it_z);
  }

  auto new_snap = std::make_shared<GpuSnapshot>();

  new_snap->num_points = n;
  
  double *d_x, *d_y, *d_z;
  
  int *d_nn;

  bool success = true;

  success &= check_cuda_error(cudaMalloc(&d_x, n * sizeof(double)), "Malloc X");

  success &= check_cuda_error(cudaMalloc(&d_y, n * sizeof(double)), "Malloc Y");
  
  success &= check_cuda_error(cudaMalloc(&d_z, n * sizeof(double)), "Malloc Z");
  
  success &= check_cuda_error(cudaMalloc(&d_nn, n * sizeof(int)), "Malloc NN");


  if (!success) 
  {
    if(d_x) cudaFree(d_x); if(d_y) cudaFree(d_y); if(d_z) cudaFree(d_z); if(d_nn) cudaFree(d_nn);
    return;
  }

  new_snap->x = std::shared_ptr<double>(d_x, [](double* p){ cudaFree(p); });

  new_snap->y = std::shared_ptr<double>(d_y, [](double* p){ cudaFree(p); });

  new_snap->z = std::shared_ptr<double>(d_z, [](double* p){ cudaFree(p); });

  new_snap->nn_indices = std::shared_ptr<int>(d_nn, [](int* p){ cudaFree(p); });

  cudaMemcpy(d_x, host_x.data(), n * sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_y, host_y.data(), n * sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_z, host_z.data(), n * sizeof(double), cudaMemcpyHostToDevice);

  build_spatial_index(d_x, d_y, d_z, d_hashes_ptr, d_indices_ptr, 
                      d_starts_ptr, d_ends_ptr, n, grid_config_, 
                      hash_table_size_, compute_stream_
  );

  find_nearest_neighbors(d_x, d_y, d_z, d_indices_ptr, 
                         d_starts_ptr, d_ends_ptr, d_nn, n, grid_config_, 
                         hash_table_size_
  );

  cudaStreamSynchronize(compute_stream_);

  std::atomic_store(
    &current_snapshot_, 
    std::shared_ptr<const GpuSnapshot>(new_snap)
  );

  pop_nvtx_range();
}


// --- Helper Methods ---
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
  
  tf2::Vector3 v_world = tf2::quatRotate(
    q.normalized(), 
    tf2::Vector3(
      request->agent_velocity.x, 
      request->agent_velocity.y, 
      request->agent_velocity.z
    )
  );

  double3 vel = make_double3(
    v_world.x(), 
    v_world.y(), 
    v_world.z()
  );

  return {pos, vel, goal, 
          agent_radius, request->detect_shell_rad, 
          request->k_force, request->max_allowable_force
  };
}


void FieldsComputer::process_response(
  const double3& net_force, const geometry_msgs::msg::Pose& pose,
  std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response> res)
{
  tf2::Quaternion q;

  tf2::fromMsg(pose.orientation, q);
  
  tf2::Vector3 v_agent = tf2::quatRotate(
    q.normalized().inverse(), 
    tf2::Vector3(
      net_force.x, net_force.y, net_force.z
    ));
  
  res->circ_force.x = v_agent.x(); 
  
  res->circ_force.y = v_agent.y(); 
  
  res->circ_force.z = v_agent.z();
  
  res->not_null = true;
}

void FieldsComputer::setupParamsAndServices()
{
  std::map<std::string, double*> double_params = { 
    {"point_radius", &point_radius} 
  };

  for (auto const& [name, ptr] : double_params) {
    this->declare_parameter(name, 0.01);
    
    this->get_parameter(name, *ptr);
  }
  
  this->declare_parameter("show_netforce_output", false);

  this->get_parameter("show_netforce_output", show_netforce_output);

  this->declare_parameter("show_processing_delay", false);

  this->get_parameter("show_processing_delay", show_processing_delay);

  bool disable_dist = this->declare_parameter("disable_min_obstacle_distance", false);

  if (!disable_dist) 

  service_min_obstacle_distance = this->create_service<percept_interfaces::srv::AgentPoseToMinObstacleDist>(
      "/get_min_obstacle_distance", 
      std::bind(&FieldsComputer::handle_min_obstacle_distance, this, std::placeholders::_1, std::placeholders::_2),
      rclcpp::ServicesQoS(), service_cb_group_);

  struct HeuristicEntry {
    std::string topic;

    std::string disable_param;

    std::function<void(const std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Request>,
                       std::shared_ptr<percept_interfaces::srv::AgentStateToCircForce::Response>)> callback;
  };

  std::vector<HeuristicEntry> entries = {
    {"/get_apf_heuristic_circforce", "disable_apf_heuristic", 
      [this](auto req, auto res) { handle_heuristic(req, res, artificial_potential_field_kernel, "APF"); }},

    {"/get_velocity_heuristic_circforce", "disable_velocity_heuristic", 
      [this](auto req, auto res) { handle_heuristic(req, res, velocity_heuristic_kernel, "Velocity"); }},

    {"/get_goal_heuristic_circforce", "disable_goal_heuristic", 
      [this](auto req, auto res) { handle_heuristic(req, res, goal_heuristic_kernel, "Goal"); }}

    // {"/get_obstacle_heuristic_circforce", "disable_obstacle_heuristic", 
    //   [this](auto req, auto res) { handle_heuristic(req, res, obstacle_heuristic::launch_kernel, "Obstacle"); }},

    // {"/get_goalobstacle_heuristic_circforce", "disable_goalobstacle_heuristic", 
    //   [this](auto req, auto res) { handle_heuristic(req, res, goalobstacle_heuristic::launch_kernel, "GoalObstacle"); }},

    // {"/get_random_heuristic_circforce", "disable_random_heuristic", 
    //   [this](auto req, auto res) { handle_heuristic(req, res, random_heuristic::launch_kernel, "Random"); }}

  };

  for (const auto& entry : entries) 
  {
    bool disabled = this->declare_parameter(entry.disable_param, false);
  
    this->get_parameter(entry.disable_param, disabled);
  
    if (!disabled) 
    {
      heuristic_services_.push_back(  
        this->create_service<percept_interfaces::srv::AgentStateToCircForce>(
            entry.topic, 
            entry.callback, 
            rclcpp::ServicesQoS(), 
            service_cb_group_
        ));
    }

  } // load all services

}


void FieldsComputer::setupDevice()
{
  int deviceCount = 0;

  if (cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0) {

    cudaSetDevice(0); // Simplification for brevity

    cudaStreamCreate(&compute_stream_);
  }
}

bool FieldsComputer::check_cuda_error(cudaError_t err, const char* op) {

  if (err != cudaSuccess) {

    RCLCPP_ERROR(this->get_logger(), "CUDA %s failed: %s", op, cudaGetErrorString(err));

    return false;
  }

  return true;
}

int main(int argc, char **argv) 
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<FieldsComputer>();
  
  rclcpp::executors::MultiThreadedExecutor executor(
    rclcpp::ExecutorOptions(), 
    std::thread::hardware_concurrency()
  );

  executor.add_node(node);

  executor.spin();
  
  rclcpp::shutdown();

  return 0;
}