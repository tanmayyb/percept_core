#include "percept/fields_computer.hpp"
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/point.hpp>
#include "percept_interfaces/srv/pose_to_vector.hpp"  

// using namespace std::chrono_literals;

class FieldsComputer : public rclcpp::Node
{
public:
    FieldsComputer()
        : Node("fields_computer")
    {
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/primitives", 10,
            std::bind(&FieldsComputer::pointcloud_callback, this, std::placeholders::_1));
        cuda_kernel::hello_cuda_world();

        service_ = this->create_service<percept_interfaces::srv::PoseToVector>(
            "/get_heuristic_circforce",
            std::bind(&FieldsComputer::handle_pose_to_vector, this,
                     std::placeholders::_1, std::placeholders::_2));
    }

private:
    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // Get number of points
        size_t num_points = msg->width * msg->height;
        
        // Resize array to store points
        points_array_.resize(num_points * 3);  // x,y,z for each point
        
        // Create iterators for x,y,z fields
        sensor_msgs::PointCloud2Iterator<float> iter_x(*msg, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(*msg, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(*msg, "z");

        // Copy points to array
        for (size_t i = 0; i < num_points; ++i, ++iter_x, ++iter_y, ++iter_z)
        {
            points_array_[i * 3] = *iter_x;     // x
            points_array_[i * 3 + 1] = *iter_y; // y
            points_array_[i * 3 + 2] = *iter_z; // z
        }
        // RCLCPP_INFO(this->get_logger(), "Received %zu points", num_points);
    }

    void handle_pose_to_vector(
        const std::shared_ptr<percept_interfaces::srv::PoseToVector::Request> request,
        std::shared_ptr<percept_interfaces::srv::PoseToVector::Response> response)
    {
        // For now, just return the position as the vector (you can modify this logic)
        response->vector = request->pose.position;
        response->within_radius = true;  // Add your radius check logic here
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Service<percept_interfaces::srv::PoseToVector>::SharedPtr service_;
    std::vector<float> points_array_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FieldsComputer>());
    rclcpp::shutdown();
    return 0;
}