#include <memory>
#include <optional>
#include "rclcpp/rclcpp.hpp"
#include "percept_interfaces/msg/pointcloud.hpp"

class ZeroCopySubscriber : public rclcpp::Node {
public:
    ZeroCopySubscriber() : Node("zero_copy_subscriber") {
        rclcpp::SubscriptionOptions options;
        // Optimization: Use a custom callback group if processing is heavy
        
        m_subscription = this->create_subscription<percept_interfaces::msg::Pointcloud1M>(
            "out_cloud", 
            rclcpp::QoS(1).keep_last(1), // Lossy: keep only newest
            std::bind(&ZeroCopySubscriber::topic_callback, this, std::placeholders::_1),
            options
        );

        m_timer = this->create_wall_timer(std::chrono::milliseconds(10), std::bind(&ZeroCopySubscriber::process_latest, this));
    }

private:
    void topic_callback(const percept_interfaces::msg::Pointcloud1M::SharedPtr msg) {
        // Update the "Latest" buffer
        // In ROS 2, SharedPtr management ensures zero-copy if RMW supports it
        std::lock_guard<std::mutex> lock(m_mutex);
        m_latest_cloud = msg;
    }

    void process_latest() {
        percept_interfaces::msg::Pointcloud1M::SharedPtr cloud_to_process;
        
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (!m_latest_cloud) return;
            cloud_to_process = m_latest_cloud;
        }

        // Processing cloud_to_process by reference
        RCLCPP_INFO(this->get_logger(), "Processing cloud with %u points", cloud_to_process->num_points);
    }

    rclcpp::Subscription<percept_interfaces::msg::Pointcloud1M>::SharedPtr m_subscription;
    rclcpp::TimerBase::SharedPtr m_timer;
    percept_interfaces::msg::Pointcloud1M::SharedPtr m_latest_cloud;
    std::mutex m_mutex;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ZeroCopySubscriber>());
    rclcpp::shutdown();
    return 0;
}