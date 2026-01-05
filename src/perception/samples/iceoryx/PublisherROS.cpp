#include <chrono>
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "percept_interfaces/msg/pointcloud.hpp"

using namespace std::chrono_literals;

class ZeroCopyPublisher : public rclcpp::Node {
public:
    ZeroCopyPublisher() : Node("zero_copy_publisher") {
        // Publisher must be configured for zero-copy via RMW settings
        m_publisher = this->create_publisher<percept_interfaces::msg::Pointcloud1M>("out_cloud", 10);
        m_timer = this->create_wall_timer(30ms, std::bind(&ZeroCopyPublisher::publish_cloud, this));
    }
		size_t num_points = 1000000;

private:
	void publish_cloud() {
		// Loan memory from the middleware
		auto loaned_msg = m_publisher->borrow_loaned_message();

		if (loaned_msg.is_valid()) {
			auto& msg = loaned_msg.get();
				
			// Populate data directly in shared memory
			msg.num_points = num_points;
			for (uint32_t i = 0; i < msg.num_points; ++i) {
				msg.x[i] = static_cast<float>(i);
				msg.y[i] = 0.5f;
				msg.z[i] = 1.2f;
			}

			// Transfers ownership back to middleware
			m_publisher->publish(std::move(loaned_msg));
		} else {
			RCLCPP_ERROR(this->get_logger(), "Failed to loan message memory");
		}

		if (num_points>1){
			--num_points;
		}
		else{
			num_points = 1000000;
		}
	}

    rclcpp::Publisher<percept_interfaces::msg::Pointcloud1M>::SharedPtr m_publisher;
    rclcpp::TimerBase::SharedPtr m_timer;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ZeroCopyPublisher>());
    rclcpp::shutdown();
    return 0;
}