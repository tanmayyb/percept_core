#include <memory>
#include <atomic>
#include <chrono>
#include "rclcpp/rclcpp.hpp"
#include "percept_interfaces/msg/pointcloud.hpp"

template<typename T>
class SharedMailbox {
public:
    using MessagePtr = std::shared_ptr<T>;

    SharedMailbox() : latest_idx_(1), producer_idx_(0), consumer_idx_(2) {
        for (int i = 0; i < 3; ++i) {
            buffers_[i] = nullptr;
        }
    }

    /**
     * @brief Transitions a new message into the mailbox.
     * @param new_msg The SharedPtr received from the RMW.
     * Swaps the producer index with the latest index and clears 
     * the stale buffer to release SHM reference counts.
     */
    void commit(MessagePtr new_msg) {
        buffers_[producer_idx_] = std::move(new_msg);
        producer_idx_ = latest_idx_.exchange(producer_idx_, std::memory_order_release);
        buffers_[producer_idx_] = nullptr; 
    }

    /**
     * @brief Retrieves the most recent message from the mailbox.
     * @return MessagePtr to the SHM segment.
     */
    MessagePtr consume() {
        consumer_idx_ = latest_idx_.exchange(consumer_idx_, std::memory_order_acquire);
        return buffers_[consumer_idx_];
    }

private:
    MessagePtr buffers_[3];
    std::atomic<int> latest_idx_;
    int producer_idx_;
    int consumer_idx_;
};



class ZeroCopySubscriber : public rclcpp::Node {
public:
    ZeroCopySubscriber() : Node("zero_copy_subscriber") {
        // RMW must be configured (e.g., Iceoryx) to enable zero-copy transport
        m_subscription = this->create_subscription<percept_interfaces::msg::Pointcloud>(
            "out_cloud", 
            rclcpp::QoS(1).keep_last(1),
            std::bind(&ZeroCopySubscriber::topic_callback, this, std::placeholders::_1)
        );

        m_timer = this->create_wall_timer(
            std::chrono::milliseconds(10), 
            std::bind(&ZeroCopySubscriber::process_latest, this)
        );
    }

private:
    void topic_callback(const percept_interfaces::msg::Pointcloud::SharedPtr msg) {
        // Lock-free handoff to the mailbox
        m_mailbox.commit(msg);
    }

    void process_latest() {
        // Atomic acquisition of the latest message pointer
        auto cloud_to_process = m_mailbox.consume();
        
        if (!cloud_to_process) {
            return;
        }

        // Processing logic operates on pinned shared memory
        RCLCPP_INFO(this->get_logger(), "Processing cloud with %u points", cloud_to_process->num_points);
    }

    rclcpp::Subscription<percept_interfaces::msg::Pointcloud>::SharedPtr m_subscription;
    rclcpp::TimerBase::SharedPtr m_timer;
    SharedMailbox<percept_interfaces::msg::Pointcloud> m_mailbox;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ZeroCopySubscriber>());
    rclcpp::shutdown();
    return 0;
}