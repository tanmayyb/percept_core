#pragma once

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

namespace cuda_kernel {
    void hello_cuda_world();
}

namespace percept{
    class FieldsComputer : public rclcpp::Node{
        public:
            FieldsComputer();
        private:
            rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    };
}