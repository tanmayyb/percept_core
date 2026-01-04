#include "PointCloud.hpp"
#include <iceoryx_posh/popo/subscriber.hpp>
#include <iceoryx_posh/runtime/posh_runtime.hpp>
#include <optional>
#include <iostream>

class CloudPipeline {
public:
    CloudPipeline() {
        iox::runtime::PoshRuntime::initRuntime("iox-cloud-subscriber");
        
        iox::popo::SubscriberOptions options;
        options.queueCapacity = 1U;
        options.queueFullPolicy = iox::popo::QueueFullPolicy::DISCARD_OLDEST_DATA;
        
        m_subscriber.emplace(iox::capro::ServiceDescription{"Sensor", "LiDAR", "PointCloud"}, options);
    }

    void run() {
        while (true) {
            updateLatest();
            process();
        }
    }

private:
    void updateLatest() {
        while (m_subscriber->take().and_then([this](auto& sample) {
            // Assigning to optional moves the Sample handle
            m_latest = std::move(sample); 
        })) {
        }
    }

    void process() {
        if (m_latest.has_value()) {
            // (*m_latest) accesses the Sample
            // ->width accesses the PointCloud member
            std::cout << "Processing cloud with " << (*m_latest)->width << " points" << std::endl;
        }
    }

    std::optional<iox::popo::Subscriber<PointCloud>> m_subscriber;
    // Optional wrapper enables deferred initialization
    std::optional<iox::popo::Sample<const PointCloud>> m_latest; 
};

int main() {
    CloudPipeline pipeline;
    pipeline.run();
    return 0;
}
