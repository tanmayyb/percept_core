#pragma once

#include <atomic>
#include <array>
#include <vector>
#include <memory>

#include <iostream>

template<typename T>
class Mailbox 
{
	private:
		std::vector<T> buffers_[3];

		std::atomic<int> latest_idx_;

		int producer_idx_;

		int consumer_idx_;

		public:
			Mailbox(size_t batch_size, size_t n_points) 
					: latest_idx_(1), producer_idx_(0), consumer_idx_(2) 
			{
				const size_t total_elements = batch_size * n_points * 3;

				for (int i = 0; i < 3; ++i) {
					buffers_[i].resize(total_elements);
				}
			}

			std::vector<T>& get_producer_buffer() {
				return buffers_[producer_idx_];
			}

			void commit() {
				producer_idx_ = latest_idx_.exchange(producer_idx_, std::memory_order_release);
			}

			std::vector<T>& consume() {
				consumer_idx_ = latest_idx_.exchange(consumer_idx_, std::memory_order_acquire);

				return buffers_[consumer_idx_];
			}
};