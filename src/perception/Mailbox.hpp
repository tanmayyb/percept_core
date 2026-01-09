#pragma once

#include <atomic>
#include <array>
#include <vector>
#include <memory>

#include <iostream>


// // Basic Triple Buffer 
// // Suffers from Jitters when Consumer is fater than Producer
// template<typename T>
// class Mailbox 
// {
// 	public:
// 		Mailbox(size_t batch_size, size_t n_points) 
// 			: latest_idx_(1), producer_idx_(0), consumer_idx_(2) 
// 			{
// 				const size_t total_elements = batch_size * n_points * 3;

// 				for (int i = 0; i < 3; ++i) 
// 				{
// 					buffers_[i].resize(total_elements);
// 				}
// 			}

// 		std::vector<T>& get_producer_buffer() 
// 		{
// 			return buffers_[producer_idx_];
// 		}

// 		void commit() 
// 		{
// 			producer_idx_ = latest_idx_.exchange(producer_idx_, std::memory_order_release);
// 		}

// 		std::vector<T>& consume() 
// 		{
// 			consumer_idx_ = latest_idx_.exchange(consumer_idx_, std::memory_order_acquire);

// 			return buffers_[consumer_idx_];
// 		}

// 	private:
// 		std::vector<T> buffers_[3];

// 		std::atomic<int> latest_idx_;

// 		int producer_idx_;

// 		int consumer_idx_;
// };

// template<typename T>
// class SharedMailbox
// {
// 	public:
// 		// using MessagePtr = std::shared_ptr<T>;

// 		SharedMailbox() 
// 			: latest_idx_(1), producer_idx_(0), consumer_idx_(2) 
// 		{
// 			for (int i = 0; i < 3; ++i) buffers_[i] = nullptr;
// 		}

// 		void commit(std::shared_ptr<T> new_msg)
// 		{
// 			buffers_[producer_idx_] = new_msg;

// 			producer_idx_ = latest_idx_.exchange(producer_idx_, std::memory_order_release);

// 			buffers_[producer_idx_] = nullptr;
// 		}

// 		std::shared_ptr<T> consume()
// 		{
// 			consumer_idx_ = latest_idx_.exchange(consumer_idx_, std::memory_order_acquire);

// 			return buffers_[consumer_idx_];
// 		}

// 		private:
// 		std::shared_ptr<T> buffers_[3];

// 		std::atomic<int> latest_idx_;

// 		int producer_idx_;

// 		int consumer_idx_;
// };



// State based Triple Buffering
// Persistent CAS loop
template<typename T>
class Mailbox
{
	private:
		std::vector<T> buffers_[3];

		std::atomic<uint8_t> state_;

		uint8_t producer_idx_;

		uint8_t consumer_idx_;
		
		static constexpr uint8_t FLAG_MASK = 0x40;

	public:
		Mailbox(size_t batch_size, size_t n_points)
			: producer_idx_(0), consumer_idx_(2), state_(0x24)
			{
				const size_t total_elements = batch_size * n_points * 3;

				for (int i = 0; i < 3; ++i) buffers_[i].resize(total_elements);
			}
		
		std::vector<T>& get_producer_buffer()
		{
			return buffers_[producer_idx_];
		}

		void commit()
		{
			uint8_t current_state = state_.load(std::memory_order_relaxed);

			uint8_t next_state;

			do
			{
				uint8_t latest_idx_ = (current_state >> 2) & 0x03;

				next_state = (current_state & 0x30) | (producer_idx_ << 2) | latest_idx_ | FLAG_MASK;

				producer_idx_ = latest_idx_;

			} while (
				!state_.compare_exchange_weak(
					current_state, next_state, std::memory_order_relaxed)
			);	
		}

		std::vector<T>& consume()
		{
			uint8_t current_state = state_.load(std::memory_order_relaxed);

			if (!(current_state & FLAG_MASK))
			{
				return buffers_[consumer_idx_];
			}

			uint8_t next_state;

			do
			{
				uint8_t latest_idx_ = (current_state >> 2) & 0x03;

				next_state = (latest_idx_ << 4) | (consumer_idx_ << 2) | (current_state & 0x03);

				consumer_idx_ = latest_idx_;
			} while (
				!state_.compare_exchange_weak(
					current_state, next_state, std::memory_order_relaxed)
			);

			return buffers_[consumer_idx_];
		}
};
