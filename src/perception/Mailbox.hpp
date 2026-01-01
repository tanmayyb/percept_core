#pragma once

#include <atomic>
#include <array>
#include <vector>
#include <memory>

#include <iostream>

// template<typename T>
// class Mailbox
// {
// 	private:
// 		// std::array<T, 3> buffers_;

// 		T buffers_[3];

// 		std::atomic<int> latest_idx_;

// 		int producer_idx_;

// 		int consumer_idx_;

// 	public:
// 		Mailbox(size_t n_points): latest_idx_(1), producer_idx_(0), consumer_idx_(2) 
// 		{
// 			for(int i = 0; i < 3; ++i)
// 			{
// 				buffers_[i].init(n_points);
// 			}
// 		}

// 		void produce(const T& data)
// 		{
// 			buffers_[producer_idx_] = data;

// 			producer_idx_ = latest_idx_.exchange(producer_idx_, std::memory_order_release);
		
// 		}

// 		T& consume()
// 		{
// 			consumer_idx_ = latest_idx_.exchange(consumer_idx_, std::memory_order_acquire);

// 			return buffers_[consumer_idx_];
// 		}
// };


template<typename T>
class Mailbox
{
	private:
		std::vector<T> buffers_[3];

		std::atomic<int> latest_idx_;

		int producer_idx_;

		int consumer_idx_;

	public:
		Mailbox(size_t batch_size, size_t n_points): latest_idx_(1), producer_idx_(0), consumer_idx_(2) 
		{
			const size_t total_elements = batch_size * n_points;

			for (int i = 0; i < 3; ++i) 
			{
				buffers_[i].resize(total_elements);
			}
		}

		void produce(const std::vector<T>& data)
		{
			buffers_[producer_idx_] = data;		

			producer_idx_ = latest_idx_.exchange(producer_idx_, std::memory_order_release);

			// std::cout<<"producer stored data at "<<producer_idx_<<std::endl;
		}

		std::vector<T>& consume()
		{
			consumer_idx_ = latest_idx_.exchange(consumer_idx_, std::memory_order_acquire);

			return buffers_[consumer_idx_];
		}
};



// template<typename T, std::size_t BatchSize, std::size_t NPoints>
// class	Mailbox
// {
// 	private:
// 		static constexpr std::size_t TotalSize = BatchSize * NPoints;

// 		using BufferElement = std::array<float, TotalSize>;

// 		BufferElement buffers_[3];

// 		std::atomic<int> latest_idx_;

// 		int producer_idx_;

// 		int consumer_idx_;

// 	public:
// 		Mailbox(): latest_idx_(1), producer_idx_(0), consumer_idx_(2) {}

// 	void produce(const )
// }