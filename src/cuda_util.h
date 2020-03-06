#pragma once
#include <string>
#include <iostream>
#include <memory>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include "math_util.h"


namespace CudaUtil {
	inline void Check(cudaError_t err)
	{
		if (err != cudaSuccess) {
			std::cerr << "cuda error code: " << std::to_string(err) << std::endl;
			std::abort();
		}
	}

	inline void Check(curandStatus_t err)
	{
		if (err != CURAND_STATUS_SUCCESS) {
			std::cerr << "curand error code: " << std::to_string(err) << std::endl;
			std::abort();
		}
	}

	inline void Check()
	{
		Check(cudaGetLastError());
	}

	static const int THREAD1D_CAP = 256;
	static const dim3 THREAD2D_CAP = dim3(16, 16);

	class Event
	{
	public:
		Event() { Check(cudaEventCreate(&m_event)); }
		~Event() { Check(cudaEventDestroy(m_event)); }
		Event(const Event& _) = delete;
		Event& operator=(const Event& _) = delete;

		cudaEvent_t Get() { return m_event; }
		void Record() { Check(cudaEventRecord(m_event)); }
		void Synchronize() { Check(cudaEventSynchronize(m_event)); }
	private:
		cudaEvent_t m_event;
	};


	class Stream
	{
	public:
		Stream() { Check(cudaStreamCreate(&m_stream)); }
		~Stream() { Check(cudaStreamDestroy(m_stream)); }
		Stream(const Stream& _) = delete;
		Stream& operator=(const Stream& _) = delete;

		cudaStream_t Get() { return m_stream; }
		void Synchronize() { Check(cudaStreamSynchronize(m_stream)); }
	private:
		cudaStream_t m_stream;
	};


	template<typename T>
	class Arr
	{
	public:
		Arr() {}
		Arr(const size_t& _size) { Resize(_size); }
		~Arr() {
			if (m_data != nullptr)
				Check(cudaFree(m_data));
		}

		Arr(const Arr& _arr) = delete;
		Arr& operator=(const Arr& _) = delete;

		T* Get() { return m_data; }
		T const* Get() const { return m_data; }
		size_t Size() const { return size; }
		void Resize(const size_t& _size) {
			if (size > 0) {
				Check(cudaFree(m_data));
				m_data = nullptr;
			}
			size = _size;
			if (_size > 0)
				Check(cudaMalloc((void**)&m_data, _size * sizeof(T)));
		}

		void Upload(const T* const src, const size_t& bytes, cudaStream_t stream) {
			Check(cudaMemcpyAsync((void*)m_data, src, bytes, cudaMemcpyHostToDevice, stream));
		}

		void Upload(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m, cudaStream_t stream) {
			Upload(m.data(), m.size() * sizeof(T), stream);
		}

		void Upload(const cv::Mat& m, cudaStream_t stream) {
			Upload((const T* const)m.data, m.channels()*m.rows*m.cols * sizeof(T), stream);
		}


		void Download(void* dst, const size_t& bytes, cudaStream_t stream) const {
			Check(cudaMemcpyAsync(dst, (void*)m_data, bytes, cudaMemcpyDeviceToHost, stream));
		}

		void Download(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m, cudaStream_t stream) const {
			Download((void*)m.data(), m.size() * sizeof(T), stream);
		}

		void Download(const cv::Mat& m, cudaStream_t stream) const {
			Download((void*)m.data, m.channels()*m.rows*m.cols * sizeof(T), stream);
		}

		void SetZero(cudaStream_t stream) {
			CudaUtil::Check(cudaMemsetAsync(m_data, 0, size * sizeof(T), stream));
		}

	private:
		T* m_data = nullptr;
		size_t size = 0;
	};

}




