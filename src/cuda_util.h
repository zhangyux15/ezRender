#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>


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
}

