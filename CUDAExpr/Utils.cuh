#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define _dev __host__ __device__ __forceinline__ inline


template <typename T>
_dev T min(const T a, const T b)
{
	return a < b ? a : b;
}

template <typename T>
_dev T max(const T a, const T b)
{
	return a > b ? a : b;
}

