#pragma once

#ifndef __CUDACC_RTC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

#define _dev __host__ __device__ __forceinline__ 


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


// An array of fixed size that can be passed by value in function calls.
template <typename T, size_t Tsize>
struct Array {
	T mElements[Tsize];

	T& operator[] (const int idx) { return mElements[idx]; };
	const T& operator[] (const int idx) const { return mElements[idx]; };
	size_t size() const { return Tsize; }
};
