#pragma once

#ifndef __CUDACC_RTC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#endif

#ifdef ENABLE_CALL_TRACE
#define KERNEL_TRACE(msg)   \
	{ if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && \
          blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) \
          { printf("Kernel: "); printf(msg); printf("\n"); } \
      __syncthreads(); }
#define HOST_TRACE(msg)   { printf("Host:  "); printf(msg); printf("\n"); }
#else
#define KERNEL_TRACE(msg) {}
#define HOST_TRACE(msg) {}
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


_dev size_t divCeil(const size_t a, const size_t b)
{
	return (a + b - 1) / b;
}

// An array of fixed size that can be passed by value in function calls.
template <typename T, size_t Tsize>
struct Array {
	T mElements[Tsize];

	_dev T& operator[] (const int idx) { return mElements[idx]; };
	const _dev T& operator[] (const int idx) const { return mElements[idx]; };
	size_t _dev size() const { return Tsize; }
};


