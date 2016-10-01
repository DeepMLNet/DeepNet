#pragma once

#ifndef __CUDACC_RTC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#endif

#define _dev __host__ __device__ __forceinline__ 
#define _devonly __device__ __forceinline__

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

// dummy functions for IntelliSense
#ifndef __CUDACC__ 
int atomicAdd(int* address, int val); 
unsigned int atomicAdd(unsigned int* address, unsigned int val); 
unsigned long long int atomicAdd(unsigned long long int* address, unsigned long long int val); 
float atomicAdd(float* address, float val);

template <typename T> T tex1D(cudaTextureObject_t texObj, float x);
template <typename T> T tex2D(cudaTextureObject_t texObj, float x, float y);
template <typename T> T tex3D(cudaTextureObject_t texObj, float x, float y, float z);
#endif


typedef unsigned int idx_t;


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

template <typename T>
_dev T signt(const T a)
{
	return (a >= ((T)0)) ? ((T)1) : ((T)-1);
}

_dev idx_t divCeil(const idx_t a, const idx_t b)
{
	return (a + b - 1) / b;
}

// An array of fixed size that can be passed by value in function calls.
template <typename T, idx_t Tsize>
struct Array {
	T mElements[Tsize];

	_dev T& operator[] (const int idx) { return mElements[idx]; };
	const _dev T& operator[] (const int idx) const { return mElements[idx]; };
	idx_t _dev size() const { return Tsize; }
};


