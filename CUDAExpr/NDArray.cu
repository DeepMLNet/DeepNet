
#include <array>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

template <size_t shape, size_t stride, size_t... Args>
class NDArray
{
public:
	float *data;

	__forceinline__ __device__ static size_t index()
	{
		return 0;
	}

	__forceinline__ __device__ static size_t index(size_t pos)
	{
		return stride * pos;
	}

	__forceinline__ __device__ static size_t index(size_t pos, Args... rpos)
	{
		return stride * pos + NDArray<Args...>::index(rpos...);
	}

	//__forceinline__ __device__ float &element(const size_t *(&pos))
	//{
	//	size_t idx = 0;
	//	#pragma unroll
	//	for (size_t d = 0; d < nDim; d++)
	//		idx += pos[d] * stride[d];
	//	return data[idx];
	//}

	__forceinline__ __device__ float &element()
	{
		return data[0];
	}
};



__device__ void sayHi0()
{
	printf("Cuda Kernel Hello Word.\n");

}

template <size_t nDim>
__device__ void scalarConst(NDArray<nDim> *target, const float value)
{
	
	target->element() = value;
}

template <size_t nDim>
__device__ void DiagonalOne(NDArray<nDim> )

extern "C" {
	__global__ void sayHi() { sayHi0(); }
	__global__ void scalarConst0(NDArray<0> *target, float value) 
		{ return scalarConst(target, value); }
	__global__ void scalarConst1(NDArray<1> *target, float value)
	{
		return scalarConst(target, value);
	}

}

