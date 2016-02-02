#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define _dev __host__ __device__ __forceinline__ inline

template <size_t stride, size_t ...rStrides>
class Stride
{
public:
	template<class ...size_ts>
	_dev static size_t offset(size_t pos, size_ts... rpos)
	{
		return stride * pos + Stride<rStrides...>::offset(rpos...);
	}

	_dev static size_t nDim()
	{
		return 1 + Stride<rStrides...>::nDim();
	}
};

template <size_t stride>
class Stride<stride>
{
public:
	_dev static size_t offset(size_t pos)
	{
		return stride * pos;
	}

	_dev static size_t nDim()
	{
		return 1;
	}
};


template <typename TStride, float *Tdata>
class NDArrayFixed
{
public:
	template<class ...size_ts>
	_dev float &element(size_ts... pos)
	{
		return data()[TStride::offset(pos...)];
	}

	_dev TStride stride()
	{
		TStride s;
		return s;
	}

	_dev float *data()
	{
		return Tdata;
	}

	_dev size_t nDim()
	{
		return TStride::nDim();
	}
};

template <typename TStride>
class NDArrayPointer
{
public:
	template<class ...size_ts>
	_dev float &element(size_ts... pos)
	{
		return data()[TStride::offset(pos...)];
	}

	_dev TStride stride()
	{
		TStride s;
		return s;
	}

	_dev float *data()
	{
		// yes, we are sick.
		return reinterpret_cast<float *>(this);
	}

	_dev size_t nDim()
	{
		return TStride::nDim();
	}
};



struct ConstOneElementwiseOp_t
{
	_dev float operator() (const float &a)
	{
		return 1.0f;
	}
};


struct NegateElementwiseOp_t
{
	_dev float operator() (const float &a)
	{
		return -a;
	}
};


struct LogElementwiseOp_t
{
	_dev float operator() (const float &a)
	{
		return logf(a);
	}
};


struct ExpElementwiseOp_t
{
	_dev float operator() (const float &a)
	{
		return expf(a);
	}
};



struct AddBinaryElementwiseOp_t
{
	_dev float operator() (const float &a, const float &b)
	{
		return a + b;
	}
};

struct SubstractBinaryElementwiseOp_t
{
	_dev float operator() (const float &a, const float &b)
	{
		return a - b;
	}
};

struct MultiplyBinaryElementwiseOp_t
{
	_dev float operator() (const float &a, const float &b)
	{
		return a * b;
	}
};

struct DivideBinaryElementwiseOp_t
{
	_dev float operator() (const float &a, const float &b)
	{
		return a / b;
	}
};


template <typename TUnaryElementwiseOp, typename TTarget, typename TA>
__global__ void elementwiseUnary(TTarget *target, const TA *a)
{
	TUnaryElementwiseOp op;
	switch (target->nDim())
	{
	case 1:
	{
		size_t d0 = threadIdx.x + blockIdx.x * blockDim.x;
		target->element(d0) = op(a->element(d0));
		break;
	}
	case 2:
	{
		size_t d0 = threadIdx.x + blockIdx.x * blockDim.x;
		size_t d1 = threadIdx.y + blockIdx.y * blockDim.y;
		target->element(d0, d1) = op(a->element(d0, d1));
		break;
	}
	case 3:
	{
		size_t d0 = threadIdx.x + blockIdx.x * blockDim.x;
		size_t d1 = threadIdx.y + blockIdx.y * blockDim.y;
		size_t d2 = threadIdx.z + blockIdx.z * blockDim.z;
		target->element(d0, d1, d2) = op(a->element(d0, d1, d2));
		break;
	}
	default:
		printf("elementwiseUnary cannot work with %d dimensions\n", target->nDim());
	}
}



template <typename TBinaryElementwiseOp, typename TTarget, typename TA, typename TB>
_dev void elementwiseBinary(TTarget *target, const TA *a, const TB *b)
{
	TBinaryElementwiseOp op;
	switch (target->nDim())
	{
	case 1:
	{
		size_t d0 = threadIdx.x + blockIdx.x * blockDim.x;
		target->element(d0) = op(a->element(d0), b->element(d0));
		break;
	}
	case 2:
	{
		size_t d0 = threadIdx.x + blockIdx.x * blockDim.x;
		size_t d1 = threadIdx.y + blockIdx.y * blockDim.y;
		target->element(d0, d1) = op(a->element(d0, d1), b->element(d0, d1));
		break;
	}
	case 3:
	{
		size_t d0 = threadIdx.x + blockIdx.x * blockDim.x;
		size_t d1 = threadIdx.y + blockIdx.y * blockDim.y;
		size_t d2 = threadIdx.z + blockIdx.z * blockDim.z;
		target->element(d0, d1, d2) = op(a->element(d0, d1, d2), b->element(d0, d1, d2));
		break;
	}
	default:
		printf("elementwiseBinary cannot work with %d dimensions\n", target->nDim());
	}
}


/*
template <size_t nDim>
__device__ void scalarConst(NDArray<nDim> *target, const float value)
{
	
	target->element() = value;
}
*/

//template <size_t nDim>
//__device__ void DiagonalOne(NDArray<nDim> )

/*
extern "C" {
	__global__ void sayHi() { sayHi0(); }
	__global__ void scalarConst0(NDArray<0> *target, float value) 
		{ return scalarConst(target, value); }
	__global__ void scalarConst1(NDArray<1> *target, float value)
	{
		return scalarConst(target, value);
	}
}
*/

