#pragma once

#include <functional>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Tensor.h"

#define _dev __host__ __device__ __forceinline__ 
#define _devonly __device__ __forceinline__

struct WorkFunc1 {
	Tensor<float, 3> trgt;
	Tensor<float, 3> src1;
	void operator() (Idxs<3> pos) {
		trgt[pos] = src1[pos];
	}
};


template<dim_t TWorkDims, dim_t TDim>
inline _dev void FillPosFromRest(const Idxs<TWorkDims> &workSize, Idxs<TWorkDims> &pos, idx_t rest) {
	idx_t size = 1;
	for (dim_t d = TDim + 1; d < TWorkDims - 2; d++)
		size *= workSize[d];
	pos[TDim] = rest / size;
	rest -= pos[TDim] * size;
	if (TDim < TWorkDims - 2)
		FillPosFromRest<TWorkDims, TDim + 1>(workSize, pos, rest);
}


template<typename TWorkFn, dim_t TWorkDims>
__device__ void PerformWork(TWorkFn &workFn, const Idxs<TWorkDims> &workSize) {
	Idxs<TWorkDims> pos;
	if (TWorkDims == 0) {
		workFn(pos);
	}
	else if (TWorkDims == 1) {
		for (pos[0] = threadIdx.x + blockIdx.x * blockDim.x;
			 pos[0] < workSize[0];
			 pos[0] += gridDim.x * blockDim.x) {
			workFn(pos);
		}
	}
	else if (TWorkDims == 2) {
		for (pos[0] = threadIdx.y + blockIdx.y * blockDim.y;
			 pos[0] < workSize[0];
			 pos[0] += gridDim.y * blockDim.y) {
			for (pos[1] = threadIdx.x + blockIdx.x * blockDim.x;
				 pos[1] < workSize[1];
				 pos[1] += gridDim.x * blockDim.x) {
				workFn(pos);
			}
		}
	}
	else if (TWorkDims == 3) {
		for (pos[0] = threadIdx.z + blockIdx.z * blockDim.z;
			 pos[0] < workSize[0];
			 pos[0] += gridDim.z * blockDim.z) {
			for (pos[1] = threadIdx.y + blockIdx.y * blockDim.y;
				 pos[1] < workSize[1];
				 pos[1] += gridDim.y * blockDim.y) {
				for (pos[2] = threadIdx.x + blockIdx.x * blockDim.x;
					 pos[2] < workSize[2];
					 pos[2] += gridDim.x * blockDim.x) {
					workFn(pos);
				}
			}
		}
	}
	else {
		idx_t restWork = 1;
		for (dim_t d = 0; d < TWorkDims - 2; d++)
			restWork *= workSize[d];
		for (idx_t rest = threadIdx.z + blockIdx.z * blockDim.z;
			 rest < restWork;
			 rest += gridDim.z * blockDim.z) {
			FillPosFromRest<TWorkDims, 0>(workSize, pos, rest);
			for (pos[TWorkDims - 2] = threadIdx.y + blockIdx.y * blockDim.y;
				 pos[TWorkDims - 2] < workSize[TWorkDims - 2];
				 pos[TWorkDims - 2] += gridDim.y * blockDim.y) {
				for (pos[TWorkDims - 1] = threadIdx.x + blockIdx.x * blockDim.x;
				 	 pos[TWorkDims - 1] < workSize[TWorkDims - 1];
					 pos[TWorkDims - 1] += gridDim.x * blockDim.x) {
					workFn(pos);
				}
			}
		}
	}
};



// - generic work function for homogeneous work is written
// - write a kernel that uses it for testing and instantiate the template

//template<dim_t TWorkDims>
//_dev void CopyWorkFn(const Idxs<TWorkDims> &pos) {
//
//}

template<typename TTrgtT, typename TSrc1T, dim_t TWorkDims>
struct UnaryElemwiseApplyWorkFn {
	std::function<TTrgtT(TSrc1T)> ElemwiseFn;
	Tensor<TTrgtT, TWorkDims> Trgt;
	const Tensor<TSrc1T, TWorkDims> Src1;

	UnaryElemwiseApplyWorkFn(
		std::function<TTrgtT(TSrc1T)> elemwiseFn,
		Tensor<TTrgtT, TWorkDims> trgt,
		Tensor<TSrc1T, TWorkDims> src1)
		: ElemwiseFn(elemwiseFn), Trgt(trgt), Src1(src1) {

	}

	void operator() (const Idxs<TWorkDims> &pos) {
		Trgt[pos] = ElemwiseFn(Src1[pos]);
	}
};


template<typename TTrgtT, typename TSrc1T, dim_t TWorkDims>
_dev void FillElementwise(
	Tensor<TTrgtT, TWorkDims> &trgt,
	const Tensor<TSrc1T, TWorkDims> &src1) {

	UnaryElemwiseApplyWorkFn<TTrgtT, TSrc1T, TWorkDims> workFn 
		([](TSrc1T val1) { return val1; }, trgt, src1);
	PerformWork<UnaryElemwiseApplyWorkFn<TTrgtT, TSrc1T, TWorkDims>, TWorkDims> 
		(workFn, trgt.Shape());
};





/*
template<typename TUnaryOp, typename TTrgt, typename TA>
void FillElementwise(TUnaryOp &op, TTrgt &trgt, TA &a) {
	// need generic loop over arbitrary number of dimensions.
	// also should this be a kernel or a function
	// - if it is a kernel, then there should be no loop but index need to be derived
	//   from threadIdx and blockIdx
	// - actually we need a generic function for translating the threadIdx and blockIdx
	//   into a position
	// - but this also might involve a loop if there is too much work
	// - so we need this loop as a generic function
	// - but actually it could be a device function template, accepting a functor
	// - and then we call it from approproiate kernels

}
*/


