#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Tensor.h"


// Now what needs to be done?
// - we need a copy routine for CUDA
// - we also need elementwise operations for CUDA
// - we should write a generic elementwise function
// - make it part of Tensor
// - optimizing compiler can only auto vectorize loops if it knows the strides in advance
// - we would need special handling for that in host code
// - for CUDA it does not matter...
// - so we should not make the function part of the struct but keep it externally

//wo

struct WorkFunc1 {
	Tensor<float, 3> trgt;
	Tensor<float, 3> src1;
	void operator() (Idxs<3> pos) {
		trgt[pos] = src1[pos];
	}
};


template<typename TWorkFn, typename TWorkSize>
void PerformWork(TWorkFunc &workFn, const TWorkSize &workSize) {

	const dim_t workDims = workSize.NDims();
	Idxs<workDims> pos;
	Idxs<workDims> incr;

	if (workDims = 0) {
		// nothing to initialize
	}
	else if (workDims = 1) {
		pos[0] = threadIdx.x + blockIdx.x * blockDim.x;
		incr[0] = gridDim.x * blockDim.x;
	}
	else if (workDims = 2) {
		pos[0] = threadIdx.y + blockIdx.y * blockDim.y;
		incr[0] = gridDim.y * blockDim.y;
		pos[1] = threadIdx.x + blockIdx.x * blockDim.x;
		incr[1] = gridDim.x * blockDim.x;
	}
	else if (workDims = 3) {
		pos[0] = threadIdx.z + blockIdx.z * blockDim.z;
		incr[0] = gridDim.z * blockDim.z;
		pos[1] = threadIdx.y + blockIdx.y * blockDim.y;
		incr[1] = gridDim.y * blockDim.y;
		pos[2] = threadIdx.x + blockIdx.x * blockDim.x;
		incr[2] = gridDim.x * blockDim.x;
	}
	else {

	}

	for (dim_t d = 0; d < workDims; d++) {

	}

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