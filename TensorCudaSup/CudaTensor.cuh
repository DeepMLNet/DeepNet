#pragma once

#include <nvfunctional>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Tensor.h"


template<dim_t TWorkDims, dim_t TDim, dim_t TRestDims>
struct FillRestPos {
	static inline _dev_ 
	void Do(const Idxs<TWorkDims> &workSize, Idxs<TWorkDims> &pos, idx_t rest) {
		idx_t size = 1;
		for (dim_t d = TDim + 1; d < TWorkDims - 2; d++)
			size *= workSize[d];
		pos[TDim] = rest / size;
		rest -= pos[TDim] * size;
		FillRestPos<TWorkDims, TDim + 1, TRestDims - 1>::Do(workSize, pos, rest);
	}
};

template<dim_t TWorkDims, dim_t TDim>
struct FillRestPos<TWorkDims, TDim, (dim_t)0> {
	static inline _dev_ 
	void Do(const Idxs<TWorkDims> &workSize, Idxs<TWorkDims> &pos, idx_t rest) {
	}
};


template<typename TWorkFn, dim_t TWorkDims> _dev_
void PerformWork(TWorkFn &workFn, const Idxs<TWorkDims> &workSize) {
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
			FillRestPos<TWorkDims, 0, TWorkDims-2>::Do(workSize, pos, rest);
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


template<typename TTrgtT, typename TSrc1T, dim_t TWorkDims>
struct UnaryElemwiseApplyWorkFn {
	nvstd::function<TTrgtT(TSrc1T)> ElemwiseFn;
	Tensor<TTrgtT, TWorkDims> Trgt;
	const Tensor<TSrc1T, TWorkDims> Src1;

	inline _dev_ 
	UnaryElemwiseApplyWorkFn(nvstd::function<TTrgtT(TSrc1T)> elemwiseFn,
	                         Tensor<TTrgtT, TWorkDims> trgt,
	                         Tensor<TSrc1T, TWorkDims> src1)
		: ElemwiseFn(elemwiseFn), Trgt(trgt), Src1(src1) {

	}

	inline _dev_ 
	void operator() (const Idxs<TWorkDims> &pos) {
		Trgt[pos] = ElemwiseFn(Src1[pos]);
	}
};

template<typename TTrgtT, typename TSrc1T, dim_t TWorkDims> _dev_
void PerfomUnaryElemwiseWork(nvstd::function<TTrgtT(TSrc1T)> elemwiseFn,
							 Tensor<TTrgtT, TWorkDims> &trgt,
                             const Tensor<TSrc1T, TWorkDims> &src1) {
	UnaryElemwiseApplyWorkFn<TTrgtT, TSrc1T, TWorkDims> workFn (elemwiseFn, trgt, src1);
	PerformWork(workFn, trgt.Shape);
}

template<typename TTrgtT, typename TSrcT, dim_t TWorkDims> _dev_
void Copy(Tensor<TTrgtT, TWorkDims> &trgt, const Tensor<TSrcT, TWorkDims> &src) {
	PerfomUnaryElemwiseWork<TTrgtT, TSrcT, TWorkDims> ([](TSrcT x) { return x; }, 
		                                               trgt, src);
};




