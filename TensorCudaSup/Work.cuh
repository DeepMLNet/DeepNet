#pragma once

#include "Common.cuh"
#include "Idxs.cuh"


template<dim_t TWorkDims>
using WorkFn = nvstd::function<void(const Idxs<TWorkDims> &)>;

template<dim_t TWorkDims, dim_t TDim, dim_t TRestDims>
struct FillRestPosT {
	static inline _dev_ 
		void Do(const Idxs<TWorkDims> &workSize, Idxs<TWorkDims> &pos, idx_t rest) {
		idx_t size = 1;
		for (dim_t d = TDim + 1; d < TWorkDims - 2; d++)
			size *= workSize[d];
		pos[TDim] = rest / size;
		FillRestPosT<TWorkDims, TDim+1, TRestDims-1>::Do(workSize, pos, rest - pos[TDim] * size);
	}
};

template<dim_t TWorkDims, dim_t TDim>
struct FillRestPosT<TWorkDims, TDim, (dim_t)0> {
	static inline _dev_ 
		void Do(const Idxs<TWorkDims> &workSize, Idxs<TWorkDims> &pos, idx_t rest) { }
};


template<dim_t TWorkDims> inline _dev_
void PerformWork(const Idxs<TWorkDims> &workSize, const WorkFn<TWorkDims> &workFn) {
	// TWorkDims >= 4
	Idxs<TWorkDims> pos;
	idx_t restWork = 1;
	for (dim_t d = 0; d < TWorkDims - 2; d++)
		restWork *= workSize[d];
	for (idx_t rest = threadIdx.z + blockIdx.z * blockDim.z;
		rest < restWork;
		rest += gridDim.z * blockDim.z) {
		FillRestPosT<TWorkDims, 0, TWorkDims-2>::Do(workSize, pos, rest);
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
};

template<> inline _dev_
void PerformWork<(dim_t)0>(const Idxs<0> &workSize, const WorkFn<0> &workFn) {
	Idxs<0> pos;
	workFn(pos);
};

template<> inline _dev_
void PerformWork<(dim_t)1>(const Idxs<1> &workSize, const WorkFn<1> &workFn) {
	Idxs<1> pos;
	for (pos[0] = threadIdx.x + blockIdx.x * blockDim.x;
		pos[0] < workSize[0];
		pos[0] += gridDim.x * blockDim.x) {
		workFn(pos);
	}
};

template<> inline _dev_
void PerformWork<(dim_t)2>(const Idxs<2> &workSize, const WorkFn<2> &workFn) {
	Idxs<2> pos;
	for (pos[0] = threadIdx.y + blockIdx.y * blockDim.y;
		pos[0] < workSize[0];
		pos[0] += gridDim.y * blockDim.y) {
		for (pos[1] = threadIdx.x + blockIdx.x * blockDim.x;
			pos[1] < workSize[1];
			pos[1] += gridDim.x * blockDim.x) {
			workFn(pos);
		}
	}
};

template<> inline _dev_
void PerformWork<(dim_t)3>(const Idxs<3> &workSize, const WorkFn<3> &workFn) {
	Idxs<3> pos;
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
};



