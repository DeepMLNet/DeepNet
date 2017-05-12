#pragma once

#include "Common.cuh"
#include "Idxs.cuh"
#include "Tensor.cuh"
#include "Work.cuh"


template<dim_t TNDims, dim_t TNIdxs>
struct IdxTensors {
	Tensor<idx_t, TNDims> idxs[TNIdxs];
	bool specified[TNIdxs];
};

template<dim_t TNDims>
struct IdxTensors<TNDims, (dim_t)0> {
	Tensor<idx_t, TNDims> idxs[1];
	bool specified[1];
};


template<typename T, dim_t TTrgtDims, dim_t TSrcDims>
struct GatherFn {
	Tensor<T, TTrgtDims> trgt;
	IdxTensors<TTrgtDims, TSrcDims> srcIdxs;
	Tensor<T, TSrcDims> src;

	_dev_ void operator() (Idxs<TTrgtDims> &trgtPos) {
		Idxs<TSrcDims> srcPos;
		for (dim_t d = 0; d < TSrcDims; d++) {
			if (srcIdxs.specified[d])
				srcPos[d] = srcIdxs.idxs[d][trgtPos];
			else
				srcPos[d] = trgtPos[d];
		}
		if (src.PosValid(srcPos))
			trgt[trgtPos] = src[srcPos];
		else {
			printf("Gather: invalid source position ");
			PrintIdxs(srcPos);
			printf(" for shape ");
			PrintIdxs(src.Shape());
			printf(".\n");
			__threadfence();
			__syncthreads();
			asm("trap;");
		}
	}
};

template<typename T, dim_t TTrgtDims, dim_t TSrcDims> _dev_
void Gather(Tensor<T, TTrgtDims> &trgt, 
	        IdxTensors<TTrgtDims, TSrcDims> &srcIdxs, 
			Tensor<T, TSrcDims> &src) {
	GatherFn<T, TTrgtDims, TSrcDims> workFn = {trgt, srcIdxs, src};
	PerformWork(trgt.Shape(), workFn);
};




