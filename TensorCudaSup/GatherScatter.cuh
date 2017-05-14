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


// =============================================================================================
// gather and scatter
// =============================================================================================

template<typename T, dim_t TTrgtDims, dim_t TSrcDims>
struct GatherFn {
	Tensor<T, TTrgtDims> trgt;
	IdxTensors<TTrgtDims, TSrcDims> srcIdxs;
	Tensor<T, TSrcDims> src;
	int *error;
	bool trapOnError;

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
			if (atomicExch(error, 1) != 1) {
				printf("Gather: invalid source position ");
				PrintIdxs(srcPos);
				printf(" for shape ");
				PrintIdxs(src.Shape());
				printf(".\n");
				if (trapOnError) {
					__threadfence();
					__syncthreads();
					asm("trap;");
				}
			}
		}
	}
};

template<typename T, dim_t TTrgtDims, dim_t TSrcDims> _dev_
void Gather(Tensor<T, TTrgtDims> &trgt, 
	        IdxTensors<TTrgtDims, TSrcDims> &srcIdxs, 
			Tensor<T, TSrcDims> &src,
			ptr_t error, bool trapOnError) {
	GatherFn<T, TTrgtDims, TSrcDims> workFn = {trgt, srcIdxs, src, (int *)error, trapOnError};
	PerformWork(trgt.Shape(), workFn);
};



template<typename T, dim_t TTrgtDims, dim_t TSrcDims>
struct ScatterFn {
	Tensor<T, TTrgtDims> trgt;
	IdxTensors<TSrcDims, TTrgtDims> trgtIdxs;
	Tensor<T, TSrcDims> src;
	int *error;
	bool trapOnError;

	_dev_ void operator() (Idxs<TSrcDims> &srcPos) {
		Idxs<TTrgtDims> trgtPos;
		for (dim_t d = 0; d < TTrgtDims; d++) {
			if (trgtIdxs.specified[d])
				trgtPos[d] = trgtIdxs.idxs[d][srcPos];
			else
				trgtPos[d] = srcPos[d];
		}
		if (trgt.PosValid(trgtPos))
			atomicAdd(&trgt[trgtPos], src[srcPos]);
		else {
			if (atomicExch(error, 1) != 1) {
				printf("Scatter: invalid target position ");
				PrintIdxs(trgtPos);
				printf(" for shape ");
				PrintIdxs(trgt.Shape());
				printf(".\n");
				if (trapOnError) {
					__threadfence();
					__syncthreads();
					asm("trap;");
				}
			}
		}
	}
};

template<typename T, dim_t TTrgtDims, dim_t TSrcDims> _dev_
void Scatter(Tensor<T, TTrgtDims> &trgt, 
			 IdxTensors<TSrcDims, TTrgtDims> &trgtIdxs, 
			 Tensor<T, TSrcDims> &src,
			 ptr_t error, bool trapOnError) {
	ScatterFn<T, TTrgtDims, TSrcDims> workFn = {trgt, trgtIdxs, src, (int *)error, trapOnError};
	PerformWork(src.Shape(), workFn);
};



