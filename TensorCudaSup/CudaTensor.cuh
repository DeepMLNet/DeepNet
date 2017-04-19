#pragma once

#include "Common.cuh"
#include "Idxs.cuh"
#include "Tensor.cuh"
#include "Work.cuh"



template<typename T, dim_t TDims> _dev_
void Copy(Tensor<T, TDims> &trgt, const Tensor<T, TDims> &src) {
	WorkFn<TDims> workFn = [&trgt, &src](auto pos) { trgt[pos] = src[pos]; };
	PerformWork(trgt.Shape(), workFn);
};


template<typename T, dim_t TTrgtDims, dim_t TSrcDims> _dev_
void CopyHeterogenous(Tensor<T, TTrgtDims> &trgt, const Tensor<T, TSrcDims> &src) {
	constexpr dim_t trgtDims = TTrgtDims;
	constexpr dim_t srcDims = TSrcDims;
	WorkFn<1> workFn = [trgtDims, srcDims, &trgt, &src](auto pos) { 
		auto trgtPos = Idxs<trgtDims>::FromLinearPos(trgt.Shape(), pos[0]);
		auto srcPos = Idxs<srcDims>::FromLinearPos(src.Shape(), pos[0]);
		trgt[trgtPos] = src[srcPos]; 
	};

	assert(trgt.Size() == src.Size());
	Idxs<1> workSize {{trgt.Size()}};
	PerformWork(workSize, workFn);
};


