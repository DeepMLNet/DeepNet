#pragma once

#include "Utils.cuh"
#include "Ops.cuh"


template <typename TDyn, typename TBase, typename TIdx>
_dev TDyn dynamicSubtensor(TBase &base, const TIdx &idx) {
	TDyn dyn;
	dyn.mData = const_cast<typename TDyn::DataType *>(base.data());
	dyn.mOffset = base.offset();
	for (size_t dim = 0; dim < dyn.nDim(); dim++) {
		if (idx[dim])
			dyn.mOffset += base.stride(dim) * (*(idx[dim]));
		dyn.mStride[dim] = base.stride(dim);
	}
	return dyn;
}

template <typename TDyn, typename TBase, typename TIdx>
_dev const TDyn dynamicSubtensor(const TBase &base, const TIdx &idx) {
	return dynamicSubtensor(const_cast<TBase>(base), idx);
}


template <typename TTrgt, typename TBaseSrc, typename TDynSrc, size_t nTrgtIdxs>
_dev void copyFromDynamicSubtensor(TTrgt &trgt, 
								   const TBaseSrc &baseSrc, const Array<size_t *, nTrgtIdxs> &srcIdx)
{
	IdEOp_t copyOp;
	TDynSrc dynSrc = dynamicSubtensor<TDynSrc>(baseSrc, srcIdx);
	TTrgt::elemwise1Ary(copyOp, trgt, dynSrc);
}


template <typename TBaseTrgt, typename TDynTrgt, size_t nTrgtIdxs, typename TSrc>
_dev void copyToDynamicSubtensor(TBaseTrgt &baseTrgt, const Array<size_t *, nTrgtIdxs> &trgtIdx,
								 const TSrc &src)
{
	IdEOp_t copyOp;
	TDynTrgt dynTrgt = dynamicSubtensor<TDynTrgt>(baseTrgt, trgtIdx);
	TDynTrgt::elemwise1Ary(copyOp, dynTrgt, src);
}

