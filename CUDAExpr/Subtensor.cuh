#pragma once

#include "Utils.cuh"
#include "Ops.cuh"




template <typename TElemwiseOp, typename TTarget, typename TSrc0>
struct TElemwise1Ary {
	typedef void type(const TElemwiseOp &op, TTarget &trgt, const TSrc0 &src0);
};


template <typename TDyn, typename TBase, typename TIdx>
_dev TDyn dynamicSubtensor(const TBase &base, const TIdx &idx) {
	TDyn dyn;
	dyn.mData = base.data();
	dyn.mOffset = base.offset();
	for (size_t dim = 0; dim < dyn.nDim(); dim++) {
		if (idx[dim])
			dyn.mOffset += base.stride(dim) * (*(idx[dim]));
		dyn.mStride[dim] = base.stride(dim);
	}
	return dyn;
}

template <typename TTrgt, typename TBaseSrc, typename TDynSrc, size_t nDims,
		  TElemwise1Ary<IdEOp_t, TTrgt, TDynSrc>::type copyFun>
_dev void copyFromDynamicSubtensor(TTrgt &trgt, 
								   const TBaseSrc &baseSrc, const Array<size_t, nDims> &srcIdx)
{
	IdEOp_t copyOp;
	TDynSrc dynSrc = dynamicSubtensor(baseSrc, srcIdx);
	copyFun(copyOp, trgt, dynSrc);
}

template <typename TBaseTrgt, typename TDynTrgt, size_t nDims, typename TSrc,
		  TElemwise1Ary<IdEOp_t, TDynTrgt, TSrc>::type copyFun>
_dev void copyToDynamicSubtensor(TBaseTrgt &baseTrgt, const Array<size_t, nDims> &trgtIdx,
								 const TSrc &src)
{
	IdEOp_t copyOp;
	TDynTrgt dynTrgt = dynamicSubtensor(baseTrgt, trgtIdx);
	copyFun(copyOp, dynTrgt, src);
}


