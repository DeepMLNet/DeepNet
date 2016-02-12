#pragma once

#include "Utils.cuh"
#include "ThrustInterface.cuh"


/// Sums all elements in src and stores them into the first element of trgt.
template <typename TTarget, typename TSrc>
void sum(TTarget &trgt, TSrc &src) {
	// linear indexer
	typedef thrust::counting_iterator<size_t> LinearIndexIteratorT;
	LinearIndexIteratorT linearIdxer(0);

	// iterator over input 
	typedef LinearIndexToElement<TSrc, float> SrcTransformT;
	typedef thrust::transform_iterator<SrcTransformT, LinearIndexIteratorT> SrcElementIteratorT;
	SrcElementIteratorT srcElems(linearIdxer, SrcTransformT(src));

	// constant key iterator
	thrust::constant_iterator<int> cnstKey(0);

	// iterator over output
	typedef LinearIndexToElement<TTarget, float> TrgtTransformT;
	typedef thrust::transform_iterator<TrgtTransformT, LinearIndexIteratorT> TrgtElementIteratorT;
	TrgtElementIteratorT trgtElems(linearIdxer, TrgtTransformT(trgt));

	thrust::device_vector<float> srcIn(5);
	thrust::device_vector<float> keyOut(5);
	thrust::device_vector<float> valOut(5);

	// perform sum
	thrust::reduce_by_key(cnstKey,
						  cnstKey + srcIn.size(),
					      //srcIn.begin(), 
						  srcElems,
						  keyOut.begin(),
						  valOut.begin());

	//thrust::reduce_by_key(cnstKey,
	//					  cnstKey + src.size(),
	//				      srcElems,
	//					  thrust::make_discard_iterator(),
	//					  trgtElems);
					
}


/// converts a linear index into a key that is constant for all elements which have
/// all positions equal except the last one
template <typename TArrayND>
struct LinearIndexToSumAxisKey : public thrust::unary_function<size_t, size_t> {
	TArrayND arrayND;

	LinearIndexToSumAxisKey(TArrayND ary) : arrayND(ary)  { }

	_dev size_t operator() (size_t linearIdx) {
		return arrayND.idxToPosWithLastDimSetToZero(linearIdx).toIdx();
	}
};


/// Sums over the last axis in src and stores the partial sums into trgt.
template <typename TTarget, typename TSrc>
void sumLastAxis(TTarget &trgt, const TSrc &src) {
	// linear indexer
	typedef thrust::counting_iterator<size_t> LinearIndexIteratorT;
	LinearIndexIteratorT linearIdxer(0);

	// iterator over input 
	typedef LinearIndexToElement<const TSrc, const float *> SrcTransformT;
	typedef thrust::transform_iterator<SrcTransformT, LinearIndexIteratorT> SrcElementIteratorT;
	SrcElementIteratorT srcElems(linearIdxer, SrcTransformT(src));

	// key iterator that assigns same key to elements that have same position without 
	// taking into account the last dimension
	typedef LinearIndexToSumAxisKey<const TSrc> IdxToKeyT;
	typedef thrust::transform_iterator<IdxToKeyT, LinearIndexIteratorT> KeyIteratorT;
	KeyIteratorT sumKeys(linearIdxer, KeyIteratorT(src));

	// iterator over output
	typedef LinearIndexToElement<TTarget, float *> TrgtTransformT;
	typedef thrust::transform_iterator<TrgtTransformT, LinearIndexIteratorT> TrgtElementIteratorT;
	TrgtElementIteratorT trgtElems(linearIdxer, TrgtTransformT(trgt));

	// perform sum by axis
	thrust::reduce_by_key(sumKeys,
						  sumKeys + src.size(),
						  srcElems,
						  thrust::make_discard_iterator(),
		                  trgtElems);
}


