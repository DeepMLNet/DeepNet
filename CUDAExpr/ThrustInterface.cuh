#pragma once

#pragma warning (push)
#pragma warning (disable : 4267)
#include <thrust/device_vector.h>
#pragma warning (pop)

#include "Utils.cuh"

// what is necessary for Thrust support?
// let us try to implement summation of an NDArray

// so our transform iterators must do the following:
//   - convert continous indexing into position
//   - convert position into memory address


/// converts a linear index into a pointer to a ArrayND element
template <typename TArrayND, typename TValue>
struct LinearIndexToElement : public thrust::unary_function<size_t, thrust::device_ptr<TValue>> {
	TArrayND arrayND;

	LinearIndexToElement(TArrayND ary) : arrayND(ary)  { }

	_dev thrust::device_ptr<TValue> operator() (size_t linearIdx) {
		return thrust::device_pointer_cast(&arrayND.element(arrayND.idxToPos(linearIdx)));
	}
};


/// Sums all elements in src and stores them into the first element of trgt.
template <typename TTarget, typename TSrc>
void sum(TTarget &trgt, const TSrc &src) {
	// linear indexer
	typedef thrust::counting_iterator<size_t> LinearIndexIteratorT;
	LinearIndexIteratorT linearIdxer(0);

	// iterator over input 
	typedef LinearIndexToElement<const TSrc, const float *> SrcTransformT;
	typedef thrust::transform_iterator<SrcTransformT, LinearIndexIteratorT> SrcElementIteratorT;
	SrcElementIteratorT srcElems(linearIdxer, SrcTransformT(src));

	// constant key iterator
	thrust::constant_iterator<int> cnstKey(0);

	// iterator over output
	typedef LinearIndexToElement<TTarget, float *> TrgtTransformT;
	typedef thrust::transform_iterator<TrgtTransformT, LinearIndexIteratorT> TrgtElementIteratorT;
	TrgtElementIteratorT trgtElems(linearIdxer, TrgtTransformT(trgt));

	// perform sum
	thrust::reduce_by_key(cnstKey,
						  cnstKey + src.size(),
					      srcElems,
						  thrust::make_discard_iterator(),
						  trgtElems);
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

