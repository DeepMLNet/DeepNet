#pragma once

#include <cstdio>
#include <iostream>
#include <cuda.h>

#include "Utils.cuh"
#include "ThrustInterface.cuh"


/// Sums all elements in src and stores them into the first element of trgt.
template <typename TTarget, typename TSrc>
void sum(TTarget &trgt, TSrc &src, 
	     CUstream &stream, char *tmp_buffer, idx_t tmp_buffer_size) {
	//std::printf("entering sum with trgt=%p and src=%p\n",
	//			trgt.data(), src.data()); 

	buffer_allocator alloc("sum", tmp_buffer, tmp_buffer_size);

	ArrayNDRange<TSrc> srcRange(src);
	ArrayNDRange<TTarget> trgtRange(trgt);

	thrust::constant_iterator<int> cnstKey(0);
	thrust::reduce_by_key(thrust::cuda::par(alloc).on(stream),
						  cnstKey,
						  cnstKey + src.size(),
						  srcRange.begin(),
						  thrust::make_discard_iterator(),
						  trgtRange.begin());				
}

/// converts a linear index into a key that is constant for all elements which have
/// all positions equal except the last one
template <typename TArrayND>
struct LinearIndexToSumAxisKey : public thrust::unary_function<idx_t, idx_t> {
	TArrayND arrayND;

	LinearIndexToSumAxisKey(TArrayND ary) : arrayND(ary)  { }

	_dev idx_t operator() (idx_t linearIdx) {
		return arrayND.linearIdxToPosWithLastDimSetToZero(linearIdx).toLinearIdx(arrayND);
	}
};


/// Sums over the last axis in src and stores the partial sums into trgt.
template <typename TTarget, typename TSrc>
void sumLastAxis(TTarget &trgt, TSrc &src,
	             CUstream &stream, char *tmp_buffer, idx_t tmp_buffer_size) {

	buffer_allocator alloc("sumLastAxis", tmp_buffer, tmp_buffer_size);

	ArrayNDRange<TSrc> srcRange(src);
	ArrayNDRange<TTarget> trgtRange(trgt);

	// key iterator that assigns same key to elements that have same position without 
	// taking into account the last dimension
	typedef thrust::counting_iterator<idx_t> LinearIndexIteratorT;
	typedef LinearIndexToSumAxisKey<TSrc> IdxToKeyT;
	typedef thrust::transform_iterator<IdxToKeyT, LinearIndexIteratorT> KeyIteratorT;
	KeyIteratorT sumKeys(LinearIndexIteratorT(0), IdxToKeyT(src));
	typedef typename ArrayNDRange<TTarget>::iterator iterator;

	// perform sum by axis
	thrust::pair<thrust::discard_iterator<>, iterator> result = 
		thrust::reduce_by_key(thrust::cuda::par(alloc).on(stream),
							  sumKeys,
							  sumKeys + src.size(),
							  srcRange.begin(),
							  thrust::make_discard_iterator(),
							  trgtRange.begin());

	iterator outputLast = result.second;
	if (outputLast > trgtRange.end()) {
		printf("!!!!!! sumLastAxis overflow !!!!!\n");
	}
}


