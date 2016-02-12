#pragma once

#pragma warning (push)
#pragma warning (disable : 4267)
#include <thrust/device_vector.h>
#pragma warning (pop)

#include "Utils.cuh"


// TODO: change to permutation iterator

/// converts a linear index into a pointer to a ArrayND element
template <typename TArrayND, typename TValue>
struct LinearIndexToElement : public thrust::unary_function<size_t, thrust::device_ptr<TValue>> {
	TArrayND arrayND;

	LinearIndexToElement(TArrayND ary) : arrayND(ary)  { }

	_dev thrust::device_ptr<TValue> operator() (size_t linearIdx) {
		return thrust::device_pointer_cast(0);
		//return thrust::device_pointer_cast(&arrayND.element(arrayND.idxToPos(linearIdx)));
	}
};

