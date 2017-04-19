#pragma once

#include <cstdint>

#ifdef __NVCC__
#include <assert.h>
#define _dev_ __host__ __device__
#define _devonly_ __device__ 
#else
#include <cassert>
#define _dev_
#endif

typedef int32_t dim_t;
typedef int64_t idx_t;

template<dim_t TNDims>
struct Idxs
{
	idx_t Data[TNDims];

	inline _dev_ dim_t NDims() const { 
		return TNDims; 
	}
	inline _dev_ idx_t &operator[] (dim_t dim) {
		assert(0 <= dim && dim < TNDims);
		return Data[dim]; 
	}
	inline _dev_ const idx_t &operator[] (dim_t dim) const {
		assert(0 <= dim && dim < TNDims);
		return Data[dim];
	}
};


template<typename T, dim_t TNDims>
struct Tensor
{
	typedef T ElemType;
	typedef Idxs<TNDims> TIdxs;

	T * const Base;
	const idx_t Offset;
	const TIdxs Shape;
	const TIdxs Stride;

	_dev_ Tensor(T *base, idx_t offset, TIdxs shape, TIdxs stride)
		: Base(base), Offset(offset), Shape(shape), Stride(stride)
	{
		for (dim_t d = 0; d < TNDims; d++) {
			assert(0 <= Shape[d]);
		}
	}

	inline _dev_ dim_t NDims() const { return TNDims; }
	inline _dev_ size_t ElemSize() const { return sizeof(T); }

	inline _dev_ idx_t LinearPos(const TIdxs &pos) const {
		idx_t linearPos = Offset;
		for (dim_t d = 0; d < TNDims; d++) {
			assert(0 <= pos[d] && pos[d] < Shape[d]);
			linearPos += pos[d] * Stride[d];
		}
		return linearPos;
	}

	inline _dev_ T &operator[] (const TIdxs &pos) {
		return Base[LinearPos(pos)]; 
	}

	inline _dev_ T const &operator[] (const TIdxs &pos) const {
		return Base[LinearPos(pos)];
	}

};






