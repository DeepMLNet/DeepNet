#pragma once

#include <cstdint>
#include <cassert>

#ifdef TENSORCUDASUP_EXPORTS
#define API __declspec(dllexport)
#else
#define API __declspec(dllimport)
#endif



typedef int32_t dim_t;
typedef int64_t idx_t;

template<dim_t TNDims>
struct Idxs
{
	idx_t Data[TNDims];

	inline dim_t NDims() { return TNDims; } const
	inline idx_t &operator[] (dim_t dim) { 
		assert(0 <= dim && dim < TNDims);
		return Data[dim]; 
	}
	inline const idx_t &operator[] (dim_t dim) const {
		assert(0 <= dim && dim < TNDims);
		return Data[dim];
	}
};


template<typename T, dim_t TNDims>
struct Tensor
{
	typedef T ElemType;
	typedef Idxs<TNDims> TIdxs;

	T *Base;
	idx_t Offset;
	TIdxs Shape;
	TIdxs Stride;


	Tensor(T *base, idx_t offset, TIdxs shape, TIdxs stride) 
		: Base(base), Offset(offset), Shape(shape), Stride(stride)
	{
		for (dim_t d = 0; d < TNDims; d++) {
			assert(0 <= Shape[d]);
		}
	}

	inline dim_t NDims() { return TNDims; }
	inline size_t ElemSize() { return sizeof(T); }

	inline idx_t LinearPos(const TIdxs &pos) const {
		idx_t linearPos = Offset;
		for (dim_t d = 0; d < TNDims; d++) {
			assert(0 <= pos[d] && pos[d] < Shape[d]);
			linearPos += pos[d] * Stride[d];
		}
		return linearPos;
	}

	inline T &operator[] (const TIdxs &pos) { return Base[LinearPos(pos)]; }

};






