#pragma once

#include "Common.cuh"
#include "Idxs.cuh"


template<typename T, dim_t TNDims>
struct Tensor {
	typedef T ElemType;
	typedef Idxs<TNDims> TIdxs;

	T * const _Base;
	const idx_t _Offset;
	const TIdxs _Shape;
	const TIdxs _Stride;

	_dev_ Tensor(T *base, idx_t offset, TIdxs shape, TIdxs stride)
		: _Base(base), _Offset(offset), _Shape(shape), _Stride(stride)
	{
		for (dim_t d = 0; d < TNDims; d++) {
			assert(0 <= _Shape[d]);
		}
	}

	_dev_ T *Base() const { return _Base; }
	_dev_ idx_t Offset() const { return _Offset; }
	_dev_ const TIdxs &Shape() const { return _Shape; }
	_dev_ const TIdxs &Stride() const { return _Stride; }
	_dev_ dim_t NDims() const { return TNDims; }
	_dev_ size_t ElemSize() const { return sizeof(T); }

	_dev_ idx_t LinearPos(const TIdxs &pos) const {
		idx_t linearPos = Offset();
		for (dim_t d = 0; d < TNDims; d++) {
			assert(0 <= pos[d] && pos[d] < Shape()[d]);
			linearPos += pos[d] * Stride()[d];
		}
		return linearPos;
	}

	_dev_ T &operator[] (const TIdxs &pos) {
		return Base()[LinearPos(pos)]; 
	}

	_dev_ T const &operator[] (const TIdxs &pos) const {
		return Base()[LinearPos(pos)];
	}

	_dev_ idx_t Size() const {
		idx_t size = 1;
		for (dim_t d = 0; d < TNDims; d++) 
			size *= Shape()[d];
		return size;
	}
};




