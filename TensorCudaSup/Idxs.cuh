#pragma once

#include "Common.cuh"

template<dim_t TNDims>
struct Idxs {
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

	inline _dev_ idx_t ToLinearPos(const Idxs<TNDims> &shape) const {
		idx_t incr[TNDims];
		incr[TNDims-1] = 1;
		for (dim_t d = TNDims-2; d >= 0; d--)
			incr[d] = incr[d+1] * shape[d+1];

		idx_t linearPos = 0;
		for (dim_t d = 0; d < TNDims; d++) {
			linearPos += (*this)[d] * incr[d];
		}
		return linearPos;
	}

	inline _dev_ static Idxs<TNDims> FromLinearPos(const Idxs<TNDims> &shape, idx_t linearPos) {
		idx_t incr[TNDims];
		incr[TNDims-1] = 1;
		for (dim_t d = TNDims-2; d >= 0; d--)
			incr[d] = incr[d+1] * shape[d+1];

		Idxs<TNDims> idxs;
		for (dim_t d = 0; d < TNDims; d++) {
			idxs[d] = linearPos / incr[d];
			linearPos -= idxs[d] * incr[d];
		}
		return idxs;
	}
};

