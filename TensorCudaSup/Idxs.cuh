#pragma once

#include "Common.cuh"

template<dim_t TNDims>
struct Idxs {
	idx_t Data[TNDims];

	_dev_ dim_t NDims() const { 
		return TNDims; 
	}

	_dev_ idx_t &operator[] (dim_t dim) {
		assert(0 <= dim && dim < TNDims);
		return Data[dim]; 
	}

	_dev_ const idx_t &operator[] (dim_t dim) const {
		assert(0 <= dim && dim < TNDims);
		return Data[dim];
	}

	_dev_ idx_t ToLinearPos(const Idxs<TNDims> &shape) const {
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

	_dev_ static Idxs<TNDims> FromLinearPos(const Idxs<TNDims> &shape, idx_t linearPos) {
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


template<>
struct Idxs<(dim_t)0> {
	idx_t Dummy;

	_dev_ dim_t NDims() const { 
		return 0; 
	}

	_dev_ idx_t &operator[] (dim_t dim) {
		assert(false);
		return Dummy; 
	}

	_dev_ const idx_t &operator[] (dim_t dim) const {
		assert(false);
		return Dummy; 
	}

	_dev_ idx_t ToLinearPos(const Idxs<(dim_t)0> &shape) const {
		return (idx_t)0;
	}

	_dev_ static Idxs<(dim_t)0> FromLinearPos(const Idxs<(dim_t)0> &shape, idx_t linearPos) {
		Idxs<(dim_t)0> idxs;
		idxs.Dummy = (idx_t)0;
		return idxs;
	}
};


_dev_ void PrintIdxs(const Idxs<0> &pos) { printf("[]"); }
_dev_ void PrintIdxs(const Idxs<1> &pos) { printf("[%ld]", pos[0]); }
_dev_ void PrintIdxs(const Idxs<2> &pos) { printf("[%ld; %ld]", pos[0], pos[1]); }
_dev_ void PrintIdxs(const Idxs<3> &pos) { printf("[%ld; %ld; %ld]", pos[0], pos[1], pos[2]); }
_dev_ void PrintIdxs(const Idxs<4> &pos) { printf("[%ld; %ld; %ld; %ld]", pos[0], pos[1], pos[2], pos[3]); }
_dev_ void PrintIdxs(const Idxs<5> &pos) { printf("[%ld; %ld; %ld; %ld; %ld]", pos[0], pos[1], pos[2], pos[3], pos[4]); }
_dev_ void PrintIdxs(const Idxs<6> &pos) { printf("[%ld; %ld; %ld; %ld; %ld; %ld]", pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]); }
_dev_ void PrintIdxs(const Idxs<7> &pos) { printf("[%ld; %ld; %ld; %ld; %ld; %ld; %ld]", pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]); }
_dev_ void PrintIdxs(const Idxs<8> &pos) { printf("[%ld; %ld; %ld; %ld; %ld; %ld; %ld; %ld]", pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6], pos[7]); }
_dev_ void PrintIdxs(const Idxs<9> &pos) { printf("[%ld; %ld; %ld; %ld; %ld; %ld; %ld; %ld; %ld]", pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6], pos[7], pos[8]); }


