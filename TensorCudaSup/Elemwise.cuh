#pragma once

#include "Common.cuh"
#include "Idxs.cuh"
#include "Tensor.cuh"
#include "Work.cuh"


// =============================================================================================
// no-ary operations (no inputs)
// =============================================================================================

template<typename T, dim_t TDims>
struct FillConstFn {
	T value;
	Tensor<T, TDims> trgt;
	_dev_ void operator() (Idxs<TDims> &pos) {
		trgt[pos] = value;
	}
};

template<typename T, dim_t TDims> _dev_
void FillConst(T value, Tensor<T, TDims> &trgt) {
	FillConstFn<T, TDims> workFn = {value, trgt};
	PerformWork(trgt.Shape(), workFn);
};



// =============================================================================================
// unary operations (one input)
// =============================================================================================

#define UNARY_OP(Name, op)  \
	template<typename T, dim_t TDims> \
	struct Name##Fn { \
		Tensor<T, TDims> trgt; \
		Tensor<T, TDims> src; \
		_dev_ void operator() (Idxs<TDims> &pos) { \
			trgt[pos] = op(src[pos]); \
		} \
	}; \
	template<typename T, dim_t TDims> _dev_ \
	void Name(Tensor<T, TDims> &trgt, const Tensor<T, TDims> &src) { \
		Name##Fn<T, TDims> workFn = {trgt, src}; \
		PerformWork(trgt.Shape(), workFn); \
	};

#define FLOAT_DOUBLE_OVERLOAD(func) \
	_dev_ double ol_##func(double x) { return func(x); } \
	_dev_ float  ol_##func(float  x) { return func##f(x); }

template<typename T> T _dev_ id(T x) { return x; };
template<typename T> T _dev_ unaryPlus(T x)  { return +x; };
template<typename T> T _dev_ unaryMinus(T x) { return -x; };

template<typename T>
T _dev_ sgn(T x) {
	if (x == (T)0)
		return 0;
	else if (x < (T)0)
		return -1;
	else
		return 1;
}

FLOAT_DOUBLE_OVERLOAD(fabs);
_dev_ int8_t   ol_fabs(int8_t x)   { return x < 0 ? -x : x; }
_dev_ int16_t  ol_fabs(int16_t x)  { return x < 0 ? -x : x; }
_dev_ int32_t  ol_fabs(int32_t x)  { return x < 0 ? -x : x; }
_dev_ int64_t  ol_fabs(int64_t x)  { return x < 0 ? -x : x; }
_dev_ uint8_t  ol_fabs(uint8_t x)  { return x; }
_dev_ uint16_t ol_fabs(uint16_t x) { return x; }
_dev_ uint32_t ol_fabs(uint32_t x) { return x; }
_dev_ uint64_t ol_fabs(uint64_t x) { return x; }

FLOAT_DOUBLE_OVERLOAD(log);
FLOAT_DOUBLE_OVERLOAD(log10);
FLOAT_DOUBLE_OVERLOAD(exp);
FLOAT_DOUBLE_OVERLOAD(sin);
FLOAT_DOUBLE_OVERLOAD(cos);
FLOAT_DOUBLE_OVERLOAD(tan);
FLOAT_DOUBLE_OVERLOAD(asin);
FLOAT_DOUBLE_OVERLOAD(acos);
FLOAT_DOUBLE_OVERLOAD(atan);
FLOAT_DOUBLE_OVERLOAD(sinh);
FLOAT_DOUBLE_OVERLOAD(cosh);
FLOAT_DOUBLE_OVERLOAD(tanh);
FLOAT_DOUBLE_OVERLOAD(sqrt);
FLOAT_DOUBLE_OVERLOAD(ceil);
FLOAT_DOUBLE_OVERLOAD(floor);
FLOAT_DOUBLE_OVERLOAD(round);
FLOAT_DOUBLE_OVERLOAD(trunc);

_dev_ bool negate (bool x) { return !x; }


UNARY_OP(Copy,			id);
UNARY_OP(UnaryPlus,		unaryPlus);
UNARY_OP(UnaryMinus,	unaryMinus);
UNARY_OP(Abs,			ol_fabs);
UNARY_OP(Sgn,			sgn);
UNARY_OP(Log,			ol_log);
UNARY_OP(Log10,			ol_log10);
UNARY_OP(Exp,			ol_exp);
UNARY_OP(Sin,			ol_sin);
UNARY_OP(Cos,			ol_cos);
UNARY_OP(Tan,			ol_tan);
UNARY_OP(Asin,			ol_asin);
UNARY_OP(Acos,			ol_acos);
UNARY_OP(Atan,			ol_atan);
UNARY_OP(Sinh,			ol_sinh);
UNARY_OP(Cosh,			ol_cosh);
UNARY_OP(Tanh,			ol_tanh);
UNARY_OP(Sqrt,			ol_sqrt);
UNARY_OP(Ceiling,		ol_ceil);
UNARY_OP(Floor,			ol_floor);
UNARY_OP(Round,			ol_round);
UNARY_OP(Truncate,		ol_trunc);
UNARY_OP(Negate,	    negate); 



// =============================================================================================
// binary operations (two inputs)
// =============================================================================================

#define BINARY_OP(Name, op)  \
	template<typename T, dim_t TDims> \
	struct Name##Fn { \
		Tensor<T, TDims> trgt; \
		Tensor<T, TDims> src1; \
		Tensor<T, TDims> src2; \
		_dev_ void operator() (Idxs<TDims> &pos) { \
			trgt[pos] = op(src1[pos], src2[pos]); \
		} \
	}; \
	template<typename T, dim_t TDims> _dev_ \
	void Name(Tensor<T, TDims> &trgt, const Tensor<T, TDims> &src1, const Tensor<T, TDims> &src2) { \
		Name##Fn<T, TDims> workFn = {trgt, src1, src2}; \
		PerformWork(trgt.Shape(), workFn); \
	};


template <typename T> _dev_ T add (T a, T b)      { return a + b; };
template <typename T> _dev_ T subtract (T a, T b) { return a - b; };
template <typename T> _dev_ T multiply (T a, T b) { return a * b; };
template <typename T> _dev_ T divide (T a, T b)   { return a / b; };
template <typename T> _dev_ T min(T a, T b)       { return a < b ? a : b; };
template <typename T> _dev_ T max(T a, T b)       { return a > b ? a : b; };

_dev_ double ol_pow (double a, double b) { return pow(a, b); };
_dev_ float  ol_pow (float a, float b)   { return powf(a, b); };

_dev_ double   ol_fmod (double   a, double   b) { return fmod(a, b); };
_dev_ float    ol_fmod (float    a, float    b) { return fmodf(a, b); };
_dev_ int8_t   ol_fmod (int8_t   a, int8_t   b) { return a % b; }
_dev_ int16_t  ol_fmod (int16_t  a, int16_t  b) { return a % b; }
_dev_ int32_t  ol_fmod (int32_t  a, int32_t  b) { return a % b; }
_dev_ int64_t  ol_fmod (int64_t  a, int64_t  b) { return a % b; }
_dev_ uint8_t  ol_fmod (uint8_t  a, uint8_t  b) { return a % b; }
_dev_ uint16_t ol_fmod (uint16_t a, uint16_t b) { return a % b; }
_dev_ uint32_t ol_fmod (uint32_t a, uint32_t b) { return a % b; }
_dev_ uint64_t ol_fmod (uint64_t a, uint64_t b) { return a % b; }

_dev_ bool and_fn (bool a, bool b) { return a && b; }
_dev_ bool or_fn  (bool a, bool b) { return a || b; }
_dev_ bool xor_fn (bool a, bool b) { return a != b; }

BINARY_OP(Add,			add);	
BINARY_OP(Subtract,		subtract);
BINARY_OP(Multiply,		multiply);
BINARY_OP(Divide,		divide);
BINARY_OP(Modulo,		ol_fmod);
BINARY_OP(Power,		ol_pow);
BINARY_OP(MinElemwise,  min);
BINARY_OP(MaxElemwise,	max);
BINARY_OP(And,          and_fn);
BINARY_OP(Or,           or_fn);
BINARY_OP(Xor,          xor_fn);


// =============================================================================================
// unary comparisions
// =============================================================================================

#define UNARY_COMPARISON(Name, op)  \
	template<typename T, dim_t TDims> \
	struct Name##Fn { \
		Tensor<bool, TDims> trgt; \
		Tensor<T, TDims> src1; \
		_dev_ void operator() (Idxs<TDims> &pos) { \
			trgt[pos] = op(src1[pos]); \
		} \
	}; \
	template<typename T, dim_t TDims> _dev_ \
	void Name(Tensor<bool, TDims> &trgt, const Tensor<T, TDims> &src1) { \
		Name##Fn<T, TDims> workFn = {trgt, src1}; \
		PerformWork(trgt.Shape(), workFn); \
	};

_dev_ bool ol_isfinite (double   x) { return isfinite(x); }
_dev_ bool ol_isfinite (float    x) { return isfinite(x); }
_dev_ bool ol_isfinite (int8_t   x) { return true; }
_dev_ bool ol_isfinite (int16_t  x) { return true; }
_dev_ bool ol_isfinite (int32_t  x) { return true; }
_dev_ bool ol_isfinite (int64_t  x) { return true; }
_dev_ bool ol_isfinite (uint8_t  x) { return true; }
_dev_ bool ol_isfinite (uint16_t x) { return true; }
_dev_ bool ol_isfinite (uint32_t x) { return true; }
_dev_ bool ol_isfinite (uint64_t x) { return true; }

UNARY_COMPARISON(IsFinite, ol_isfinite);

// =============================================================================================
// binary comparisions
// =============================================================================================

#define BINARY_COMPARISON(Name, op)  \
	template<typename T, dim_t TDims> \
	struct Name##Fn { \
		Tensor<bool, TDims> trgt; \
		Tensor<T, TDims> src1; \
		Tensor<T, TDims> src2; \
		_dev_ void operator() (Idxs<TDims> &pos) { \
			trgt[pos] = op(src1[pos], src2[pos]); \
		} \
	}; \
	template<typename T, dim_t TDims> _dev_ \
	void Name(Tensor<bool, TDims> &trgt, const Tensor<T, TDims> &src1, const Tensor<T, TDims> &src2) { \
		Name##Fn<T, TDims> workFn = {trgt, src1, src2}; \
		PerformWork(trgt.Shape(), workFn); \
	};


template <typename T> _dev_ bool equal (T a, T b)				{ return a == b; };
template <typename T> _dev_ bool not_equal (T a, T b)			{ return a != b; };
template <typename T> _dev_ bool less (T a, T b)				{ return a <  b; };
template <typename T> _dev_ bool less_or_equal (T a, T b)		{ return a <= b; };
template <typename T> _dev_ bool greater (T a, T b)				{ return a >  b; };
template <typename T> _dev_ bool greater_or_equal (T a, T b)	{ return a >= b; };

BINARY_COMPARISON(Equal,			equal);
BINARY_COMPARISON(NotEqual,			not_equal);
BINARY_COMPARISON(Less,				less);
BINARY_COMPARISON(LessOrEqual,		less_or_equal);
BINARY_COMPARISON(Greater,			greater);
BINARY_COMPARISON(GreaterOrEqual,	greater_or_equal);





template<typename T, dim_t TTrgtDims, dim_t TSrcDims> _dev_
void CopyHeterogenous(Tensor<T, TTrgtDims> &trgt, const Tensor<T, TSrcDims> &src) {
	constexpr dim_t trgtDims = TTrgtDims;
	constexpr dim_t srcDims = TSrcDims;
	auto workFn = [trgtDims, srcDims, &trgt, &src](Idxs<1> &pos) { 
		Idxs<trgtDims> trgtPos = Idxs<trgtDims>::FromLinearPos(trgt.Shape(), pos[0]);
		Idxs<srcDims> srcPos = Idxs<srcDims>::FromLinearPos(src.Shape(), pos[0]);
		trgt[trgtPos] = src[srcPos]; 
	};

	assert(trgt.Size() == src.Size());
	Idxs<1> workSize {{trgt.Size()}};
	PerformWork(workSize, workFn);
};

