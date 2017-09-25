#pragma once

#include "Common.cuh"
#include "Idxs.cuh"
#include "Tensor.cuh"
#include "Work.cuh"
#include "Elemwise.cuh"


template<typename TTrgt, dim_t TTrgtDims, typename TSrc,
	     typename TState, typename TInitialFn, typename TUpdateFn, typename TResultFn> 
struct AxisReduceFn { 
	Tensor<TTrgt, TTrgtDims> trgt; 
	Tensor<TSrc, TTrgtDims+1> src; 

	TInitialFn initialFn;
	TUpdateFn updateFn;
	TResultFn resultFn;

	_dev_ void operator() (const Idxs<TTrgtDims> &trgtPos) { 
		Idxs<TTrgtDims+1> srcPos;
		for (dim_t d=0; d<TTrgtDims; d++)
			srcPos[d] = trgtPos[d];

		TState state = initialFn();
		for (srcPos[TTrgtDims] = 0; srcPos[TTrgtDims] < src.Shape()[TTrgtDims]; srcPos[TTrgtDims]++) {
			state = updateFn(state, src[srcPos], srcPos[TTrgtDims]);
		}
		trgt[trgtPos] = resultFn(state);
	} 
}; 

template<typename TTrgt, dim_t TTrgtDims, typename TSrc,
   	     typename TState, typename TInitialFn, typename TUpdateFn, typename TResultFn> 
_dev_ void AxisReduce(Tensor<TTrgt, TTrgtDims> &trgt, const Tensor<TSrc, TTrgtDims+1> &src,
	                  TInitialFn &initialFn, TUpdateFn &updateFn, TResultFn &resultFn) { 
	AxisReduceFn<TTrgt, TTrgtDims, TSrc, TState, TInitialFn, TUpdateFn, TResultFn> workFn = 
		{trgt, src, initialFn, updateFn, resultFn};
	PerformWork(trgt.Shape(), workFn); 
};

// =============================================================================================
// axis reduce operations (result has same type as source)
// =============================================================================================

#define AXIS_REDUCE(Name, op) \
	template<typename T> struct Name##InitialFn { \
		T value; \
		_dev_ T operator() () { return value; } \
	}; \
	template<typename T> struct Name##UpdateFn { \
		_dev_ T operator() (T state, T value, idx_t pos) { return op(state, value); } \
	}; \
	template<typename T> struct Name##ResultFn { \
		_dev_ T operator() (T state) { return state; } \
	}; \
	template<typename T, dim_t TTrgtDims> _dev_ \
	void Name(T initial, Tensor<T, TTrgtDims> &trgt, const Tensor<T, TTrgtDims+1> &src) { \
		Name##InitialFn<T> initialFn = {initial}; \
		Name##UpdateFn<T> updateFn; \
		Name##ResultFn<T> resultFn; \
		AxisReduce<T, TTrgtDims, T, T, Name##InitialFn<T>, Name##UpdateFn<T>, Name##ResultFn<T>> \
			(trgt, src, initialFn, updateFn, resultFn); \
	};


AXIS_REDUCE(MinLastAxis,		min);
AXIS_REDUCE(MaxLastAxis,		max);
AXIS_REDUCE(SumLastAxis,		add);
AXIS_REDUCE(ProductLastAxis,	multiply);
AXIS_REDUCE(AllLastAxis,		and_fn);
AXIS_REDUCE(AnyLastAxis,		or_fn);


// =============================================================================================
// axis reduce operations to index
// =============================================================================================

template<typename T> struct ArgReduceState {
	T value;
	idx_t pos;
	_dev_ ArgReduceState(T value, idx_t pos)
		: value(value), pos(pos) { }
};

#define ARG_AXIS_REDUCE(Name, compareOp) \
	template<typename T> struct Name##InitialFn { \
		T initialValue; \
		_dev_ ArgReduceState<T> operator() () { \
			return ArgReduceState<T>(initialValue, -1); \
		} \
	}; \
	template<typename T> struct Name##UpdateFn { \
		_dev_ ArgReduceState<T> operator() (ArgReduceState<T> state, T value, idx_t pos) { \
			if (compareOp(value, state.value)) \
				return ArgReduceState<T>(value, pos); \
			else \
				return state; \
		} \
	}; \
	template<typename T> struct Name##ResultFn { \
		_dev_ idx_t operator() (ArgReduceState<T> state) { \
			return state.pos; \
		} \
	}; \
	template<typename T, dim_t TTrgtDims> _dev_ \
	void Name (T initial, Tensor<idx_t, TTrgtDims> &trgt, const Tensor<T, TTrgtDims+1> &src) { \
		Name##InitialFn<T> initialFn = {initial}; \
		Name##UpdateFn<T> updateFn; \
		Name##ResultFn<T> resultFn; \
		AxisReduce<idx_t, TTrgtDims, T, ArgReduceState<T>, Name##InitialFn<T>, \
				   Name##UpdateFn<T>, Name##ResultFn<T>> (trgt, src, initialFn, updateFn, resultFn); \
	};

ARG_AXIS_REDUCE(ArgMinLastAxis,	less);
ARG_AXIS_REDUCE(ArgMaxLastAxis,	greater);


// =============================================================================================
// find value
// =============================================================================================

// must match the value of TensorLayoutTypes.NotFound
static const idx_t NotFound = -9223372036854775804LL;

struct FindLastAxisInitialFn { 
    _dev_ idx_t operator() () {         
        return NotFound;
    } 
}; 

template<typename T> struct FindLastAxisUpdateFn {
    T searchValue;
    _dev_ idx_t operator() (idx_t foundPos, T value, idx_t pos) {
        if (foundPos <> NotFound)
            return foundPos;
        else {
            if (value == searchValue) 
                return pos; 
            else 
                return NotFound;
        } 
    } 
}; 

struct FindLastAxisResultFn { 
    _dev_ idx_t operator() (idx_t foundPos) { 
        return foundPos; 
    } 
}; 

template<typename T, dim_t TTrgtDims> _dev_ 
void FindLastAxis (T searchValue, Tensor<idx_t, TTrgtDims> &trgt, const Tensor<T, TTrgtDims+1> &src) { 
    FindLastAxisInitialFn initialFn; 
    FindLastAxisUpdateFn<T> updateFn = {searchValue}; 
    FindLastAxisResultFn resultFn; 
    AxisReduce<idx_t, TTrgtDims, T, idx_t, FindLastAxisInitialFn, 
               FindLastAxisUpdateFn<T>, FindLastAxisResultFn> (trgt, src, initialFn, updateFn, resultFn); 
};



