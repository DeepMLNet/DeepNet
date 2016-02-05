#pragma once

#include "Utils.cuh"


struct ConstOneElementwiseOp_t
{
	_dev float operator() (float a)
	{
		return 1.0f;
	}
};


struct NegateElementwiseOp_t
{
	_dev float operator() (float a)
	{
		return -a;
	}
};


struct LogElementwiseOp_t
{
	_dev float operator() (float a)
	{
		return logf(a);
	}
};


struct ExpElementwiseOp_t
{
	_dev float operator() (float a)
	{
		return expf(a);
	}
};

struct CopyElementwiseOp_t
{
	_dev float operator() (float a)
	{
		return a;
	}
};


struct AddBinaryElementwiseOp_t
{
	_dev float operator() (float a, float b)
	{
		return a + b;
	}
};

struct SubstractBinaryElementwiseOp_t
{
	_dev float operator() (float a, float b)
	{
		return a - b;
	}
};

struct MultiplyBinaryElementwiseOp_t
{
	_dev float operator() (float a, float b)
	{
		return a * b;
	}
};

struct DivideBinaryElementwiseOp_t
{
	_dev float operator() (float a, float b)
	{
		return a / b;
	}
};

