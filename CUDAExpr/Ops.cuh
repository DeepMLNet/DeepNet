#pragma once

#include "Utils.cuh"


struct DiagonalOneIEOp_t {
	_dev float operator() (const size_t *pos, const size_t dims) const {
		if (dims == 0) {
			return 1.0;
		} else {
			bool allEqual = true;
			for (size_t dim = 1; dim <= dims; dim++) {
				if (pos[0] != pos[dim]) {
					allEqual = false;
					break;
				}
			}
			if (allEqual)
				return 1.0;
			else
				return 0.0;
		}
	}
};




struct ConstEOp_t
{
	_dev ConstEOp_t(float value) : value(value) {}

	_dev float operator() () const
	{
		return value;
	}

	float value;
};


struct NegateEOp_t
{
	_dev float operator() (float a) const
	{
		return -a;
	}
};


struct LogEOp_t
{
	_dev float operator() (float a) const
	{
		return logf(a);
	}
};


struct ExpEOp_t
{
	_dev float operator() (float a) const
	{
		return expf(a);
	}
};

struct IdEOp_t
{
	_dev float operator() (float a) const
	{
		return a;
	}
};


struct AddEOp_t
{
	_dev float operator() (float a, float b) const
	{
		return a + b;
	}
};

struct SubstractEOp_t
{
	_dev float operator() (float a, float b) const
	{
		return a - b;
	}
};

struct MultiplyEOp_t
{
	_dev float operator() (float a, float b) const
	{
		return a * b;
	}
};

struct DivideEOp_t
{
	_dev float operator() (float a, float b) const
	{
		return a / b;
	}
};

struct PowerEOp_t
{
	_dev float operator() (float a, float b) const
	{
		return powf(a, b);
	}
};




// TODO: dummy
struct Dot
{
	_dev float operator() (float a, float b) const
	{
		return 111.1f;
	}
};

