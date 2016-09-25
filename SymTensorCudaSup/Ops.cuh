#pragma once

#include "Utils.cuh"



// ============================= leaf ops ==============================================

struct ConstEOp_t
{
	const float value;
	_dev float operator() () const
	{
		return value;
	}
};


struct ZerosEOp_t
{
	_dev float operator() () const
	{
		return 0.0f;
	}
};


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


// ============================= unary ops ==============================================

struct IdEOp_t
{
	_dev float operator() (float a) const
	{
		return a;
	}
};


struct NegateEOp_t
{
	_dev float operator() (float a) const
	{
		return -a;
	}
};


struct AbsEOp_t
{
	_dev float operator() (float a) const
	{
		return fabsf(a);
	}
};

struct SignTEOp_t
{
	_dev float operator() (float a) const
	{
		return a > 0.0f ? 1.0f : 0.0f;
	}
};

struct LogEOp_t
{
	_dev float operator() (float a) const
	{
		return logf(a);
	}
};

struct Log10EOp_t
{
	_dev float operator() (float a) const
	{
		return log10f(a);
	}
};

struct ExpEOp_t
{
	_dev float operator() (float a) const
	{
		return expf(a);
	}
};

struct SinEOp_t
{
	_dev float operator() (float a) const
	{
		return sinf(a);
	}
};

struct CosEOp_t
{
	_dev float operator() (float a) const
	{
		return cosf(a);
	}
};

struct TanEOp_t
{
	_dev float operator() (float a) const
	{
		return tanf(a);
	}
};

struct AsinEOp_t
{
	_dev float operator() (float a) const
	{
		return asinf(a);
	}
};

struct AcosEOp_t
{
	_dev float operator() (float a) const
	{
		return acosf(a);
	}
};

struct AtanEOp_t
{
	_dev float operator() (float a) const
	{
		return atanf(a);
	}
};

struct SinhEOp_t
{
	_dev float operator() (float a) const
	{
		return sinhf(a);
	}
};

struct CoshEOp_t
{
	_dev float operator() (float a) const
	{
		return coshf(a);
	}
};

struct TanhEOp_t
{
	_dev float operator() (float a) const
	{
		return tanhf(a);
	}
};

struct SqrtEOp_t
{
	_dev float operator() (float a) const
	{
		return sqrtf(a);
	}
};

struct CeilEOp_t
{
	_dev float operator() (float a) const
	{
		return ceilf(a);
	}
};

struct FloorEOp_t
{
	_dev float operator() (float a) const
	{
		return floorf(a);
	}
};

struct RoundEOp_t
{
	_dev float operator() (float a) const
	{
		return roundf(a);
	}
};

struct TruncateEOp_t
{
	_dev float operator() (float a) const
	{
		return truncf(a);
	}
};

struct NotEOp_t
{
	_dev float operator() (bool a) const
	{
		return !a;
	}
};

struct CheckFiniteIEOp_t {
	int * const nonFiniteCountPtr;
	const char name[50];

	_devonly float operator() (const size_t *pos, const size_t dims, float a) const {
		if (!isfinite(a)) {
			atomicAdd(nonFiniteCountPtr, 1);

			switch (dims) {
			case 0:	printf("Non-finite element in %s at [].\n", name); break;
			case 1: printf("Non-finite element in %s at [%llu].\n", name, pos[0]); break;
			case 2: printf("Non-finite element in %s at [%llu; %llu].\n", name, pos[0], pos[1]); break;
			case 3: printf("Non-finite element in %s at [%llu; %llu; %llu].\n", name, pos[0], pos[1], pos[2]); break;
			case 4: printf("Non-finite element in %s at [%llu; %llu; %llu; %llu].\n", name, pos[0], pos[1], pos[2], pos[3]); break;
			case 5: printf("Non-finite element in %s at [%llu; %llu; %llu; %llu; %llu].\n", name, pos[0], pos[1], pos[2], pos[3], pos[4]); break;
			default: printf("Non-finite element in %s.", name);
			}			
		}
		return a;
	}
};


// ============================= binary ops ==============================================

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

struct MaxEOp_t
{
	_dev float operator() (float a, float b) const
	{
		return max(a, b);
	}
};

struct MinEOp_t
{
	_dev float operator() (float a, float b) const
	{
		return min(a, b);
	}
};

struct EqualEOp_t
{
	_dev bool operator() (float a, float b) const
	{
		return a == b;
	}
};

struct LessEOp_t
{
	_dev bool operator() (float a, float b) const
	{
		return a < b;
	}
};

struct LessEqualEOp_t
{
	_dev bool operator() (float a, float b) const
	{
		return a <= b;
	}
};

struct GreaterEOp_t
{
	_dev bool operator() (float a, float b) const
	{
		return a > b;
	}
};

struct GreaterEqualEOp_t
{
	_dev bool operator() (float a, float b) const
	{
		return a >= b;
	}
};

struct NotEqualEOp_t
{
	_dev bool operator() (float a, float b) const
	{
		return a != b;
	}
};

struct AndEOp_t
{
	_dev bool operator() (bool a, bool b) const
	{
		return a && b;
	}
};

struct OrEOp_t
{
	_dev bool operator() (bool a, bool b) const
	{
		return a || b;
	}
};


// ============================= tertiary ops ==============================================

struct IfThenElseEOp_t
{
	_dev float operator() (float ifTrue, float ifFalse, bool cond) const
	{
		return cond ? ifTrue : ifFalse;
	}
};






