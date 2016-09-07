#pragma once

#include "Utils.cuh"


// dummy functions for IntelliSense
#ifndef __CUDACC__ 
template <typename T> T tex1D(cudaTextureObject_t texObj, float x);
template <typename T> T tex2D(cudaTextureObject_t texObj, float x, float y);
template <typename T> T tex3D(cudaTextureObject_t texObj, float x, float y, float z);
#endif


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



struct ZerosEOp_t
{
	_dev float operator() () const
	{
		return 0.0f;
	}
};


struct Interpolate1DEOp_t
{
	_devonly float operator() (float a0) const 
	{
		float idx0 = (a0 - minArg0) / resolution0 + offset;
		printf("minArg %f resolution %f offset %f\n", minArg0, resolution0, offset);
		return tex1D<float>(tbl, 1);
		//return tex1D<float>(tbl, idx0);
	}

	cudaTextureObject_t tbl;
	float minArg0;
	float resolution0;
	float offset;
};

struct Interpolate2DEOp_t
{
	_devonly float operator() (float a0, float a1) const 
	{
		float idx0 = (a0 - minArg0) / resolution0 + offset;
		float idx1 = (a1 - minArg1) / resolution1 + offset;
		return tex2D<float>(tbl, idx0, idx1);
	}

	cudaTextureObject_t tbl;
	float minArg0;
	float resolution0;
	float minArg1;
	float resolution1;
	float offset;
};

struct Interpolate3DEOp_t
{
	_devonly float operator() (float a0, float a1, float a2) const 
	{
		float idx0 = (a0 - minArg0) / resolution0 + offset;
		float idx1 = (a1 - minArg1) / resolution1 + offset;
		float idx2 = (a2 - minArg2) / resolution2 + offset;
		return tex3D<float>(tbl, idx0, idx1, idx2);
	}

	cudaTextureObject_t tbl;
	float minArg0;
	float resolution0;
	float minArg1;
	float resolution1;
	float minArg2;
	float resolution2;
	float offset;
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


