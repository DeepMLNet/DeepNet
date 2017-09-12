#pragma once

#include "Utils.cuh"

struct Interpolate1DEOp_t
{
	_devonly float operator() (float a0) const 
	{
		float idx0 = (a0 - minArg0) / resolution0 + offset;
		return tex1D<float>(tbl, idx0);
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
		return tex2D<float>(tbl, idx1, idx0);
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
		return tex3D<float>(tbl, idx2, idx1, idx0);
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
