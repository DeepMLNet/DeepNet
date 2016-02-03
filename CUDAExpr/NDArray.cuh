#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define _dev __host__ __device__ __forceinline__ inline


template <typename T>
_dev T min(const T a, const T b)
{
	return a < b ? a : b;
}

template <typename T>
_dev T max(const T a, const T b)
{
	return a > b ? a : b;
}




template <size_t shape0, size_t shape1, size_t shape2>
class Shape3D
{
public:
	_dev static size_t shape(const size_t dim)
	{
		switch (dim)
		{
		case 0:	return shape0;
		case 1:	return shape1;
		case 2:	return shape2;
		default: return 0;
		}
	}
};


template <size_t stride0, size_t stride1, size_t stride2>
class Stride3D
{
public:
	_dev static size_t stride(const size_t dim)
	{
		switch (dim)
		{
		case 0:	return stride0;
		case 1:	return stride1;
		case 2:	return stride2;
		default: return 0;
		}
	}

	_dev static size_t offset(const size_t pos0, const size_t pos1, const size_t pos2) 
	{ 
		return stride0 * pos0 + stride1 * pos1 + stride2 * pos2;	
	}
};


//template <typename TShape, typename TStride, float *Tdata>
//class NDArray3DFixed
//{
//public:
//	typedef TShape Shape;
//	typedef TStride Stride;
//
//	_dev size_t shape(const size_t dim) const
//	{
//		return TShape::shape(dim);
//	}
//
//	_dev size_t stride(const size_t dim) const
//	{
//		return TStride::stride(dim);
//	}
//
//	_dev size_t size() const
//	{
//		return shape(0) * shape(1) * shape(2);
//	}
//
//	_dev static size_t allocation() const
//	{
//		return max(shape(0) * stride(0), max(shape(1) * stride(1), shape(2) * stride(2)))
//	}
//
//	_dev float *data()
//	{
//		return Tdata;
//	}
//
//	_dev const float *data() const
//	{
//		return Tdata;
//	}
//
//	_dev float &element(const size_t pos0, const size_t pos1, const size_t pos2)
//	{
//		return data()[TStride::offset(pos0, pos1, pos2)];
//	}
//
//	_dev const float &element(const size_t pos0, const size_t pos1, const size_t pos2) const
//	{
//		return data()[TStride::offset(pos0, pos1, pos2)];
//	}
//};


template <typename TShape, typename TStride>
class NDArray3DPointer
{
public:
	typedef TShape Shape;
	typedef TStride Stride;

	_dev static size_t shape(const size_t dim) 
	{
		return TShape::shape(dim);
	}

	_dev static size_t stride(const size_t dim) 
	{
		return TStride::stride(dim);
	}

	_dev static size_t size() 
	{
		return shape(0) * shape(1) * shape(2);
	}

	_dev static size_t allocation() 
	{
		return max(shape(0) * stride(0), max(shape(1) * stride(1), shape(2) * stride(2))) * sizeof(float);
	}

	_dev float *data()
	{
		return reinterpret_cast<float *>(this);
	}

	_dev const float *data() const
	{
		return reinterpret_cast<const float *>(this);
	}

	_dev float &element(const size_t pos0, const size_t pos1, const size_t pos2)
	{
		return data()[TStride::offset(pos0, pos1, pos2)];
	}

	_dev const float &element(const size_t pos0, const size_t pos1, const size_t pos2) const
	{
		return data()[TStride::offset(pos0, pos1, pos2)];
	}
};


