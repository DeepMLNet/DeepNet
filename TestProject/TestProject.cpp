// TestProject.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <iostream>

using namespace std;





template <size_t stride, size_t ...rStrides>
class Indexer
{
public:
	template<class ...size_ts>
	static inline size_t index(size_t pos, size_ts... rpos)
	{	
		return stride * pos + Indexer<rStrides...>::index(rpos...);
	}

	//__forceinline__ __device__ float &element(const size_t *(&pos))
	//{
	//	size_t idx = 0;
	//	#pragma unroll
	//	for (size_t d = 0; d < nDim; d++)
	//		idx += pos[d] * stride[d];
	//	return data[idx];
	//}

	//__forceinline__ __device__ float &element()
	//{
	//	return data[0];
	//}
};


template <size_t stride>
class Indexer<stride>
{
public:
	static size_t index(size_t pos)
	{
		return stride * pos;
	}
};


template <typename TIndexer>
class NDArray
{
	public:
		float *data;
		typedef TIndexer idx;
};


int main()
{
	cout << "NDArray<10>::index(2) = " << Indexer<10>::index(2) << endl;
	cout << "NDArray<10,20>::index(2,3) = " << Indexer<10, 20>::index(2, 3) << endl;
	cout << "NDArray<10,20,30>::index(2,3,4) = " << Indexer<10,20,30>::index(2, 3, 4) << endl;
    return 0;
}

