// TestProject.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <iostream>

using namespace std;


template <size_t stride, size_t ...rStrides>
class Stride
{
public:
	template<class ...size_ts>
	static inline size_t offset(size_t pos, size_ts... rpos)
	{	
		return stride * pos + Stride<rStrides...>::offset(rpos...);
	}
};


template <size_t stride>
class Stride<stride>
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
	cout << "NDArray<10>::index(2) = " << Stride<10>::index(2) << endl;
	cout << "NDArray<10,20>::index(2,3) = " << Stride<10, 20>::offset(2, 3) << endl;
	cout << "NDArray<10,20,30>::index(2,3,4) = " << Stride<10,20,30>::offset(2, 3, 4) << endl;
    return 0;
}

