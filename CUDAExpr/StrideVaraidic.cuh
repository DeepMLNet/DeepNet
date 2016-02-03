#pragma once

template <size_t stride, size_t ...rStrides>
class Stride
{
public:
	template<class ...size_ts>
	_dev static size_t offset(size_t pos, size_ts... rpos)
	{
		return stride * pos + Stride<rStrides...>::offset(rpos...);
	}

	_dev static size_t nDim()
	{
		return 1 + Stride<rStrides...>::nDim();
	}
};

template <size_t stride>
class Stride<stride>
{
public:
	_dev static size_t offset(size_t pos)
	{
		return stride * pos;
	}

	_dev static size_t nDim()
	{
		return 1;
	}
};

