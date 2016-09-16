#pragma once

#pragma warning (disable : 4503) 

#pragma warning (push)
#pragma warning (disable : 4267 4244 4503) 
#include <thrust/device_vector.h>
#include <thrust/system/cuda/vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/pair.h>
#pragma warning (pop)

#include <cstdlib>
#include <iostream>
#include <map>
#include <cassert>
#include <string>

#include "Utils.cuh"


//#define TRACE_BUFFER_ALLOCATOR
//#define PRINT_BUFFER_ALLOCATOR_STATS


/// Thrust allocator that assigns memory sequentially from a preallocated buffer.
class buffer_allocator 
{
  private:
	static const std::ptrdiff_t alignment = 32;

	std::string name;
	char *buffer;
	const size_t size;

	size_t allocs = 0;
	size_t pos = 0;

  public:
    typedef char value_type;

    buffer_allocator(std::string name, char *buffer, size_t size) 
		: name(name), buffer(buffer), size(size)
	{
	}

    ~buffer_allocator()
    {
		#ifdef PRINT_BUFFER_ALLOCATOR_STATS
		std::cout << "destructing ";
		print_statistics();
		#endif
    }

	void print_statistics()
	{
		std::cout << "buffer_allocator " << name << ": ";
		std::cout << "allocations=" << std::dec << allocs << "  ";
		std::cout << "size=" << std::dec << size << " Bytes  ";
		std::cout << "used=" << std::dec << pos << " Bytes  ";
		std::cout << "remaining=" << std::dec << (size - pos) << " bytes  ";
	}

    char *allocate(std::ptrdiff_t num_bytes)
    {
		// check if enough memory is available
		if (pos + num_bytes >= size)
		{	
			std::cerr << "buffer_allocator " << name << " is out of memory " <<
				"while processing request of size " << std::dec <<  num_bytes << "bytes" << std::endl;
			print_statistics();
			throw std::bad_alloc();
		}

		// perform allocation, rounding up to satisfy alignment
		char *ptr = buffer + pos;
		std::ptrdiff_t padded = (num_bytes / alignment + 1) * alignment;
		pos += padded;
		allocs++;

		#ifdef TRACE_BUFFER_ALLOCATOR
		std::cout << "buffer allocator " << name << " allocated " << std::dec << num_bytes << " bytes " <<
			"at 0x" << std::hex << static_cast<void *>(ptr) << std::endl;
		#endif	

		return ptr;
    }

    void deallocate(char *ptr, size_t num_bytes)
    {
		// we do not free any memory
		#ifdef TRACE_BUFFER_ALLOCATOR
		std::cout << "buffer allocator " << name << " freed " <<
			"at 0x" << std::hex << static_cast<void *>(ptr) << std::endl;
		#endif	
    }
};


/// thrust range for iterating over an ArrayND with arbitrary strides and offset
template <typename TArrayND>
class ArrayNDRange
{
protected:
	TArrayND BaseArray;

public:
	typedef float TValue;
	typedef typename thrust::device_ptr<TValue> BaseIterator;
    typedef typename thrust::iterator_difference<BaseIterator>::type difference_type;

    struct IndexFunctor : public thrust::unary_function<difference_type, difference_type>
    {
        TArrayND BaseArray;
		
		IndexFunctor(TArrayND baseArray) : BaseArray(baseArray) {}

        _dev difference_type operator()(const difference_type& linearIdx) const
        { 
			// return the element position index given a linear index
			return BaseArray.index(BaseArray.linearIdxToPos(linearIdx));
        }
    };

    typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
    typedef typename thrust::transform_iterator<IndexFunctor, CountingIterator>   TransformIterator;
    typedef typename thrust::permutation_iterator<BaseIterator,TransformIterator> PermutationIterator;
    typedef PermutationIterator iterator;

    ArrayNDRange(TArrayND baseArray) : BaseArray(baseArray) {}
   
    iterator begin(void) const
    {
        return PermutationIterator(thrust::device_pointer_cast((TValue *)BaseArray.data()), 
								   TransformIterator(CountingIterator(0), IndexFunctor(BaseArray)));
    }

    iterator end(void) const
    {
        return begin() + BaseArray.size();
    }
    
};