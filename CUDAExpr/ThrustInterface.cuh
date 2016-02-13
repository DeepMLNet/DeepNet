#pragma once

#pragma warning (disable : 4503) 

#pragma warning (push)
#pragma warning (disable : 4267 4244 4503) 
#include <thrust/device_vector.h>
#pragma warning (pop)


#include "Utils.cuh"


/// thrust range for iterating over a ArrayND with arbitrary strides and offset
template <typename TArrayND>
class ArrayNDRange
{
public:
	typedef float TValue;
	typedef typename thrust::device_ptr<TValue> BaseIterator;
    typedef typename thrust::iterator_difference<BaseIterator>::type difference_type;

    struct IndexFunctor : public thrust::unary_function<difference_type, difference_type>
    {
        IndexFunctor(TArrayND baseArray) : BaseArray(baseArray) {}
        _dev difference_type operator()(const difference_type& linearIdx) const
        { 
			// return the element position index given a linear index
			return BaseArray.index(BaseArray.linearIdxToPos(linearIdx));
        }

        TArrayND BaseArray;
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
    
protected:
	TArrayND BaseArray;
};