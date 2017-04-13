#pragma once

#include <cstdint>

#ifdef TENSORCUDASUP_EXPORTS
#define API __declspec(dllexport)
#else
#define API __declspec(dllimport)
#endif



// how to define variable dimensions?
// perhaps limit number of dimensions and use fixed arrays for shape and stride?
// otherwise we would need templates, which has drawbacks...
// or we would need to instantiate the appropriate code, but this would mean
// generating all templates for all functions that might possibly use Tensor.
// so bad idea...

// but we need to make Tensor a template anyway
// or 


struct Tensor
{
		
};


