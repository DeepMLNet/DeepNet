#include "CudaTensor.cuh"

#define _dllkernel_ extern "C" __declspec(dllexport) __global__

extern "C" __global__
void Copy_Float_Float_3(Tensor<float, 3> trgt, Tensor<float, 3> src) {
	Copy(trgt, src);
}



