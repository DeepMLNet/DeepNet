#include "CudaTensor.cuh"

extern "C"
__global__ void FillElementwise_Float_Float_3(
	Tensor<float, 3> trgt,
	Tensor<float, 3> src1) {

	FillElementwise(trgt, src1);
}


