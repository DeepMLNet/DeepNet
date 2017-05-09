#include "CudaTensor.cuh"


extern "C" __global__
void Copy_Float_3(Tensor<float, 3> trgt, Tensor<float, 3> src) {
	Copy(trgt, src);
}

extern "C" __global__
void CopyHeterogenous_Float_3_3(Tensor<float, 3> trgt, Tensor<float, 3> src) {
	CopyHeterogenous(trgt, src);
}


extern "C" __global__
void CopyHeterogenous_Float_3_4(Tensor<float, 3> trgt, Tensor<float, 4> src) {
	CopyHeterogenous(trgt, src);
}


int main(int argc, char **argv) {
	return 0;
}