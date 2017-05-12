// The purpose of main.cu is to have .cu file with a main() function for 
// standalone compilation using nvcc. 
// This compilation is just done to catch compile errors during the build process.
// However the resulting binaries are not used in any way.
// Tensor is embedding the *.cuh headers directly as resources and uses
// the NVRTC compiler for run-time compilation.

#include "Elemwise.cuh"
#include "Reduction.cuh"
#include "GatherScatter.cuh"

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