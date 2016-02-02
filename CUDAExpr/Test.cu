#include <cuda.h>
#include <iostream>
#include <cassert>

#include "NDArray.cuh"


using namespace std;


// This will output the proper CUDA error strings
// in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(CUresult err, const char *file, const int line)
{
	if (CUDA_SUCCESS != err) {
		fprintf(stderr,
			"CUDA Driver API error = %04d from file <%s>, line %i.\n",
			err, file, line);
		exit(-1);
	}
}

__global__ void sayHi()
{
	printf("Cuda Kernel Hello Word.\n");
}


int main()
{
	CUresult res;
	CUdevice device;
	CUcontext context;
	char name[100];

	checkCudaErrors(cuInit(0));
	checkCudaErrors(cuDeviceGet(&device, 0));
	cuDeviceGetName(name, sizeof(name), device);
	printf("> Using device 0: %s\n", name);
	checkCudaErrors(cuCtxCreate(&context, 0, device));


	cout << "NDArray<10>::index(2) = " << Stride<10>::offset(2) << endl;
	cout << "NDArray<10,20>::index(2,3) = " << Stride<10, 20>::offset(2, 3) << endl;
	cout << "NDArray<10,20,30>::index(2,3,4) = " << Stride<10, 20, 30>::offset(2, 3, 4) << endl;


	size_t dataDim0Size = 5;
	size_t dataDim1Size = 3;
	CUdeviceptr dataGpuPtr;
	const size_t dataDim0Stride = 512;
	size_t dataDim0StrideDummy;
	const size_t dataDim1Stride = 1 * sizeof(float);

	res = cuMemAllocPitch(&dataGpuPtr, &dataDim0StrideDummy,
		dataDim0Size * sizeof(float), dataDim1Size * sizeof(float), sizeof(float));
	
	cout << "Allocated array of size " << dataDim0Size << "x" << dataDim1Size << endl;
	cout << "result: " << res << endl;
	cout << "got: addr: " << dataGpuPtr << "  strides: " << dataDim0StrideDummy << "x" << dataDim1Stride << endl;

	float *dataGpuData = reinterpret_cast<float *>(dataGpuPtr);
	typedef NDArrayPointer<Stride<dataDim0Stride, dataDim1Stride> > *TPNDArray;
	TPNDArray data = reinterpret_cast<TPNDArray>(dataGpuData);

	dim3 grid(1, 1, 1);
	dim3 block(1, 1, 1);
	elementwiseUnary<ConstOneElementwiseOp_t><<<grid, block>>>(data, data);


	//float data[10];
	//NDArrayPointer<Stride<1, 2>> nd(data);
	//
	//nd.element(1, 2) = 1;
	//cout << "nd.element(1, 2): " << nd.element(1, 2) << endl;
	//cout << "calling elementwiseunary" << endl;
	//elementwiseUnary<NegateElementwiseOp_t>()
	//cout << "nd.element(1, 2): " << nd.element(1, 2) << endl;

	return 0;
}