#include <cuda.h>
#include <iostream>
#include <cassert>

#include "Ops.cuh"
#include "NDSupport.cuh"
//#include "Reduce.cuh"
#include "Subtensor.cuh"

using namespace std;


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


template <typename TNDArray>
void printNDArray(TNDArray *ndArray)
{
	for (size_t pos0 = 0; pos0 < ndArray->shape(0); pos0++)
	{
		printf("[\n");
		for (size_t pos1 = 0; pos1 < ndArray->shape(1); pos1++)
		{
			for (size_t pos2 = 0; pos2 < ndArray->shape(2); pos2++)
			{
				printf("%.3f\t", ndArray->element(pos0, pos1, pos2));
			}
			printf("\n");
		}
		printf("]\n");
	}
}


template <typename TNDArray>
TNDArray *post(const TNDArray *ndArray)
{
	size_t bytes = ndArray->allocation();
	CUdeviceptr dataDevice;
	checkCudaErrors(cuMemAlloc(&dataDevice, bytes));
	const float *dataHost = reinterpret_cast<const float *>(ndArray);
	cuMemcpyHtoD(dataDevice, dataHost, bytes);
	return reinterpret_cast<TNDArray *>(dataDevice);
}

template <typename TNDArray>
TNDArray *gather(const TNDArray *ndArray)
{
	size_t bytes = ndArray->allocation();
	CUdeviceptr dataDevice = reinterpret_cast<CUdeviceptr>(ndArray);
	float *dataHost = reinterpret_cast<float *>(malloc(bytes));
	cuMemcpyDtoH(dataHost, dataDevice, bytes);
	return reinterpret_cast<TNDArray *>(dataHost);
}

template <typename TNDArray>
TNDArray *allocDevice()
{
	size_t bytes = TNDArray::allocation();
	CUdeviceptr dataDevice;
	checkCudaErrors(cuMemAlloc(&dataDevice, bytes));
	return reinterpret_cast<TNDArray *>(dataDevice);
}

int main()
{
	//CUresult res;
	CUdevice device;
	CUcontext context;
	char name[100];

	checkCudaErrors(cuInit(0));
	checkCudaErrors(cuDeviceGet(&device, 0));
	cuDeviceGetName(name, sizeof(name), device);
	printf("> Using device 0: %s\n", name);
	checkCudaErrors(cuCtxCreate(&context, 0, device));


	//const size_t dataDim0Size = 1, dataDim1Size = 3, dataDim2Size = 4;
	//const size_t dataDim0Stride = dataDim1Size * dataDim2Size, dataDim1Stride = dataDim2Size, dataDim2Stride = 1;
	//typedef NDArray3DPointer<Shape3D<dataDim0Size, dataDim1Size, dataDim2Size>,
	//	Stride3D<dataDim0Stride, dataDim1Stride, dataDim2Stride> > TNDArray;
	//TNDArray *aDevice = allocDevice<TNDArray>();
	//TNDArray *bDevice = allocDevice<TNDArray>();
	//TNDArray *cDevice = allocDevice<TNDArray>();

	//CUdeviceptr dataGpuPtr;
	//size_t dataDim1StrideDummy;
	//res = cuMemAllocPitch(&dataGpuPtr, &dataDim1StrideDummy,
	//	dataDim2Size * sizeof(float), dataDim0Size * dataDim1Size * sizeof(float), sizeof(float));

	//cout << "Allocated array of size " << dataDim0Size << "x" << dataDim1Size << endl;
	//cout << "result: " << res << endl;
	//cout << "got: addr: " << dataGpuPtr << "  strides: " << dataDim1StrideDummy << "x" << dataDim2Stride << endl;
	//float *dataGpuData = reinterpret_cast<float *>(dataGpuPtr);

	//dim3 grid(1, 1, 1);
	//dim3 block(1, 10, 10);

	//elementwiseUnary<ConstOneElementwiseOp_t> <<<grid, block>>>(aDevice, aDevice);
	//elementwiseUnary<ConstOneElementwiseOp_t> <<<grid, block>>>(bDevice, bDevice);

	//elementwiseBinary<AddBinaryElementwiseOp_t> <<<grid, block>>>(cDevice, aDevice, bDevice);
	//elementwiseBinary<AddBinaryElementwiseOp_t> <<<grid, block>>>(cDevice, cDevice, cDevice);

	//TNDArray *dataHost = gather(cDevice);
	//cout << "data=" << endl;
	//printNDArray(dataHost);

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


