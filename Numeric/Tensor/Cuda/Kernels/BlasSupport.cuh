#pragma once

#include "Common.cuh"


_dev_ void CheckBlasInfo(ptr_t _info, int batchSize, ptr_t _error, bool trapOnError) {
	const int *info = (const int *)_info;
	int *error = (int *)_error;
	for (int batch = threadIdx.x + blockIdx.x * blockDim.x;
		 batch < batchSize;
		 batch += gridDim.x * blockDim.x) {
		if (info[batch] != 0) {
			if (info[batch] < 0) 
				printf("CUBLAS: parameter %d has illegal value.\n", -info[batch]);
			if (info[batch] > 0) 
				printf("CUBLAS: matrix batch %d is singular.\n", batch);

			*error=1;
			if (trapOnError) {
				__threadfence();
				__syncthreads();
				asm("trap;");
			}
		}
	}
}

