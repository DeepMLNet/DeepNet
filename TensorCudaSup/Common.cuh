#pragma once

#include <nvfunctional>
#include <cstdint>
#include <initializer_list>
#include <cassert>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifdef __NVCC__
#define _dev_ __host__ __device__
#define _devonly_ __device__ 
#else
#define _dev_
#endif

typedef int32_t dim_t;
typedef int64_t idx_t;

