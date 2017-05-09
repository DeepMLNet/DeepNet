#pragma once

#ifndef __CUDACC_RTC__

#include <nvfunctional>
#include <cstdint>
#include <initializer_list>
#include <cassert>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#endif

#define _dev_ __device__ __forceinline__

#ifdef __CUDACC_RTC__
// integer types
typedef char int8_t;
typedef unsigned char uint8_t;
typedef short int16_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
#else
#include <stdint.h>
#endif

typedef int32_t dim_t;
typedef int64_t idx_t;

