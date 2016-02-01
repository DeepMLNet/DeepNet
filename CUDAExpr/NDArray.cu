
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


class NDArray
{
public:
	size_t *shape;
	size_t *stride;
	float *data;
};


__global__ void scalarConst(NDArray *target, const float value)
{
	*(target->data) = value;
}
