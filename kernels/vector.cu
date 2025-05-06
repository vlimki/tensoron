#include <cuda_runtime.h>

// Dot product
extern "C" __global__ void dot_product(float* a, float* b, float* c, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < n) {
		atomicAdd(c, a[idx] * b[idx]);
	}
}
