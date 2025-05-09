#include <cuda_runtime.h>

// KERNELS
extern "C" __global__ void mul_float(float* a, float* b, float* c, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < n) {
		atomicAdd(c, a[idx] * b[idx]);
	}
}

extern "C" __global__ void mul_double(double* a, double* b, double* c, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < n) {
		atomicAdd(c, a[idx] * b[idx]);
	}
}
