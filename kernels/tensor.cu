#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

/*
 * General tensor functions that work for all ranks.
 */

// KERNELS
extern "C" __global__ void scale_float(float* a, float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        a[idx] = a[idx] * b[0];
    }
}

extern "C" __global__ void add_float(float* m1, float* m2, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < n) {
		m1[idx] = m1[idx] + m2[idx];
	}
}

extern "C" __global__ void relu_float(float* m1, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < n) {
		m1[idx] = (m1[idx] + fabsf(m1[idx])) / 2;
	}
}

extern "C" __global__ void tanh_float(float* m1, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < n) {
		m1[idx] = tanhf(m1[idx]);
	}
}

extern "C" __global__ void sigmoid_float(float* m1, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < n) {
		m1[idx] = 1/(1 + expf(-m1[idx]));
	}
}
