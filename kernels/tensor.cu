#include <cuda_runtime.h>

/*
 * General tensor functions that work for all ranks.
 */

extern "C" __global__ void scale(float* a, float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        a[idx] = a[idx] * b[0];
    }
}

extern "C" __global__ void add(float* m1, float* m2, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < n) {
		m1[idx] = m1[idx] + m2[idx];
	}
}

