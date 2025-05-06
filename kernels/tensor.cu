#include <cuda_runtime.h>

// General tensor functions that work for all ranks.
extern "C" __global__ void scale(float* a, float b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        c[idx] = a[idx] * b;
    }
}
