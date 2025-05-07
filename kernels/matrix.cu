#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda.h>

struct dimensions_t {
	uint32_t m1_rows;
	uint32_t m1_cols;
	uint32_t m2_rows;
	uint32_t m2_cols;
};

// KERNELS
extern "C" __global__ void mul_float(float* m1, float* m2, float* result, struct dimensions_t dims) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if(col < dims.m2_cols && row < dims.m1_rows) {
		float sum = 0.0;

		for(int i = 0; i < dims.m1_cols; i++) {
			sum += m1[row * dims.m1_cols + i] * m2[col + i * dims.m2_cols];
		}

		result[row * dims.m2_cols + col] = sum;
	}
}
