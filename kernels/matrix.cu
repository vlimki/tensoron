#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda.h>

struct dimensions_t {
	uint32_t rows;
	uint32_t cols;
};

// KERNELS
extern "C" __global__ void mul_float(float* m1, float* m2, float* result, struct dimensions_t dims1, struct dimensions_t dims2) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if(col < dims2.cols && row < dims1.rows) {
		float sum = 0.0;

		for(int i = 0; i < dims1.cols; i++) {
			sum += m1[row * dims1.cols + i] * m2[col + i * dims2.cols];
		}

		result[row * dims2.cols + col] = sum;
	}
}

extern "C" __global__ void transpose_float(float* in, float* out, struct dimensions_t dims) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < dims.rows * dims.cols) {
		int r = idx / dims.cols;
		int c = idx % dims.cols;

		out[c * dims.rows + r] = in[r * dims.cols + c];
	}
}

extern "C" __global__ void mul_double(double* m1, double* m2, double* result, struct dimensions_t dims1, struct dimensions_t dims2) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if(col < dims2.cols && row < dims1.rows) {
		double sum = 0.0;

		for(int i = 0; i < dims1.cols; i++) {
			sum += m1[row * dims1.cols + i] * m2[col + i * dims2.cols];
		}

		result[row * dims2.cols + col] = sum;
	}
}

extern "C" __global__ void transpose_double(double* in, double* out, struct dimensions_t dims) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < dims.rows * dims.cols) {
		int r = idx / dims.cols;
		int c = idx % dims.cols;

		out[c * dims.rows + r] = in[r * dims.cols + c];
	}
}
