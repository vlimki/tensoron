#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda.h>

struct dimensions_t {
	uint32_t m1_rows;
	uint32_t m1_cols;
	uint32_t m2_rows;
	uint32_t m2_cols;
};

__global__ void matmul_kernel(float* m1, float* m2, float* result, struct dimensions_t dims) {
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

/*float* matmul_cuda(float* m1, float* m2, struct dimensions_t dims) {
	if(dims.m1_cols != dims.m2_rows) {
		printf("Invalid dimensions.\n");
		exit(-1);
	}

	float* result = (float*) malloc(sizeof(float) * dims.m1_rows * dims.m2_cols);

	float* cuda_m1;
	float* cuda_m2;
	float* cuda_result;

	cudaMalloc(&cuda_m1, sizeof(float) * dims.m1_rows * dims.m1_cols);
	cudaMalloc(&cuda_m2, sizeof(float) * dims.m2_rows * dims.m2_cols);
	cudaMalloc(&cuda_result, sizeof(float) * dims.m1_rows * dims.m2_cols);

	cudaMemcpy(cuda_m1, m1, sizeof(float) * dims.m1_rows * dims.m1_cols, cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_m2, m2, sizeof(float) * dims.m2_rows * dims.m2_cols, cudaMemcpyHostToDevice);

	dim3 blockSz = dim3(16, 16, 1);
	dim3 gridSz = dim3(ceil(dims.m2_cols/16.0), ceil(dims.m1_rows/16.0), 1);

	matmul_kernel<<<gridSz, blockSz>>>(cuda_m1, cuda_m2, cuda_result, dims);

	cudaMemcpy(result, cuda_result, sizeof(float) * dims.m1_rows * dims.m2_cols, cudaMemcpyDeviceToHost);

	cudaFree(cuda_m1);
	cudaFree(cuda_m2);
	cudaFree(cuda_result);

	return result;
}

int main() {
	const int N = 2500;

	float* m1 = (float*) malloc(sizeof(float) * N * N);
	float* m2 = (float*) malloc(sizeof(float) * N * N);
	
	struct dimensions_t dims {
		.m1_rows = N,
		.m1_cols = N,
		.m2_rows = N,
		.m2_cols = N,
	};

	for(int i = 0; i < N * N; i++) {
		m1[i] = (float) rand() / RAND_MAX - 0.5;
		m2[i] = (float) rand() / RAND_MAX - 0.5;
	}

	float* r = matmul_cuda(m1, m2, dims);

	printf("[ ");
	for(int i = 0; i < N * N; i++) {
		printf("%f, ", r[i]);
	}
	printf("]\n");

	free(m1);
	free(m2);
}*/
