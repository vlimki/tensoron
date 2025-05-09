#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

/*
 * General tensor functions that work for all ranks.
 */

// KERNELS
extern "C" __global__ void scale_float(float* a, float* b, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        output[idx] = a[idx] * b[0];
    }
}

extern "C" __global__ void add_float(float* m1, float* m2, float* output, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < n) {
		output[idx] = m1[idx] + m2[idx];
	}
}

extern "C" __global__ void sub_float(float* m1, float* m2, float* output, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < n) {
		output[idx] = m1[idx] - m2[idx];
	}
}

extern "C" __global__ void cmul_float(float* m1, float* m2, float* output, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < n) {
		output[idx] = m1[idx] * m2[idx];
	}
}


extern "C" __global__ void relu_float(float* m1, float* output, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < n) {
		output[idx] = (m1[idx] + fabsf(m1[idx])) / 2;
	}
}

extern "C" __global__ void tanh_float(float* m1, float* output, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < n) {
		output[idx] = tanhf(m1[idx]);
	}
}

extern "C" __global__ void sigmoid_float(float* m1, float* output, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < n) {
		output[idx] = 1/(1 + expf(-m1[idx]));
	}
}

extern "C" __global__ void sigmoid_derivative_float(float* m1, float* output, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < n) {
		float sigmoid_x =  1/(1 + expf(-m1[idx]));
		output[idx] = sigmoid_x * (1 - sigmoid_x);
	}
}

extern "C" __global__ void relu_derivative_float(float* m1, float* output, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < n) {
		output[idx] = m1[idx] < 0 ? 0 : 1;
	}
}

extern "C" __global__ void scale_double(double* a, double* b, double* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        output[idx] = a[idx] * b[0];
    }
}

extern "C" __global__ void add_double(double* m1, double* m2, double* output, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < n) {
		output[idx] = m1[idx] + m2[idx];
	}
}

extern "C" __global__ void sub_double(double* m1, double* m2, double* output, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < n) {
		output[idx] = m1[idx] - m2[idx];
	}
}

extern "C" __global__ void cmul_double(double* m1, double* m2, double* output, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < n) {
		output[idx] = m1[idx] * m2[idx];
	}
}


extern "C" __global__ void relu_double(double* m1, double* output, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < n) {
		output[idx] = (m1[idx] + fabsf(m1[idx])) / 2;
	}
}

extern "C" __global__ void tanh_double(double* m1, double* output, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < n) {
		output[idx] = tanhf(m1[idx]);
	}
}

extern "C" __global__ void sigmoid_double(double* m1, double* output, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < n) {
		output[idx] = 1/(1 + expf(-m1[idx]));
	}
}

extern "C" __global__ void sigmoid_derivative_double(double* m1, double* output, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < n) {
		float sigmoid_x =  1/(1 + expf(-m1[idx]));
		output[idx] = sigmoid_x * (1 - sigmoid_x);
	}
}

extern "C" __global__ void relu_derivative_double(double* m1, double* output, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < n) {
		output[idx] = m1[idx] < 0 ? 0 : 1;
	}
}
