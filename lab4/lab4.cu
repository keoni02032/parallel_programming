#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <iomanip>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

struct comparator {
	__host__ __device__ bool operator()(double a, double b) {
		return fabs(a) < fabs(b);
	}
};

__global__ void swap_gen(double* data, int n, int m, int j, int max) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;
	for (int i = idx; i < m + 1; i += offset) {
		thrust::swap(data[i * n + j], data[i * n + max]);
	}
}

__global__ void back(double* data, int* b, int j, int n, int m, double* result) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;
	result[j] = data[n * m + b[j] - 1] / data[j * n + b[j] - 1];
	for (int i = j - idx - 1; i >= 0; i -= offset) {
		if (b[i] > 0) {
			int w = b[i];
			data[n * m + w - 1] -= result[j] * data[j * n + w - 1];
		}
	}
}

__global__ void kernel(double* data, int i, int jj, int n, int m) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	for (int j = idx + i + 1; j < n; j += offsetx) {
		for (int k = idy + jj + 1; k < m + 1; k += offsety) {
			data[k * n + j] -= data[k * n + i] * data[jj * n + j] / data[jj * n + i];
		}
	}
}

int main() {
	std::ios_base::sync_with_stdio(0);
	std::cin.tie(0);
	int n, m;
	std::cin >> n >> m;
	double* data, * dev_data, * dev_result;
	int t = m + 1, * dev_b;
	data = (double*)malloc(sizeof(double) * n * t);
	int* b = (int*)malloc(sizeof(int) * m);
	double* result = (double*)malloc(sizeof(double) * m);
	memset(result, 0, sizeof(double) * m);
	for (int i = 0; i < m; ++i) {
		b[i] = 0;
	}
	CSC(cudaMalloc(&dev_data, sizeof(double) * n * t));
	CSC(cudaMalloc(&dev_b, sizeof(int) * m));
	CSC(cudaMalloc(&dev_result, sizeof(double) * m));
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			std::cin >> data[i + j * n];
		}
	}
	for (int i = 0; i < n; ++i) {
		std::cin >> data[i + n * m];
	}
	CSC(cudaMemcpy(dev_data, data, sizeof(double) * n * t, cudaMemcpyHostToDevice));
	comparator comp;
	double tmp;
	int max;
	thrust::device_ptr<double> p_arr, res;
	for (int j = 0, i = 0; i < n && j < m; ++i, ++j) {
		p_arr = thrust::device_pointer_cast(dev_data + j * n);
		res = thrust::max_element(p_arr + i, p_arr + n, comp);
		max = res - p_arr;
		tmp = fabs(*res);
		if (tmp > 1e-7) {
			if (i != max) {
				swap_gen << <512, 512 >> > (dev_data, n, m, i, max);
				CSC(cudaGetLastError());
			}
			b[j] = i + 1;
			kernel << <dim3(32, 32), dim3(32, 32) >> > (dev_data, i, j, n, m);
			CSC(cudaGetLastError());
		}
		else {
			--i;
		}
	}
	CSC(cudaMemcpy(dev_b, b, sizeof(int) * m, cudaMemcpyHostToDevice));
	CSC(cudaMemcpy(dev_result, result, sizeof(double) * m, cudaMemcpyHostToDevice));
	for (int i = m - 1; i >= 0; --i) {
		if (b[i] > 0) {
			back << <512, 512 >> > (dev_data, dev_b, i, n, m, dev_result);
			CSC(cudaGetLastError());
		}
	}
	CSC(cudaMemcpy(result, dev_result, sizeof(double) * m, cudaMemcpyDeviceToHost));
	for (int i = 0; i < m; ++i) {
		printf("%.10e ", result[i]);
	}
	free(data);
	return 0;
}