#include <stdio.h>
#include <stdlib.h>

__global__ void dif(double* a, double* b, double* c, int N) {
	int i, idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;
	// printf("%lf\n", a[0]);
	for (i = idx; i < N; i += offset) {
		c[i] = a[i] - b[i];
	}
}

int main(void) {
	int N;
	scanf("%d", &N);

	double *a = (double*) malloc(sizeof(double) * N);
	double *b = (double*) malloc(sizeof(double) * N);
	double *c = (double*) malloc(sizeof(double) * N);
	double *dev_a, *dev_b, *dev_c;

	cudaMalloc((void**)&dev_a, sizeof(double) * N);
	cudaMalloc((void**)&dev_b, sizeof(double) * N);
	cudaMalloc((void**)&dev_c, sizeof(double) * N);

	for(int i = 0; i < N; i++) {
		scanf("%lf", &a[i]);
	}

	// for(int i = 0; i < N; i++) {
	// 	printf("%lf ", a[i]);
	// }

	for(int i = 0; i < N; i++) {
		scanf("%lf", &b[i]);
	}

	// for(int i = 0; i < N; i++) {
	// 	printf("%lf ", b[i]);
	// }

	cudaMemcpy(dev_a, a, sizeof(double) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, sizeof(double) * N, cudaMemcpyHostToDevice);

	dif<<<256, 256>>>(dev_a, dev_b, dev_c, N);

	cudaMemcpy(c, dev_c, sizeof(double) * N, cudaMemcpyDeviceToHost);

	for(int i = 0; i < N; i++) {
		printf("%10.10lf ", c[i]);
	}

	cudaFree(dev_c);
	cudaFree(dev_b);
	cudaFree(dev_a);
	free(a);
	free(b);
	free(c);

	return 0;
}