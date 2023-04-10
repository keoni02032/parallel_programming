#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <vector>


#define BLO 512
#define BB 250000
#define uint unsigned int

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

__global__ void digit_of_a_number(uint* dev_data, int size, int i, uint* dev_b) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;
	for (int j = idx; j < size; j += offsetx) {
		dev_b[j] = (dev_data[j] >> i) & 1;
	}
}

#define _index(i) ((i) + ((i) >> 8))

__global__ void par_scan(uint* dev_b, int size, uint* displace) {
	__shared__ int sdata[BLO + 120];
	int idx = threadIdx.x;
	int offsetx = (gridDim.x * blockIdx.y + blockIdx.x) * BLO;
	uint tmp;
	int index, of, s;
	if (offsetx + idx < size) {
		sdata[_index(idx)] = dev_b[offsetx + idx];
		for (s = 1; s <= BLO / 2; s <<= 1) {
			__syncthreads();
			of = s - 1;
			index = 2 * s * idx;
			if (index < BLO)
				sdata[_index(of + index + s)] += sdata[_index(of + index)];
		}
		if (idx == 0) {
			displace[gridDim.x * blockIdx.y + blockIdx.x] = sdata[_index(BLO - 1)];
			sdata[_index(BLO - 1)] = 0;
		}
		for (s = BLO / 2; s > 0; s >>= 1) {
			__syncthreads();
			of = s - 1;
			index = 2 * s * idx;
			if (index < BLO) {
				tmp = sdata[_index(of + index + s)];
				sdata[_index(of + index + s)] += sdata[_index(of + index)];
				sdata[_index(of + index)] = tmp;
			}
		}
		__syncthreads();
		dev_b[offsetx + idx] = sdata[_index(idx)];
	}
}

__global__ void back(uint* dev_b, int size, int blocks, uint* displace) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = gridDim.x * blockDim.x;
	for (int i = idx; i < size; i += offset) {
		dev_b[i] += displace[i / BLO];
	}
}


void scan(uint* dev_b, int size) {
	int blocks = (size - 1) / BLO + 1;
	uint* displace;
	cudaMalloc(&displace, blocks * sizeof(uint));
	if (blocks < BLO) {
		par_scan << <dim3(BLO, BLO), dim3(BLO) >> > (dev_b, size, displace);
		std::cerr << blocks << "(1) ";
		CSC(cudaGetLastError());
	}
	else {
		par_scan << <dim3(BLO, BLO), dim3(BLO) >> > (dev_b, size, displace);
		std::cerr << blocks << "(2) ";
		CSC(cudaGetLastError());
	}
	if (blocks == 1) {
		CSC(cudaFree(displace));
		return;
	}
	scan(displace, blocks);
	back << <BLO, BLO >> > (dev_b, size, blocks, displace);
	std::cerr << blocks << "(3) ";
	CSC(cudaGetLastError());
	CSC(cudaFree(displace));
}

__global__ void radix_sort(uint* dev_s, int size, int i, uint* dev_data, uint* dev_b) {
	uint ter = 0;
	if (((dev_s[size - 1] >> i) & 1) == 1) {
		ter = 1;
	}
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;
	for (int j = idx; j < size; j += offsetx) {
		if (((dev_s[j] >> i) & 1) == 0) {
			dev_data[j - dev_b[j]] = dev_s[j];
		}
		else {
			dev_data[dev_b[j] + (size - (dev_b[size - 1] + ter))] = dev_s[j];
		}
	}
}

int main() {
	std::ios_base::sync_with_stdio(0);
	std::cin.tie(0);
	int size;
	fread(&size, sizeof(int), 1, stdin);
	 // std::cin >> size;
	if (size == 0) {
		return 0;
	}
	uint* data = new uint[size];
	fread(data, sizeof(uint), size, stdin);
	 // for (int i = 0; i < size; ++i) {
	 // 	std::cin >> data[i];
	 // }
	std::cerr << size << " ";
	//for (int i = 0; i < size; ++i) {
	//	std::cerr << data[i] << " ";
	//}
	uint* dev_data, * dev_s, * dev_b, * tmp;
	CSC(cudaMalloc(&dev_data, size * sizeof(uint)));
	CSC(cudaMalloc(&dev_s, size * sizeof(uint)));
	CSC(cudaMalloc(&dev_b, size * sizeof(uint)));
	CSC(cudaMemcpy(dev_data, data, size * sizeof(uint), cudaMemcpyHostToDevice));
	CSC(cudaMemcpy(dev_s, data, size * sizeof(uint), cudaMemcpyHostToDevice));
	for (int j = 0; j < 32; j++) {

		std::cerr << j << " " << "!!!!!!!!" << std::endl;

		digit_of_a_number << <BLO, BLO >> > (dev_data, size, j, dev_b);
		CSC(cudaGetLastError());
		//	CSC(cudaMemcpy(data, dev_b, size * sizeof(uint), cudaMemcpyDeviceToHost));
		//	for (int i = 0; i < size; ++i) {
		//		std::cout << data[i] << " ";
		//	}
		//	std::cout << std::endl;
		scan(dev_b, size);

		//	CSC(cudaMemcpy(data, dev_b, size * sizeof(uint), cudaMemcpyDeviceToHost));
		//	for (int i = 0; i < size; ++i) {
		//		std::cout << data[i] << " ";
		//	}
		//	std::cout << std::endl;
		//	CSC(cudaMemcpy(data, dev_data, size * sizeof(uint), cudaMemcpyDeviceToHost));
		//	for (int i = 0; i < size; ++i) {
		//		std::cout << data[i] << " ";
		//	}
		//	std::cout << std::endl;


		tmp = dev_data;
		dev_data = dev_s;
		dev_s = tmp;


		radix_sort << <BLO, BLO >> > (dev_s, size, j, dev_data, dev_b);


		//	CSC(cudaMemcpy(data, dev_data, size * sizeof(uint), cudaMemcpyDeviceToHost));
		//	for (int i = 0; i < size; ++i) {
		//		std::cout << data[i] << " ";
		//	}
		//	std::cout << std::endl;
		//	std::cout << std::endl;
		//	std::cout << std::endl;
		CSC(cudaGetLastError());
	}

	//scan(dev_data, size);


	CSC(cudaMemcpy(data, dev_data, size * sizeof(uint), cudaMemcpyDeviceToHost));
	fwrite(data, sizeof(uint), size, stdout);
	 // for (int i = 0; i < size; ++i) {
	 // 	std::cout << data[i] << " ";
	 // }
	cudaFree(dev_data);
	cudaFree(dev_b);
	cudaFree(dev_s);
	free(data);
	return 0;
}

// 11
// 0 1 1 0 1 0 0 1 1 0 1
