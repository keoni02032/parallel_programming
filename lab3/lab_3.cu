#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <vector>

typedef struct {
	int x;
	int y;
} point;

typedef struct {
	float x = 0;
	float y = 0;
	float z = 0;
} cord_float;

__constant__ float avg[32][3];
__constant__ float norm_avg[32][3];

__global__ void kernel(uchar4* pixels, int w, int h, int count_clases) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int x, y, i, max_p;
	cord_float rgb;
	float max_val;
	float result[32];
	for(y = idy; y < h; y += offsety) {
		for(x = idx; x < w; x += offsetx) {
			uchar4 p;
			p.x = pixels[y * w + x].x;
			p.y = pixels[y * w + x].y;
			p.z = pixels[y * w + x].z;
			for (i = 0; i < count_clases; ++i) {
				rgb.x = p.x * norm_avg[i][0];
				rgb.y = p.y * norm_avg[i][1];
				rgb.z = p.z * norm_avg[i][2];
				result[i] = rgb.x + rgb.y + rgb.z;
			}
			max_val = result[0];
			max_p = 0;
			for (i = 1; i < count_clases; ++i) {
				if (max_val < result[i]) {
					max_val = result[i];
					max_p = i;
				}
			}
			pixels[y * w + x].w = (char)max_p;
		}
	}
}

int main(int argc, const char* argv[])
{
	std::string in, out;
	int w, h, count_clases, size;
	std::cin >> in >> out >> count_clases;
	std::vector<std::vector<point>> vv(count_clases);
	for (int i = 0; i < count_clases; ++i) {
		std::cin >> size;
		vv[i].resize(size);
		for (int j = 0; j < size; ++j) {
			point t;
			std::cin >> t.x >> t.y;
			vv[i][j] = t;
		}
	}
	FILE *fp = fopen(in.c_str(), "rb");
	fread(&w, sizeof(int), 1, fp);
	fread(&h, sizeof(int), 1, fp);
	uchar4 *data;
	data = (uchar4*)malloc(sizeof(uchar4) * w * h);
	fread(data, sizeof(uchar4), w * h, fp);
	fclose(fp);uchar4 *dev_out;
	cudaMalloc(&dev_out, sizeof(uchar4) * w * h);
	cudaMemcpy(dev_out, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice);
	

	float cpu_avg[32][3];
	for (int i = 0; i < count_clases; ++i) {
		point c;
		cpu_avg[i][0] = 0;
		cpu_avg[i][1] = 0;
		cpu_avg[i][2] = 0;
		for (int j = 0; j < vv[i].size(); ++j) {
			c.x = vv[i][j].x;
			c.y = vv[i][j].y;
			cpu_avg[i][0] += data[c.y * w + c.x].x;
			cpu_avg[i][1] += data[c.y * w + c.x].y;
			cpu_avg[i][2] += data[c.y * w + c.x].z;
		}
		cpu_avg[i][0] /= vv[i].size();
		cpu_avg[i][1] /= vv[i].size();
		cpu_avg[i][2] /= vv[i].size();
	}
	
	float norm_cpu_avg[32][3];
	for (int i = 0; i < count_clases; ++i) {
		norm_cpu_avg[i][0] = cpu_avg[i][0] / std::sqrt(cpu_avg[i][0] * cpu_avg[i][0] + cpu_avg[i][1] * cpu_avg[i][1] + cpu_avg[i][2] * cpu_avg[i][2]);
		norm_cpu_avg[i][1] = cpu_avg[i][1] / std::sqrt(cpu_avg[i][0] * cpu_avg[i][0] + cpu_avg[i][1] * cpu_avg[i][1] + cpu_avg[i][2] * cpu_avg[i][2]);
		norm_cpu_avg[i][2] = cpu_avg[i][2] / std::sqrt(cpu_avg[i][0] * cpu_avg[i][0] + cpu_avg[i][1] * cpu_avg[i][1] + cpu_avg[i][2] * cpu_avg[i][2]);
	}
	
	cudaMemcpyToSymbol(avg, cpu_avg, sizeof(float) * 32 * 3);
	cudaMemcpyToSymbol(norm_avg, norm_cpu_avg, sizeof(float) * 32 * 3);
    kernel<<<dim3(32, 32), dim3(32, 32)>>>(dev_out, w, h, count_clases);
	cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost);
	cudaFree(dev_out);
	FILE *fp1;
	fp1 = fopen(out.c_str(), "wb");
	fwrite(&w, sizeof(int), 1, fp1);
	fwrite(&h, sizeof(int), 1, fp1);
	fwrite(data, sizeof(uchar4), w * h, fp1);
	fclose(fp1);
	free(data);
	return 0;
}