#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
// #include <bits/stdc++.h>
// using namespace std;

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

// текстурная ссылка <тип элементов, размерность, режим нормализации>
texture<uchar4, 2, cudaReadModeElementType> tex;


__global__ void kernel(uchar4 *out, int w, int h, int r) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int x, y, i, j, x_0, y_0, x_n, y_n, pos;
	int counts1[257], counts2[257], counts3[257];
	int k;
	uchar4 p_1;
	// uchar4 p;

	for(y = idy; y < h; y += offsety) {
		for(x = idx; x < w; x += offsetx) {
			x_0 = max(0, x - r);
			x_n = min(w - 1, x + r);
			y_0 = max(0, y - r);
			y_n = min(h - 1, y + r);
			memset(counts1, 0, sizeof(int) * 257);
			memset(counts2, 0, sizeof(int) * 257);
			memset(counts3, 0, sizeof(int) * 257);
			for (i = x_0; i <= x_n; ++i) {
				for (j = y_0; j <= y_n; ++j) {
					p_1 = tex2D(tex, i, j);
					counts1[p_1.x]++;
					counts2[p_1.y]++;
					counts3[p_1.z]++;
				}
			}
			pos = (x_n - x_0 + 1) * (y_n - y_0 + 1) / 2;
			uchar4 res = make_uchar4(0, 0, 0, 0);
			k = 0;
			for (i = 0; i < 257; ++i) {
				k += counts1[i];
				if (k >= pos + 1) {
					res.x = i;
					break;
				}
			}
			k = 0;
			for (i = 0; i < 257; ++i) {
				k += counts2[i];
				if (k >= pos + 1) {
					res.y = i;
					break;
				}
			}
			k = 0;
			for (i = 0; i < 257; ++i) {
				k += counts3[i];
				if (k >= pos + 1) {
					res.z = i;
					break;
				}
			}
			out[y * w + x] = make_uchar4(res.x, res.y, res.z, 0);

		}
	}
}

int main(int argc, const char* argv[]) {

	std::string in, out;
	int w, h, r;
	std::cin >> in >> out >> r;



	FILE *fp = fopen(in.c_str(), "rb");
	fread(&w, sizeof(int), 1, fp);
	fread(&h, sizeof(int), 1, fp);
	uchar4 *data;
	data = (uchar4*)malloc(sizeof(uchar4) * w * h);
	fread(data, sizeof(uchar4), w * h, fp);
	// std::cout << w << " " << h << " " << std::endl;
	fclose(fp);
	cudaArray *arr;
	cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
	cudaMallocArray(&arr, &ch, w, h);

	cudaMemcpyToArray(arr, 0, 0, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice);

	// Подготовка текстурной ссылки, настройка интерфейса работы с данными
	tex.addressMode[0] = cudaAddressModeClamp;	// Политика обработки выхода за границы по каждому измерению
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.channelDesc = ch;
	tex.filterMode = cudaFilterModePoint;		// Без интерполяции при обращении по дробным координатам
	tex.normalized = false;						// Режим нормализации координат: без нормализации

	// Связываем интерфейс с данными
	cudaBindTextureToArray(tex, arr, ch);

	uchar4 *dev_out;
	cudaMalloc(&dev_out, sizeof(uchar4) * w * h);

	kernel<<<dim3(32, 32), dim3(32, 32)>>>(dev_out, w, h, r);
	cudaGetLastError();

	cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost);

	// Отвязываем данные от текстурной ссылки
	cudaUnbindTexture(tex);

	cudaFreeArray(arr);
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
