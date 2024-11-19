#include <stdio.h>

__device__ void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

__device__ void bitonicMerge(int *data, int low, int cnt, int dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        for (int i = low; i < low + k; i++) {
            if (dir == (data[i] > data[i + k])) {
                swap(&data[i], &data[i + k]);
            }
        }
        bitonicMerge(data, low, k, dir);
        bitonicMerge(data, low + k, k, dir);
    }
}

__device__ void bitonicSort(int *data, int low, int cnt, int dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        bitonicSort(data, low, k, 1);   // Sort in ascending order
        bitonicSort(data, low + k, k, 0); // Sort in descending order
        bitonicMerge(data, low, cnt, dir); // Merge the result
    }
}

__global__ void bitonicSortKernel(int *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int step, j, k;

    for (step = 2; step <= n; step *= 2) {
        for (j = step / 2; j > 0; j /= 2) {
            k = idx ^ j;
            if (k > idx) {
                if ((idx & step) == 0) {
                    if (data[idx] > data[k]) {
                        swap(&data[idx], &data[k]);
                    }
                } else {
                    if (data[idx] < data[k]) {
                        swap(&data[idx], &data[k]);
                    }
                }
            }
            __syncthreads();
        }
    }
}

void bitonicSort(int *data, int n) {
    int *d_data;
    size_t size = n * sizeof(int);

    // Allocate device memory
    cudaMalloc((void**)&d_data, size);
    cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    bitonicSortKernel<<<blocks, threadsPerBlock>>>(d_data, n);

    // Copy back to host
    cudaMemcpy(data, d_data, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_data);
}

int main() {
    const int ARRAY_SIZE = 16;
    int data[ARRAY_SIZE] = {12, 5, 8, 1, 9, 4, 3, 15, 7, 6, 11, 14, 2, 0, 10, 13};

    printf("Original array:\n");
    for (int i = 0; i < ARRAY_SIZE; i++) {
        printf("%d ", data[i]);
    }
    printf("\n");

    bitonicSort(data, ARRAY_SIZE);

    printf("Sorted array:\n");
    for (int i = 0; i < ARRAY_SIZE; i++) {
        printf("%d ", data[i]);
    }
    printf("\n");

    return 0;
}
