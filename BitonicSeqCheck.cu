#include <stdio.h>

__device__ bool isBitonicSegment(int *data, int start, int end) {
    bool increasing = true;
    bool decreasing = false;

    for (int i = start; i < end; i++) {
        if (increasing) {
            if (data[i] > data[i + 1]) {
                increasing = false;
                decreasing = true;
            } else if (data[i] == data[i + 1]) {
                return false; // Not strictly increasing or decreasing
            }
        } else if (decreasing) {
            if (data[i] < data[i + 1]) {
                return false; // If we start increasing again
            }
        }
    }
    return true; // If we finish the loop, the segment is bitonic
}

__global__ void checkBitonicKernel(int *data, bool *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread checks one possible segment of the sequence
    if (idx < n - 1) {
        if (isBitonicSegment(data, idx, n - 1)) {
            *result = true; // At least one bitonic segment found
        }
    }
}

bool isBitonic(int *data, int n) {
    int *d_data;
    bool *d_result;
    bool h_result = false;

    // Allocate device memory
    cudaMalloc((void**)&d_data, n * sizeof(int));
    cudaMalloc((void**)&d_result, sizeof(bool));
    cudaMemcpy(d_data, data, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(bool), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    checkBitonicKernel<<<blocks, threadsPerBlock>>>(d_data, d_result, n);

    // Copy result back to host
    cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_data);
    cudaFree(d_result);

    return h_result;
}

int main() {
    const int ARRAY_SIZE = 10;
    int data[ARRAY_SIZE] = {1, 3, 5, 7, 6, 4, 2};

    printf("Checking if the sequence is bitonic...\n");

    if (isBitonic(data, ARRAY_SIZE)) {
        printf("The sequence is bitonic.\n");
    } else {
        printf("The sequence is not bitonic.\n");
    }

    return 0;
}
