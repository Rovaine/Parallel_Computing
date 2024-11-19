#include <iostream>
#include <cuda_runtime.h>

__global__ void generateBitonicSequence(int *array, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n) {
        // Simple bitonic pattern generation
        // Create a bitonic sequence: first increasing then decreasing
        if (idx < n / 2) {
            array[idx] = idx; // Increasing part
        } else {
            array[idx] = n - 1 - idx; // Decreasing part
        }
    }
}

void createBitonicSequence(int *array, int n) {
    int *d_array;
    size_t size = n * sizeof(int);

    cudaMalloc(&d_array, size);
    
    // Launch kernel with enough threads
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    generateBitonicSequence<<<numBlocks, blockSize>>>(d_array, n);
    
    cudaMemcpy(array, d_array, size, cudaMemcpyDeviceToHost);
    cudaFree(d_array);
}

int main() {
    const int SIZE = 8;
    int h_array[SIZE];

    createBitonicSequence(h_array, SIZE);

    std::cout << "Bitonic Sequence: ";
    for (int i = 0; i < SIZE; i++) {
        std::cout << h_array[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
