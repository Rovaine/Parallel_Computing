#include <stdio.h>
#include <cuda_runtime.h>
//to run >nvcc arraysum.cu -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.34.31933\bin\Hostx64\x64", then a.exe on CLI
// CUDA kernel to compute element-wise sum of two arrays
__global__ void arraySum(int *a, int *b, int *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    int n = 10;  // Size of the arrays
    int size = n * sizeof(int);
    
    // Host arrays
    int h_a[10], h_b[10], h_c[10];

    // Initialize the arrays with some values
    for (int i = 0; i < n; ++i) {
        h_a[i] = i;           // Array h_a = [0, 1, 2, ..., 9]
        h_b[i] = i * 2;       // Array h_b = [0, 2, 4, ..., 18]
    }

    // Device arrays
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy the host arrays to device memory
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Define block and grid size
    int blockSize = 256; // Number of threads per block
    int gridSize = (n + blockSize - 1) / blockSize; // Number of blocks needed

    // Launch the kernel
    arraySum<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    // Copy the result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Display the result
    printf("Resultant array:\n");
    for (int i = 0; i < n; ++i) {
        printf("%d ", h_c[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
