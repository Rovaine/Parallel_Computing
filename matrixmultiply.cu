#include <stdio.h>
#include <cuda_runtime.h>

/*h_a =
[ 0  1  2  3 ]
[ 1  2  3  4 ]
[ 2  3  4  5 ]
[ 3  4  5  6 ]

h_b =
[  0  -1  -2  -3 ]
[  1   0  -1  -2 ]
[  2   1   0  -1 ]
[  3   2   1   0 ]

*/

// Matrix dimensions (can be adjusted)
#define N 16  // Matrix size N x N

// CUDA kernel for matrix multiplication
__global__ void matrixMultiply(int *a, int *b, int *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

int main() {
    int size = N * N * sizeof(int);
    
    // Host matrices
    int h_a[N][N], h_b[N][N], h_c[N][N];

    // Initialize matrices with some values
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            h_a[i][j] = i + j;
            h_b[i][j] = i - j;
        }
    }

    // Device matrices
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy host matrices to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Define the grid and block sizes
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);

    // Launch kernel
    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Copy result matrix from device to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Display result
    printf("Result matrix:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%d ", h_c[i][j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
