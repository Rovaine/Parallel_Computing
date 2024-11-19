#include <stdio.h>
#include <cuda_runtime.h>

#define MASK_WIDTH 3
#define TILE_WIDTH 16
#define WIDTH 32  // You can set the image size here
#define HEIGHT 32 // You can set the image size here

// CUDA kernel for 2D convolution
__global__ void convolution2D(float *input, float *output, float *mask, int width, int height) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y * TILE_WIDTH + ty;
    int col_o = blockIdx.x * TILE_WIDTH + tx;
    int row_i = row_o - MASK_WIDTH / 2;
    int col_i = col_o - MASK_WIDTH / 2;

    __shared__ float sharedMem[TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1];

    if (row_i >= 0 && row_i < height && col_i >= 0 && col_i < width) {
        sharedMem[ty][tx] = input[row_i * width + col_i];
    } else {
        sharedMem[ty][tx] = 0.0f;  // Zero padding
    }

    __syncthreads();

    float outputValue = 0.0f;
    if (tx < TILE_WIDTH && ty < TILE_WIDTH) {
        for (int i = 0; i < MASK_WIDTH; i++) {
            for (int j = 0; j < MASK_WIDTH; j++) {
                outputValue += sharedMem[ty + i][tx + j] * mask[i * MASK_WIDTH + j];
            }
        }

        if (row_o < height && col_o < width) {
            output[row_o * width + col_o] = outputValue;
        }
    }
}

int main() {
    int width = WIDTH;
    int height = HEIGHT;

    float h_input[WIDTH * HEIGHT], h_output[WIDTH * HEIGHT];
    float h_mask[MASK_WIDTH * MASK_WIDTH] = {0, -1, 0, -1, 5, -1, 0, -1, 0};  // Example sharpening kernel

    // Initialize input image
    for (int i = 0; i < width * height; i++) {
        h_input[i] = static_cast<float>(i % 256); // Dummy data
    }

    // Allocate device memory
    float *d_input, *d_output, *d_mask;
    cudaMalloc((void**)&d_input, width * height * sizeof(float));
    cudaMalloc((void**)&d_output, width * height * sizeof(float));
    cudaMalloc((void**)&d_mask, MASK_WIDTH * MASK_WIDTH * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, MASK_WIDTH * MASK_WIDTH * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_WIDTH - 1) / TILE_WIDTH);

    // Launch kernel
    convolution2D<<<dimGrid, dimBlock>>>(d_input, d_output, d_mask, width, height);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the output (for validation)
    for (int i = 0; i < width * height; i++) {
        if (i % width == 0) printf("\n");
        printf("%0.2f ", h_output[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mask);

    return 0;
}
