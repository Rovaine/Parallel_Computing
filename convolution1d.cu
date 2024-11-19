#include <stdio.h>
#include <cuda_runtime.h>

__global__ void convolution1D(float *input, float *kernel, float *output, int inputSize, int kernelSize) {
    int halfKernelSize = kernelSize / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= halfKernelSize && idx < inputSize - halfKernelSize) {
        float result = 0.0f;

        for (int k = -halfKernelSize; k <= halfKernelSize; k++) {
            result += input[idx + k] * kernel[halfKernelSize + k];
        }
        output[idx] = result;
    }
}

void hostConvolution(float *input, float *kernel, float *output, int inputSize, int kernelSize) {
    float *d_input, *d_kernel, *d_output;
    int inputBytes = inputSize * sizeof(float);
    int kernelBytes = kernelSize * sizeof(float);
    int outputBytes = inputSize * sizeof(float);

    cudaMalloc(&d_input, inputBytes);
    cudaMalloc(&d_kernel, kernelBytes);
    cudaMalloc(&d_output, outputBytes);

    cudaMemcpy(d_input, input, inputBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelBytes, cudaMemcpyHostToDevice);

    int blockSize = 256; // You can change this to optimize performance
    int gridSize = (inputSize + blockSize - 1) / blockSize;

    convolution1D<<<gridSize, blockSize>>>(d_input, d_kernel, d_output, inputSize, kernelSize);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, outputBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}

int main() {
    const int inputSize = 10;
    const int kernelSize = 3;

    // Input array
    float input[inputSize] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    // Kernel (for example, a simple averaging filter)
    float kernel[kernelSize] = {1.0f/3, 1.0f/3, 1.0f/3};
    // Output array
    float output[inputSize] = {0};

    hostConvolution(input, kernel, output, inputSize, kernelSize);

    // Print the output
    printf("Input:  ");
    for (int i = 0; i < inputSize; i++) {
        printf("%f ", input[i]);
    }
    printf("\n");

    printf("Kernel: ");
    for (int i = 0; i < kernelSize; i++) {
        printf("%f ", kernel[i]);
    }
    printf("\n");

    printf("Output: ");
    for (int i = 0; i < inputSize; i++) {
        printf("%f ", output[i]);
    }
    printf("\n");

    return 0;
}
