#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define MAX_FILTER_SIZE 25  // Support up to 5×5 filters

// Store filter in constant memory for fast access
__constant__ float d_filter[MAX_FILTER_SIZE];

// CUDA Kernel for convolution
__global__ void convolutionGPU(float *image, float *output, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < M && col < M) {
        float sum = 0.0f;
        int pad = N / 2;
        
        // Apply filter
        for (int fi = 0; fi < N; fi++) {
            for (int fj = 0; fj < N; fj++) {
                int ii = row - pad + fi;
                int jj = col - pad + fj;
                
                // Zero padding for boundaries
                if (ii >= 0 && ii < M && jj >= 0 && jj < M) {
                    sum += image[ii * M + jj] * d_filter[fi * N + fj];
                }
            }
        }
        
        output[row * M + col] = sum;
    }
}

// Create filters (same as CPU version)
void createSobelX(float *filter) {
    float sobel[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    for (int i = 0; i < 9; i++) filter[i] = sobel[i];
}

void createSobelY(float *filter) {
    float sobel[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    for (int i = 0; i < 9; i++) filter[i] = sobel[i];
}

void createBoxBlur(float *filter, int N) {
    float value = 1.0f / (N * N);
    for (int i = 0; i < N * N; i++) filter[i] = value;
}

void createSharpen(float *filter) {
    float sharpen[9] = {0, -1, 0, -1, 5, -1, 0, -1, 0};
    for (int i = 0; i < 9; i++) filter[i] = sharpen[i];
}

int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 512;
    int N = (argc > 2) ? atoi(argv[2]) : 3;
    
    size_t image_size = M * M * sizeof(float);
    size_t filter_size = N * N * sizeof(float);
    
    // Allocate host memory
    float *h_image = (float *)malloc(image_size);
    float *h_filter = (float *)malloc(filter_size);
    float *h_output = (float *)malloc(image_size);
    
    // Initialize
    for (int i = 0; i < M * M; i++) {
        h_image[i] = (float)(rand() % 256);
    }
    
    if (N == 3) {
        createSobelX(h_filter);
    } else {
        createBoxBlur(h_filter, N);
    }
    
    // Allocate device memory
    float *d_image, *d_output;
    cudaMalloc((void**)&d_image, image_size);
    cudaMalloc((void**)&d_output, image_size);
    
    // Copy data to GPU
    cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice);
    
    // Copy filter to constant memory
    cudaMemcpyToSymbol(d_filter, h_filter, filter_size);
    
    // Launch configuration
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((M + 15) / 16, (M + 15) / 16);
    
    // Warm up
    convolutionGPU<<<numBlocks, threadsPerBlock>>>(d_image, d_output, M, N);
    cudaDeviceSynchronize();
    
    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    convolutionGPU<<<numBlocks, threadsPerBlock>>>(d_image, d_output, M, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy result back
    cudaMemcpy(h_output, d_output, image_size, cudaMemcpyDeviceToHost);
    
    printf("GPU Convolution time (M=%d, N=%d): %.4f ms\n", M, N, milliseconds);
    
    // Cleanup
    cudaFree(d_image);
    cudaFree(d_output);
    free(h_image);
    free(h_filter);
    free(h_output);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}

/*
================================================================================
8.3 Filter Kernels Implemented

**1. Sobel Edge Detection**
```
Sobel X (Vertical edges):    Sobel Y (Horizontal edges):
[-1  0  1]                   [-1 -2 -1]
[-2  0  2]                   [ 0  0  0]
[-1  0  1]                   [ 1  2  1]
```

**2. Box Blur (N×N)**
```
1/(N²) × [1  1  ...  1]
         [1  1  ...  1]
         [...      ...]
         [1  1  ...  1]
```

**3. Sharpen**
```
[ 0 -1  0]
[-1  5 -1]
[ 0 -1  0]
```

**4. Laplacian Edge Detection**
```
[-1 -1 -1]
[-1  8 -1]
[-1 -1 -1]
```
================================================================================
*/