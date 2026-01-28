#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_WIDTH 16
#define MAX_FILTER_SIZE 49  // Support up to 7×7 filters

// Constant memory for filter
__constant__ float d_filter[MAX_FILTER_SIZE];

// Matrix multiplication kernel with tiling
__global__ void matrixMultiplyTiled(float *A, float *B, float *C, int N) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    
    float Pvalue = 0.0;
    
    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        if (Row < N && (m * TILE_WIDTH + tx) < N)
            ds_A[ty][tx] = A[Row * N + m * TILE_WIDTH + tx];
        else
            ds_A[ty][tx] = 0.0f;
        
        if (Col < N && (m * TILE_WIDTH + ty) < N)
            ds_B[ty][tx] = B[(m * TILE_WIDTH + ty) * N + Col];
        else
            ds_B[ty][tx] = 0.0f;
        
        __syncthreads();
        
        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += ds_A[ty][k] * ds_B[k][tx];
        
        __syncthreads();
    }
    
    if (Row < N && Col < N)
        C[Row * N + Col] = Pvalue;
}

// Convolution kernel
__global__ void convolutionGPU(float *image, float *output, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < M && col < M) {
        float sum = 0.0f;
        int pad = N / 2;
        
        for (int fi = 0; fi < N; fi++) {
            for (int fj = 0; fj < N; fj++) {
                int ii = row - pad + fi;
                int jj = col - pad + fj;
                
                if (ii >= 0 && ii < M && jj >= 0 && jj < M) {
                    sum += image[ii * M + jj] * d_filter[fi * N + fj];
                }
            }
        }
        
        output[row * M + col] = sum;
    }
}

extern "C" {
    // Matrix multiplication - basic version
    __declspec(dllexport) void gpu_matrix_multiply(float *h_A, float *h_B, float *h_C, int N) {
        size_t size = N * N * sizeof(float);
        
        float *d_A, *d_B, *d_C;
        cudaMalloc((void**)&d_A, size);
        cudaMalloc((void**)&d_B, size);
        cudaMalloc((void**)&d_C, size);
        
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
        
        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
        dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);
        
        matrixMultiplyTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
        
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
    
    // Matrix multiplication - timed version (FIXED)
    __declspec(dllexport) float gpu_matrix_multiply_timed(float *h_A, float *h_B, float *h_C, int N) {
        size_t size = N * N * sizeof(float);
        
        float *d_A, *d_B, *d_C;
        cudaMalloc((void**)&d_A, size);
        cudaMalloc((void**)&d_B, size);
        cudaMalloc((void**)&d_C, size);
        
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
        
        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
        dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);
        
        // Time only kernel execution
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        matrixMultiplyTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
        
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        return milliseconds;  // FIXED: Added return statement
    }
    
    // Convolution - basic version
    __declspec(dllexport) void gpu_convolution(float *h_image, float *h_filter, float *h_output, int M, int N) {
        size_t image_size = M * M * sizeof(float);
        size_t filter_size = N * N * sizeof(float);
        
        float *d_image, *d_output;
        cudaMalloc((void**)&d_image, image_size);
        cudaMalloc((void**)&d_output, image_size);
        
        cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(d_filter, h_filter, filter_size);
        
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((M + 15) / 16, (M + 15) / 16);
        
        convolutionGPU<<<numBlocks, threadsPerBlock>>>(d_image, d_output, M, N);
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_output, d_output, image_size, cudaMemcpyDeviceToHost);
        
        cudaFree(d_image);
        cudaFree(d_output);
    }
    
    // Convolution - timed version
    __declspec(dllexport) float gpu_convolution_timed(float *h_image, float *h_filter, float *h_output, int M, int N) {
        size_t image_size = M * M * sizeof(float);
        size_t filter_size = N * N * sizeof(float);
        
        float *d_image, *d_output;
        cudaMalloc((void**)&d_image, image_size);
        cudaMalloc((void**)&d_output, image_size);
        
        cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(d_filter, h_filter, filter_size);
        
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((M + 15) / 16, (M + 15) / 16);
        
        // Time only kernel execution
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        convolutionGPU<<<numBlocks, threadsPerBlock>>>(d_image, d_output, M, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        cudaMemcpy(h_output, d_output, image_size, cudaMemcpyDeviceToHost);
        
        cudaFree(d_image);
        cudaFree(d_output);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        return milliseconds;
    }
}