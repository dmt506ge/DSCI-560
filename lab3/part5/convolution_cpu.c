#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// CPU Convolution: Apply N×N filter to M×M image
void convolutionCPU(float *image, float *filter, float *output, int M, int N) {
    int pad = N / 2;  // Padding size
    
    // For each output pixel
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            float sum = 0.0f;
            
            // Apply filter
            for (int fi = 0; fi < N; fi++) {
                for (int fj = 0; fj < N; fj++) {
                    int ii = i - pad + fi;
                    int jj = j - pad + fj;
                    
                    // Handle boundary (zero padding)
                    if (ii >= 0 && ii < M && jj >= 0 && jj < M) {
                        sum += image[ii * M + jj] * filter[fi * N + fj];
                    }
                }
            }
            
            output[i * M + j] = sum;
        }
    }
}

// Create edge detection filter (Sobel X)
void createSobelX(float *filter) {
    float sobel[9] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };
    for (int i = 0; i < 9; i++) {
        filter[i] = sobel[i];
    }
}

// Create edge detection filter (Sobel Y)
void createSobelY(float *filter) {
    float sobel[9] = {
        -1, -2, -1,
         0,  0,  0,
         1,  2,  1
    };
    for (int i = 0; i < 9; i++) {
        filter[i] = sobel[i];
    }
}

// Create blur filter (Box blur)
void createBoxBlur(float *filter, int N) {
    float value = 1.0f / (N * N);
    for (int i = 0; i < N * N; i++) {
        filter[i] = value;
    }
}

// Create sharpen filter
void createSharpen(float *filter) {
    float sharpen[9] = {
         0, -1,  0,
        -1,  5, -1,
         0, -1,  0
    };
    for (int i = 0; i < 9; i++) {
        filter[i] = sharpen[i];
    }
}

int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 512;  // Image size
    int N = (argc > 2) ? atoi(argv[2]) : 3;    // Filter size
    
    size_t image_size = M * M * sizeof(float);
    size_t filter_size = N * N * sizeof(float);
    
    // Allocate memory
    float *image = (float *)malloc(image_size);
    float *filter = (float *)malloc(filter_size);
    float *output = (float *)malloc(image_size);
    
    // Initialize random image
    for (int i = 0; i < M * M; i++) {
        image[i] = (float)(rand() % 256);  // 0-255 grayscale
    }
    
    // Create Sobel X filter
    if (N == 3) {
        createSobelX(filter);
    } else {
        createBoxBlur(filter, N);
    }
    
    // Benchmark
    clock_t start = clock();
    convolutionCPU(image, filter, output, M, N);
    clock_t end = clock();
    
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("CPU Convolution time (M=%d, N=%d): %f seconds\n", M, N, elapsed);
    
    free(image);
    free(filter);
    free(output);
    
    return 0;
}