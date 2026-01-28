# CUDA Matrix Multiplication and Image Processing Lab

This repository contains implementations of matrix multiplication and image convolution operations comparing CPU and GPU (CUDA) performance.

## Project Structure

### Matrix Multiplication Implementations
- `matrix_cpu.c` - CPU baseline implementation
- `matrix_gpu_naive.cu` - Naive CUDA kernel (one thread per element)
- `matrix_gpu_tiled.cu` - Optimized CUDA with shared memory tiling
- `matrix_cublas.cu` - NVIDIA cuBLAS library implementation

### Image Processing (Convolution)
- `convolution_cpu.c` - CPU convolution with various filters
- `convolution_gpu.cu` - GPU-accelerated convolution with constant memory

### Python Library Interface
- `matrix_lib.cu` - Shared library exposing CUDA functions to Python
- `test_cuda_library.py` - Basic Python interface testing
- `test_cuda_library_detailed.py` - Detailed performance comparison
- `test_image_processing.py` - Image processing with Python

### Utilities
- `check_gpu.cu` - GPU detection and capability checking
- `setup_cuda_env.ps1` - Windows environment setup script
- `plot_performance.py` - Performance visualization tools

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit (12.5+ or 13.1+)
- NVIDIA Driver (555.97+ for CUDA 12.5, 570+ for CUDA 13.1)
- Windows: Visual Studio Build Tools 2022
- Python 3.x with numpy (for Python integration)

## Building

### On Windows

Initialize the Visual Studio environment first:
```powershell
.\setup_cuda_env.ps1
```

Then compile:
```powershell
# CPU versions
gcc matrix_cpu.c -o matrix_cpu.exe -O2
gcc convolution_cpu.c -o convolution_cpu.exe -O2

# GPU versions
nvcc matrix_gpu_naive.cu -o matrix_gpu_naive.exe
nvcc matrix_gpu_tiled.cu -o matrix_gpu_tiled.exe
nvcc matrix_cublas.cu -o matrix_cublas.exe -lcublas
nvcc convolution_gpu.cu -o convolution_gpu.exe

# Python shared library
nvcc -Xcompiler -fPIC -shared matrix_lib.cu -o libmatrix.dll
```

### On Linux

```bash
# CPU versions
gcc matrix_cpu.c -o matrix_cpu -O2
gcc convolution_cpu.c -o convolution_cpu -O2

# GPU versions
nvcc matrix_gpu_naive.cu -o matrix_gpu_naive
nvcc matrix_gpu_tiled.cu -o matrix_gpu_tiled
nvcc matrix_cublas.cu -o matrix_cublas -lcublas
nvcc convolution_gpu.cu -o convolution_gpu

# Python shared library
nvcc -Xcompiler -fPIC -shared matrix_lib.cu -o libmatrix.so
```

## Running

### Matrix Multiplication
```bash
./matrix_cpu 1024
./matrix_gpu_naive 1024
./matrix_gpu_tiled 2048
./matrix_cublas 2048
```

### Convolution
```bash
./convolution_cpu 1024 3    # 1024x1024 image, 3x3 filter
./convolution_gpu 2048 5    # 2048x2048 image, 5x5 filter
```

### Python Interface
```python
python test_cuda_library.py
python test_image_processing.py
```

## Performance Analysis

Expected performance hierarchy (fastest to slowest):
1. cuBLAS (highly optimized)
2. Tiled CUDA (shared memory)
3. Naive CUDA (basic parallelization)
4. CPU (single-threaded baseline)

See `CLAUDE.md` for detailed architecture information and troubleshooting.

## License

Educational project for CUDA programming course.
