import ctypes
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt

# Load CUDA library
lib = ctypes.CDLL("./matrix_lib.dll")

# Define convolution function signatures
lib.gpu_convolution.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_int
]

lib.gpu_convolution_timed.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_int
]
lib.gpu_convolution_timed.restype = ctypes.c_float




# Define filter kernels
def create_sobel_x():
    """Edge detection - vertical edges"""
    return np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

def create_sobel_y():
    """Edge detection - horizontal edges"""
    return np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32)

def create_box_blur(N):
    """Blur filter"""
    return np.ones((N, N), dtype=np.float32) / (N * N)

def create_sharpen():
    """Sharpen filter"""
    return np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32)

def create_edge_detect():
    """Laplacian edge detection"""
    return np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=np.float32)

def generate_test_image(M):
    """Generate a test image with patterns"""
    image = np.zeros((M, M), dtype=np.float32)
    
    # Add some shapes
    # Circle
    center = M // 2
    radius = M // 4
    for i in range(M):
        for j in range(M):
            if (i - center)**2 + (j - center)**2 < radius**2:
                image[i, j] = 255
    
    # Square
    start = M // 4
    end = M // 4 + M // 8
    image[start:end, start:end] = 200
    
    # Horizontal line
    image[M // 4, :] = 150
    
    # Vertical line
    image[:, 3 * M // 4] = 150
    
    return image

def apply_filter_gpu(image, filter_kernel):
    """Apply filter using GPU"""
    M = image.shape[0]
    N = filter_kernel.shape[0]
    
    output = np.zeros((M, M), dtype=np.float32)
    
    kernel_time_ms = lib.gpu_convolution_timed(
        image.ravel(),
        filter_kernel.ravel(),
        output.ravel(),
        M, N
    )
    
    return output, kernel_time_ms

def test_filters():
    """Test different filters on generated image"""
    M = 512
    
    # Generate test image
    image = generate_test_image(M)
    
    # Define filters to test
    filters = {
        'Sobel X': create_sobel_x(),
        'Sobel Y': create_sobel_y(),
        'Box Blur 5×5': create_box_blur(5),
        'Sharpen': create_sharpen(),
        'Edge Detect': create_edge_detect()
    }
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    # Show original
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Apply each filter
    print("\n" + "="*60)
    print(f"Testing Filters on {M}×{M} Image")
    print("="*60)
    
    for idx, (name, filter_kernel) in enumerate(filters.items(), start=1):
        print(f"\n{name}:")
        
        result, gpu_time = apply_filter_gpu(image, filter_kernel)
        
        print(f"  GPU kernel time: {gpu_time:.4f} ms")
        
        # Normalize for display
        result_normalized = np.abs(result)
        result_normalized = (result_normalized / result_normalized.max() * 255).astype(np.uint8)
        
        axes[idx].imshow(result_normalized, cmap='gray')
        axes[idx].set_title(f'{name}\n{gpu_time:.4f} ms')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('filter_results.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization to filter_results.png")

def benchmark_sizes():
    """Benchmark different image and filter sizes"""
    print("\n" + "="*60)
    print("Convolution Performance Benchmark")
    print("="*60)
    
    test_configs = [
        (256, 3), (512, 3), (1024, 3),
        (512, 5), (1024, 5), (512, 7)
    ]
    
    for M, N in test_configs:
        image = np.random.rand(M, M).astype(np.float32) * 255
        filter_kernel = create_box_blur(N)
        
        _, gpu_time = apply_filter_gpu(image, filter_kernel)
        
        print(f"M={M:4d}, N={N}: {gpu_time:7.4f} ms")

if __name__ == "__main__":
    print("="*60)
    print("CUDA Image Processing Demo")
    print("="*60)
    
    # Test filters with visualization
    test_filters()
    
    # Performance benchmark
    benchmark_sizes()
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)