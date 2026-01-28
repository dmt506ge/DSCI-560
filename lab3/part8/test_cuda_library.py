import ctypes
import numpy as np
import time

# Load the CUDA library
# On Windows, use .dll
lib = ctypes.CDLL("./matrix_lib.dll")

# Define the function signature
lib.gpu_matrix_multiply.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int
]

def test_matrix_multiply(N):
    """Test GPU matrix multiplication via Python"""
    print(f"\nTesting N={N}...")
    
    # Create random matrices
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    C = np.zeros((N, N), dtype=np.float32)
    
    # Measure time for GPU computation via library
    start = time.time()
    lib.gpu_matrix_multiply(A.ravel(), B.ravel(), C.ravel(), N)
    gpu_time = time.time() - start
    
    # Verify correctness with NumPy
    start = time.time()
    C_numpy = np.matmul(A, B)
    numpy_time = time.time() - start
    
    # Check if results match
    max_error = np.max(np.abs(C - C_numpy))
    
    print(f"  GPU time (via Python): {gpu_time:.4f} seconds")
    print(f"  NumPy time: {numpy_time:.4f} seconds")
    print(f"  Speedup: {numpy_time/gpu_time:.2f}×")
    print(f"  Max error: {max_error:.6f}")
    
    if max_error < 0.01:
        print(f"  ✓ Results match!")
    else:
        print(f"  ✗ Results don't match!")
    
    return gpu_time, numpy_time

if __name__ == "__main__":
    print("="*60)
    print("Testing CUDA Library from Python")
    print("="*60)
    
    sizes = [512, 1024, 2048]
    
    for N in sizes:
        test_matrix_multiply(N)
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)