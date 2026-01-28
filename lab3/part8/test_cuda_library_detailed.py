import ctypes
import numpy as np
import time

# Load library
lib = ctypes.CDLL("./matrix_lib.dll")

# Define function signatures
lib.gpu_matrix_multiply.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int
]

lib.gpu_matrix_multiply_timed.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int
]
lib.gpu_matrix_multiply_timed.restype = ctypes.c_float

def test_detailed(N):
    """Detailed performance analysis"""
    print(f"\n{'='*60}")
    print(f"Testing N={N}")
    print('='*60)
    
    # Create matrices
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    C_gpu = np.zeros((N, N), dtype=np.float32)
    
    # Test 1: Total time (including data transfer)
    start = time.time()
    lib.gpu_matrix_multiply(A.ravel(), B.ravel(), C_gpu.ravel(), N)
    total_time = time.time() - start
    
    # Test 2: Kernel time only
    kernel_time_ms = lib.gpu_matrix_multiply_timed(A.ravel(), B.ravel(), C_gpu.ravel(), N)
    kernel_time = kernel_time_ms / 1000.0  # Convert to seconds

    # Test 3: NumPy
    start = time.time()
    C_numpy = np.matmul(A, B)
    numpy_time = time.time() - start
    
    # Calculate data transfer overhead
    transfer_time = total_time - kernel_time
    
    # Verify
    max_error = np.max(np.abs(C_gpu - C_numpy))
    
    print(f"\nPerformance Breakdown:")
    print(f"  Total GPU time:        {total_time*1000:.2f} ms")
    print(f"    ├─ Kernel time:      {kernel_time*1000:.2f} ms ({kernel_time/total_time*100:.1f}%)")
    print(f"    └─ Transfer time:    {transfer_time*1000:.2f} ms ({transfer_time/total_time*100:.1f}%)")
    print(f"  NumPy time:            {numpy_time*1000:.2f} ms")
    print(f"\nSpeedups:")
    print(f"  Total speedup:         {numpy_time/total_time:.2f}×")
    print(f"  Kernel-only speedup:   {numpy_time/kernel_time:.2f}×")
    print(f"\nAccuracy:")
    print(f"  Max error:             {max_error:.6f}")
    print(f"  Status:                {'✓ PASS' if max_error < 0.01 else '✗ FAIL'}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Detailed CUDA Library Performance Analysis")
    print("="*60)
    
    for N in [512, 1024, 2048]:
        test_detailed(N)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)