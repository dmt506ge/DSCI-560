import matplotlib.pyplot as plt
import numpy as np

# Data for all implementations
matrix_sizes = [128, 256, 512, 1024, 2048, 4096]

# CPU data (in seconds, only have 512, 1024, 2048)
cpu_sizes = [512, 1024, 2048]
cpu_times_sec = [0.536, 9.773, 100.831]
cpu_times_ms = [t * 1000 for t in cpu_times_sec]  # Convert to ms

# GPU data (in milliseconds)
naive_cuda = [0.0225, 0.0389, 0.1526, 1.0834, 8.2627, 80.6685]
optimized_cuda = [0.0195, 0.0307, 0.1198, 1.1008, 6.6296, 56.5074]
cublas = [0.1423, 0.0471, 0.0338, 0.2314, 0.8158, 5.8716]

# ========== Figure 1: GPU Implementations Comparison ==========
plt.figure(figsize=(12, 7))
plt.plot(matrix_sizes, naive_cuda, 'o-', linewidth=2, markersize=8, label='Naive CUDA')
plt.plot(matrix_sizes, optimized_cuda, 's-', linewidth=2, markersize=8, label='Optimized CUDA (Tiled)')
plt.plot(matrix_sizes, cublas, '^-', linewidth=2, markersize=8, label='cuBLAS')

plt.xlabel('Matrix Size (N×N)', fontsize=14, fontweight='bold')
plt.ylabel('Execution Time (milliseconds)', fontsize=14, fontweight='bold')
plt.title('GPU Matrix Multiplication Performance Comparison', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Use log scale for better visualization
plt.xticks(matrix_sizes)
plt.tight_layout()
plt.savefig('gpu_performance_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: gpu_performance_comparison.png")

# ========== Figure 2: CPU vs GPU Comparison ==========
plt.figure(figsize=(12, 7))

# Plot GPU implementations
plt.plot(matrix_sizes, naive_cuda, 'o-', linewidth=2, markersize=8, label='Naive CUDA', alpha=0.8)
plt.plot(matrix_sizes, optimized_cuda, 's-', linewidth=2, markersize=8, label='Optimized CUDA', alpha=0.8)
plt.plot(matrix_sizes, cublas, '^-', linewidth=2, markersize=8, label='cuBLAS', alpha=0.8)

# Plot CPU (only available sizes)
plt.plot(cpu_sizes, cpu_times_ms, 'D-', linewidth=2, markersize=10, label='CPU', color='red')

plt.xlabel('Matrix Size (N×N)', fontsize=14, fontweight='bold')
plt.ylabel('Execution Time (milliseconds, log scale)', fontsize=14, fontweight='bold')
plt.title('CPU vs GPU Matrix Multiplication Performance', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.xticks(matrix_sizes)
plt.tight_layout()
plt.savefig('cpu_vs_gpu_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: cpu_vs_gpu_comparison.png")

# ========== Figure 3: Speedup Analysis ==========
plt.figure(figsize=(12, 7))

# Calculate speedups for sizes where we have CPU data
speedup_sizes = [512, 1024, 2048]
naive_speedup = [536/0.1526, 9773/1.0834, 100831/8.2627]
optimized_speedup = [536/0.1198, 9773/1.1008, 100831/6.6296]
cublas_speedup = [536/0.0338, 9773/0.2314, 100831/0.8158]

x_pos = np.arange(len(speedup_sizes))
width = 0.25

plt.bar(x_pos - width, naive_speedup, width, label='Naive CUDA', alpha=0.8)
plt.bar(x_pos, optimized_speedup, width, label='Optimized CUDA', alpha=0.8)
plt.bar(x_pos + width, cublas_speedup, width, label='cuBLAS', alpha=0.8)

plt.xlabel('Matrix Size (N×N)', fontsize=14, fontweight='bold')
plt.ylabel('Speedup vs CPU (×)', fontsize=14, fontweight='bold')
plt.title('GPU Speedup Over CPU', fontsize=16, fontweight='bold')
plt.xticks(x_pos, speedup_sizes)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, v in enumerate(naive_speedup):
    plt.text(i - width, v + 3000, f'{v:.0f}×', ha='center', fontsize=9)
for i, v in enumerate(optimized_speedup):
    plt.text(i, v + 3000, f'{v:.0f}×', ha='center', fontsize=9)
for i, v in enumerate(cublas_speedup):
    plt.text(i + width, v + 3000, f'{v:.0f}×', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('speedup_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: speedup_comparison.png")

# ========== Figure 4: Performance Scaling (UPDATED) ==========
plt.figure(figsize=(12, 7))

# Calculate time per element (normalized)
naive_normalized = [t/(n*n*n) * 1e9 for t, n in zip(naive_cuda, matrix_sizes)]  # nanoseconds per operation
optimized_normalized = [t/(n*n*n) * 1e9 for t, n in zip(optimized_cuda, matrix_sizes)]
cublas_normalized = [t/(n*n*n) * 1e9 for t, n in zip(cublas, matrix_sizes)]

plt.plot(matrix_sizes, naive_normalized, 'o-', linewidth=2, markersize=8, label='Naive CUDA')
plt.plot(matrix_sizes, optimized_normalized, 's-', linewidth=2, markersize=8, label='Optimized CUDA')
plt.plot(matrix_sizes, cublas_normalized, '^-', linewidth=2, markersize=8, label='cuBLAS')

plt.xlabel('Matrix Size (N×N)', fontsize=14, fontweight='bold')
plt.ylabel('Time per Operation (nanoseconds)', fontsize=14, fontweight='bold')
plt.title('Efficiency: Time per Floating Point Operation', fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, alpha=0.3)
plt.xticks(matrix_sizes)

# Set y-axis limits to show more range
plt.ylim(0, 80)  # Extended y-axis range from 0 to 100 nanoseconds

plt.tight_layout()
plt.savefig('efficiency_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: efficiency_comparison.png")

print("\n" + "="*60)
print("All graphs generated successfully!")
print("="*60)
print("\nGenerated files:")
print("  1. gpu_performance_comparison.png - GPU implementations only")
print("  2. cpu_vs_gpu_comparison.png - CPU vs all GPU versions")
print("  3. speedup_comparison.png - Speedup factors")
print("  4. efficiency_comparison.png - Time per operation (UPDATED)")