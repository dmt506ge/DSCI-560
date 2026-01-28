@echo off
echo Testing Optimized GPU Matrix Multiplication (Tiled)...
echo.

echo Testing N=512...
matrix_gpu_tiled.exe 512 >> gpu_tiled_results.txt

echo Testing N=1024...
matrix_gpu_tiled.exe 1024 >> gpu_tiled_results.txt

echo Testing N=2048...
matrix_gpu_tiled.exe 2048 >> gpu_tiled_results.txt

echo Testing N=4096...
matrix_gpu_tiled.exe 4096 >> gpu_tiled_results.txt

echo.
echo Results saved to gpu_tiled_results.txt
type gpu_tiled_results.txt
