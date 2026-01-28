@echo off
echo Testing Naive GPU Matrix Multiplication...
echo.

echo Testing N=512...
matrix_gpu_naive.exe 512 >> gpu_results.txt

echo Testing N=1024...
matrix_gpu_naive.exe 1024 >> gpu_results.txt

echo Testing N=2048...
matrix_gpu_naive.exe 2048 >> gpu_results.txt

echo Testing N=4096...
matrix_gpu_naive.exe 4096 >> gpu_results.txt

echo.
echo Results saved to gpu_results.txt
type gpu_results.txt
