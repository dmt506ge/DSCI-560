@echo off
echo Testing cuBLAS Matrix Multiplication...
echo.

echo Testing N=512...
matrix_cublas.exe 512 >> cublas_results.txt

echo Testing N=1024...
matrix_cublas.exe 1024 >> cublas_results.txt

echo Testing N=2048...
matrix_cublas.exe 2048 >> cublas_results.txt

echo Testing N=4096...
matrix_cublas.exe 4096 >> cublas_results.txt

echo.
echo Results saved to cublas_results.txt
type cublas_results.txt
