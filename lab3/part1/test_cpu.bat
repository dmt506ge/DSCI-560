@echo off
echo Testing CPU Matrix Multiplication...
echo.

echo Testing N=512...
matrix_cpu.exe 512 >> cpu_results.txt

echo Testing N=1024...
matrix_cpu.exe 1024 >> cpu_results.txt

echo Testing N=2048...
matrix_cpu.exe 2048 >> cpu_results.txt

echo Testing N=4096...
matrix_cpu.exe 4096 >> cpu_results.txt

echo.
echo Results saved to cpu_results.txt
type cpu_results.txt
