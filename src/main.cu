#include <cuda_runtime.h>

// Declarações externas
void cpu_sequential(float* A, float* B, float* C, int N);
__global__ void cuda_naive(float* A, float* B, float* C, int N);
__global__ void gemm_cuda_tiled(float* A, float* B, float* C, int N);

extern "C" void run_benchmarks(int N, float* h_A, float* h_B, float* h_C_cpu, float* h_C_naive, float* h_C_tiled) {
    // 1. Executa o teste sequencial na CPU primeiro
    cpu_sequential(h_A, h_B, h_C_cpu, N);

    // 2. Alocação e execução na GPU (CUDA)
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threads(32, 32);
    dim3 blocks((N + 31) / 32, (N + 31) / 32);

    // Executa Naive
    cuda_naive<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C_naive, d_C, size, cudaMemcpyDeviceToHost);

    // Limpa a GPU para o próximo teste
    cudaMemset(d_C, 0, size);

    // Executa Tiled
    gemm_cuda_tiled<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C_tiled, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
