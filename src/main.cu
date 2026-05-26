#include <cuda_runtime.h>
#include <chrono>

// Declarações externas
void cpu_sequential(float* A, float* B, float* C, int N);
__global__ void cuda_naive(float* A, float* B, float* C, int N);
__global__ void gemm_cuda_tiled(float* A, float* B, float* C, int N);

extern "C" void run_benchmarks(
    int N, float* h_A, float* h_B, 
    float* h_C_cpu, float* h_C_naive, float* h_C_tiled,
    float* t_cpu, float* t_naive, float* t_tiled
) {
    // 1. BENCHMARK CPU SEQUENCIAL
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpu_sequential(h_A, h_B, h_C_cpu, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> diff_cpu = end_cpu - start_cpu;
    *t_cpu = diff_cpu.count(); // Tempo em segundos

    // PREPARAÇÃO DA GPU
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threads(32, 32);
    dim3 blocks((N + 31) / 32, (N + 31) / 32);

    // Eventos CUDA para medição de tempo precisa na GPU
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    float milliseconds = 0;

    // 2. BENCHMARK CUDA NAIVE (Memória Global)
    cudaEventRecord(start_event);
    cuda_naive<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop_event);
    cudaDeviceSynchronize();
    
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);
    *t_naive = milliseconds / 1000.0f; // Converte milissegundos para segundos
    cudaMemcpy(h_C_naive, d_C, size, cudaMemcpyDeviceToHost);

    // Limpa a GPU para o próximo teste
    cudaMemset(d_C, 0, size);

    // 3. BENCHMARK CUDA TILED (Shared Memory)
    cudaEventRecord(start_event);
    gemm_cuda_tiled<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop_event);
    cudaDeviceSynchronize();
    
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);
    *t_tiled = milliseconds / 1000.0f; // Converte milissegundos para segundos
    cudaMemcpy(h_C_tiled, d_C, size, cudaMemcpyDeviceToHost);

    // Liberação de memória e destruição de eventos
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
