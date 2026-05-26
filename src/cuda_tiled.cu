#define TILE_WIDTH 32

__global__ void gemm_cuda_tiled(float* A, float* B, float* C, int N) {
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x; int ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    float sum = 0.0f;
    int numTiles = (N + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int ph = 0; ph < numTiles; ++ph) {
        if (row < N && (ph * TILE_WIDTH + tx) < N)
            sA[ty][tx] = A[row * N + ph * TILE_WIDTH + tx];
        else
            sA[ty][tx] = 0.0f;

        if (col < N && (ph * TILE_WIDTH + ty) < N)
            sB[ty][tx] = B[(ph * TILE_WIDTH + ty) * N + col];
        else
            sB[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
