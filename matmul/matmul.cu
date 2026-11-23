// ============================================================
// matmul —— C = A × B（N×N float 方阵）
//
// 四个版本，一条完整的优化阶梯：
//   naive  一个线程算一个输出          每 FMA 配 2 条 global load
//   tiled  shared memory 接住重复读    读的地方从 global 换到 shared，次数没变
//   reg4x4 每线程 16 个累加器          每 FMA 的 load 从 2.0 降到 0.5
//   reg8x8 每线程 64 个累加器          load 再降到 0.25，AI 跨过 ridge
//
// Roofline（RTX 4070 Ti Super，FP32）：
//   峰值算力 44.1 TFLOPS     DRAM 带宽 672 GB/s     ridge point 65.6 FLOP/byte
//
// 用法：  matmul.exe [N]
//         N 默认 4096，必须是 256 的倍数
// ============================================================

#include "cuda_check.h"
#include <algorithm>
#include <cmath>

// ============================================================
// v0 —— naive：一个线程算一个输出
// 每线程 2N 次 global load，总读量 2N³；每 1 次 FMA 要 2 条 load 养活
// ============================================================
__global__ void matmul(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;     // .y 管行
    int col = blockIdx.x * blockDim.x + threadIdx.x;     // .x 管列

    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// ============================================================
// v1 —— tiled：block 先把 32×32 的子块读进 shared
//
// global 读量从 2N³ 降到 2N³/32。但【读的次数没变】，只是换了地方，
// 所以加速有限（实测 1.29x）。
//
// 两个 __syncthreads()：填完后的防 RAW（读完才写下一轮），本轮读完的防 WAR
// 不需要 padding：As[k][ty] 同 warp 内地址相同 → 广播；
//                Bs[k][tx] 同 warp 内 bank = tx，铺满 0-31 → 无冲突
// ============================================================
__global__ void matmul_tile(float* A, float* B, float* C, int N) {
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];

    int row = blockIdx.y * 32 + threadIdx.y;
    int col = blockIdx.x * 32 + threadIdx.x;

    float sum = 0.0f;
    for (int t = 0; t < N / 32; t++) {
        As[threadIdx.y][threadIdx.x] = A[row * N + (t * 32 + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(t * 32 + threadIdx.y) * N + col];
        __syncthreads();

        for (int k = 0; k < 32; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    C[row * N + col] = sum;
}

// ============================================================
// v2 —— register tiling 4×4：每线程 16 个累加器
//
// block(32,8) = 256 线程，每 block 出 32 行 × 128 列
// 每 k 迭代：8 次 shared load 养 16 次 FMA（tiled 是 2 次养 1 次）
//
// 坐标映射（三层，y 管行 x 管列）：
//   C 的行 = blockIdx.y*32 + threadIdx.y*4 + i
//   C 的列 = blockIdx.x*128 + threadIdx.x + j*32
//
// 列用 tx + j*32 而不是 tx*4+j：让一个 warp 的 32 个线程落在 32 个不同 bank 上
// ============================================================
__global__ void matmul_reg(float* A, float* B, float* C, int N) {
    __shared__ float As[32][32];
    __shared__ float Bs[32][128];

    int block_row = blockIdx.y * 32;
    int block_col = blockIdx.x * 128;
    int thread_row = threadIdx.y * 4;

    float acc[4][4] = {0};
    float a[4];
    float b[4];

    for (int t = 0; t < N / 32; t++) {
        for (int i = 0; i < 4; i++) {
            As[thread_row + i][threadIdx.x] =
                A[(block_row + thread_row + i) * N + t * 32 + threadIdx.x];
        }
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                Bs[thread_row + i][threadIdx.x + j * 32] =
                    B[(t * 32 + thread_row + i) * N + block_col + threadIdx.x + j * 32];
            }
        }
        __syncthreads();

        for (int k = 0; k < 32; k++) {
            for (int i = 0; i < 4; i++) a[i] = As[thread_row + i][k];
            for (int j = 0; j < 4; j++) b[j] = Bs[k][threadIdx.x + j * 32];
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            C[(block_row + thread_row + i) * N + block_col + threadIdx.x + j * 32] = acc[i][j];
        }
    }
}

// ============================================================
// v3 —— register tiling 8×8：每线程 64 个累加器
//
// block(32,8)，每 block 出 64 行 × 256 列
// 每 k 迭代：16 次 shared load 养 64 次 FMA（0.25）
//
// ⚠️ 两个 y 偏移不一样：As 用 ty*8，Bs 用 ty*4
//    （As 有 64 行，Bs 有 32 行，而 ty 都只有 8 种取值）
// ⚠️ 静态 __shared__ 每个 block 上限 48 KB，这里是 40 KB，刚好在下面
//
// 代价：128 reg/thread → 每 SM 只能开 2 个 block → occupancy 33%
//       但实测比 4×4 快 1.78x —— occupancy 不直接决定性能
// ============================================================
__global__ void matmul_reg8(float* A, float* B, float* C, int N) {
    __shared__ float As[64][32];
    __shared__ float Bs[32][256];

    int block_row = blockIdx.y * 64;
    int block_col = blockIdx.x * 256;

    float acc[8][8] = {0.0f};
    float a[8];
    float b[8];

    for (int t = 0; t < N / 32; t++) {
        for (int i = 0; i < 8; i++) {
            As[threadIdx.y * 8 + i][threadIdx.x] =
                A[(block_row + threadIdx.y * 8 + i) * N + t * 32 + threadIdx.x];
        }
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 8; j++) {
                Bs[threadIdx.y * 4 + i][threadIdx.x + j * 32] =
                    B[(t * 32 + threadIdx.y * 4 + i) * N + block_col + threadIdx.x + j * 32];
            }
        }
        __syncthreads();

        for (int k = 0; k < 32; k++) {
            for (int i = 0; i < 8; i++) a[i] = As[threadIdx.y * 8 + i][k];
            for (int j = 0; j < 8; j++) b[j] = Bs[k][threadIdx.x + j * 32];
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            C[(block_row + threadIdx.y * 8 + i) * N + block_col + threadIdx.x + j * 32] = acc[i][j];
        }
    }
}

// ============================================================
// 基准测试
// ============================================================

struct Result { float ms; double gflops; bool ok; };

static Result bench(void (*kernel)(float*, float*, float*, int),
                    dim3 block, dim3 grid, int N,
                    float* d_a, float* d_b, float* d_c,
                    float* h_a, float* h_b, float* h_c)
{
    cudaEvent_t s0 = nullptr, s1 = nullptr;
    CUDA_CHECK(cudaEventCreate(&s0));
    CUDA_CHECK(cudaEventCreate(&s1));

    float min_ms = 1e12f, ms = 0.0f;
    for (int rep = 0; rep < 4; rep++) {
        if (rep == 0) {
            kernel<<<grid, block>>>(d_a, d_b, d_c, N);
        } else {
            CUDA_CHECK(cudaEventRecord(s0));
            kernel<<<grid, block>>>(d_a, d_b, d_c, N);
            CUDA_CHECK(cudaEventRecord(s1));
            CUDA_CHECK(cudaEventSynchronize(s1));
            CUDA_CHECK(cudaEventElapsedTime(&ms, s0, s1));
            if (ms < min_ms) min_ms = ms;
        }
    }
    CUDA_CHECK_LAUNCH();

    CUDA_CHECK(cudaEventDestroy(s0));
    CUDA_CHECK(cudaEventDestroy(s1));
    CUDA_CHECK(cudaMemcpy(h_c, d_c, (size_t)N * N * sizeof(float), cudaMemcpyDeviceToHost));

    // 抽三点对拍（CPU 双精度）
    int rr[3] = { 0, 0, N - 1 };
    int cc[3] = { 0, 1, N - 1 };
    bool ok = true;
    for (int t = 0; t < 3; t++) {
        double ref = 0.0;
        for (int k = 0; k < N; k++) {
            ref += (double)h_a[rr[t] * N + k] * (double)h_b[k * N + cc[t]];
        }
        double got = (double)h_c[rr[t] * N + cc[t]];
        if (fabs(got - ref) > 1e-3 * (1.0 + fabs(ref))) ok = false;
    }

    Result r;
    r.ms      = min_ms;
    r.gflops  = 2.0 * (double)N * N * N / (min_ms * 1e-3) / 1e9;
    r.ok      = ok;
    return r;
}

int main(int argc, char** argv) {
    int N = argc > 1 ? atoi(argv[1]) : 4096;

    if (N % 256 != 0) {
        printf("N 必须是 256 的倍数\n");
        return 1;
    }

    print_device_info();

    const size_t sz = (size_t)N * N * sizeof(float);

    float* h_a = new float[(size_t)N * N];
    float* h_b = new float[(size_t)N * N];
    float* h_c = new float[(size_t)N * N];
    for (size_t i = 0; i < (size_t)N * N; i++) {
        h_a[i] = (float)(i % 7);
        h_b[i] = (float)(i % 5);
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_a, sz));
    CUDA_CHECK(cudaMalloc((void**)&d_b, sz));
    CUDA_CHECK(cudaMalloc((void**)&d_c, sz));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, sz, cudaMemcpyHostToDevice));

    const double peak = 44.1e3;      // GFLOPS，FP32 峰值

    printf("%-22s %10s %10s %9s %8s %7s\n",
           "version", "min_ms", "GFLOPS", "vs naive", "peak", "check");
    printf("----------------------------------------------------------------------\n");

    float base_ms = 0.0f;

    struct V { const char* name; void (*k)(float*, float*, float*, int); dim3 block, grid; };

    const V vs[] = {
        { "naive",   matmul,      dim3(32, 32), dim3(N / 32,  N / 32) },
        { "tiled",   matmul_tile, dim3(32, 32), dim3(N / 32,  N / 32) },
        { "reg 4x4", matmul_reg,  dim3(32, 8),  dim3(N / 128, N / 32) },
        { "reg 8x8", matmul_reg8, dim3(32, 8),  dim3(N / 256, N / 64) },
    };

    for (int i = 0; i < 4; i++) {
        Result r = bench(vs[i].k, vs[i].block, vs[i].grid, N,
                         d_a, d_b, d_c, h_a, h_b, h_c);
        if (i == 0) base_ms = r.ms;
        printf("%-22s %10.3f %10.1f %8.2fx %7.1f%% %7s\n",
               vs[i].name, r.ms, r.gflops, base_ms / r.ms,
               100.0 * r.gflops / peak, r.ok ? "OK" : "FAIL");
    }

    printf("----------------------------------------------------------------------\n");
    printf("N = %d    FLOP = 2N³ = %.3e\n", N, 2.0 * N * N * N);
    printf("FP32 峰值 44100 GFLOPS / DRAM 672 GB/s / ridge point 65.6 FLOP/byte\n");

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return 0;
}
