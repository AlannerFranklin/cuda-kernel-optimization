// ============================================================
// histogram —— 统计每个 bin 出现多少次（256 个 bin）
//
// 两个版本：
//   naive   每个元素直接 atomicAdd 到 global 的 bin
//   shared  每个 block 建私有直方图，最后合并
//
// 核心对比：naive 是 1600 万次原子加打在 256 个全局地址上（激烈竞争），
// shared 版把竞争限制在 block 内，同时用 grid-stride 把全局原子加的次数
// 压掉 64 倍。
//
// 另一个重点：这是【唯一一个必须用 atomicAdd】的课题。所以 harness
// 每次 launch 前都要 memset —— atomicAdd 是累加的，不清零会翻 N 倍。
//
// 用法：  histogram.exe [N]
//         N 默认 16777216（64 MB，超过 48 MB 的 L2）
// ============================================================

#include "cuda_check.h"
#include <algorithm>
#include <cmath>

constexpr int NUM_BINS = 256;
constexpr int TPB      = 256;

// ------------------------------------------------------------
// v0 —— naive：每线程处理一个元素，直接 atomicAdd 到 global
//
// 两个符号各管一件事：
//   a[i]          取出的值（bin 编号）
//   &out[a[i]]    那个 bin 的【地址】—— atomicAdd 要地址，因为它要读-改-写
// ------------------------------------------------------------
__global__ void histogram_naive(int* a, int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        atomicAdd(&out[a[i]], 1);
    }
}

// ------------------------------------------------------------
// v1 —— shared 私有直方图 + grid-stride
//
// ⚠️ 只加 shared 是【没用的】：
//    65536 个 block × 256 次合并 = 1677 万次全局原子加，一次没少。
//    必须同时用 grid-stride 把 block 数降下来（这里固定 1024 个）。
//
// 三个循环，只有中间那个需要 grid-stride：
//   第 1 个：本 block 把 256 个 bin 清零
//   第 2 个：遍历【输入元素】(0..N-1)，跨 block 分活
//   第 3 个：遍历【bin】(0..255)，每 block 各自合并自己那份
// ------------------------------------------------------------
__global__ void histogram_shared(int* a, int* out, int N) {
    __shared__ int hist[NUM_BINS];

    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) hist[i] = 0;
    __syncthreads();

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        atomicAdd(&hist[a[i]], 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        if (hist[i] > 0) atomicAdd(&out[i], hist[i]);      // 跳过空 bin
    }
}

// ============================================================
// 基准测试
// ============================================================

static float bench(void (*kernel)(int*, int*, int),
                   dim3 grid, dim3 block, int N,
                   int* d_a, int* d_out, int* h_out, const int* ref,
                   bool* ok_out, double* gbps_out)
{
    cudaEvent_t s0 = nullptr, s1 = nullptr;
    CUDA_CHECK(cudaEventCreate(&s0));
    CUDA_CHECK(cudaEventCreate(&s1));

    float min_ms = 1e12f, ms = 0.0f;

    for (int rep = 0; rep < 4; rep++) {
        // ★ 必须每次都清零：atomicAdd 是累加的，而这里 launch 了 4 次
        CUDA_CHECK(cudaMemset(d_out, 0, NUM_BINS * sizeof(int)));

        if (rep == 0) {
            kernel<<<grid, block>>>(d_a, d_out, N);
        } else {
            CUDA_CHECK(cudaEventRecord(s0));
            kernel<<<grid, block>>>(d_a, d_out, N);
            CUDA_CHECK(cudaEventRecord(s1));
            CUDA_CHECK(cudaEventSynchronize(s1));
            CUDA_CHECK(cudaEventElapsedTime(&ms, s0, s1));
            if (ms < min_ms) min_ms = ms;
        }
    }
    CUDA_CHECK_LAUNCH();

    CUDA_CHECK(cudaEventDestroy(s0));
    CUDA_CHECK(cudaEventDestroy(s1));
    CUDA_CHECK(cudaMemcpy(h_out, d_out, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < NUM_BINS; i++) {
        if (h_out[i] != ref[i]) { ok = false; break; }
    }

    *ok_out   = ok;
    *gbps_out = (double)N * sizeof(int) / (min_ms * 1e-3) / 1e9;
    return min_ms;
}

int main(int argc, char** argv) {
    int N = argc > 1 ? atoi(argv[1]) : 16777216;

    print_device_info();

    int* h_a   = new int[N];
    int* h_out = new int[NUM_BINS];

    // 值域 [0,255]，7 与 256 互质 → 各 bin 均匀分布
    for (int i = 0; i < N; i++) h_a[i] = (i * 7 + 3) % NUM_BINS;

    int* ref = new int[NUM_BINS]();
    for (int i = 0; i < N; i++) ref[h_a[i]]++;

    int* d_a   = nullptr;
    int* d_out = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_a,   (size_t)N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_out, NUM_BINS * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, (size_t)N * sizeof(int), cudaMemcpyHostToDevice));

    printf("%-26s %10s %10s %9s %7s\n", "version", "min_ms", "GB/s", "vs naive", "check");
    printf("------------------------------------------------------------------\n");

    bool ok = false;
    double gbps = 0.0;

    // ---- naive：每 block 256 个元素 ----
    float t0 = bench(histogram_naive, dim3(N / TPB), dim3(TPB), N,
                     d_a, d_out, h_out, ref, &ok, &gbps);
    printf("%-26s %10.6f %10.1f %8.2fx %7s\n",
           "naive (global atomic)", t0, gbps, 1.0, ok ? "OK" : "FAIL");

    // ---- shared：固定 1024 个 block，每线程 grid-stride ----
    float t1 = bench(histogram_shared, dim3(1024), dim3(TPB), N,
                     d_a, d_out, h_out, ref, &ok, &gbps);
    printf("%-26s %10.6f %10.1f %8.2fx %7s\n",
           "shared (+grid-stride)", t1, gbps, t0 / t1, ok ? "OK" : "FAIL");

    printf("------------------------------------------------------------------\n");
    printf("N = %d (%.1f MB)   峰值 672 GB/s\n", N, (double)N * sizeof(int) / 1e6);
    printf("naive 慢是因为 %d 次原子加打在 256 个全局地址上\n", N);

    delete[] h_a;
    delete[] h_out;
    delete[] ref;
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
