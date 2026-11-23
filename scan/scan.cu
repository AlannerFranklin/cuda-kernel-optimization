// ============================================================
// scan —— 前缀和（exclusive scan）
//
// 输入 a，输出 out，要求 out[i] = a[0] + a[1] + ... + a[i-1]（不含自己）
//
// 四个版本：
//   v0  Blelloch（上扫 + 下扫），【分段】—— block 之间不传递
//   v1  三 kernel 两遍法，修成【全局】
//   v2  块内换成 Hillis-Steele + warp shuffle
//   v3  每线程 4 个元素
//
// 前缀和是「输出依赖前缀」这类问题的通用解，用途：
//   stream compaction / radix sort 的分桶偏移 / 稀疏矩阵的行指针
//
// 用法：  scan.exe [N]
//         N 默认 262144（= 256 × 1024），保证 nBlocks = 1024 ≤ 一个 block 能扫完
// ============================================================

#include "cuda_check.h"
#include <algorithm>
#include <cmath>

constexpr int TILE        = 256;      // 每 block 线程数（v0/v1/v2 也是一次处理的元素数）
constexpr int BLOCK_ELEMS = 1024;     // v3 每 block 覆盖的元素数 = TILE × 4

// ============================================================
// v0 —— Blelloch：上扫 + 根清零 + 下扫（单 block，分段）
//
// 上扫：a[i] += a[i-d]，第 d 层活跃位置减半
// 下扫：t = a[i1]; a[i1] = a[i2]; a[i2] = t + a[i2]
//
// ⚠️ 这版只扫自己那 256 个，block 之间不传递 —— 输出是【分段前缀和】
// ============================================================
__global__ void scan_v0(float* a, float* out, int N) {
    __shared__ float tile[TILE];
    const int n = TILE;

    tile[threadIdx.x] = a[blockIdx.x * n + threadIdx.x];
    __syncthreads();

    for (int d = 1; d < n; d *= 2) {                 // 上扫
        int i = threadIdx.x * 2 * d;
        if (i + 2 * d - 1 < n) {
            tile[i + 2 * d - 1] += tile[i + d - 1];
        }
        __syncthreads();
    }

    tile[n - 1] = 0;                                 // 根清零：树根的「总和」→「前面的总和」
    __syncthreads();

    for (int d = n / 2; d > 0; d /= 2) {             // 下扫
        int i = threadIdx.x * 2 * d;
        if (i + 2 * d - 1 < n) {
            float t = tile[i + d - 1];
            tile[i + d - 1]   = tile[i + 2 * d - 1];
            tile[i + 2 * d - 1] = t + tile[i + 2 * d - 1];
        }
        __syncthreads();
    }

    out[blockIdx.x * n + threadIdx.x] = tile[threadIdx.x];
}

// ============================================================
// 下面三个 kernel 是 v1 / v2 / v3 共用的
// ============================================================

// ------------------------------------------------------------
// scan_offsets —— 对 block_sums 做一次 exclusive scan（单 block，最多 1024 个）
// 调用时 block 大小传 nBlocks：<<<1, nBlocks>>>
// ------------------------------------------------------------
__global__ void scan_offsets(float* block_sums, int nBlocks) {
    __shared__ float tile[1024];
    const int n = nBlocks;

    tile[threadIdx.x] = block_sums[threadIdx.x];
    __syncthreads();

    for (int d = 1; d < n; d *= 2) {
        int i = threadIdx.x * 2 * d;
        if (i + 2 * d - 1 < n) {
            tile[i + 2 * d - 1] += tile[i + d - 1];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) tile[n - 1] = 0;
    __syncthreads();

    for (int d = n / 2; d > 0; d /= 2) {
        int i = threadIdx.x * 2 * d;
        if (i + 2 * d - 1 < n) {
            float t = tile[i + d - 1];
            tile[i + d - 1]   = tile[i + 2 * d - 1];
            tile[i + 2 * d - 1] = t + tile[i + 2 * d - 1];
        }
        __syncthreads();
    }

    block_sums[threadIdx.x] = tile[threadIdx.x];
}

// ------------------------------------------------------------
// scan_add —— 把块偏移加回每个元素
// elemsPerBlock：v1/v2 传 TILE(256)，v3 传 BLOCK_ELEMS(1024)
//
// 用「线程跨步」而不是每线程连续几个：同一 warp 每轮读连续 32 个 float
// （128 字节 = 1 个 cache line），合并访存。
// ------------------------------------------------------------
__global__ void scan_add(float* out, float* block_sums, int elemsPerBlock) {
    int base    = blockIdx.x * elemsPerBlock;
    float delta = block_sums[blockIdx.x];

    for (int i = threadIdx.x; i < elemsPerBlock; i += blockDim.x) {
        out[base + i] += delta;
    }
}

// ------------------------------------------------------------
// v1 —— 本地 Blelloch + 记录本块总和
// 上扫结束时 tile[TILE-1] 正好是本块总和，必须在清零根之前存下来
// ------------------------------------------------------------
__global__ void scan_v1_local(float* a, float* out, float* block_sums, int N) {
    __shared__ float tile[TILE];
    const int n = TILE;

    tile[threadIdx.x] = a[blockIdx.x * TILE + threadIdx.x];
    __syncthreads();

    for (int d = 1; d < n; d *= 2) {
        int i = threadIdx.x * 2 * d;
        if (i + 2 * d - 1 < n) {
            tile[i + 2 * d - 1] += tile[i + d - 1];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) block_sums[blockIdx.x] = tile[TILE - 1];
    __syncthreads();

    tile[TILE - 1] = 0;
    __syncthreads();

    for (int d = n / 2; d > 0; d /= 2) {
        int i = threadIdx.x * 2 * d;
        if (i + 2 * d - 1 < n) {
            float t = tile[i + d - 1];
            tile[i + d - 1]   = tile[i + 2 * d - 1];
            tile[i + 2 * d - 1] = t + tile[i + 2 * d - 1];
        }
        __syncthreads();
    }

    out[blockIdx.x * TILE + threadIdx.x] = tile[threadIdx.x];
}

// ------------------------------------------------------------
// v2 —— 块内换成 Hillis-Steele + warp shuffle
//
// 每层 x[i] += x[i-off]，【必须用上一层的旧值】——所以先 shfl 读、后 += 写。
// 产出是【包含式】，最后用 shfl_up(x,1) 错一位转成排除式。
// lane 0 的「前一位」是本 warp 前面的所有 warp 之和，正好在 warp_sums[warp] 里。
// ------------------------------------------------------------
__global__ void scan_v2_local(float* a, float* out, float* block_sums, int N) {
    __shared__ float warp_sums[TILE / 32];

    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    float x = a[blockIdx.x * TILE + threadIdx.x];

    for (int off = 1; off < 32; off *= 2) {          // warp 内 Hillis-Steele
        float y = __shfl_up_sync(0xffffffff, x, off);
        if (lane >= off) x += y;
    }

    if (lane == 31) warp_sums[warp] = x;
    __syncthreads();

    if (threadIdx.x == 0) {                          // 串行扫 8 个 warp 总和（就地变排斥式）
        float run = 0.0f;
        for (int k = 0; k < TILE / 32; k++) {
            float t = warp_sums[k];
            warp_sums[k] = run;
            run += t;
        }
    }
    __syncthreads();

    x += warp_sums[warp];                            // 块内包含式
    float prev = __shfl_up_sync(0xffffffff, x, 1);   // 错一位 → 排斥式
    if (lane == 0) prev = warp_sums[warp];           // lane 0 的前一位 = 本 warp 的独占偏移

    if (threadIdx.x == TILE - 1) block_sums[blockIdx.x] = x;
    out[blockIdx.x * TILE + threadIdx.x] = prev;
}

// ------------------------------------------------------------
// v3 —— 每线程 4 个元素
//
// 线程内先串行 scan → 用 x[3] 做 warp scan → 错位转排斥式 → 加回
// 输出用 carry + (i == 0 ? 0 : x[i-1])：x[] 是包含式，错一位就是排斥式
// ------------------------------------------------------------
__global__ void scan_v3_local(float* a, float* out, float* block_sums, int N) {
    float x[4];
    __shared__ float warp_sums[TILE / 32];

    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    for (int i = 0; i < 4; i++) {
        x[i] = a[blockIdx.x * BLOCK_ELEMS + threadIdx.x * 4 + i];
    }
    __syncthreads();

    for (int i = 1; i < 4; i++) {                    // 线程内串行前缀和
        x[i] += x[i - 1];
    }

    float t = x[3];                                  // 本线程总和
    for (int off = 1; off <= 16; off *= 2) {         // warp 内 Hillis-Steele
        float y = __shfl_up_sync(0xffffffff, t, off);
        if (lane >= off) t += y;
    }

    float carry = __shfl_up_sync(0xffffffff, t, 1);  // 错一位 → 排斥式
    if (lane == 0) carry = 0;
    __syncthreads();

    if (lane == 31) warp_sums[warp] = t;
    __syncthreads();

    if (threadIdx.x == 0) {
        float run = 0.0f;
        for (int k = 0; k < TILE / 32; k++) {
            float v = warp_sums[k];
            warp_sums[k] = run;
            run += v;
        }
    }
    __syncthreads();

    carry += warp_sums[warp];
    __syncthreads();

    for (int i = 0; i < 4; i++) {
        out[blockIdx.x * BLOCK_ELEMS + threadIdx.x * 4 + i] =
            carry + (i == 0 ? 0.0f : x[i - 1]);
    }
    __syncthreads();

    if (threadIdx.x == TILE - 1) block_sums[blockIdx.x] = carry + x[3];
}

// ============================================================
// 基准测试
// ============================================================

// v0：单 kernel，输出是分段前缀和
static float bench_v0(int N, float* d_a, float* d_out) {
    dim3 block(TILE);
    dim3 grid(N / TILE);

    cudaEvent_t s0 = nullptr, s1 = nullptr;
    CUDA_CHECK(cudaEventCreate(&s0));
    CUDA_CHECK(cudaEventCreate(&s1));

    float min_ms = 1e12f, ms = 0.0f;
    for (int rep = 0; rep < 4; rep++) {
        if (rep == 0) {
            scan_v0<<<grid, block>>>(d_a, d_out, N);
        } else {
            CUDA_CHECK(cudaEventRecord(s0));
            scan_v0<<<grid, block>>>(d_a, d_out, N);
            CUDA_CHECK(cudaEventRecord(s1));
            CUDA_CHECK(cudaEventSynchronize(s1));
            CUDA_CHECK(cudaEventElapsedTime(&ms, s0, s1));
            if (ms < min_ms) min_ms = ms;
        }
    }
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaEventDestroy(s0));
    CUDA_CHECK(cudaEventDestroy(s1));
    return min_ms;
}

// v1/v2/v3：三次启动包在一个计时窗口里
static float bench_three(void (*local)(float*, float*, float*, int),
                         int N, int nBlocks, int elemsPerBlock,
                         float* d_a, float* d_out, float* d_bs)
{
    dim3 block(TILE);
    dim3 grid(nBlocks);

    cudaEvent_t s0 = nullptr, s1 = nullptr;
    CUDA_CHECK(cudaEventCreate(&s0));
    CUDA_CHECK(cudaEventCreate(&s1));

    float min_ms = 1e12f, ms = 0.0f;

    for (int rep = 0; rep < 4; rep++) {
        if (rep != 0) CUDA_CHECK(cudaEventRecord(s0));

        local<<<grid, block>>>(d_a, d_out, d_bs, N);
        scan_offsets<<<1, nBlocks>>>(d_bs, nBlocks);
        scan_add<<<grid, block>>>(d_out, d_bs, elemsPerBlock);

        if (rep != 0) {
            CUDA_CHECK(cudaEventRecord(s1));
            CUDA_CHECK(cudaEventSynchronize(s1));
            CUDA_CHECK(cudaEventElapsedTime(&ms, s0, s1));
            if (ms < min_ms) min_ms = ms;
        }
    }
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaEventDestroy(s0));
    CUDA_CHECK(cudaEventDestroy(s1));
    return min_ms;
}

// 全域对比：out[i] 应等于 a[0..i-1] 的和
static bool check_global(const float* h_a, const float* h_out, int N) {
    double run = 0.0;
    int bad = 0;
    for (int i = 0; i < N; i++) {
        double err = fabs((double)h_out[i] - run);
        if (err > 1e-3 * (1.0 + fabs(run))) bad++;
        run += (double)h_a[i];
    }
    return bad == 0;
}

// 分段对比：每 TILE 个元素独立算一次
static bool check_segmented(const float* h_a, const float* h_out, int N, int chunk) {
    int bad = 0;
    for (int base = 0; base + chunk <= N; base += chunk) {
        double run = 0.0;
        for (int i = 0; i < chunk; i++) {
            double err = fabs((double)h_out[base + i] - run);
            if (err > 1e-3 * (1.0 + fabs(run))) bad++;
            run += (double)h_a[base + i];
        }
    }
    return bad == 0;
}

int main(int argc, char** argv) {
    int N = argc > 1 ? atoi(argv[1]) : 262144;

    if (N % BLOCK_ELEMS != 0) {
        printf("N 必须是 %d 的倍数\n", BLOCK_ELEMS);
        return 1;
    }
    int nBlocks = N / BLOCK_ELEMS;
    if (nBlocks > 1024) {
        printf("nBlocks = %d > 1024，scan_offsets 单 block 扫不完\n", nBlocks);
        return 1;
    }

    print_device_info();

    float* h_a   = new float[N];
    float* h_out = new float[N];
    for (int i = 0; i < N; i++) h_a[i] = (float)(i % 7) - 3.0f;

    float* d_a   = nullptr;
    float* d_out = nullptr;
    float* d_bs  = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_a,   (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_out, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_bs,  (size_t)nBlocks * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));

    const double bytes = 2.0 * N * sizeof(float);     // 读一遍 + 写一遍

    printf("%-26s %10s %10s %9s %7s\n", "version", "min_ms", "GB/s", "vs v0", "check");
    printf("------------------------------------------------------------------\n");

    float base_ms = 0.0f;

    // ---- v0（分段）----
    {
        float ms = bench_v0(N, d_a, d_out);
        base_ms = ms;
        CUDA_CHECK(cudaMemcpy(h_out, d_out, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = check_segmented(h_a, h_out, N, TILE);
        printf("%-26s %10.6f %10.1f %8.2fx %7s\n",
               "v0  blelloch (segmented)", ms, bytes / (ms * 1e-3) / 1e9, 1.0,
               ok ? "OK" : "FAIL");
    }

    // ---- v1 / v2 / v3（全局）----
    struct V { const char* name; void (*local)(float*, float*, float*, int); int elems; };
    const V vs[] = {
        { "v1  two-pass (3 kernels)", scan_v1_local, TILE        },
        { "v2  hillis-steele",        scan_v2_local, TILE        },
        { "v3  4 elems/thread",       scan_v3_local, BLOCK_ELEMS },
    };

    for (int i = 0; i < 3; i++) {
        float ms = bench_three(vs[i].local, N, nBlocks, vs[i].elems, d_a, d_out, d_bs);
        CUDA_CHECK(cudaMemcpy(h_out, d_out, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = check_global(h_a, h_out, N);
        printf("%-26s %10.6f %10.1f %8.2fx %7s\n",
               vs[i].name, ms, bytes / (ms * 1e-3) / 1e9, base_ms / ms,
               ok ? "OK" : "FAIL");
    }

    printf("------------------------------------------------------------------\n");
    printf("N = %d    nBlocks = %d    峰值 672 GB/s\n", N, nBlocks);
    printf("v0 是分段前缀和（block 间不传递），其余三版是全局\n");

    delete[] h_a;
    delete[] h_out;
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_bs));
    return 0;
}
