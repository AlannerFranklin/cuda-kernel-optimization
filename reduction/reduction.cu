// ============================================================
// reduction —— 把 N 个 float 求和成一个标量
//
// 七个版本，从朴素实现推到「单个 kernel 直接出标量」。
// 归约是最基础的并行模式，但它同时踩到 SIMT 的四个坑：
//   分支发散、同步开销、bank conflict、跨 block 合并
//
// 用法：  reduction.exe [N]
//         N 默认 16777216（= 2^24，64 MB）
//
// ⚠️ 为什么不用 10485760：那是 41.9 MB，比 L2 的 48 MB 小，
//    整块数据会留在 L2 里，测出来的就不是 DRAM 带宽了
//    （之前一度量到 1847 GB/s，超过 DRAM 峰值 672 的两倍）。
//    64 MB > 48 MB，才能测到真正的显存带宽。
//
// 硬件：RTX 4070 Ti Super（Ada, CC 8.9），DRAM 672 GB/s
// ============================================================

#include "cuda_check.h"
#include <algorithm>
#include <cmath>

constexpr int TPB = 256;      // 每 block 线程数

// ============================================================
// v0 —— 基线：shared memory 树形归约
//
// 每线程读 1 个元素进 shared，然后 8 轮折半相加。
// 问题：9 次 __syncthreads；干活的线程逐轮减半
//       128 → 64 → 32 → … → 1，有效线程-轮次只有 255/(8×256) = 12.45%
// ============================================================
__global__ void reduction_v0(float* a, float* out, int N) {
    __shared__ float tile[TPB];

    int base = blockIdx.x * blockDim.x;
    tile[threadIdx.x] = a[base + threadIdx.x];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (threadIdx.x < s) {
            tile[threadIdx.x] += tile[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = tile[0];
}

// ============================================================
// v1 —— 展开：s ≤ 32 之后只用一个 warp
//
// 从 s = 32 起，干活的线程全在 warp 0 里，块级屏障是浪费。
// 循环砍到 s > 32，剩下 6 轮用一个 warp + __syncwarp() 收尾。
// 同步次数：9 → 3。
// ============================================================
__global__ void reduction_v1(float* a, float* out, int N) {
    __shared__ float tile[TPB];

    int base = blockIdx.x * blockDim.x;
    tile[threadIdx.x] = a[base + threadIdx.x];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s /= 2) {
        if (threadIdx.x < s) {
            tile[threadIdx.x] += tile[threadIdx.x + s];
        }
        __syncthreads();
    }

    // 最后 6 轮：只有一个 warp 参与，用 warp 级同步
    if (threadIdx.x < 32) {
        volatile float* t = tile;
        for (int i = 32; i > 0; i /= 2) {
            __syncwarp();
            t[threadIdx.x] += t[threadIdx.x + i];
        }
    }
    if (threadIdx.x == 0) out[blockIdx.x] = tile[0];
}

// ============================================================
// v2 —— 每线程 4 个元素
//
// 一个 block 覆盖 1024 个元素，block 数除以 4。
// 读入用【跨步】而不是每线程连续 4 个：
//   同一 warp 在 i=0 那一轮读的是连续 32 个 float（128 字节 = 1 个 cache line）
//   写成连续版会跨 512 字节，同一条指令要发 4 倍的访存请求
// ============================================================
__global__ void reduction_v2(float* a, float* out, int N) {
    __shared__ float tile[TPB];

    int base = blockIdx.x * (blockDim.x * 4);
    float sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        sum += a[base + i * blockDim.x + threadIdx.x];
    }
    tile[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s /= 2) {
        if (threadIdx.x < s) {
            tile[threadIdx.x] += tile[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x < 32) {
        volatile float* t = tile;
        for (int i = 32; i > 0; i /= 2) {
            __syncwarp();
            t[threadIdx.x] += t[threadIdx.x + i];
        }
    }
    if (threadIdx.x == 0) out[blockIdx.x] = tile[0];
}

// ============================================================
// v3 —— 最后 32 个改用 warp shuffle
//
// 6 次 shared 往返 → 5 步寄存器传递，不需要任何同步。
// 注意循环条件是 s >= 32（不是 s > 32）：
//   树必须把有效范围从 256 缩到【收尾能处理的 32】，
//   所以 s = 32 那一轮必须执行。写 s > 32 会让 tile[32..63] 的
//   那 128 个元素没人收——这是本项目里犯过两次的错。
// ============================================================
__global__ void reduction_v3(float* a, float* out, int N) {
    __shared__ float tile[TPB];

    int base = blockIdx.x * (blockDim.x * 4);
    float sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        sum += a[base + i * blockDim.x + threadIdx.x];
    }
    tile[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s >= 32; s /= 2) {
        if (threadIdx.x < s) {
            tile[threadIdx.x] += tile[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x < 32) {
        float v = tile[threadIdx.x];
        for (int off = 16; off > 0; off /= 2) {
            v += __shfl_down_sync(0xffffffff, v, off);
        }
        if (threadIdx.x == 0) out[blockIdx.x] = v;
    }
}

// ============================================================
// v4 —— 整棵 shared 树都干掉
//
// 8 个 warp 各自用 shuffle 缩成 1 个值 → 8 个值进 shared
// → warp 0 再缩一次。shared 只要 8 个 float，同步只要 1 次。
//
// 第二段的 (lane < 8) ? ... : 0.0f 不能省：
//   既防 buf 越界，又保证 32 个线程全员参与 shuffle
// ============================================================
__global__ void reduction_v4(float* a, float* out, int N) {
    __shared__ float warp_sums[TPB / 32];

    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    int nwarp = blockDim.x / 32;

    int base = blockIdx.x * (blockDim.x * 4);
    float sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        sum += a[base + i * blockDim.x + threadIdx.x];
    }

    for (int off = 16; off > 0; off /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, off);
    }
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        sum = (lane < nwarp) ? warp_sums[lane] : 0.0f;
        for (int off = nwarp / 2; off > 0; off /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, off);
        }
        if (lane == 0) out[blockIdx.x] = sum;
    }
}

// ============================================================
// v5 —— float4 向量化
//
// 4 条 load 指令 → 1 条。一次搬 16 字节，warp 的 32 个线程共 512 字节
// （= 4 个 cache line），一条指令搞定。
// 要求地址 16 字节对齐——cudaMalloc 保证至少 256 字节对齐。
// ============================================================
__global__ void reduction_v5(float* a, float* out, int N) {
    __shared__ float warp_sums[TPB / 32];

    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    int nwarp = blockDim.x / 32;

    const float4* a4 = reinterpret_cast<const float4*>(a);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float4 v = a4[idx];
    float sum = v.x + v.y + v.z + v.w;

    for (int off = 16; off > 0; off /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, off);
    }
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        sum = (lane < nwarp) ? warp_sums[lane] : 0.0f;
        for (int off = nwarp / 2; off > 0; off /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, off);
        }
        if (lane == 0) out[blockIdx.x] = sum;
    }
}

// ============================================================
// v6 —— 单个 kernel 直接出标量
//
// 前五版都由 host 把 grid.x 个部分和加起来，归约其实没在 GPU 上做完。
// 这里用 atomicAdd 跨 block 合并——block 之间没有 shared memory，
// 唯一的交汇点是 global memory，而 atomicAdd 保证「读-改-写」不被撕开。
//
// ⚠️ atomicAdd 是【累加】，所以 harness 必须每次 launch 前把 out 清零。
//    否则（如果 harness 复用同一个 buffer 跑多次）结果会翻倍。
//    本文件的 bench_one() 已经在每次 launch 前 memset。
// ============================================================
__global__ void reduction_v6(float* a, float* out, int N) {
    __shared__ float warp_sums[TPB / 32];

    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    int nwarp = blockDim.x / 32;

    const float4* a4 = reinterpret_cast<const float4*>(a);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float4 v = a4[idx];
    float sum = v.x + v.y + v.z + v.w;

    for (int off = 16; off > 0; off /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, off);
    }
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        sum = (lane < nwarp) ? warp_sums[lane] : 0.0f;
        for (int off = nwarp / 2; off > 0; off /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, off);
        }
        if (lane == 0) atomicAdd(out, sum);      // ← 唯一和前几版不同的地方
    }
}

// ============================================================
// 基准测试框架
// ============================================================

struct Result {
    float  ms;
    double gbps;
    bool   ok;
};

// 跑一个 kernel：预热 1 次 + 计时 3 次取最小，并对拍。
// out 每次 launch 前清零——v6 的 atomicAdd 依赖这一点。
static Result bench_one(void (*kernel)(float*, float*, int),
                        int N, int gridn,
                        float* d_a, float* d_out,
                        float* h_out, double ref_sum)
{
    cudaEvent_t start = nullptr, stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const dim3 block(TPB);
    const dim3 grid(gridn);

    float min_ms = 1e12f, ms = 0.0f;

    for (int rep = 0; rep < 4; rep++) {
        // 每次 launch 前清零：v6 的 atomicAdd 是累加的，不清会翻 N 倍
        CUDA_CHECK(cudaMemset(d_out, 0, (size_t)gridn * sizeof(float)));

        if (rep == 0) {
            kernel<<<grid, block>>>(d_a, d_out, N);
        } else {
            CUDA_CHECK(cudaEventRecord(start));
            kernel<<<grid, block>>>(d_a, d_out, N);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            if (ms < min_ms) min_ms = ms;
        }
    }
    CUDA_CHECK_LAUNCH();

    CUDA_CHECK(cudaMemcpy(h_out, d_out, (size_t)gridn * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // 汇总 + 对拍
    double sum = 0.0;
    for (int i = 0; i < gridn; i++) sum += h_out[i];
    bool ok = fabs(sum - ref_sum) / fabs(ref_sum) < 1e-3;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    Result r;
    r.ms   = min_ms;
    r.gbps = (double)N * sizeof(float) / (min_ms * 1e-3) / 1e9;
    r.ok   = ok;
    return r;
}

int main(int argc, char** argv) {
    int N = argc > 1 ? atoi(argv[1]) : 16777216;

    if (N % 1024 != 0) {
        printf("N 必须是 1024 的倍数（v2 以后每 block 处理 1024 个元素）\n");
        return 1;
    }

    print_device_info();

    // ---- 输入：值域很小的整数，避免 float 累加误差 ----
    float* h_a = new float[N];
    for (int i = 0; i < N; i++) h_a[i] = (float)(i % 7) - 3.0f;

    // ---- CPU 参考值（double 累加）----
    double ref = 0.0;
    for (int i = 0; i < N; i++) ref += (double)h_a[i];

    // ---- 设备内存 ----
    const int max_gridn = N / TPB;         // v0/v1 用
    float* d_a   = nullptr;
    float* d_out = nullptr;
    float* h_out = new float[max_gridn];
    CUDA_CHECK(cudaMalloc((void**)&d_a,   (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_out, (size_t)max_gridn * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));

    struct Version {
        const char* name;
        void (*kernel)(float*, float*, int);
        int div;                            // grid = N / div
    };

    const Version versions[] = {
        { "v0  baseline tree",    reduction_v0, 256  },
        { "v1  unroll (1 warp)",  reduction_v1, 256  },
        { "v2  4 elems/thread",   reduction_v2, 1024 },
        { "v3  + warp shuffle",   reduction_v3, 1024 },
        { "v4  full warp-level",  reduction_v4, 1024 },
        { "v5  + float4",         reduction_v5, 1024 },
        { "v6  single kernel",    reduction_v6, 1024 },
    };
    const int nver = (int)(sizeof(versions) / sizeof(versions[0]));

    printf("%-22s %10s %10s %9s %7s\n", "version", "min_ms", "GB/s", "vs v0", "check");
    printf("------------------------------------------------------------------------\n");

    float base_ms = 0.0f;

    for (int i = 0; i < nver; i++) {
        int gridn = N / versions[i].div;
        Result r = bench_one(versions[i].kernel, N, gridn,
                             d_a, d_out, h_out, ref);

        if (i == 0) base_ms = r.ms;

        printf("%-22s %10.6f %10.1f %8.2fx %7s\n",
               versions[i].name, r.ms, r.gbps, base_ms / r.ms,
               r.ok ? "OK" : "FAIL");
    }

    printf("------------------------------------------------------------------------\n");
    printf("N = %d (%.1f MB)   峰值带宽 672 GB/s（RTX 4070 Ti Super）\n",
           N, (double)N * sizeof(float) / 1e6);

    delete[] h_a;
    delete[] h_out;
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
