// ============================================================
// vector_add —— 最基础的逐元素运算
//
// 三个版本：
//   coalesced  相邻线程访问相邻地址 —— 基准
//   strided    相邻线程访问相隔 gridDim 个元素 —— 每个线程踩一个不同的 cache line
//   saxpy      y = a*x + y（和 vector_add 同构，顺手加上）
//
// 这个课题要说明的问题：**同一个算法、同样的指令数，访存模式一变，
// 有效带宽差一个数量级。**
//
// 用法：  vector_add.exe [N]
//         N 默认 16777216（= 2^24，每个数组 64 MB，远大于 48 MB 的 L2）
// ============================================================

#include "cuda_check.h"
#include <algorithm>
#include <cmath>

constexpr int TPB = 256;

// ------------------------------------------------------------
// 合并访存（coalesced）
// 同一个 warp 的 32 个线程访问地址连续的 32 个元素 = 128 字节 = 1 个 cache line
// ------------------------------------------------------------
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// ------------------------------------------------------------
// 散开访存（strided）
// i = threadIdx.x * gridDim.x + blockIdx.x
// 同一个 warp 内的线程 i 相差 gridDim.x —— 每个线程踩在不同的 cache line 上
// 一条指令要发 32 倍的访存请求，但只用到其中 4 个字节
// ------------------------------------------------------------
__global__ void vector_add_strided(float* a, float* b, float* c, int n) {
    int i = threadIdx.x * gridDim.x + blockIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// ------------------------------------------------------------
// saxpy：y = a*x + y，访存形状和 vector_add 完全一样（2 读 1 写）
// ------------------------------------------------------------
__global__ void saxpy(float a, float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

// ============================================================
// 基准测试
// ============================================================

// 返回 min_ms。float 版本没有返回值，这里返回计时。
static float bench(void (*kernel)(float*, float*, float*, int),
                   int N, float* d_a, float* d_b, float* d_c)
{
    dim3 block(TPB);
    dim3 grid((N + TPB - 1) / TPB);

    cudaEvent_t start = nullptr, stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float min_ms = 1e12f, ms = 0.0f;

    for (int rep = 0; rep < 4; rep++) {
        if (rep == 0) {
            kernel<<<grid, block>>>(d_a, d_b, d_c, N);
        } else {
            CUDA_CHECK(cudaEventRecord(start));
            kernel<<<grid, block>>>(d_a, d_b, d_c, N);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            if (ms < min_ms) min_ms = ms;
        }
    }
    CUDA_CHECK_LAUNCH();

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return min_ms;
}

int main(int argc, char** argv) {
    int N = argc > 1 ? atoi(argv[1]) : 16777216;

    print_device_info();

    // ---- 主机数据 ----
    float* h_a = new float[N];
    float* h_b = new float[N];
    float* h_c = new float[N];
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // ---- 设备 ----
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_a, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_b, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_c, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));

    // vector_add 的流量：读 a + 读 b + 写 c = 3 个数组
    const double bytes = 3.0 * N * sizeof(float);

    printf("%-24s %10s %10s %9s %7s\n", "version", "min_ms", "GB/s", "vs base", "check");
    printf("----------------------------------------------------------------\n");

    // ---- 合并版 ----
    float t_coal = bench(vector_add, N, d_a, d_b, d_c);
    CUDA_CHECK(cudaMemcpy(h_c, d_c, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
    bool ok_coal = (h_c[0] == 3.0f && h_c[N - 1] == 3.0f);
    printf("%-24s %10.6f %10.1f %8.2fx %7s\n",
           "coalesced", t_coal, bytes / (t_coal * 1e-3) / 1e9, 1.0, ok_coal ? "OK" : "FAIL");

    // ---- 散开版 ----
    float t_str = bench(vector_add_strided, N, d_a, d_b, d_c);
    CUDA_CHECK(cudaMemcpy(h_c, d_c, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
    bool ok_str = (h_c[0] == 3.0f && h_c[N - 1] == 3.0f);
    printf("%-24s %10.6f %10.1f %8.2fx %7s\n",
           "strided", t_str, bytes / (t_str * 1e-3) / 1e9, t_coal / t_str, ok_str ? "OK" : "FAIL");

    // ---- saxpy ----
    dim3 block(TPB);
    dim3 grid((N + TPB - 1) / TPB);

    cudaEvent_t s0 = nullptr, s1 = nullptr;
    CUDA_CHECK(cudaEventCreate(&s0));
    CUDA_CHECK(cudaEventCreate(&s1));
    float t_saxpy = 1e12f, ms = 0.0f;
    for (int rep = 0; rep < 4; rep++) {
        if (rep == 0) {
            saxpy<<<grid, block>>>(2.0f, d_a, d_b, N);
        } else {
            CUDA_CHECK(cudaEventRecord(s0));
            saxpy<<<grid, block>>>(2.0f, d_a, d_b, N);
            CUDA_CHECK(cudaEventRecord(s1));
            CUDA_CHECK(cudaEventSynchronize(s1));
            CUDA_CHECK(cudaEventElapsedTime(&ms, s0, s1));
            if (ms < t_saxpy) t_saxpy = ms;
        }
    }
    CUDA_CHECK(cudaEventDestroy(s0));
    CUDA_CHECK(cudaEventDestroy(s1));

    // ⚠️ saxpy 是【非幂等】的（y = a*x + y），而 bench 会 launch 4 次。
    //    检查前必须先把 d_b 复位、再单独跑一次；否则读到的是累加 4 次后的值（10 而非 4）。
    //    这是「kernel 与 harness 的幂等契约」的又一个实例（同 reduction_v6 的 atomicAdd）。
    CUDA_CHECK(cudaMemcpy(d_b, h_b, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));
    saxpy<<<grid, block>>>(2.0f, d_a, d_b, N);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaMemcpy(h_c, d_b, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
    bool ok_saxpy = (h_c[0] == 4.0f);            // y = 2*1 + 2 = 4
    printf("%-24s %10.6f %10.1f %8.2fx %7s\n",
           "saxpy", t_saxpy, bytes / (t_saxpy * 1e-3) / 1e9, t_coal / t_saxpy,
           ok_saxpy ? "OK" : "FAIL");

    printf("----------------------------------------------------------------\n");
    printf("N = %d (%.1f MB/array)   峰值 672 GB/s（RTX 4070 Ti Super） \n",
           N, (double)N * sizeof(float) / 1e6);
    printf("coalesced / strided = %.2f 倍 —— 有效带宽的差距 \n", t_str / t_coal);

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return 0;
}
