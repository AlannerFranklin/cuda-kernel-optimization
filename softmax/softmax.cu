// ============================================================
// softmax —— out_i = exp(x_i - m) / Σ exp(x_j - m)
//
// 一个 block 处理一行，三次遍历：
//   ① 求 max        ← 归约（求最大值版）
//   ② 求 Σ exp      ← 归约（求和版）
//   ③ 写回
//
// 为什么必须减 max：exp(90) = 1.2e39 超过 float 上限 3.4e38，
// 直接算会得到 inf，然后 inf/inf = NaN。
// 减去最大值后 x_i - m ≤ 0，exp ≤ 1，不会溢出；
// 数学上分子分母同除 exp(m)，结果不变。
//
// 两次归约用的是同一个模式 —— 归约 = 通信 + 合并：
//   __shfl_down_sync 只负责把邻居的值搬过来，加还是比大小由你决定。
//
// 用法：  softmax.exe [N]
//         N 是一行的元素数，默认 1024；行数固定 16384
// ============================================================

#include "cuda_check.h"
#include <algorithm>
#include <cmath>

constexpr int TPB = 256;

// ============================================================
// block 级归约：256 个线程的值 → 1 个，广播给全部线程
//
// 分四步：
//   ① 每个 warp 用 shuffle 缩成 1 个（5 步：off = 16,8,4,2,1）
//   ② 每个 warp 的 lane 0 写进 buf[warp]
//   ③ warp 0 把这 8 个值再缩一次（3 步：off = 4,2,1）
//   ④ 所有线程读 buf[0]
//
// 为什么每步 off 减半：每步之后每个线程代表的元素个数翻倍（1→2→4→…→32）
// ============================================================
__device__ float block_sum(float v, float* buf) {
    int lane  = threadIdx.x % 32;
    int warp  = threadIdx.x / 32;
    int nwarp = blockDim.x / 32;

    // ① warp 内归约
    for (int off = 16; off > 0; off /= 2) {
        v += __shfl_down_sync(0xffffffff, v, off);
    }

    // ② 每个 warp 的结果进 shared
    if (lane == 0) buf[warp] = v;
    __syncthreads();

    // ③ warp 0 归约这 nwarp 个值
    if (warp == 0) {
        v = (lane < nwarp) ? buf[lane] : 0.0f;      // 防越界 + 保证 32 线程全员参与
        for (int off = nwarp / 2; off > 0; off /= 2) {
            v += __shfl_down_sync(0xffffffff, v, off);
        }
        if (lane == 0) buf[0] = v;
    }
    __syncthreads();

    // ④ shared 对 block 内所有线程可见，直接读
    return buf[0];
}

// 求最大值版：只把「合并算子」和「中性元」换掉
//   0.0f 是加法的中性元；max 的中性元是 -INFINITY
//   （输入全为负时，用 0.0f 填充会把最大值算成 0）
__device__ float block_max(float v, float* buf) {
    int lane  = threadIdx.x % 32;
    int warp  = threadIdx.x / 32;
    int nwarp = blockDim.x / 32;

    for (int off = 16; off > 0; off /= 2) {
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
    }

    if (lane == 0) buf[warp] = v;
    __syncthreads();

    if (warp == 0) {
        v = (lane < nwarp) ? buf[lane] : -INFINITY;
        for (int off = nwarp / 2; off > 0; off /= 2) {
            v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
        }
        if (lane == 0) buf[0] = v;
    }
    __syncthreads();

    return buf[0];
}

// ============================================================
// softmax kernel：一个 block 处理一行
// ============================================================
__global__ void softmax(float* x, float* out, int N) {
    __shared__ float buf[32];

    float* row = x + blockIdx.x * N;

    // ① 求 max
    float m = -INFINITY;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        m = fmaxf(m, row[i]);
    }
    m = block_max(m, buf);
    __syncthreads();          // block_max 结尾所有线程要读 buf[0]，而 block_sum 会覆盖它

    // ② 求 Σ exp(x - m)
    float s = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        s += expf(row[i] - m);
    }
    s = block_sum(s, buf);

    // ③ 写回
    float inv = 1.0f / s;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        out[blockIdx.x * N + i] = expf(row[i] - m) * inv;
    }
}

// ============================================================
// 基准测试
// ============================================================

int main(int argc, char** argv) {
    const int M = 16384;                                  // 行数
    int N = argc > 1 ? atoi(argv[1]) : 1024;              // 每行元素数
    const int total = M * N;

    print_device_info();

    float* h_x   = new float[total];
    float* h_out = new float[total];

    // 值域 [-30, 90]：exp(90) 会溢出 float，只有「先减最大值」的写法能算对
    for (int i = 0; i < total; i++) {
        h_x[i] = (float)((i % 13) * 10 - 30);
    }

    float* d_x   = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_x,   (size_t)total * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_out, (size_t)total * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, (size_t)total * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t s0 = nullptr, s1 = nullptr;
    CUDA_CHECK(cudaEventCreate(&s0));
    CUDA_CHECK(cudaEventCreate(&s1));

    const dim3 block(TPB);
    const dim3 grid(M);

    float min_ms = 1e12f, ms = 0.0f;
    for (int rep = 0; rep < 4; rep++) {
        if (rep == 0) {
            softmax<<<grid, block>>>(d_x, d_out, N);
        } else {
            CUDA_CHECK(cudaEventRecord(s0));
            softmax<<<grid, block>>>(d_x, d_out, N);
            CUDA_CHECK(cudaEventRecord(s1));
            CUDA_CHECK(cudaEventSynchronize(s1));
            CUDA_CHECK(cudaEventElapsedTime(&ms, s0, s1));
            if (ms < min_ms) min_ms = ms;
        }
    }
    CUDA_CHECK_LAUNCH();

    CUDA_CHECK(cudaEventDestroy(s0));
    CUDA_CHECK(cudaEventDestroy(s1));
    CUDA_CHECK(cudaMemcpy(h_out, d_out, (size_t)total * sizeof(float), cudaMemcpyDeviceToHost));

    // ---------- 检查 ①：每行总和 = 1（全部行，不依赖参考实现）----------
    int bad_sum = 0;
    float worst_sum = 0.0f;
    for (int r = 0; r < M; r++) {
        double s = 0.0;
        for (int c = 0; c < N; c++) s += h_out[(size_t)r * N + c];
        float err = fabsf((float)s - 1.0f);
        if (err > 1e-3f) { bad_sum++; if (err > worst_sum) worst_sum = err; }
    }

    // ---------- 检查 ②：逐元素，抽 64 行 ----------
    int bad_elem = 0;
    float worst_elem = 0.0f;
    for (int r = 0; r < M; r += M / 64) {
        float mx = -INFINITY;
        for (int c = 0; c < N; c++) mx = fmaxf(mx, h_x[(size_t)r * N + c]);
        double sum = 0.0;
        for (int c = 0; c < N; c++) sum += expf(h_x[(size_t)r * N + c] - mx);
        for (int c = 0; c < N; c++) {
            float expect = expf(h_x[(size_t)r * N + c] - mx) / (float)sum;
            float err = fabsf(h_out[(size_t)r * N + c] - expect);
            if (err > 1e-5f) { bad_elem++; if (err > worst_elem) worst_elem = err; }
        }
    }

    double bytes = 2.0 * total * sizeof(float);
    double gbps  = bytes / (min_ms * 1e-3) / 1e9;

    printf("%-24s %10s %10s %7s\n", "version", "min_ms", "GB/s", "check");
    printf("--------------------------------------------------\n");
    printf("%-24s %10.6f %10.1f %7s\n",
           "softmax", min_ms, gbps,
           (bad_sum == 0 && bad_elem == 0) ? "OK" : "FAIL");
    printf("--------------------------------------------------\n");
    printf("M = %d rows × N = %d    %.1f MB in + %.1f MB out\n",
           M, N, (double)total * sizeof(float) / 1e6, (double)total * sizeof(float) / 1e6);
    printf("row_sum : %s  (%d 行不符，最大误差 %.2e)\n",
           bad_sum == 0 ? "OK" : "FAILED", bad_sum, worst_sum);
    printf("element : %s  (%d 个元素不符，最大误差 %.2e)\n",
           bad_elem == 0 ? "OK" : "FAILED", bad_elem, worst_elem);

    delete[] h_x;
    delete[] h_out;
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
