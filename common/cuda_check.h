#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// 所有 CUDA 调用的错误检查。出错时打印文件/行号/错误串，然后退出。
//
// 用法：CUDA_CHECK(cudaMalloc(&p, bytes));
#define CUDA_CHECK(call)                                        \
    do {                                                        \
        cudaError_t err_ = (call);                              \
        if (err_ != cudaSuccess) {                              \
            printf("[CUDA error] %s:%d\n  %s (code=%d)\n",      \
                __FILE__, __LINE__,                             \
                cudaGetErrorString(err_), (int)err_);           \
            exit(1);                                            \
        }                                                       \
    } while (0)

// kernel 启动之后立刻查一次：<<<>>> 的错误是异步的，不会当场抛异常，
// 必须显式查，否则会跑到后面某个无关的 API 才炸，极难定位。
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())

// 设备信息：SM 数、寄存器、shared、L2、带宽。每个可执行文件开头打一次，
// 让基准数字带上硬件上下文。
inline void print_device_info() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));

    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));

    printf("device      : %s\n", p.name);
    printf("compute     : %d.%d  (%d SMs)\n", p.major, p.minor, p.multiProcessorCount);
    printf("regs/SM     : %d\n", p.regsPerMultiprocessor);
    printf("smem/SM     : %zu KB\n", p.sharedMemPerMultiprocessor / 1024);
    printf("L2          : %d KB\n", p.l2CacheSize / 1024);
    printf("\n");
}
