# ncu 实测汇总

> 采集：2026-11-23，管理员终端运行 `tools\collect_ncu.bat`。
> Nsight Compute 2024.3.1 + RTX 4070 Ti Super + CUDA 12.6。
> **采集用小规模 N**（只为取计数器，性能数字见 [../perf-report.md](../perf-report.md)）。

## 汇总表

| kernel | 计算吞吐 % | DRAM 吞吐 % | load sector 有效率 % | DRAM 字节 |
|---|---|---|---|---|
| vector_add（合并） | 8.34 | **89.26** | **100.00** | 47.6 MB |
| vector_add_strided（散开） | 2.46 | 26.37 | **12.50** | 56.5 MB |
| reduction_v0（树形） | **84.85** | 52.45 | 100 | 18.8 MB |
| reduction_v6（单 kernel） | 12.49 | **89.81** | 100 | 17.3 MB |
| matmul（naive） | **95.24** | 3.64 | 82.50 | 21.8 MB |
| matmul_reg8（8×8） | 60.23 | 12.10 | 100 | 11.4 MB |
| histogram_naive | 0.43 | 2.73 | 100 | 21.0 MB |
| histogram_shared | 10.77 | **79.45** | 100 | 18.5 MB |
| softmax | 28.83 | **91.59** | 100 | 126.9 MB |

（计算吞吐 = `sm__throughput...pct`；DRAM 吞吐 = `gpu__dram_throughput...pct`；
sector 有效率 = `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct`）

## 判读

**① 访存模式：100% vs 12.5%**

```
vector_add           100.00%   ← 32 线程读连续 128 字节 = 4 个 sector 全用满
vector_add_strided    12.50%   ← 每 sector 32 字节只用 4 字节
                     -------
                    恰好 8 倍流量差
```

这是"访存模式决定性能"最直接的证据，与学习仓库的实测完全一致。

**② reduction：v0 是计算受限，v6 是带宽受限**

- v0 计算吞吐 **84.85%**、DRAM 只有 52%——**瓶颈在 shared 树形归约的指令与
  `__syncthreads()`**，不在内存。这解释了 v0 为什么只有 350 GB/s。
- v6 计算吞吐 12.5%、DRAM **89.8%**——**纯 memory-bound**，已经贴着带宽上限跑。
  这也说明 v6 之后的优化空间几乎为零（实测 618 GB/s ≈ 峰值 92%）。

**③ matmul naive：计算吞吐 95%，却只有 5% 的 FP32 峰值**

这是最值得讲的一条：

```
naive:  sm__throughput = 95.24%   （SM 几乎满载）
        实际 GFLOPS 仅 2258 / 44100 = 5.1%

矛盾吗？不矛盾——SM 的吞吐被【访存指令】吃满了。
每 1 次 FMA 要配 2 次 global load，流水线忙着发 load，FFMA 反而发不出去。
```

**这正是 register tiling 的动机**：把每 FMA 的 load 从 2.0 降到 0.25，
SM 才有余力发 FFMA。reg8x8 的计算吞吐降到 60%（因为 occupancy 只有 33%），
但 GFLOPS 反而涨到 46% 峰值——**"计算吞吐高"不等于"算得快"**，
还要看这些吞吐用在了哪类指令上。

**④ histogram_naive：0.43% 计算吞吐、2.7% DRAM**

两种吞吐都极低 = **不是在算，也不是在搬，是在等**。
1677 万次原子加打在 256 个全局地址上，线程全部卡在原子操作的串行化上。
加 shared 私有化后 DRAM 吞吐升到 79%，带宽从 12.9 GB/s 涨到 595.8 GB/s。

**⑤ softmax：DRAM 91.6%**

六个课题里最贴近带宽上限的 kernel，与 perf-report 里 627 GB/s（峰值 93%）一致。

## 复现

```bat
:: 管理员终端
tools\collect_ncu.bat
```

原始 CSV 在本目录下（`*.csv`），每个文件包含完整的 kernel 元数据
（block/grid 形状、寄存器、所有采集到的指标列）。

## 两个坑

1. **`gld_efficiency` 是旧工具（nvprof / Nsight Visual Profiler）的指标名，
   ncu 里不存在。** 对应指标是
   `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct`。
2. **ncu 报的 `Duration` 不能当性能数字**（会重放 kernel 多次），
   性能永远看程序自己的 `cudaEvent`。
