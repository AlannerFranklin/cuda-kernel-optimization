# ncu 数据采集

> Nsight Compute 需要**管理员权限**才能读取 GPU 性能计数器，否则报
> `ERR_NVGPUCTRPERM`。本文记录采集命令与判读方法。

## 采集

**用管理员终端**执行仓库根目录的 `tools\collect_ncu.bat`：

```bat
cd /d <仓库路径>
tools\collect_ncu.bat
```

脚本会对每个课题的代表性 kernel 采集 Speed of Light 的两个百分比并导出 CSV 到本目录：

| 文件 | 含义 |
|---|---|
| `vector_add_coalesced.csv` / `vector_add_strided.csv` | 合并 vs 散开访存 |
| `reduction_v0.csv` / `reduction_v6.csv` | 归约首尾两版 |
| `matmul_naive.csv` / `matmul_reg8x8.csv` | matmul 首尾两版 |
| `histogram_naive.csv` / `histogram_shared.csv` | 直方图两版 |
| `softmax.csv` | softmax |

## 判读要点

| 指标 | 含义 | 看什么 |
|---|---|---|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | 计算单元利用率（%） | 高 → compute-bound |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | DRAM 带宽利用率（%） | 高 → memory-bound |
| `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` | **global load 的 sector 有效率**（旧工具里叫 `gld_efficiency`） | 合并≈100%，散开≈12.5% |
| `dram__bytes.sum` | 实际走的 DRAM 字节数 | Roofline 的 AI 分母必须用它 |

## 两个坑

1. **`gld_efficiency` 是 nvprof / Nsight Visual Profiler 的名字，ncu 里不存在。**
   ncu 的对应指标是 `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct`。
2. **ncu 报的 `Duration` 不能当性能数字**（它会重放 kernel 多次），
   性能永远看程序自己的 `cudaEvent`。

## 已知实测（学习仓库，2026-11-21）

| kernel | 指标 | 值 |
|---|---|---|
| vector_add（合并） | sector 有效率 | **100.00%** |
| vector_add_strided（散开） | sector 有效率 | **12.50%** |

100 / 12.5 = **8 倍流量差**——这是"访存模式决定性能"的直接证据。
