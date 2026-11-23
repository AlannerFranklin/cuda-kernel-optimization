# 实测性能报告

> 采集环境：RTX 4070 Ti Super（Ada, CC 8.9，66 SM，16 GB）+ CUDA 12.6 +
> Windows 11 + MSVC 19.44。计时一律 `cudaEvent`（预热 1 次 + 计时 3 次取最小），
> `cudaMemcpy` 不进计时窗口。
>
> 采集时间：2026-11-23 晚（家机）。命令：`build.bat` 后依次运行各课题的 exe。

---

## 1. vector_add —— 访存模式决定带宽

| 版本 | min_ms | GB/s | 相对基线 | 正确性 |
|---|---|---|---|---|
| coalesced（合并） | 0.324544 | **620.3** | 1.00x | OK |
| strided（散开） | 1.571840 | 128.1 | 0.21x | OK |
| saxpy（a·x + y） | 0.321536 | 626.1 | 1.01x | OK |

- N = 16,777,216（67.1 MB/数组），**远大于 48 MB 的 L2**，测的是真 DRAM
- coalesced 620 GB/s ≈ 峰值 672 的 **92%**
- 合并 / 散开 = **4.84 倍**

> **与学习仓库 11.3 倍实验的差别**：那里 N=1e6（12 MB），数据整个装进 L2，
> 散开版测到的是 L2 内部的无效率访问；这里 67 MB 走 DRAM，随机访问被 DRAM
> 的行激活/预充电掩盖了一部分差距。**同一现象，两级存储，两个数字。**

---

## 2. reduction —— 七版归约，1.76x

| 版本 | min_ms | GB/s | 相对 v0 | 正确性 |
|---|---|---|---|---|
| v0 基础树形 | 0.191488 | 350.5 | 1.00x | OK |
| v1 展开（单 warp 收尾） | 0.126976 | 528.5 | 1.51x | OK |
| v2 每线程 4 元素 | 0.109504 | 612.8 | 1.75x | OK |
| v3 + warp shuffle | 0.108544 | 618.3 | 1.76x | OK |
| v4 全 warp 级 | 0.109568 | 612.5 | 1.75x | OK |
| v5 + float4 向量化 | 0.108544 | 618.3 | 1.76x | OK |
| v6 单 kernel（atomicAdd） | 0.108544 | 618.3 | 1.76x | OK |

- N = 16,777,216（67.1 MB），峰值带宽 672 GB/s
- 从 v2 起就**卡在 ~618 GB/s = 峰值的 92%**——归约是纯 memory-bound，
  后面几版的优化没有空间可赚

---

## 3. scan —— 四版前缀和

| 版本 | min_ms | GB/s | 相对 v0 | 正确性 |
|---|---|---|---|---|
| v0 Blelloch（分段） | 0.011264 | 186.2 | 1.00x | OK |
| v1 两遍法（3 kernel） | 0.018432 | 113.8 | 0.61x | OK |
| v2 Hillis-Steele | 0.014336 | 146.3 | 0.79x | OK |
| v3 每线程 4 元素 | 0.013312 | 157.5 | 0.85x | OK |

- N = 262,144，nBlocks 按版本不同（v1/v2 用 1024，v3 用 256）
- **v0 看起来最快，但它只是分段前缀和**（block 之间不传递），比其余三版少做两遍
  全局读写，不具可比性
- v1 最慢：三个 kernel 串行，中间两次全量读写 `block_sums`
- v2/v3 把块内扫描做成单 kernel（shuffle），省掉了第一遍

---

## 4. matmul —— 四版，8.99x

| 版本 | min_ms | GFLOPS | 相对 naive | 峰值占比 | 正确性 |
|---|---|---|---|---|---|
| naive | 60.854 | 2,258.5 | 1.00x | 5.1% | OK |
| tiled | 46.660 | 2,945.6 | 1.30x | 6.7% | OK |
| reg 4×4 | 10.327 | 13,308.6 | 5.89x | 30.2% | OK |
| **reg 8×8** | **6.769** | **20,305.3** | **8.99x** | **46.0%** | OK |

- N = 4096，FLOP = 2N³ = 1.374e11；FP32 峰值 44,100 GFLOPS；
  ridge point = 44,100 / 672 = **65.6 FLOP/byte**
- Roofline（ncu 实测 DRAM 字节数，见 [../matmul/](../matmul/)）：

| 版本 | DRAM 字节 | AI (FLOP/byte) | 落区 | 完成度 |
|---|---|---|---|---|
| naive | 9.35 GB | 14.7 | memory-bound | 23% |
| tiled | 9.07 GB | 15.2 | memory-bound | 29% |
| reg 4×4 | 4.35 GB | 31.6 | memory-bound | 60% |
| reg 8×8 | 1.19 GB | **115.5** | **compute-bound** | 51% |

- **8×8 的 occupancy 只有 33%（4×4 是 83%），却快 1.78 倍**：
  occupancy 只是"藏延迟的能力"，8×8 每 FMA 只需 0.25 次 load，不需要那么多线程
- SASS 实测（`cuobjdump -sass`）：编译器把 `As[r][k]` 的读取**向量化**了
  （`LDS.128`），所以 4×4 实际的 load/FMA 是 0.31，比设计值 0.5 更低

---

## 5. histogram —— 46x

| 版本 | min_ms | GB/s | 相对 naive | 正确性 |
|---|---|---|---|---|
| naive（全局 atomicAdd） | 5.190656 | 12.9 | 1.00x | OK |
| shared（私有直方图 + grid-stride） | 0.112640 | **595.8** | **46.08x** | OK |

- N = 16,777,216（67.1 MB），256 个 bin
- naive：1677 万次原子加全部打在 256 个**全局**地址上
- shared 版达到峰值的 **89%**
- **只加 shared 没用**：65536 个 block × 256 次合并 = 1677 万次全局原子加，一次没少。
  必须同时用 grid-stride 把 block 数降下来

---

## 6. softmax —— 627 GB/s（峰值的 93%）

| 指标 | 值 |
|---|---|
| min_ms | 0.214016 |
| GB/s | **627.1** |
| row_sum 检查 | OK（16384 行，最大误差 0） |
| element 检查 | OK（最大误差 0） |

- 规模：16384 行 × 1024 列，**67.1 MB 输入 + 67.1 MB 输出**
- 三次遍历：求 max → 求 Σexp(x−max) → 写回
- 输入范围刻意取 [-30, 90]：`exp(90)` 会溢出 float，**忘了减最大值就会立刻出 NaN**，
  两个正确性检查会发现——内置的陷阱检测
- 未做 corner case 优化（行内两次重复扫描靠 L1/L2 吸收），
  剩余优化空间主要在第二、三遍的读回

---

## 复现方法

```bat
build.bat
build\vector_add\vector_add.exe
build\reduction\reduction.exe
build\scan\scan.exe
build\matmul\matmul.exe
build\histogram\histogram.exe
build\softmax\softmax.exe
```

ncu 数据（计算/DRAM 吞吐、sector 有效率、DRAM 字节数）见 [ncu/summary.md](ncu/summary.md)，采集方法见 [ncu/README.md](ncu/README.md)。
