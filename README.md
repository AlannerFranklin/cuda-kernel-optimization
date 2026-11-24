# CUDA Kernel Optimization

从朴素实现一路推到接近硬件上限，六个基础算子的完整优化过程。

每一个版本都保留、都测、都解释——**不是「我写了个归约」，是「归约从 353 GB/s 优化到
近峰值，每一步为什么」**。

硬件：**RTX 4070 Ti Super**（Ada，CC 8.9，66 SM）。FP32 峰值 **44.1 TFLOPS**，
DRAM 带宽 **672 GB/s**。

## 快速开始

```bat
build.bat                              :: 全部编译（自动探测 MSVC，用 Ninja + Release）
build\reduction\reduction.exe          :: 单跑某个课题
build\matmul\matmul.exe 4096           :: 带参数
```

每个可执行文件会打印**一张对比表**——所有版本在同一台机器、同一次运行下横向比较。

依赖：CUDA Toolkit + Visual Studio（MSVC）。编译时带 `-Xptxas -v`，
每个 kernel 的寄存器 / shared / spill 用量会直接打出来。

## 性能总表

> 下表是一次**统一运行**的结果（2026-11-23 晚，RTX 4070 Ti Super + CUDA 12.6）。
> 全部规模都**大于 48 MB 的 L2**，测的是真 DRAM。
> 完整数据与判读见 [docs/perf-report.md](docs/perf-report.md) 和 [docs/ncu/summary.md](docs/ncu/summary.md)。

| 课题 | 版本跨度 | 最佳性能 | 峰值占比 | 加速比 |
|---|---|---|---|---|
| **vector_add** | 合并 → 散开 | 620.3 GB/s / 128.1 GB/s | 92% / 19% | **4.84x** |
| **reduction** | v0 → v6（七版） | 350.5 → 618.3 GB/s | 92% | **1.76x** |
| **scan** | v0 → v3（四版） | 186.2 → 157.5 GB/s | 28% | 见下 |
| **matmul** | naive → reg8x8 | 2,258 → **20,305 GFLOPS** | **46%** | **8.99x** |
| **histogram** | naive → shared | 12.9 → **595.8 GB/s** | **89%** | **46.1x** |
| **softmax** | 单版 | **627.1 GB/s** | **93%** | — |

**几个值得单独说的：**

**① 归约只快了 1.76x，不是 5.8x——这是一个自我纠正**

早期在 N=41.9 MB（**小于 L2 的 48 MB**）上测出 5.8x。换成 67.1 MB 之后只有 1.76x。

原因：从 v2 起就撞上了 DRAM 带宽上限（618 GB/s = 峰值的 92%），**后面几版没有空间可赚**。
那个 5.8x 是「数据在缓存里」时的数字，不是真实场景。

> **这条本身就是一课：同一个 kernel，数据放不放得进 L2，测出来的「加速比」可以差三倍。**

**② softmax 93%、histogram 89%、vector_add 92%——三个都基本到顶了**

再优化也没有空间，**识别「这个已经到极限」和识别「这里还能优化 10 倍」一样重要**。

**③ matmul 的 8×8 只占峰值 46%，但它是唯一进入 compute-bound 区的**

AI = 115.5，跨过 ridge point 65.6。前三个版本都在 memory-bound 区。

**④ 32 路 bank conflict 实测比无冲突慢 18.3 倍**（见下面的微基准）

**⑤ ncu 里最反直觉的一条：naive matmul 的计算吞吐 95%，却只有 5% 的 FP32 峰值**

SM 几乎满载——但满载在**发访存指令**上，不是在算。这正是 register tiling 的动机。

## 各课题

### [vector_add](vector_add/) —— 访存模式决定一切

同一道加法，两个版本只差地址计算方式：

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;                    // 合并：warp 读连续 128 字节
int i = threadIdx.x * gridDim.x + blockIdx.x;                     // 散开：warp 读 32 个不同 cache line
```

在 **67 MB（超过 L2）** 的规模下实测差 **4.84 倍**（620.3 vs 128.1 GB/s）。

> ⚠️ 但同一个实验在 **12 MB（L2 常驻）** 的规模下测出来是 **11.3 倍**。
> **同一现象，两级存储，两个数字**——散开访问在 L2 内部浪费得更彻底，
> 到了 DRAM 反而被行激活/预充电掩盖了一部分。**比较加速比之前，先确认数据落在哪一级。**

ncu 的对应指标是 `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct`：

| 版本 | 每个 sector 用到的字节 |
|---|---|
| coalesced | **100%** |
| strided | **12.5%**（4 字节 / 32 字节 sector） |

100 / 12.5 = **8 倍流量膨胀**，由 profiler 直接量出。

### [reduction](reduction/) —— 七个版本，1.76x

| 版本 | 做法 | 解决的问题 |
|---|---|---|
| v0 | shared 树形 | 基线：9 次 `__syncthreads`，有效线程-轮次仅 12.45% |
| v1 | 展开 | `s ≤ 32` 后只用一个 warp，同步降到 3 次 |
| v2 | 每线程 4 个元素 | block 数除以 4，跨步读保证合并 |
| v3 | warp shuffle 收尾 | 6 次 shared 往返 → 5 步寄存器传递 |
| v4 | 整棵树干掉 | 8 个 warp 各自 shuffle，shared 只留 8 个 float、1 次同步 |
| v5 | `float4` | 4 条 load 指令 → 1 条 |
| v6 | `atomicAdd` | 跨 block 合并，单 kernel 直接出标量 |

**两个踩过的坑（都在注释里标了）：**

1. **`s > 32` 是错的**——树必须把有效范围从 256 缩到收尾能处理的 32，
   所以 `s = 32` 那一轮必须执行。写 `> 32` 会让 `tile[32..63]` 那 128 个元素没人收
2. **`atomicAdd` 是累加的**——benchmark 会 launch 多次，所以每次 launch 前必须 memset

### [scan](scan/) —— Blelloch 与 Hillis-Steele

前缀和 `out[i] = a[0..i-1]`。两种算法都实现了：

| 算法 | 工作量 | 深度 | 实现 |
|---|---|---|---|
| Blelloch | **O(n)** | O(log n) | shared 树，上扫 + 下扫 |
| Hillis-Steele | O(n log n) | O(log n) | warp shuffle，一遍往上加 |

Blelloch 的**根清零**是关键一步：树根的「总和」换成「前面的总和」——它前面什么都没有，所以是 0。

v0 是**分段**前缀和（block 之间不传递），v1 用三个 kernel 修成全局。

**坑**：Hillis-Steele 产出的是包含式，对拍要的是排除式——整体错一位。
用 `__shfl_up_sync(x, 1)` 往回挪一个 lane 即可，lane 0 取本 warp 的独占偏移。

### [matmul](matmul/) —— 四版，8.99x

| 版本 | 每线程输出 | 每 FMA 的 load | GFLOPS | 峰值占比 | Roofline |
|---|---|---|---|---|---|
| naive | 1 | 2.0（global） | 2,258 | 5.1% | memory-bound |
| tiled | 1 | 2.0（shared） | 2,946 | 6.7% | memory-bound |
| reg 4×4 | 16 | 0.5 | 13,309 | 30.2% | memory-bound |
| reg 8×8 | 64 | **0.25** | **20,305** | **46.0%** | **compute-bound** |

> N = 4096，min_ms：60.85 / 46.66 / 10.33 / 6.77。

**四版在 Roofline 上走了一条完整的路线**，8×8 是唯一跨过 ridge point（65.6）的：

| 版本 | AI (FLOP/byte) | 天花板 | 完成度 |
|---|---|---|---|
| naive | 14.7 | 9,878 | 23% |
| tiled | 15.2 | 10,181 | 29% |
| reg 4×4 | 31.6 | 21,229 | 60% |
| reg 8×8 | **115.5** | 44,100（算力峰值） | 51% |

**8×8 的 occupancy 只有 33%，却比 83% 的 4×4 快 1.78 倍**——
occupancy 不直接决定性能，每 FMA 配多少 load 才是。

详细的瓶颈分析与全部实测数据见 [docs/perf-report.md](docs/perf-report.md)（含 Roofline 表）；ncu 采集方法与判读见 [docs/ncu/README.md](docs/ncu/README.md)。

### [histogram](histogram/) —— 46.1x，达峰值 89%

```
naive    12.9 GB/s    1677 万次原子加，全部打在 256 个全局地址上
shared  595.8 GB/s    私有直方图 + grid-stride（1024 blocks）+ 跳过空 bin
```

**只加 shared 是没用的**：65536 个 block × 256 次合并 = 1677 万次全局原子加，一次没少。
必须同时用 grid-stride 把 block 数降下来。

**这是唯一必须用 `atomicAdd` 的课题**，所以 harness 每次 launch 前 memset。

### [softmax](softmax/) —— 两次归约的应用

```cpp
m = max(row)                    // 归约（求最大值版）
s = Σ exp(row[i] - m)           // 归约（求和版）
out[i] = exp(row[i] - m) / s
```

**为什么必须减 max**：`exp(90) = 1.2e39` 超过 float 上限 `3.4e38`，直接算出 `inf/inf = NaN`。
减完之后 `x_i - m ≤ 0`，`exp ≤ 1`，不会溢出；数学上分子分母同除 `exp(m)` 抵消，结果不变。

两次归约用的是同一套 `block_sum` / `block_max`——**只换合并算子**
（`+=` → `fmaxf`）、**换中性元**（`0.0f` → `-INFINITY`）。

## 两个贯穿全项目的原则

### 1. 归约 = 通信 + 合并

`__shfl_down_sync` **不做任何运算**，它只把邻居的值搬过来。加法、取最大、取最小、
求积全用它，换的只是合并算子：

```cpp
v += __shfl_down_sync(0xffffffff, v, off);            // 求和
v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));   // 求最大
```

前提：合并算子必须满足**结合律**（计算顺序被打乱了）。

### 2. 优化必须对着瓶颈

- `matmul` 前两版卡在「每 FMA 配几条 load」，加 tile 没用，要加寄存器分块
- `histogram` 卡在原子竞争，光搬进 shared 没用，要减 block 数
- `CUDA Graph` 能消掉 launch 开销（实测 10.4x），但对**大量大 kernel** 的负载毫无帮助
  （提交开销只占 0.3%）

## 目录结构

```
common/cuda_check.h      CUDA_CHECK 宏 + 设备信息
vector_add/              coalesced / strided / saxpy
reduction/               v0-v6 七个版本
scan/                    Blelloch / 两遍法 / Hillis-Steele / 每线程 4 元素
matmul/                  naive / tiled / reg4x4 / reg8x8
histogram/               naive / shared
softmax/                 block 归约 + 三次遍历
docs/                    ncu 截图、瓶颈分析
```

## 编译细节

- `sm_89`（Ada）。换卡改 `CMakeLists.txt` 里的 `CMAKE_CUDA_ARCHITECTURES`
- `-Xptxas -v`：编译时打印每个 kernel 的寄存器 / shared / spill
- `/utf-8`：源码有中文注释，MSVC 默认按 GBK 读会乱码
