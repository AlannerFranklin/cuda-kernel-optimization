# 读 SASS：从机器码验证优化设计

> 工具：`cuobjdump`（CUDA 13.4）
> 目标架构：sm_89（RTX 4070 Ti Super）

## 为什么读 SASS

`ptxas -v` 只给数字（寄存器数、spill 字节）。**SASS 给真相**：

- 循环到底展开了没有
- `if (threadIdx.x < s)` 编译成了**分支**还是**谓词**
- 一段代码里到底有几条真正的 load/store
- 编译器的实际表现和你的设计差多少

面试里被问「你怎么知道编译器没把它优化掉」，答案是**「我去看了 SASS」**。

## 命令

```bat
cuobjdump -sass build\matmul\matmul.exe > sass.txt
```

一个 exe 里所有 kernel 都会导出，用 `grep "Function :"` 找名字。
单看某一个 kernel 可以用 `-fun <mangled_name>`。

## 一行 SASS 怎么读

```
/*0130*/    @!P1    LDS    R4, [R7.X4]     ;
  ↑          ↑      ↑       ↑
指令地址   谓词    指令名    操作数
```

- `/*0130*/` 是这条指令在代码里的字节偏移
- `@!P1` 是**谓词**：「P1 为假时才执行」——这就是 `if` 的编译结果
- `R4` 是寄存器；`[R7.X4]` 是 shared 地址（R7 × 4 字节）

## matmul 四个版本的指令配比

统计各 kernel 里的关键指令（`cuobjdump -sass` + `grep -oE` + `sort | uniq -c`）：

| kernel | FFMA | LDS | LDG | 每 FMA 的 load | 源码设计值 |
|---|---|---|---|---|---|
| `matmul`（naive） | 29 | — | **58** | **2.00** | 2.0 ✅ |
| `matmul_tile` | 32 | 40 | 2 | 1.25 | 2.0 |
| `matmul_reg`（4×4） | **512** | **160** | 20 | **0.31** | 0.5 |
| `matmul_reg8`（8×8） | **2048** | **320** | 40 | **0.156** | 0.25 |

**naive 的 2.00 完全吻合设计**：58 条 `LDG` / 29 条 `FFMA`——每做一次乘加就要两条全局读。
这是「naive 慢」的机器码证据，不再是纸上推算。

## 发现：编译器把 load 向量化了

4×4 版按源码设计应该是每 k 迭代 8 次 load / 16 次 FMA = **0.5**，实测 **0.31**。

拆开看 LDS 的两种形态：

```
128 条 LDS      一次搬 1 个 float
 32 条 LDS.128  一次搬 4 个 float      ← 编译器干的
```

`32 × 4 = 128` 个 float，正好是**全部 As 的读取量**。

编译器发现 `a[i] = As[thread_row + i][k]` 在 k 连续时，`As[r][k]` 与 `As[r][k+1]`
在 shared 里**相邻**，于是把 4 个 k 一次性搬进来：

```
源码设计：256 条 load 指令
实际机器码：160 条        ← 省了 96 条
```

8×8 版同理：512 → 320（`256 条 LDS + 64 条 LDS.128`）。

**`Bs[k][tx + j*32]` 没有被向量化**——k 连续时地址隔 128 字节，不连续，只能一条一条来。

**结论：源码里的「每 FMA 0.5 条 load」在机器上实际是 0.31。**
这是编译器自己做的优化，源码里看不出来，只有读 SASS 才知道。

## 另外三条从 SASS 直接得到的结论

### ① 循环展开情况

| kernel | 展开程度 | 依据 |
|---|---|---|
| `matmul` | 未完全展开 | 29 FFMA，循环还在 |
| `matmul_reg` | **完全展开** | 512 = 32 轮 × 16，一轮不少 |
| `matmul_reg8` | **完全展开** | 2048 = 32 轮 × 64 |

### ② 没有掉进 local memory

四个 kernel 里 **`LDL` / `STL` 一条都没有**，与 ptxas 报告的 `spill 0` 一致。
`acc[4][4]`（16 个 float）和 `acc[8][8]`（64 个 float）确实全在寄存器里——
**寄存器分块名副其实**。

### ③ `if` 是谓词，不是分支

`reduction_v0` 里的 `if (threadIdx.x < s)` 编译成：

```asm
ISETP.GE.U32.AND P1, PT, R7, R3     ; P1 = (threadIdx.x >= s)
@!P1 LDS  R4, [R7.X4]               ; 只有 !P1 的线程才执行
@!P1 LDS  R5, [R2]
@!P1 FADD R4, R4, R5
@!P1 STS  [R7.X4], R4
```

**没有 `BRA`。** 32 个线程走同一条指令流，不满足条件的线程结果被丢弃。

所以 warp divergence 的代价是**浪费发射槽**，不是跳转开销。

### ④ 操作数缓存标记

```asm
FFMA R10, R16.reuse, R45, R10 ;
```

`.reuse` 表示「这个寄存器和上一条指令用的是同一个，直接从操作数缓存取」。
编译器在帮忙减少操作数读取压力。

## 常用指令速查

| 指令 | 含义 |
|---|---|
| `LDG` / `STG` | LoaD / STore Global（显存） |
| `LDS` / `STS` | LoaD / STore Shared |
| `LDL` / `STL` | LoaD / STore Local ← **出现就说明 spill 了** |
| `FFMA` | 浮点乘加（一次做两次运算） |
| `FADD` / `FMUL` | 浮点加 / 乘 |
| `IMAD` | 整数乘加（算地址用得最多） |
| `ISETP` | 整数比较，结果进谓词 |
| `S2R` | 读特殊寄存器（threadIdx / blockIdx / laneid） |
| `BAR.SYNC` | `__syncthreads()` |
| `SHFL` | warp shuffle |
| `BRA` | 跳转 |
| `.128` 后缀 | 一条指令搬 128 位（4 个 float） |
| `.reuse` 标记 | 操作数复用上一条的寄存器 |
