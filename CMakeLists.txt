cmake_minimum_required(VERSION 3.24)
project(cuda_kernel_optimization LANGUAGES CXX CUDA)

# ============================================================
# 编译配置
#   - C++17
#   - sm_89：RTX 4070 Ti Super（Ada, CC 8.9）
#     换卡时改这一行即可，或者用 -DCMAKE_CUDA_ARCHITECTURES=native
#   - /utf-8：源码里有中文注释，MSVC 默认按 GBK 读会乱码
#   - -Xptxas -v：编译时打印每个 kernel 的寄存器用量 / shared 用量 / spill
#     这是本项目的基本功——寄存器数目以 ptxas 为准，不靠估
# ============================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /utf-8")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=\"/utf-8\" -Xptxas -v")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 公共头文件目录：common/cuda_check.h
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)

add_subdirectory(vector_add)
add_subdirectory(reduction)
add_subdirectory(scan)
add_subdirectory(matmul)
add_subdirectory(histogram)
add_subdirectory(softmax)
