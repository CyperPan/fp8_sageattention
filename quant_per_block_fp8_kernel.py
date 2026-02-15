import torch
import triton
import triton.language as tl
import time

# ============================================================
# 1. FP8 量化内核（被测代码）
# ============================================================
@triton.jit
def quant_per_block_fp8_kernel(Input, Output, Scale, L,
                               stride_iz, stride_ih, stride_in,
                               stride_oz, stride_oh, stride_on,
                               stride_sz, stride_sh,
                               sm_scale,
                               C: tl.constexpr, BLK: tl.constexpr):
    off_blk = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)
    offs_n = off_blk * BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, C)
    input_ptrs = Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None] * stride_in + offs_k[None, :]
    output_ptrs = Output + off_b * stride_oz + off_h * stride_oh + offs_n[:, None] * stride_on + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk
    x = tl.load(input_ptrs, mask=offs_n[:, None] < L)
    x = x.to(tl.float32)
    x *= sm_scale
    scale = tl.max(tl.abs(x)) / 448.0
    x_fp8 = x / scale
    x_fp8 = x_fp8.to(tl.float8e4nv)
    tl.store(output_ptrs, x_fp8, mask=offs_n[:, None] < L)
    tl.store(scale_ptrs, scale)