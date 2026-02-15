def bench_triton_do_bench():
    print("=" * 60)
    print("方法一：triton.testing.do_bench")
    print("=" * 60)

    configs = [
        # (batch, heads, seq_len, head_dim)
        (1, 32, 1024, 128),
        (1, 32, 4096, 128),
        (1, 32, 8192, 128),
        (2, 32, 4096, 128),
        (1, 32, 16384, 128),
    ]

    for b, h, seq, d in configs:
        q = torch.randn(b, h, seq, d, dtype=torch.float16, device="cuda")
        k = torch.randn(b, h, seq, d, dtype=torch.float16, device="cuda")

        # warmup + benchmark, 返回毫秒
        ms = triton.testing.do_bench(lambda: per_block_fp8(q, k))

        # 计算吞吐量: 需要读 Q+K (FP16) + 写 Q_fp8+K_fp8 (FP8) + scales
        num_elements = q.numel() + k.numel()
        read_bytes = num_elements * 2           # FP16 = 2 bytes
        write_bytes = num_elements * 1           # FP8  = 1 byte
        total_bytes = read_bytes + write_bytes
        bandwidth_gbps = total_bytes / (ms * 1e-3) / 1e9

        print(f"  B={b}, H={h}, Seq={seq}, D={d}  |  {ms:.3f} ms  |  {bandwidth_gbps:.1f} GB/s")


# ============================================================
# 3. 方法二：使用 CUDA Event（手动计时，更灵活）
# ============================================================
def bench_cuda_events():
    print("\n" + "=" * 60)
    print("方法二：CUDA Events")
    print("=" * 60)

    b, h, seq, d = 1, 32, 4096, 128
    q = torch.randn(b, h, seq, d, dtype=torch.float16, device="cuda")
    k = torch.randn(b, h, seq, d, dtype=torch.float16, device="cuda")

    warmup = 10
    repeat = 100

    # Warmup
    for _ in range(warmup):
        per_block_fp8(q, k)

    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(repeat):
        per_block_fp8(q, k)
    end_event.record()
    torch.cuda.synchronize()

    avg_ms = start_event.elapsed_time(end_event) / repeat
    print(f"  Config: B={b}, H={h}, Seq={seq}, D={d}")
    print(f"  Average: {avg_ms:.3f} ms  ({repeat} iterations)")


# ============================================================
# 4. 方法三：triton.testing.Benchmark（自动生成对比图）
# ============================================================
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[2**i for i in range(10, 16)],  # 1024 ~ 32768
        x_log=True,
        line_arg="provider",
        line_vals=["fp8_quant", "naive_fp16_cast"],
        line_names=["FP8 Quantization", "Naive FP16→FP8 Cast"],
        styles=[("blue", "-"), ("red", "--")],
        ylabel="ms",
        plot_name="FP8 Quantization Benchmark",
        args={"batch": 1, "heads": 32, "head_dim": 128},
    )
)
def bench_plot(batch, heads, seq_len, head_dim, provider):
    q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
    k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")

    if provider == "fp8_quant":
        fn = lambda: per_block_fp8(q, k)
    else:
        # 对比：简单的 FP16 → FP8 直接转换（无 scale）
        fn = lambda: (q.to(torch.float8_e4m3fn), k.to(torch.float8_e4m3fn))

    ms = triton.testing.do_bench(fn)
    return ms


# ============================================================
# 5. 方法四：与 INT8 原版对比
# ============================================================
@triton.jit
def quant_per_block_int8_kernel(Input, Output, Scale, L,
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
    scale = tl.max(tl.abs(x)) / 127.
    x_int8 = x / scale
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(tl.int8)
    tl.store(output_ptrs, x_int8, mask=offs_n[:, None] < L)
    tl.store(scale_ptrs, scale)

def per_block_int8(q, k, km=None, BLKQ=128, BLKK=64, sm_scale=None):
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)
    if km is not None:
        k = k - km
    b, h_qo, qo_len, head_dim = q.shape
    _, h_kv, kv_len, _ = k.shape
    q_scale = torch.empty((b, h_qo, (qo_len + BLKQ - 1) // BLKQ), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((b, h_kv, (kv_len + BLKK - 1) // BLKK), device=q.device, dtype=torch.float32)
    if sm_scale is None:
        sm_scale = head_dim ** -0.5
    grid = ((qo_len + BLKQ - 1) // BLKQ, h_qo, b)
    quant_per_block_int8_kernel[grid](
        q, q_int8, q_scale, qo_len,
        q.stride(0), q.stride(1), q.stride(2),
        q_int8.stride(0), q_int8.stride(1), q_int8.stride(2),
        q_scale.stride(0), q_scale.stride(1),
        sm_scale=(sm_scale * 1.44269504), C=head_dim, BLK=BLKQ)
    grid = ((kv_len + BLKK - 1) // BLKK, h_kv, b)
    quant_per_block_int8_kernel[grid](
        k, k_int8, k_scale, kv_len,
        k.stride(0), k.stride(1), k.stride(2),
        k_int8.stride(0), k_int8.stride(1), k_int8.stride(2),
        k_scale.stride(0), k_scale.stride(1),
        sm_scale=1.0, C=head_dim, BLK=BLKK)
    return q_int8, q_scale, k_int8, k_scale

def bench_fp8_vs_int8():
    print("\n" + "=" * 60)
    print("方法四：FP8 vs INT8 对比")
    print("=" * 60)
    print(f"  {'Config':<35} | {'INT8 (ms)':>10} | {'FP8 (ms)':>10} | {'Speedup':>8}")
    print("-" * 75)

    configs = [
        (1, 32, 1024, 128),
        (1, 32, 4096, 128),
        (1, 32, 8192, 128),
        (1, 32, 16384, 128),
        (1, 32, 32768, 128),
    ]

    for b, h, seq, d in configs:
        q = torch.randn(b, h, seq, d, dtype=torch.float16, device="cuda")
        k = torch.randn(b, h, seq, d, dtype=torch.float16, device="cuda")

        ms_int8 = triton.testing.do_bench(lambda: per_block_int8(q, k))
        ms_fp8  = triton.testing.do_bench(lambda: per_block_fp8(q, k))
        speedup = ms_int8 / ms_fp8

        label = f"B={b}, H={h}, Seq={seq}, D={d}"
        print(f"  {label:<35} | {ms_int8:>10.3f} | {ms_fp8:>10.3f} | {speedup:>7.2f}x")


# ============================================================
# 运行所有 benchmark
# ============================================================
if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton: {triton.__version__}")
    print()

    # 方法一：快速测多个配置
    bench_triton_do_bench()

    # 方法二：CUDA Event 手动计时
    bench_cuda_events()

    # 方法三：生成对比图（会保存 .png 文件）
    # bench_plot.run(save_path="./benchmark_results/", print_data=True)

    # 方法四：FP8 vs INT8 对比
    bench_fp8_vs_int8()