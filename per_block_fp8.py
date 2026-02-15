def per_block_fp8(q, k, km=None, BLKQ=128, BLKK=64, sm_scale=None, tensor_layout="HND"):
    q_fp8 = torch.empty(q.shape, dtype=torch.float8_e4m3fn, device=q.device)
    k_fp8 = torch.empty(k.shape, dtype=torch.float8_e4m3fn, device=k.device)
    if km is not None:
        k = k - km
    b, h_qo, qo_len, head_dim = q.shape
    _, h_kv, kv_len, _ = k.shape
    q_scale = torch.empty((b, h_qo, (qo_len + BLKQ - 1) // BLKQ), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((b, h_kv, (kv_len + BLKK - 1) // BLKK), device=q.device, dtype=torch.float32)
    if sm_scale is None:
        sm_scale = head_dim ** -0.5
    grid = ((qo_len + BLKQ - 1) // BLKQ, h_qo, b)
    quant_per_block_fp8_kernel[grid](
        q, q_fp8, q_scale, qo_len,
        q.stride(0), q.stride(1), q.stride(2),
        q_fp8.stride(0), q_fp8.stride(1), q_fp8.stride(2),
        q_scale.stride(0), q_scale.stride(1),
        sm_scale=(sm_scale * 1.44269504), C=head_dim, BLK=BLKQ)
    grid = ((kv_len + BLKK - 1) // BLKK, h_kv, b)
    quant_per_block_fp8_kernel[grid](
        k, k_fp8, k_scale, kv_len,
        k.stride(0), k.stride(1), k.stride(2),
        k_fp8.stride(0), k_fp8.stride(1), k_fp8.stride(2),
        k_scale.stride(0), k_scale.stride(1),
        sm_scale=1.0, C=head_dim, BLK=BLKK)
    return q_fp8, q_scale, k_fp8, k_scale
