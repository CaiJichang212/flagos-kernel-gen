import torch
import triton
import triton.language as tl


@triton.jit
def _w4a8_group_gemm_moe_kernel(
    x_q_ptr,
    x_scale_ptr,
    w_q4_packed_ptr,
    w_scale_ptr,
    w_zero_ptr,
    expert_offsets_ptr,
    out_ptr,
    N,
    K,
    GROUP_SIZE: tl.constexpr,
    stride_x_m,
    stride_x_k,
    stride_w_e,
    stride_w_n,
    stride_w_k,
    stride_ws_e,
    stride_ws_n,
    stride_ws_g,
    stride_wz_e,
    stride_wz_n,
    stride_wz_g,
    stride_out_m,
    stride_out_n,
    NUM_GROUPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HALF_GROUP: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    expert_start = tl.load(expert_offsets_ptr + pid_e)
    expert_end = tl.load(expert_offsets_ptr + pid_e + 1)
    expert_m = expert_end - expert_start

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_hg = tl.arange(0, HALF_GROUP)

    rows = expert_start + offs_m
    mask_m = offs_m < expert_m
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    w_base = w_q4_packed_ptr + pid_e * stride_w_e
    ws_base = w_scale_ptr + pid_e * stride_ws_e
    wz_base = w_zero_ptr + pid_e * stride_wz_e

    offs_n_ws = offs_n * stride_ws_n
    offs_n_wz = offs_n * stride_wz_n
    offs_n_w = offs_n * stride_w_n

    row_x_base = rows * stride_x_m
    offs_hg_wk = offs_hg * stride_w_k
    offs_hg_even_xk = (offs_hg * 2) * stride_x_k
    offs_hg_odd_xk = (offs_hg * 2 + 1) * stride_x_k

    mask_n_2d = mask_n[None, :]
    mask_m_2d = mask_m[:, None]

    MASK_LOW: tl.constexpr = 0x0F

    for group_id in range(0, NUM_GROUPS):
        k_base_xk = (group_id * GROUP_SIZE) * stride_x_k
        packed_base_wk = (group_id * HALF_GROUP) * stride_w_k

        x_even_ptrs = x_q_ptr + row_x_base[:, None] + (k_base_xk + offs_hg_even_xk[None, :])
        x_odd_ptrs = x_q_ptr + row_x_base[:, None] + (k_base_xk + offs_hg_odd_xk[None, :])
        x_even = tl.load(x_even_ptrs, mask=mask_m_2d, other=0)
        x_odd = tl.load(x_odd_ptrs, mask=mask_m_2d, other=0)

        w_packed_ptrs = w_base + (packed_base_wk + offs_hg_wk[:, None]) + offs_n_w[None, :]
        w_packed = tl.load(w_packed_ptrs, mask=mask_n_2d, other=0)

        w_zero_vals = tl.load(wz_base + offs_n_wz + group_id * stride_wz_g, mask=mask_n, other=0)

        w_packed_i16 = w_packed.to(tl.int16)
        w_zero_i16 = w_zero_vals.to(tl.int16)
        w_low_f = ((w_packed_i16 & MASK_LOW) - w_zero_i16[None, :]).to(tl.float16)
        w_high_f = (((w_packed_i16 >> 4) & MASK_LOW) - w_zero_i16[None, :]).to(tl.float16)

        x_even_f = x_even.to(tl.float16)
        x_odd_f = x_odd.to(tl.float16)

        partial = tl.dot(x_even_f, w_low_f) + tl.dot(x_odd_f, w_high_f)

        w_scale_vals = tl.load(ws_base + offs_n_ws + group_id * stride_ws_g, mask=mask_n, other=0.0)
        acc += partial * w_scale_vals[None, :]

    x_scale_vals = tl.load(x_scale_ptr + rows, mask=mask_m, other=0.0)
    acc *= x_scale_vals[:, None]

    out_ptrs = out_ptr + rows[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    out_mask = mask_m_2d & mask_n[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


def w4a8_group_gemm_moe(x_q: torch.Tensor, x_scale: torch.Tensor, w_q4_packed: torch.Tensor, w_scale: torch.Tensor, w_zero: torch.Tensor, expert_offsets: torch.Tensor, out: torch.Tensor, group_size: int) -> torch.Tensor:
    M_total, K = x_q.shape
    E, N, _ = w_q4_packed.shape
    num_groups = K // group_size
    half_group = group_size // 2

    expert_counts = expert_offsets[1:] - expert_offsets[:-1]
    max_m_per_expert = int(expert_counts.max().item())
    if max_m_per_expert == 0:
        return out

    if max_m_per_expert <= 16:
        BLOCK_M = 16
    elif max_m_per_expert <= 32:
        BLOCK_M = 32
    else:
        BLOCK_M = 64

    if N <= 64:
        BLOCK_N = 32
        num_warps = 4
    elif N <= 256:
        BLOCK_N = 64
        num_warps = 4
    else:
        BLOCK_N = 128
        num_warps = 8

    num_stages = 2

    grid = (E, triton.cdiv(max_m_per_expert, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _w4a8_group_gemm_moe_kernel[grid](
        x_q, x_scale, w_q4_packed, w_scale, w_zero, expert_offsets, out,
        N=N, K=K,
        GROUP_SIZE=group_size,
        stride_x_m=x_q.stride(0), stride_x_k=x_q.stride(1),
        stride_w_e=w_q4_packed.stride(0), stride_w_n=w_q4_packed.stride(1), stride_w_k=w_q4_packed.stride(2),
        stride_ws_e=w_scale.stride(0), stride_ws_n=w_scale.stride(1), stride_ws_g=w_scale.stride(2),
        stride_wz_e=w_zero.stride(0), stride_wz_n=w_zero.stride(1), stride_wz_g=w_zero.stride(2),
        stride_out_m=out.stride(0), stride_out_n=out.stride(1),
        NUM_GROUPS=num_groups,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HALF_GROUP=half_group,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
