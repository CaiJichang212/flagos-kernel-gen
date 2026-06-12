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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_PACKED_K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    expert_start = tl.load(expert_offsets_ptr + pid_e).to(tl.int32)
    expert_end = tl.load(expert_offsets_ptr + pid_e + 1).to(tl.int32)
    expert_m = expert_end - expert_start

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_pk = tl.arange(0, BLOCK_PACKED_K)

    mask_m = offs_m < expert_m
    mask_n = offs_n < N
    rows = expert_start + offs_m

    x_base = x_q_ptr + rows.to(tl.int64)[:, None] * stride_x_m
    w_base = w_q4_packed_ptr + pid_e * stride_w_e
    ws_base = w_scale_ptr + pid_e * stride_ws_e
    wz_base = w_zero_ptr + pid_e * stride_wz_e

    w_n_offsets = offs_n.to(tl.int64) * stride_w_n
    ws_n_offsets = offs_n.to(tl.int64) * stride_ws_n
    wz_n_offsets = offs_n.to(tl.int64) * stride_wz_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for group_id in range(0, NUM_GROUPS):
        pk_offsets = group_id * BLOCK_PACKED_K + offs_pk
        k_even = group_id * GROUP_SIZE + offs_pk * 2
        k_odd = k_even + 1

        x_even = tl.load(
            x_base + k_even[None, :].to(tl.int64) * stride_x_k,
            mask=mask_m[:, None],
            other=0,
            eviction_policy="evict_last",
        )
        x_odd = tl.load(
            x_base + k_odd[None, :].to(tl.int64) * stride_x_k,
            mask=mask_m[:, None],
            other=0,
            eviction_policy="evict_last",
        )

        w_packed = tl.load(
            w_base
            + pk_offsets[:, None].to(tl.int64) * stride_w_k
            + w_n_offsets[None, :],
            mask=mask_n[None, :],
            other=0,
            eviction_policy="evict_last",
        )

        w_zero_vals = tl.load(
            wz_base + group_id * stride_wz_g + wz_n_offsets,
            mask=mask_n,
            other=0,
            eviction_policy="evict_last",
        )

        w_packed_i32 = w_packed.to(tl.int32)
        w_zero_i32 = w_zero_vals.to(tl.int32)

        w_low = (w_packed_i32 & 0x0F) - w_zero_i32[None, :]
        w_high = ((w_packed_i32 >> 4) & 0x0F) - w_zero_i32[None, :]

        partial = tl.dot(x_even.to(tl.int8), w_low.to(tl.int8))
        partial += tl.dot(x_odd.to(tl.int8), w_high.to(tl.int8))

        w_scale_vals = tl.load(
            ws_base + group_id * stride_ws_g + ws_n_offsets,
            mask=mask_n,
            other=0.0,
            eviction_policy="evict_last",
        )
        acc += partial.to(tl.float32) * w_scale_vals[None, :]

    x_scale_vals = tl.load(
        x_scale_ptr + rows.to(tl.int64),
        mask=mask_m,
        other=0.0,
        eviction_policy="evict_last",
    )
    acc *= x_scale_vals[:, None]

    out_ptrs = (
        out_ptr
        + rows.to(tl.int64)[:, None] * stride_out_m
        + offs_n[None, :].to(tl.int64) * stride_out_n
    )
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def w4a8_group_gemm_moe(
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    w_q4_packed: torch.Tensor,
    w_scale: torch.Tensor,
    w_zero: torch.Tensor,
    expert_offsets: torch.Tensor,
    out: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    M_total, K = x_q.shape
    E, N, _ = w_q4_packed.shape
    num_groups = K // group_size
    block_packed_k = group_size // 2

    expert_counts = expert_offsets[1:] - expert_offsets[:-1]
    max_m_per_expert = int(expert_counts.max().item())
    if max_m_per_expert == 0:
        return out

    if max_m_per_expert <= 16:
        block_m = 16
    elif max_m_per_expert <= 32:
        block_m = 32
    else:
        block_m = 64

    if N >= 2048:
        block_n = 256
    elif N >= 1024:
        block_n = 128
    else:
        block_n = 64

    grid = (E, triton.cdiv(max_m_per_expert, block_m), triton.cdiv(N, block_n))

    _w4a8_group_gemm_moe_kernel[grid](
        x_q,
        x_scale,
        w_q4_packed,
        w_scale,
        w_zero,
        expert_offsets,
        out,
        N=N,
        K=K,
        GROUP_SIZE=group_size,
        stride_x_m=x_q.stride(0),
        stride_x_k=x_q.stride(1),
        stride_w_e=w_q4_packed.stride(0),
        stride_w_n=w_q4_packed.stride(1),
        stride_w_k=w_q4_packed.stride(2),
        stride_ws_e=w_scale.stride(0),
        stride_ws_n=w_scale.stride(1),
        stride_ws_g=w_scale.stride(2),
        stride_wz_e=w_zero.stride(0),
        stride_wz_n=w_zero.stride(1),
        stride_wz_g=w_zero.stride(2),
        stride_out_m=out.stride(0),
        stride_out_n=out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_PACKED_K=block_packed_k,
        NUM_GROUPS=num_groups,
        num_warps=8 if block_n >= 128 else 4,
        num_stages=2,
    )

    return out
