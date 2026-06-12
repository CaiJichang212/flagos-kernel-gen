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
    N: tl.constexpr,
    K: tl.constexpr,
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
):
    pid_e = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    expert_start = tl.load(expert_offsets_ptr + pid_e)
    expert_end = tl.load(expert_offsets_ptr + pid_e + 1)
    expert_m = expert_end - expert_start

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_pk = tl.arange(0, BLOCK_PACKED_K)

    rows = expert_start + offs_m
    mask_m = offs_m < expert_m
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for group_id in range(0, K // GROUP_SIZE):
        packed_k = group_id * BLOCK_PACKED_K + offs_pk
        k_even = group_id * GROUP_SIZE + offs_pk * 2
        k_odd = k_even + 1

        x_even_ptrs = (
            x_q_ptr
            + rows[:, None] * stride_x_m
            + k_even[None, :] * stride_x_k
        )
        x_odd_ptrs = (
            x_q_ptr
            + rows[:, None] * stride_x_m
            + k_odd[None, :] * stride_x_k
        )
        x_mask = mask_m[:, None]
        x_even = tl.load(x_even_ptrs, mask=x_mask, other=0)
        x_odd = tl.load(x_odd_ptrs, mask=x_mask, other=0)

        w_packed_ptrs = (
            w_q4_packed_ptr
            + pid_e * stride_w_e
            + offs_n[:, None] * stride_w_n
            + packed_k[None, :] * stride_w_k
        )
        w_packed = tl.load(w_packed_ptrs, mask=mask_n[:, None], other=0)

        w_zero_vals = tl.load(
            w_zero_ptr
            + pid_e * stride_wz_e
            + offs_n * stride_wz_n
            + group_id * stride_wz_g,
            mask=mask_n,
            other=0,
        ).to(tl.int16)

        w_low = ((w_packed & 0x0F).to(tl.int16) - w_zero_vals[:, None]).to(tl.int8)
        w_high = (((w_packed >> 4) & 0x0F).to(tl.int16) - w_zero_vals[:, None]).to(
            tl.int8
        )

        partial_even = tl.dot(x_even.to(tl.int8), tl.trans(w_low))
        partial_odd = tl.dot(x_odd.to(tl.int8), tl.trans(w_high))
        partial = (partial_even + partial_odd).to(tl.float32)

        w_scale_vals = tl.load(
            w_scale_ptr
            + pid_e * stride_ws_e
            + offs_n * stride_ws_n
            + group_id * stride_ws_g,
            mask=mask_n,
            other=0.0,
        )
        acc += partial * w_scale_vals[None, :]

    x_scale_vals = tl.load(x_scale_ptr + rows, mask=mask_m, other=0.0)
    acc *= x_scale_vals[:, None]

    out_ptrs = out_ptr + rows[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


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

    avg_m_per_expert = triton.cdiv(M_total, E)
    block_m = 16 if avg_m_per_expert <= 32 else 32
    block_n = 64
    block_packed_k = group_size // 2

    expert_counts = expert_offsets[1:] - expert_offsets[:-1]
    max_m_per_expert = int(expert_counts.max().item())
    if max_m_per_expert == 0:
        return out

    grid = (
        E,
        triton.cdiv(max_m_per_expert, block_m),
        triton.cdiv(N, block_n),
    )

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
    )

    return out
