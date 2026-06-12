import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_stages=3, num_warps=8),
    ],
    key=["N", "K", "NUM_GROUPS"],
)
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
    max_m_per_expert,
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
    NUM_EXPERTS: tl.constexpr,
):
    pid = tl.program_id(0)

    num_tiles_n: tl.constexpr = tl.cdiv(N, BLOCK_N)
    max_m_tiles = tl.cdiv(max_m_per_expert, BLOCK_M)

    tiles_per_expert = max_m_tiles * num_tiles_n
    pid_e = pid // tiles_per_expert
    remainder = pid % tiles_per_expert
    pid_m = remainder // num_tiles_n
    pid_n = remainder % num_tiles_n

    valid_expert = pid_e < NUM_EXPERTS
    safe_pid_e = tl.minimum(pid_e, NUM_EXPERTS - 1)
    expert_start = tl.load(expert_offsets_ptr + safe_pid_e).to(tl.int32)
    expert_end = tl.load(expert_offsets_ptr + safe_pid_e + 1).to(tl.int32)
    expert_m = expert_end - expert_start

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_pk = tl.arange(0, BLOCK_PACKED_K)

    mask_m = (offs_m < expert_m) & valid_expert
    mask_n = offs_n < N
    rows = expert_start + offs_m

    x_base = x_q_ptr + (rows * stride_x_m)[:, None]
    w_expert_base = w_q4_packed_ptr + safe_pid_e * stride_w_e
    ws_expert_base = w_scale_ptr + safe_pid_e * stride_ws_e
    wz_expert_base = w_zero_ptr + safe_pid_e * stride_wz_e

    w_n_offs = offs_n * stride_w_n
    ws_n_offs = offs_n * stride_ws_n
    wz_n_offs = offs_n * stride_wz_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    combined_mask_mn = mask_m[:, None] & mask_n[None, :]

    for group_id in range(0, NUM_GROUPS):
        pk_base = group_id * BLOCK_PACKED_K
        pk_offsets = pk_base + offs_pk
        k_even = (group_id * GROUP_SIZE) + offs_pk * 2
        k_odd = k_even + 1

        x_even_vals = tl.load(
            x_base + (k_even * stride_x_k)[None, :],
            mask=mask_m[:, None],
            other=0,
            eviction_policy="evict_last",
        )
        x_odd_vals = tl.load(
            x_base + (k_odd * stride_x_k)[None, :],
            mask=mask_m[:, None],
            other=0,
            eviction_policy="evict_last",
        )

        w_packed = tl.load(
            w_expert_base + (pk_offsets * stride_w_k)[:, None] + w_n_offs[None, :],
            mask=mask_n[None, :],
            other=0,
            eviction_policy="evict_first",
        )

        w_zero_vals = tl.load(
            wz_expert_base + group_id * stride_wz_g + wz_n_offs,
            mask=mask_n,
            other=0,
            eviction_policy="evict_first",
        )

        w_i32 = w_packed.to(tl.int32)
        wz_i32 = w_zero_vals.to(tl.int32)[None, :]
        w_low = ((w_i32 & 0x0F) - wz_i32).to(tl.int8)
        w_high = (((w_i32 >> 4) & 0x0F) - wz_i32).to(tl.int8)

        partial = tl.dot(x_even_vals.to(tl.int8), w_low)
        partial += tl.dot(x_odd_vals.to(tl.int8), w_high)

        w_scale_vals = tl.load(
            ws_expert_base + group_id * stride_ws_g + ws_n_offs,
            mask=mask_n,
            other=0.0,
            eviction_policy="evict_first",
        )
        acc += partial.to(tl.float32) * w_scale_vals[None, :]

    x_scale_vals = tl.load(x_scale_ptr + rows, mask=mask_m, other=0.0)
    acc *= x_scale_vals[:, None]

    out_ptrs = (
        out_ptr
        + (rows * stride_out_m)[:, None]
        + (offs_n * stride_out_n)[None, :]
    )
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=combined_mask_mn)


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

    def grid_fn(META):
        bm = META["BLOCK_M"]
        bn = META["BLOCK_N"]
        m_tiles = triton.cdiv(max_m_per_expert, bm)
        n_tiles = triton.cdiv(N, bn)
        return (E * m_tiles * n_tiles,)

    _w4a8_group_gemm_moe_kernel[grid_fn](
        x_q,
        x_scale,
        w_q4_packed,
        w_scale,
        w_zero,
        expert_offsets,
        out,
        N=N,
        K=K,
        max_m_per_expert=max_m_per_expert,
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
        BLOCK_PACKED_K=block_packed_k,
        NUM_GROUPS=num_groups,
        NUM_EXPERTS=E,
    )
    return out
