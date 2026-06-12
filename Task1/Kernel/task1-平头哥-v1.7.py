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
    num_tiles_m,
    num_tiles_n,
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
    GROUP_TILES: tl.constexpr,
):
    pid = tl.program_id(0)
    tiles_per_expert = num_tiles_m * num_tiles_n
    pid_e = pid // tiles_per_expert
    remainder = pid % tiles_per_expert

    num_pid_n = num_tiles_n
    num_pid_in_group = GROUP_TILES * num_pid_n
    group_id_sw = remainder // num_pid_in_group
    first_pid_m = group_id_sw * GROUP_TILES
    group_size_m = tl.minimum(num_tiles_m - first_pid_m, GROUP_TILES)
    pid_m = first_pid_m + (remainder % group_size_m)
    pid_n = (remainder % num_pid_in_group) // group_size_m

    expert_start = tl.load(expert_offsets_ptr + pid_e)
    expert_end = tl.load(expert_offsets_ptr + pid_e + 1)
    expert_start_i32 = expert_start.to(tl.int32)
    expert_m = expert_end.to(tl.int32) - expert_start_i32

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_pk = tl.arange(0, BLOCK_PACKED_K)

    mask_m = offs_m < expert_m
    mask_n = offs_n < N
    rows = expert_start_i32 + offs_m

    x_base = x_q_ptr + (rows * stride_x_m)[:, None]

    pid_e_i32 = pid_e.to(tl.int32)
    w_base = w_q4_packed_ptr + pid_e_i32 * stride_w_e
    ws_base = w_scale_ptr + pid_e_i32 * stride_ws_e
    wz_base = w_zero_ptr + pid_e_i32 * stride_wz_e

    w_n_offsets = (offs_n * stride_w_n)[None, :]
    ws_n_offsets = offs_n * stride_ws_n
    wz_n_offsets = offs_n * stride_wz_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for g in range(0, NUM_GROUPS):
        pk_base = g * BLOCK_PACKED_K
        pk_offsets = pk_base + offs_pk
        k_even = g * GROUP_SIZE + offs_pk * 2
        k_odd = k_even + 1

        x_even = tl.load(
            x_base + (k_even * stride_x_k)[None, :],
            mask=mask_m[:, None],
            other=0,
        )
        x_odd = tl.load(
            x_base + (k_odd * stride_x_k)[None, :],
            mask=mask_m[:, None],
            other=0,
        )

        w_packed = tl.load(
            w_base + (pk_offsets * stride_w_k)[:, None] + w_n_offsets,
            mask=mask_n[None, :],
            other=0,
        )

        w_zero_vals = tl.load(
            wz_base + g * stride_wz_g + wz_n_offsets,
            mask=mask_n,
            other=0,
            eviction_policy="evict_last",
        )
        w_scale_vals = tl.load(
            ws_base + g * stride_ws_g + ws_n_offsets,
            mask=mask_n,
            other=0.0,
            eviction_policy="evict_last",
        )

        w_packed_i32 = w_packed.to(tl.int32)
        w_zero_i32 = w_zero_vals.to(tl.int32)[None, :]

        w_low = ((w_packed_i32 & 0x0F) - w_zero_i32).to(tl.int8)
        w_high = (((w_packed_i32 >> 4) & 0x0F) - w_zero_i32).to(tl.int8)

        partial = tl.dot(x_even.to(tl.int8), w_low)
        partial += tl.dot(x_odd.to(tl.int8), w_high)

        acc += partial.to(tl.float32) * w_scale_vals[None, :]

    x_scale_vals = tl.load(x_scale_ptr + rows, mask=mask_m, other=0.0)
    acc *= x_scale_vals[:, None]

    out_ptrs = (
        out_ptr
        + (rows * stride_out_m)[:, None]
        + (offs_n * stride_out_n)[None, :]
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
    elif max_m_per_expert <= 64:
        block_m = 64
    else:
        block_m = 128

    if (N >= 1408 and max_m_per_expert >= 128) or N >= 2048:
        block_n = 256
        num_warps = 8
        num_stages = 2
    elif N >= 1024:
        block_n = 128
        num_warps = 4
        num_stages = 2
    else:
        block_n = 64
        num_warps = 4
        num_stages = 2

    if max_m_per_expert <= 32 and N >= 1024:
        block_n = 128
        num_warps = 4

    num_tiles_m = (max_m_per_expert + block_m - 1) // block_m
    num_tiles_n = (N + block_n - 1) // block_n
    total_programs = E * num_tiles_m * num_tiles_n

    group_tiles = min(8, num_tiles_m)

    grid = (total_programs,)

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
        num_tiles_m=num_tiles_m,
        num_tiles_n=num_tiles_n,
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
        GROUP_TILES=group_tiles,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
