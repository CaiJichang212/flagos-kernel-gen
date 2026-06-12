import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=2),
    ],
    key=['N', 'K', 'group_size'],
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
    N: tl.constexpr,
    K: tl.constexpr,
    group_size: tl.constexpr,
    stride_x_m, stride_x_k,
    stride_wq_e, stride_wq_n, stride_wq_k,
    stride_ws_e, stride_ws_n, stride_ws_g,
    stride_wz_e, stride_wz_n, stride_wz_g,
    stride_out_m, stride_out_n,
    E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_expert = tl.program_id(1)

    # Load expert boundaries
    expert_start = tl.load(expert_offsets_ptr + pid_expert)
    expert_end = tl.load(expert_offsets_ptr + pid_expert + 1)
    M_expert = expert_end - expert_start

    # Early exit if no tokens for this expert
    if M_expert <= 0:
        return

    # Compute tile indices for M and N dimensions
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n_tiles
    pid_n = pid % num_n_tiles

    # Check if this M tile is within bounds
    if pid_m * BLOCK_M >= M_expert:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M_expert
    mask_n = offs_n < N

    # Global row indices into x_q and out
    global_m = expert_start + offs_m

    # Accumulator in fp32
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Number of groups in K dimension
    num_groups = K // group_size
    # Number of K-blocks per group
    k_blocks_per_group = group_size // BLOCK_K

    # Base pointers for this expert's weights
    w_base = pid_expert * stride_wq_e
    ws_base = pid_expert * stride_ws_e
    wz_base = pid_expert * stride_wz_e

    # Iterate over groups
    for g in range(num_groups):
        # Load weight scale and zero for this group: [BLOCK_N]
        ws_ptrs = w_scale_ptr + ws_base + offs_n * stride_ws_n + g * stride_ws_g
        wz_ptrs = w_zero_ptr + wz_base + offs_n * stride_wz_n + g * stride_wz_g

        w_scale_g = tl.load(ws_ptrs, mask=mask_n, other=0.0)  # [BLOCK_N], fp32
        w_zero_g = tl.load(wz_ptrs, mask=mask_n, other=0).to(tl.int32)  # [BLOCK_N], int32

        # Partial accumulator for this group
        group_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)

        # Iterate over K blocks within this group
        for kb in range(k_blocks_per_group):
            k_start = g * group_size + kb * BLOCK_K
            offs_k = k_start + tl.arange(0, BLOCK_K)

            # Load x_q tile: [BLOCK_M, BLOCK_K], int8
            x_ptrs = x_q_ptr + global_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k
            x_mask = mask_m[:, None] & (offs_k[None, :] < K)
            x_tile = tle.load(x_ptrs, mask=x_mask, other=0, is_async=True)  # int8

            # Load packed w_q4: [BLOCK_N, BLOCK_K // 2], uint8
            # packed index: k // 2
            offs_k_half = k_start // 2 + tl.arange(0, BLOCK_K // 2)
            w_packed_ptrs = w_q4_packed_ptr + w_base + offs_n[:, None] * stride_wq_n + offs_k_half[None, :] * stride_wq_k
            w_mask = mask_n[:, None] & (offs_k_half[None, :] < (K // 2))
            w_packed = tle.load(w_packed_ptrs, mask=w_mask, other=0, is_async=False)  # [BLOCK_N, BLOCK_K//2], uint8

            # Unpack INT4 -> INT8
            w_low = (w_packed & 0x0F).to(tl.int8)   # [BLOCK_N, BLOCK_K//2] - even indices
            w_high = ((w_packed >> 4) & 0x0F).to(tl.int8)  # [BLOCK_N, BLOCK_K//2] - odd indices

            # Interleave to get [BLOCK_N, BLOCK_K]
            # We need to interleave w_low and w_high along the last dimension
            # w_low corresponds to even k indices, w_high to odd k indices
            # Reshape and interleave
            w_low_32 = w_low.to(tl.int32)    # [BLOCK_N, BLOCK_K//2]
            w_high_32 = w_high.to(tl.int32)  # [BLOCK_N, BLOCK_K//2]

            # Subtract zero point
            w_zero_g_expanded = w_zero_g[:, None]  # [BLOCK_N, 1]
            w_low_32 = w_low_32 - w_zero_g_expanded
            w_high_32 = w_high_32 - w_zero_g_expanded

            # Convert to int8 for dot product
            w_low_i8 = w_low_32.to(tl.int8)
            w_high_i8 = w_high_32.to(tl.int8)

            # x_tile is [BLOCK_M, BLOCK_K] int8
            # We need to split x into even and odd columns
            # x_even = x_tile[:, 0::2], x_odd = x_tile[:, 1::2]
            # But Triton doesn't support stride indexing directly
            # Instead we reshape: [BLOCK_M, BLOCK_K//2, 2]
            x_tile_reshaped = tl.reshape(x_tile, [BLOCK_M, BLOCK_K // 2, 2])
            x_even = tl.reshape(x_tile_reshaped[:, :, 0:1], [BLOCK_M, BLOCK_K // 2])  # [BLOCK_M, BLOCK_K//2]
            x_odd = tl.reshape(x_tile_reshaped[:, :, 1:2], [BLOCK_M, BLOCK_K // 2])   # [BLOCK_M, BLOCK_K//2]

            # w_low_i8^T: [BLOCK_K//2, BLOCK_N], x_even: [BLOCK_M, BLOCK_K//2]
            # partial = x_even @ w_low^T + x_odd @ w_high^T
            # tl.dot for int8 produces int32
            w_low_t = tl.trans(w_low_i8)   # [BLOCK_K//2, BLOCK_N]
            w_high_t = tl.trans(w_high_i8)  # [BLOCK_K//2, BLOCK_N]

            group_acc = tl.dot(x_even, w_low_t, group_acc)
            group_acc = tl.dot(x_odd, w_high_t, group_acc)

        # Apply weight scale for this group
        # group_acc: [BLOCK_M, BLOCK_N] int32
        # w_scale_g: [BLOCK_N] fp32
        acc += group_acc.to(tl.float32) * w_scale_g[None, :]

    # Apply per-token activation scale
    x_scale_vals = tl.load(x_scale_ptr + global_m, mask=mask_m, other=0.0)  # [BLOCK_M], fp32
    acc = acc * x_scale_vals[:, None]

    # Store output in bf16
    out_ptrs = out_ptr + global_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
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
    M_total = x_q.shape[0]
    K = x_q.shape[1]
    E = w_q4_packed.shape[0]
    N = w_q4_packed.shape[1]

    # Compute maximum possible M per expert for grid sizing
    # Use a conservative upper bound
    max_M = M_total

    def grid(META):
        num_m_tiles = triton.cdiv(max_M, META['BLOCK_M'])
        num_n_tiles = triton.cdiv(N, META['BLOCK_N'])
        return (num_m_tiles * num_n_tiles, E)

    _w4a8_group_gemm_moe_kernel[grid](
        x_q,
        x_scale,
        w_q4_packed,
        w_scale,
        w_zero,
        expert_offsets,
        out,
        N, K, group_size,
        x_q.stride(0), x_q.stride(1),
        w_q4_packed.stride(0), w_q4_packed.stride(1), w_q4_packed.stride(2),
        w_scale.stride(0), w_scale.stride(1), w_scale.stride(2),
        w_zero.stride(0), w_zero.stride(1), w_zero.stride(2),
        out.stride(0), out.stride(1),
        E,
    )

    return out