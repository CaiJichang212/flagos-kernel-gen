import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=8, num_stages=2),
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
    E: tl.constexpr,
    stride_x_m,
    stride_x_k,
    stride_w_e,
    stride_w_n,
    stride_w_k2,
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
    BLOCK_K: tl.constexpr,
):
    pid_mn = tl.program_id(0)
    pid_e = tl.program_id(1)

    # Load expert offsets
    expert_start = tl.load(expert_offsets_ptr + pid_e)
    expert_end = tl.load(expert_offsets_ptr + pid_e + 1)
    num_tokens = expert_end - expert_start

    if num_tokens <= 0:
        return

    # Compute number of M and N blocks
    num_m_blocks = tl.cdiv(num_tokens, BLOCK_M)
    num_n_blocks = tl.cdiv(N, BLOCK_N)

    # Decode pid_mn into m_block and n_block
    pid_m = pid_mn % num_m_blocks
    pid_n = pid_mn // num_m_blocks

    if pid_n >= num_n_blocks:
        return

    # Row and column offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < num_tokens
    mask_n = offs_n < N

    # Global row indices
    global_m = expert_start + offs_m

    # Number of groups along K
    num_groups = K // group_size

    # Accumulator in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over groups
    for g in range(num_groups):
        group_start = g * group_size

        # Load w_scale for this group: shape [BLOCK_N]
        # w_scale: [E, N, K//group_size], stride_ws_e, stride_ws_n, stride_ws_g
        ws_ptrs = w_scale_ptr + pid_e * stride_ws_e + offs_n * stride_ws_n + g * stride_ws_g
        w_s = tl.load(ws_ptrs, mask=mask_n, other=0.0)  # [BLOCK_N], fp32

        # Load w_zero for this group: shape [BLOCK_N]
        wz_ptrs = w_zero_ptr + pid_e * stride_wz_e + offs_n * stride_wz_n + g * stride_wz_g
        w_z = tl.load(wz_ptrs, mask=mask_n, other=0)  # [BLOCK_N], int8

        # Cast w_z to int32 for subtraction later
        w_z_i32 = w_z.to(tl.int32)

        # Accumulate partial sum for this group in int32
        group_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

        # Iterate over K within the group using BLOCK_K tiles
        num_k_iters = group_size // BLOCK_K
        for ki in range(num_k_iters):
            k_start = group_start + ki * BLOCK_K
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K

            # Load x_q: [BLOCK_M, BLOCK_K], int8
            x_ptrs = x_q_ptr + global_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k
            x_mask = mask_m[:, None] & mask_k[None, :]
            x = tl.load(x_ptrs, mask=x_mask, other=0)  # [BLOCK_M, BLOCK_K], int8

            # Load w_q4_packed: [BLOCK_N, BLOCK_K // 2], uint8
            # w_q4_packed: [E, N, K//2]
            # For K indices offs_k, the packed index is offs_k // 2
            # But we need to handle even/odd: low 4 bits = even k, high 4 bits = odd k
            # offs_k goes from k_start to k_start + BLOCK_K - 1
            # packed indices: k_start//2 to (k_start + BLOCK_K - 1)//2
            # Since BLOCK_K is a multiple of 2, we can load BLOCK_K//2 packed values
            offs_k_packed = (k_start // 2) + tl.arange(0, BLOCK_K // 2)
            mask_k_packed = offs_k_packed < (K // 2)

            w_packed_ptrs = w_q4_packed_ptr + pid_e * stride_w_e + offs_n[:, None] * stride_w_n + offs_k_packed[None, :] * stride_w_k2
            w_packed_mask = mask_n[:, None] & mask_k_packed[None, :]
            w_packed = tl.load(w_packed_ptrs, mask=w_packed_mask, other=0)  # [BLOCK_N, BLOCK_K//2], uint8

            # Unpack: low 4 bits and high 4 bits
            w_lo = (w_packed & 0x0F).to(tl.int8)  # even k positions [BLOCK_N, BLOCK_K//2]
            w_hi = ((w_packed >> 4) & 0x0F).to(tl.int8)  # odd k positions [BLOCK_N, BLOCK_K//2]

            # Interleave to get [BLOCK_N, BLOCK_K]
            # We need to reconstruct the full K dimension:
            # w_full[:, 0] = w_lo[:, 0], w_full[:, 1] = w_hi[:, 0], w_full[:, 2] = w_lo[:, 1], ...
            # Use reshape and interleave
            # Stack along last dim: [BLOCK_N, BLOCK_K//2, 2] then reshape to [BLOCK_N, BLOCK_K]
            w_lo_exp = tl.expand_dims(w_lo, 2)  # [BLOCK_N, BLOCK_K//2, 1]
            w_hi_exp = tl.expand_dims(w_hi, 2)  # [BLOCK_N, BLOCK_K//2, 1]
            w_interleaved = tl.join(w_lo_exp, w_hi_exp)  # This may not exist, use alternative

            # Alternative approach: create full weight directly using indexing
            # w_full[n, 2*i] = w_lo[n, i], w_full[n, 2*i+1] = w_hi[n, i]
            w_lo_i32 = w_lo.to(tl.int32)
            w_hi_i32 = w_hi.to(tl.int32)

            # Subtract zero point: w_z_i32 is [BLOCK_N]
            w_lo_shifted = (w_lo_i32 - w_z_i32[:, None]).to(tl.int8)  # [BLOCK_N, BLOCK_K//2]
            w_hi_shifted = (w_hi_i32 - w_z_i32[:, None]).to(tl.int8)  # [BLOCK_N, BLOCK_K//2]

            # We need to compute x @ w^T where x is [BLOCK_M, BLOCK_K] and w is [BLOCK_N, BLOCK_K]
            # Split x into even and odd columns
            # x_even = x[:, 0::2], x_odd = x[:, 1::2]
            # But Triton doesn't support slicing like that easily.
            # Instead, reshape x from [BLOCK_M, BLOCK_K] to [BLOCK_M, BLOCK_K//2, 2]
            # and split.

            # Alternative: load x in two halves corresponding to even and odd k
            # x[:, 0], x[:, 2], x[:, 4], ... correspond to w_lo
            # x[:, 1], x[:, 3], x[:, 5], ... correspond to w_hi

            # Load x as two separate loads for even and odd indices
            offs_k_even = k_start + tl.arange(0, BLOCK_K // 2) * 2
            offs_k_odd = k_start + tl.arange(0, BLOCK_K // 2) * 2 + 1
            mask_k_even = offs_k_even < K
            mask_k_odd = offs_k_odd < K

            x_even_ptrs = x_q_ptr + global_m[:, None] * stride_x_m + offs_k_even[None, :] * stride_x_k
            x_odd_ptrs = x_q_ptr + global_m[:, None] * stride_x_m + offs_k_odd[None, :] * stride_x_k
            x_even = tl.load(x_even_ptrs, mask=mask_m[:, None] & mask_k_even[None, :], other=0)  # [BLOCK_M, BLOCK_K//2]
            x_odd = tl.load(x_odd_ptrs, mask=mask_m[:, None] & mask_k_odd[None, :], other=0)  # [BLOCK_M, BLOCK_K//2]

            # Now compute: group_acc += x_even @ w_lo_shifted^T + x_odd @ w_hi_shifted^T
            # x_even: [BLOCK_M, BLOCK_K//2] int8, w_lo_shifted: [BLOCK_N, BLOCK_K//2] int8
            # tl.dot for int8 gives int32
            group_acc += tl.dot(x_even, tl.trans(w_lo_shifted))
            group_acc += tl.dot(x_odd, tl.trans(w_hi_shifted))

        # group_acc: [BLOCK_M, BLOCK_N] int32 - partial sum for this group
        # Multiply by w_scale: [BLOCK_N] fp32
        group_acc_f32 = group_acc.to(tl.float32)
        acc += group_acc_f32 * w_s[None, :]

    # Apply activation scale: x_scale [M_total], per token
    xs_ptrs = x_scale_ptr + global_m
    x_s = tl.load(xs_ptrs, mask=mask_m, other=0.0)  # [BLOCK_M], fp32

    acc = acc * x_s[:, None]

    # Store output as bf16
    acc_bf16 = acc.to(tl.bfloat16)
    out_ptrs = out_ptr + global_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc_bf16, mask=out_mask)


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
    """
    W4A8 grouped quantized GEMM for MoE inference.

    Args:
        x_q: [M_total, K], int8
        x_scale: [M_total], fp32
        w_q4_packed: [E, N, K//2], uint8
        w_scale: [E, N, K//group_size], fp32
        w_zero: [E, N, K//group_size], int8
        expert_offsets: [E+1], int32
        out: [M_total, N], bf16
        group_size: 64 or 128

    Returns:
        out: [M_total, N], bf16
    """
    assert group_size in (64, 128), "group_size must be 64 or 128"

    E, N, K_half = w_q4_packed.shape
    K = K_half * 2
    M_total = x_q.shape[0]

    # Compute maximum possible M blocks and N blocks for grid
    # We use a 2D grid: (max_mn_blocks, E)
    # Each program in dim 0 maps to a (m_block, n_block) pair
    # We need to allocate enough programs; some may exit early

    # Upper bound on tokens per expert is M_total
    # We'll compute a conservative upper bound
    max_m_blocks = triton.cdiv(M_total, 64)  # Use smallest BLOCK_M from configs
    max_n_blocks = triton.cdiv(N, 64)  # Use smallest BLOCK_N from configs
    max_mn_blocks = max_m_blocks * max_n_blocks

    # Actually, since autotune picks BLOCK_M and BLOCK_N, we need a safe upper bound
    # Use the minimum block sizes from our configs (64 for both)
    min_block_m = 64
    min_block_n = 64
    max_m_blocks = triton.cdiv(M_total, min_block_m)
    max_n_blocks = triton.cdiv(N, min_block_n)
    max_mn_blocks = max_m_blocks * max_n_blocks

    grid = (max_mn_blocks, E)

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
        group_size=group_size,
        E=E,
        stride_x_m=x_q.stride(0),
        stride_x_k=x_q.stride(1),
        stride_w_e=w_q4_packed.stride(0),
        stride_w_n=w_q4_packed.stride(1),
        stride_w_k2=w_q4_packed.stride(2),
        stride_ws_e=w_scale.stride(0),
        stride_ws_n=w_scale.stride(1),
        stride_ws_g=w_scale.stride(2),
        stride_wz_e=w_zero.stride(0),
        stride_wz_n=w_zero.stride(1),
        stride_wz_g=w_zero.stride(2),
        stride_out_m=out.stride(0),
        stride_out_n=out.stride(1),
    )

    return out