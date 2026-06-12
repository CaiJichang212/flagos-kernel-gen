import triton
import triton.language as tl
import triton.experimental.tle.language as tle
import torch


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
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
    BLOCK_K: tl.constexpr,
):
    # Program IDs: pid_expert indexes expert, pid_m indexes M tiles, pid_n indexes N tiles
    pid_expert = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Load expert token boundaries
    expert_start = tl.load(expert_offsets_ptr + pid_expert)
    expert_end = tl.load(expert_offsets_ptr + pid_expert + 1)
    expert_M = expert_end - expert_start

    # Early exit if this M tile is out of range for this expert
    if pid_m * BLOCK_M >= expert_M:
        return

    # Row offsets for this tile within the expert's token range
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Global row indices
    global_m = expert_start + offs_m
    mask_m = offs_m < expert_M

    # Column offsets for output
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    # Accumulator in float32 for numerical stability
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Number of groups along K
    num_groups_k = K // group_size
    # Number of K-tiles per group
    tiles_per_group: tl.constexpr = group_size // BLOCK_K

    # Iterate over groups along the K dimension
    for g in range(num_groups_k):
        # Load weight scale and zero point for this group: shape [BLOCK_N]
        # w_scale shape: [E, N, K//group_size]
        ws_ptrs = w_scale_ptr + pid_expert * stride_ws_e + offs_n * stride_ws_n + g * stride_ws_g
        w_scale_val = tl.load(ws_ptrs, mask=mask_n, other=0.0)

        wz_ptrs = w_zero_ptr + pid_expert * stride_wz_e + offs_n * stride_wz_n + g * stride_wz_g
        w_zero_val = tl.load(wz_ptrs, mask=mask_n, other=0)

        # Partial accumulator for this group (int32 for int8 x int8 dot products)
        group_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        # Iterate over K tiles within this group
        for t in range(tiles_per_group):
            k_start = g * group_size + t * BLOCK_K
            offs_k = k_start + tl.arange(0, BLOCK_K)

            # Load activation tile x_q: [BLOCK_M, BLOCK_K], int8
            x_ptrs = x_q_ptr + global_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k
            x_mask = mask_m[:, None] & (offs_k[None, :] < K)
            x_tile = tle.load(x_ptrs, mask=x_mask, other=0, is_async=True)

            # Load packed INT4 weights: [BLOCK_N, BLOCK_K // 2], uint8
            # w_q4_packed shape: [E, N, K//2]
            # For k index, packed index = k // 2
            offs_k_half = (k_start // 2) + tl.arange(0, BLOCK_K // 2)
            w_packed_ptrs = (w_q4_packed_ptr + pid_expert * stride_w_e +
                            offs_n[:, None] * stride_w_n +
                            offs_k_half[None, :] * stride_w_k)
            w_mask = mask_n[:, None] & (offs_k_half[None, :] < (K // 2))
            w_packed = tle.load(w_packed_ptrs, mask=w_mask, other=0, is_async=False)

            # Unpack INT4 from uint8: low nibble (even k) and high nibble (odd k)
            # w_packed shape: [BLOCK_N, BLOCK_K//2]
            w_low = (w_packed & 0xF).to(tl.int8)   # even k indices
            w_high = ((w_packed >> 4) & 0xF).to(tl.int8)  # odd k indices

            # Interleave to get [BLOCK_N, BLOCK_K]
            # We need to reconstruct the full K dimension
            # low corresponds to k=0,2,4,... and high to k=1,3,5,...
            # Reshape and interleave: stack along last dim
            # w_low: [BLOCK_N, BLOCK_K//2], w_high: [BLOCK_N, BLOCK_K//2]
            # Interleave by reshaping
            w_even = w_low   # [BLOCK_N, BLOCK_K//2]
            w_odd = w_high   # [BLOCK_N, BLOCK_K//2]

            # Broadcast zero point [BLOCK_N] -> [BLOCK_N, BLOCK_K//2]
            wz_bcast = w_zero_val[:, None].to(tl.int8)

            # Subtract zero point
            w_even = w_even - wz_bcast
            w_odd = w_odd - wz_bcast

            # Now interleave w_even and w_odd to create [BLOCK_N, BLOCK_K]
            # We need the final weight as [BLOCK_K, BLOCK_N] for dot product with x [BLOCK_M, BLOCK_K]
            # Actually, for tl.dot(a, b) where a is [M, K] and b is [K, N]:
            # x_tile is [BLOCK_M, BLOCK_K] int8
            # We need w_int8 as [BLOCK_K, BLOCK_N] int8

            # Construct full weight in transposed form [BLOCK_K, BLOCK_N]
            # Even K indices (0, 2, 4, ...): row 2*i in K dim -> w_even[:, i] for each N
            # Odd K indices (1, 3, 5, ...): row 2*i+1 in K dim -> w_odd[:, i] for each N

            # We'll compute partial dot products separately for even and odd
            # x_even = x_tile[:, 0::2] shape [BLOCK_M, BLOCK_K//2]
            # x_odd = x_tile[:, 1::2] shape [BLOCK_M, BLOCK_K//2]

            # Extract even and odd columns from x_tile
            # x_tile is [BLOCK_M, BLOCK_K]
            # Reshape to [BLOCK_M, BLOCK_K//2, 2] then slice
            # In Triton, we can use stride tricks or load separately

            # Alternative: load x in two halves corresponding to even/odd k
            # Actually, let's reconsider the packing. Each byte at position p stores:
            #   low nibble -> weight at k = 2*p (even)
            #   high nibble -> weight at k = 2*p + 1 (odd)
            # So packed index p corresponds to k = 2p, 2p+1

            # For dot product x @ W^T where W is [N, K]:
            # result[m, n] = sum_k x[m, k] * W[n, k]
            # = sum_p (x[m, 2p] * W[n, 2p] + x[m, 2p+1] * W[n, 2p+1])
            # = sum_p (x[m, 2p] * w_even[n, p] + x[m, 2p+1] * w_odd[n, p])

            # Load x for even indices: x[:, 0], x[:, 2], x[:, 4], ...
            # These are at offsets k_start, k_start+2, k_start+4, ...
            offs_k_even = k_start + tl.arange(0, BLOCK_K // 2) * 2
            offs_k_odd = k_start + tl.arange(0, BLOCK_K // 2) * 2 + 1

            x_even_ptrs = x_q_ptr + global_m[:, None] * stride_x_m + offs_k_even[None, :] * stride_x_k
            x_even_mask = mask_m[:, None] & (offs_k_even[None, :] < K)
            x_even = tl.load(x_even_ptrs, mask=x_even_mask, other=0)

            x_odd_ptrs = x_q_ptr + global_m[:, None] * stride_x_m + offs_k_odd[None, :] * stride_x_k
            x_odd_mask = mask_m[:, None] & (offs_k_odd[None, :] < K)
            x_odd = tl.load(x_odd_ptrs, mask=x_odd_mask, other=0)

            # x_even: [BLOCK_M, BLOCK_K//2] int8
            # w_even: [BLOCK_N, BLOCK_K//2] int8
            # dot: x_even @ w_even^T -> [BLOCK_M, BLOCK_N]
            # For tl.dot, need [BLOCK_M, BLOCK_K//2] @ [BLOCK_K//2, BLOCK_N]
            # So transpose w_even: w_even_t = [BLOCK_K//2, BLOCK_N]
            w_even_t = tl.trans(w_even)  # [BLOCK_K//2, BLOCK_N]
            w_odd_t = tl.trans(w_odd)    # [BLOCK_K//2, BLOCK_N]

            # Compute partial dot products
            # Cast to appropriate type for dot
            partial_even = tl.dot(x_even.to(tl.int8), w_even_t.to(tl.int8))
            partial_odd = tl.dot(x_odd.to(tl.int8), w_odd_t.to(tl.int8))

            group_acc += (partial_even + partial_odd).to(tl.float32)

        # Apply per-group weight scale: group_acc [BLOCK_M, BLOCK_N] * w_scale_val [BLOCK_N]
        acc += group_acc * w_scale_val[None, :]

    # Apply per-token activation scale
    # x_scale shape: [M_total], fp32
    token_scale = tl.load(x_scale_ptr + global_m, mask=mask_m, other=0.0)
    acc = acc * token_scale[:, None]

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
):
    """
    W4A8 Group GEMM for Mixture-of-Experts (MoE) inference.

    Args:
        x_q: INT8 quantized activations [M_total, K]
        x_scale: Per-token activation scale [M_total], fp32
        w_q4_packed: INT4 packed weights [E, N, K//2], uint8
        w_scale: Per-group weight scale [E, N, K//group_size], fp32
        w_zero: Per-group weight zero-point [E, N, K//group_size], int8
        expert_offsets: Expert boundary offsets [E+1], int32
        out: Output tensor [M_total, N], bf16
        group_size: Quantization group size (64 or 128)
    """
    M_total, K = x_q.shape
    E, N, _ = w_q4_packed.shape

    # Determine max M per expert for grid sizing
    # Use a conservative upper bound
    max_M_per_expert = M_total  # worst case: all tokens go to one expert

    # Grid: (num_experts, max_M_tiles, N_tiles)
    def grid(META):
        return (
            E,
            triton.cdiv(max_M_per_expert, META['BLOCK_M']),
            triton.cdiv(N, META['BLOCK_N']),
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
        group_size=group_size,
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
    )

    return out