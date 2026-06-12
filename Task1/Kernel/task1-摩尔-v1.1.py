import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 4}, num_warps=8, num_stages=3),
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
    stride_xs,
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
    E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_e = tl.program_id(1)

    expert_start = tl.load(expert_offsets_ptr + pid_e, eviction_policy='evict_first').to(tl.int32)
    expert_end = tl.load(expert_offsets_ptr + pid_e + 1, eviction_policy='evict_first').to(tl.int32)
    M_e = expert_end - expert_start
    if M_e <= 0:
        return

    num_m_blocks = tl.cdiv(M_e, BLOCK_M)
    num_n_blocks = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_n_blocks
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_m_blocks - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    if pid_m >= num_m_blocks:
        return
    if pid_n >= num_n_blocks:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M_e
    mask_n = offs_n < N
    global_m = expert_start + offs_m

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    K_half = K // 2
    num_k_groups = tl.cdiv(K, group_size)
    group_pairs = tl.cdiv(group_size, 2)

    offs_g_pair = tl.arange(0, group_pairs)

    x_base_m = global_m.to(tl.int64) * stride_x_m
    out_base_m = global_m.to(tl.int64) * stride_out_m
    n_offsets = offs_n.to(tl.int64)

    for k_start in range(0, K, BLOCK_K):
        k_base = k_start // 2
        for g in range(0, BLOCK_K, group_size):
            g_start_k = k_start + g
            g_idx = g_start_k // group_size
            if g_idx >= num_k_groups:
                continue

            ws = tl.load(
                w_scale_ptr + pid_e * stride_ws_e + n_offsets * stride_ws_n + g_idx * stride_ws_g,
                mask=mask_n,
                other=0.0,
                eviction_policy='evict_first',
            ).to(tl.float32)
            wz = tl.load(
                w_zero_ptr + pid_e * stride_wz_e + n_offsets * stride_wz_n + g_idx * stride_wz_g,
                mask=mask_n,
                other=0,
                eviction_policy='evict_first',
            ).to(tl.float32)

            half_start = g // 2
            k_half_offs = k_base + half_start + offs_g_pair
            mask_w = mask_n[:, None] & (k_half_offs[None, :] < K_half)

            w_ptrs = (
                w_q4_packed_ptr
                + pid_e * stride_w_e
                + n_offsets[:, None] * stride_w_n
                + k_half_offs[None, :].to(tl.int64) * stride_w_k
            )
            w_packed = tl.load(w_ptrs, mask=mask_w, other=0, eviction_policy='evict_last')

            w_low = (w_packed & 0x0F).to(tl.float32)
            w_high = ((w_packed >> 4) & 0x0F).to(tl.float32)

            w_low = (w_low - wz[:, None]) * ws[:, None]
            w_high = (w_high - wz[:, None]) * ws[:, None]

            k_even = k_start + g + 2 * offs_g_pair
            k_odd = k_even + 1
            mask_ke = (k_even < K) & (g + 2 * offs_g_pair < BLOCK_K)
            mask_ko = (k_odd < K) & (g + 2 * offs_g_pair + 1 < BLOCK_K)

            x_even = tl.load(
                x_q_ptr + x_base_m[:, None] + (k_even[None, :].to(tl.int64)) * stride_x_k,
                mask=mask_m[:, None] & mask_ke[None, :],
                other=0,
                eviction_policy='evict_last',
            ).to(tl.float32)
            x_odd = tl.load(
                x_q_ptr + x_base_m[:, None] + (k_odd[None, :].to(tl.int64)) * stride_x_k,
                mask=mask_m[:, None] & mask_ko[None, :],
                other=0,
                eviction_policy='evict_last',
            ).to(tl.float32)

            acc += tl.dot(x_even, tl.trans(w_low), allow_tf32=False)
            acc += tl.dot(x_odd, tl.trans(w_high), allow_tf32=False)

    xs = tl.load(
        x_scale_ptr + global_m.to(tl.int64) * stride_xs,
        mask=mask_m,
        other=0.0,
        eviction_policy='evict_first',
    ).to(tl.float32)

    acc = acc * xs[:, None]

    out_ptrs = out_ptr + out_base_m[:, None] + n_offsets[None, :] * stride_out_n
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
    E, N, K_half = w_q4_packed.shape
    assert K_half == K // 2

    x_q = x_q.contiguous()
    x_scale = x_scale.contiguous()
    w_q4_packed = w_q4_packed.contiguous()
    w_scale = w_scale.contiguous()
    w_zero = w_zero.contiguous()
    expert_offsets = expert_offsets.contiguous()

    BLOCK_M_MAX = 128
    BLOCK_N_MIN = 64
    max_m_blocks = (M_total + BLOCK_M_MAX - 1) // BLOCK_M_MAX
    max_n_blocks = (N + BLOCK_N_MIN - 1) // BLOCK_N_MIN
    grid = (max_m_blocks * max_n_blocks, E)

    _w4a8_group_gemm_moe_kernel[grid](
        x_q,
        x_scale,
        w_q4_packed,
        w_scale,
        w_zero,
        expert_offsets,
        out,
        N,
        K,
        group_size,
        x_q.stride(0),
        x_q.stride(1),
        x_scale.stride(0),
        w_q4_packed.stride(0),
        w_q4_packed.stride(1),
        w_q4_packed.stride(2),
        w_scale.stride(0),
        w_scale.stride(1),
        w_scale.stride(2),
        w_zero.stride(0),
        w_zero.stride(1),
        w_zero.stride(2),
        out.stride(0),
        out.stride(1),
        E,
    )
    return out
