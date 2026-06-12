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
    K_half: tl.constexpr,
    group_size: tl.constexpr,
    num_groups: tl.constexpr,
    stride_x_m,
    stride_w_e,
    stride_w_n,
    stride_ws_e,
    stride_ws_n,
    stride_wz_e,
    stride_wz_n,
    stride_out_m,
    num_experts: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, BLOCK_N)
    cumulative_blocks = 0
    expert_id = 0
    expert_start = 0
    expert_end = 0
    local_pid = 0

    for e in range(num_experts):
        e_start = tl.load(expert_offsets_ptr + e)
        e_end = tl.load(expert_offsets_ptr + e + 1)
        e_M = e_end - e_start
        e_m_blocks = tl.cdiv(e_M, BLOCK_M)
        e_total_blocks = e_m_blocks * num_n_blocks
        if pid >= cumulative_blocks and pid < cumulative_blocks + e_total_blocks:
            expert_id = e
            expert_start = e_start
            expert_end = e_end
            local_pid = pid - cumulative_blocks
        cumulative_blocks += e_total_blocks

    M_expert = expert_end - expert_start
    if M_expert <= 0:
        return

    m_block = local_pid // num_n_blocks
    n_block = local_pid % num_n_blocks
    m_start = m_block * BLOCK_M
    n_start = n_block * BLOCK_N
    rm = m_start + tl.arange(0, BLOCK_M)
    rn = n_start + tl.arange(0, BLOCK_N)
    m_mask = rm < M_expert
    n_mask = rn < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        num_k_groups = BLOCK_K // group_size
        for kg in range(num_k_groups):
            g_k_start = kg * group_size
            group_idx = (k_start + g_k_start) // group_size
            ws_ptrs = w_scale_ptr + expert_id * stride_ws_e + rn * stride_ws_n + group_idx
            wz_ptrs = w_zero_ptr + expert_id * stride_wz_e + rn * stride_wz_n + group_idx
            ws_val = tl.load(ws_ptrs, mask=n_mask, other=0.0).to(tl.float32)
            wz_val = tl.load(wz_ptrs, mask=n_mask, other=0.0).to(tl.float32)

            rk_even_abs = k_start + g_k_start + tl.arange(0, group_size // 2) * 2
            rk_odd_abs = rk_even_abs + 1
            k_even_mask = rk_even_abs < K
            k_odd_mask = rk_odd_abs < K
            rk_half_g = (k_start // 2) + (g_k_start // 2) + tl.arange(0, group_size // 2)
            k_half_g_mask = rk_half_g < K_half

            w_packed_g_ptrs = (
                w_q4_packed_ptr
                + expert_id * stride_w_e
                + rn[:, None] * stride_w_n
                + rk_half_g[None, :]
            )
            w_packed_g = tl.load(
                w_packed_g_ptrs,
                mask=n_mask[:, None] & k_half_g_mask[None, :],
                other=0,
            )
            w_low_g = (w_packed_g & 0x0F).to(tl.float32)
            w_high_g = ((w_packed_g >> 4) & 0x0F).to(tl.float32)
            w_low_deq = (w_low_g - wz_val[:, None]) * ws_val[:, None]
            w_high_deq = (w_high_g - wz_val[:, None]) * ws_val[:, None]

            x_even_ptrs = x_q_ptr + (expert_start + rm[:, None]) * stride_x_m + rk_even_abs[None, :]
            x_odd_ptrs = x_q_ptr + (expert_start + rm[:, None]) * stride_x_m + rk_odd_abs[None, :]
            x_even = tl.load(
                x_even_ptrs,
                mask=m_mask[:, None] & k_even_mask[None, :],
                other=0,
            ).to(tl.float32)
            x_odd = tl.load(
                x_odd_ptrs,
                mask=m_mask[:, None] & k_odd_mask[None, :],
                other=0,
            ).to(tl.float32)
            acc += tl.dot(x_even, tl.trans(w_low_deq))
            acc += tl.dot(x_odd, tl.trans(w_high_deq))

    xs_ptrs = x_scale_ptr + expert_start + rm
    x_scales = tl.load(xs_ptrs, mask=m_mask, other=0.0).to(tl.float32)
    acc = acc * x_scales[:, None]
    out_ptrs = out_ptr + (expert_start + rm[:, None]) * stride_out_m + rn[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


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
    E = w_q4_packed.shape[0]
    N = w_q4_packed.shape[1]
    K_half = w_q4_packed.shape[2]
    K = K_half * 2
    num_groups = K // group_size
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = group_size
    offsets_cpu = expert_offsets.cpu()
    total_blocks = 0
    for e in range(E):
        e_start = offsets_cpu[e].item()
        e_end = offsets_cpu[e + 1].item()
        e_M = max(0, e_end - e_start)
        e_m_blocks = (e_M + BLOCK_M - 1) // BLOCK_M
        num_n_blocks = (N + BLOCK_N - 1) // BLOCK_N
        total_blocks += e_m_blocks * num_n_blocks
    if total_blocks == 0:
        return out
    _w4a8_group_gemm_moe_kernel[(total_blocks,)](
        x_q,
        x_scale,
        w_q4_packed,
        w_scale,
        w_zero,
        expert_offsets,
        out,
        N,
        K,
        K_half,
        group_size,
        num_groups,
        x_q.stride(0),
        w_q4_packed.stride(0),
        w_q4_packed.stride(1),
        w_scale.stride(0),
        w_scale.stride(1),
        w_zero.stride(0),
        w_zero.stride(1),
        out.stride(0),
        E,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return out

