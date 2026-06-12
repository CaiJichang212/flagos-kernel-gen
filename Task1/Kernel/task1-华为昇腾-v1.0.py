import torch
import torch_npu
import triton
import triton.language as tl
import triton.experimental.tle as tle


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
    num_groups: tl.constexpr,
    group_size: tl.constexpr,
    E: tl.constexpr,
    M_total,
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
    BLOCK_N: tl.constexpr,
    BLOCK_GS: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_n_blocks = (N + BLOCK_N - 1) // BLOCK_N

    expert_idx = pid // num_n_blocks
    n_block_idx = pid % num_n_blocks

    if expert_idx >= E:
        return

    m_start_val = tl.load(expert_offsets_ptr + expert_idx)
    m_end_val = tl.load(expert_offsets_ptr + expert_idx + 1)

    if m_start_val >= m_end_val:
        return

    n_start = n_block_idx * BLOCK_N

    gs_ub = tle.dsa.alloc([BLOCK_GS], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    x_gs_ub = tle.dsa.alloc([BLOCK_GS], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    w_gs_ub = tle.dsa.alloc([BLOCK_GS], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    wz_gs_ub = tle.dsa.alloc([BLOCK_GS], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    prod_ub = tle.dsa.alloc([BLOCK_GS], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    w_packed_ub = tle.dsa.alloc([BLOCK_GS], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    tmp_ub = tle.dsa.alloc([BLOCK_GS], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)

    m_idx = m_start_val
    while m_idx < m_end_val:
        x_scale_val = tl.load(x_scale_ptr + m_idx).to(tl.float32)

        n_idx = n_start
        n_end = tl.minimum(n_start + BLOCK_N, N)

        while n_idx < n_end:
            acc = tl.zeros([], dtype=tl.float32)

            for g_idx in range(num_groups):
                k_base = g_idx * group_size

                w_scale_val = tl.load(
                    w_scale_ptr + expert_idx * stride_ws_e + n_idx * stride_ws_n + g_idx * stride_ws_g
                ).to(tl.float32)
                w_zero_val = tl.load(
                    w_zero_ptr + expert_idx * stride_wz_e + n_idx * stride_wz_n + g_idx * stride_wz_g
                ).to(tl.float32)

                group_acc = tl.zeros([], dtype=tl.float32)

                for k_sub in range(group_size):
                    k_pos = k_base + k_sub
                    k_packed_idx = k_pos // 2
                    is_odd = k_pos % 2

                    x_val = tl.load(
                        x_q_ptr + m_idx * stride_x_m + k_pos * stride_x_k
                    ).to(tl.float32)

                    w_packed_val = tl.load(
                        w_q4_packed_ptr + expert_idx * stride_w_e + n_idx * stride_w_n + k_packed_idx * stride_w_k2
                    ).to(tl.int32)

                    if is_odd == 1:
                        w_int4 = (w_packed_val >> 4) & 0x0F
                    else:
                        w_int4 = w_packed_val & 0x0F

                    w_dequant = w_int4.to(tl.float32) - w_zero_val

                    group_acc += x_val * w_dequant

                acc += group_acc * w_scale_val

            result = acc * x_scale_val
            result_bf16 = result.to(tl.bfloat16)

            tl.store(
                out_ptr + m_idx * stride_out_m + n_idx * stride_out_n,
                result_bf16,
            )

            n_idx += 1

        m_idx += 1


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
    M_total = x_q.shape[0]

    BLOCK_N = 1
    BLOCK_GS = group_size

    num_n_blocks = (N + BLOCK_N - 1) // BLOCK_N
    grid = (E * num_n_blocks,)

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
        K_half,
        num_groups,
        group_size,
        E,
        M_total,
        x_q.stride(0),
        x_q.stride(1),
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
        BLOCK_N=BLOCK_N,
        BLOCK_GS=BLOCK_GS,
    )

    return out
