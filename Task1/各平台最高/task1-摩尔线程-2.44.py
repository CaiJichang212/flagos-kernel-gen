import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=2),
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
    stride_xs_m,
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
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_e = tl.program_id(2)

    expert_start = tl.load(expert_offsets_ptr + pid_e).to(tl.int64)
    expert_end = tl.load(expert_offsets_ptr + pid_e + 1).to(tl.int64)
    M_e = expert_end - expert_start
    row_start = pid_m * BLOCK_M
    if row_start >= M_e:
        return

    offs_m = row_start + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M_e
    mask_n = offs_n < N
    global_m = expert_start + offs_m
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    num_k_blocks = K // BLOCK_K

    for k_block_idx in range(num_k_blocks):
        k_start = k_block_idx * BLOCK_K
        offs_k = k_start + tl.arange(0, BLOCK_K)
        group_idx = k_start // group_size
        x_ptrs = x_q_ptr + global_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k
        x_block = tl.load(x_ptrs, mask=mask_m[:, None] & (offs_k[None, :] < K), other=0).to(tl.float32)
        offs_k_packed = k_start // 2 + tl.arange(0, BLOCK_K // 2)
        w_ptrs = w_q4_packed_ptr + pid_e * stride_w_e + offs_n[:, None] * stride_w_n + offs_k_packed[None, :] * stride_w_k
        w_packed = tl.load(w_ptrs, mask=mask_n[:, None] & (offs_k_packed[None, :] < (K // 2)), other=0).to(tl.int32)
        w_low = (w_packed & 0x0F).to(tl.float32)
        w_high = ((w_packed >> 4) & 0x0F).to(tl.float32)
        w_low_exp = tl.reshape(w_low, (BLOCK_N, BLOCK_K // 2, 1))
        w_high_exp = tl.reshape(w_high, (BLOCK_N, BLOCK_K // 2, 1))
        w_interleaved = tl.join(w_low_exp, w_high_exp)
        w_unpacked = tl.reshape(w_interleaved, (BLOCK_N, BLOCK_K))
        wz_ptrs = w_zero_ptr + pid_e * stride_wz_e + offs_n * stride_wz_n + group_idx * stride_wz_g
        w_zero_val = tl.load(wz_ptrs, mask=mask_n, other=0).to(tl.float32)
        w_dequant = w_unpacked - w_zero_val[:, None]
        ws_ptrs = w_scale_ptr + pid_e * stride_ws_e + offs_n * stride_ws_n + group_idx * stride_ws_g
        w_scale_val = tl.load(ws_ptrs, mask=mask_n, other=0.0)
        partial = tl.dot(x_block, tl.trans(w_dequant), allow_tf32=False)
        acc += partial * w_scale_val[None, :]

    xs_ptrs = x_scale_ptr + global_m * stride_xs_m
    x_scale_val = tl.load(xs_ptrs, mask=mask_m, other=0.0)
    acc = acc * x_scale_val[:, None]
    out_ptrs = out_ptr + global_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
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
    E = w_q4_packed.shape[0]
    N = w_q4_packed.shape[1]
    K = x_q.shape[1]
    M_total = x_q.shape[0]
    x_q = x_q.contiguous()
    x_scale = x_scale.contiguous()
    w_q4_packed = w_q4_packed.contiguous()
    w_scale = w_scale.contiguous()
    w_zero = w_zero.contiguous()
    expert_offsets = expert_offsets.contiguous()

    def grid(meta):
        return (
            triton.cdiv(M_total, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
            E,
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
        stride_xs_m=x_scale.stride(0),
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
        E=E,
    )
    return out
