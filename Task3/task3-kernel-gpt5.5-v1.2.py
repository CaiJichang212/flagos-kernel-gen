import torch
import triton
import triton.language as tl

GROUP_SIZE = 128


@triton.jit
def _silu_grad_quant_kernel(
    x_ptr,
    grad_y_ptr,
    grad_input_q_ptr,
    grad_input_s_ptr,
    M,
    H: tl.constexpr,
    stride_x_m,
    stride_gy_m,
    stride_giq_m,
    stride_gis_m,
    stride_gis_g,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    base_m = pid_m * BLOCK_M
    offs_m = base_m + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    row_offsets = offs_m * stride_x_m
    gy_row_offsets = offs_m * stride_gy_m
    out_row_offsets = offs_m * stride_giq_m

    gate_base = x_ptr + offs_h[None, :]
    up_base = x_ptr + (offs_h[None, :] + H)
    gy_base = grad_y_ptr + offs_h[None, :]
    giq_gate_base = grad_input_q_ptr + offs_h[None, :]
    giq_up_base = grad_input_q_ptr + (offs_h[None, :] + H)

    gate = tl.load(gate_base + row_offsets[:, None]).to(tl.float32)
    up = tl.load(up_base + row_offsets[:, None]).to(tl.float32)
    grad_y = tl.load(gy_base + gy_row_offsets[:, None]).to(tl.float32)

    sigmoid = tl.sigmoid(gate)
    silu = gate * sigmoid

    d_up = grad_y * silu
    d_gate = grad_y * up * sigmoid * (1.0 + gate * (1.0 - sigmoid))

    d_gate_bf = d_gate.to(tl.bfloat16).to(tl.float32)
    d_up_bf = d_up.to(tl.bfloat16).to(tl.float32)

    gate_absmax = tl.max(tl.abs(d_gate_bf), axis=1)
    gate_scale = tl.maximum(gate_absmax * 0.007874015748031496, 1.0e-10)
    gate_q = d_gate_bf / gate_scale[:, None]
    gate_q = tl.minimum(tl.maximum(gate_q, -127.0), 127.0).to(tl.int8)

    up_absmax = tl.max(tl.abs(d_up_bf), axis=1)
    up_scale = tl.maximum(up_absmax * 0.007874015748031496, 1.0e-10)
    up_q = d_up_bf / up_scale[:, None]
    up_q = tl.minimum(tl.maximum(up_q, -127.0), 127.0).to(tl.int8)

    tl.store(giq_gate_base + out_row_offsets[:, None], gate_q)
    tl.store(giq_up_base + out_row_offsets[:, None], up_q)

    num_h_groups = H // 128
    scale_base = offs_m * stride_gis_m
    scale_gate_ptrs = grad_input_s_ptr + scale_base + pid_h * stride_gis_g
    scale_up_ptrs = grad_input_s_ptr + scale_base + (pid_h + num_h_groups) * stride_gis_g
    tl.store(scale_gate_ptrs, gate_scale)
    tl.store(scale_up_ptrs, up_scale)


@triton.jit
def _silu_y_t_quant_kernel(
    x_ptr,
    y_q_t_ptr,
    y_s_t_ptr,
    M,
    H: tl.constexpr,
    stride_x_m,
    stride_yqt_h,
    stride_yqt_m,
    stride_yst_h,
    stride_yst_g,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H

    row_offsets = offs_m * stride_x_m
    gate_base = x_ptr + offs_h[None, :]
    up_base = x_ptr + (offs_h[None, :] + H)

    gate = tl.load(gate_base + row_offsets[:, None], mask=mask_h[None, :], other=0.0).to(tl.float32)
    up = tl.load(up_base + row_offsets[:, None], mask=mask_h[None, :], other=0.0).to(tl.float32)

    y = (gate * tl.sigmoid(gate) * up).to(tl.bfloat16).to(tl.float32)

    y_absmax = tl.max(tl.abs(y), axis=0)
    y_scale = tl.maximum(y_absmax * 0.007874015748031496, 1.0e-10)
    y_q = y / y_scale[None, :]
    y_q = tl.minimum(tl.maximum(y_q, -127.0), 127.0).to(tl.int8)

    yqt_ptrs = y_q_t_ptr + offs_h[None, :] * stride_yqt_h + offs_m[:, None] * stride_yqt_m
    tl.store(yqt_ptrs, y_q, mask=mask_h[None, :])

    scale_ptrs = y_s_t_ptr + offs_h * stride_yst_h + pid_m * stride_yst_g
    tl.store(scale_ptrs, y_scale, mask=mask_h)


def silu_dot_fwd_bwd_quant_fuse(x: torch.Tensor, grad_y: torch.Tensor, grad_input_q: torch.Tensor, grad_input_s: torch.Tensor, y_q_t: torch.Tensor, y_s_t: torch.Tensor, group_size: int = 128):
    assert group_size == GROUP_SIZE
    M, two_h = x.shape
    H = two_h // 2
    assert two_h == H * 2
    assert H % GROUP_SIZE == 0
    assert M % GROUP_SIZE == 0

    grad_block_m = 8
    grad_block_h = GROUP_SIZE
    grad_grid = (triton.cdiv(M, grad_block_m), triton.cdiv(H, grad_block_h))
    _silu_grad_quant_kernel[grad_grid](
        x, grad_y, grad_input_q, grad_input_s,
        M, H,
        x.stride(0), grad_y.stride(0), grad_input_q.stride(0),
        grad_input_s.stride(0), grad_input_s.stride(1),
        BLOCK_M=grad_block_m, BLOCK_H=grad_block_h,
        num_warps=4, num_stages=2,
    )

    y_block_m = GROUP_SIZE
    y_block_h = 16
    y_grid = (triton.cdiv(M, y_block_m), triton.cdiv(H, y_block_h))
    _silu_y_t_quant_kernel[y_grid](
        x, y_q_t, y_s_t,
        M, H,
        x.stride(0), y_q_t.stride(0), y_q_t.stride(1),
        y_s_t.stride(0), y_s_t.stride(1),
        BLOCK_M=y_block_m, BLOCK_H=y_block_h,
        num_warps=4, num_stages=2,
    )

    return grad_input_q, grad_input_s, y_q_t, y_s_t
