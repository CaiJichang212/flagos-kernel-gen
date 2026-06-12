# -*- coding: utf-8 -*-
"""
Task 03: silu_dot_fwd_bwd_quant_fuse

Standalone Triton implementation for SwiGLU backward fusion with INT8 quantization.
The implementation intentionally has no PyTorch-compute fallback path: all core
math, reductions, quantization, and stores are performed by Triton kernels.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl


GROUP_SIZE = 128
_GRAD_BLOCK_M = 8
_Y_BLOCK_M = 128
_Y_BLOCK_H = 16


@triton.jit
def _silu_grad_quant_kernel(
    x_ptr,
    grad_y_ptr,
    grad_input_q_ptr,
    grad_input_s_ptr,
    H: tl.constexpr,
    H_GROUPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_g * BLOCK_N + tl.arange(0, BLOCK_N)

    x_row = offs_m[:, None] * (H * 2)
    gy_row = offs_m[:, None] * H
    cols = offs_n[None, :]

    gate = tl.load(x_ptr + x_row + cols).to(tl.float32)
    up = tl.load(x_ptr + x_row + H + cols).to(tl.float32)
    grad_y = tl.load(grad_y_ptr + gy_row + cols).to(tl.float32)

    sigmoid = tl.sigmoid(gate)
    silu = gate * sigmoid

    d_up = grad_y * silu
    d_gate = grad_y * up * sigmoid * (1.0 + gate * (1.0 - sigmoid))

    d_gate = d_gate.to(tl.bfloat16).to(tl.float32)
    d_up = d_up.to(tl.bfloat16).to(tl.float32)

    absmax_gate = tl.max(tl.abs(d_gate), axis=1)
    absmax_up = tl.max(tl.abs(d_up), axis=1)
    scale_gate = tl.maximum(absmax_gate / 127.0, 1.0e-10)
    scale_up = tl.maximum(absmax_up / 127.0, 1.0e-10)

    q_gate_f = d_gate / scale_gate[:, None]
    q_up_f = d_up / scale_up[:, None]
    q_gate_f = tl.minimum(tl.maximum(q_gate_f, -127.0), 127.0)
    q_up_f = tl.minimum(tl.maximum(q_up_f, -127.0), 127.0)

    q_gate = q_gate_f.to(tl.int8)
    q_up = q_up_f.to(tl.int8)

    tl.store(grad_input_q_ptr + x_row + cols, q_gate)
    tl.store(grad_input_q_ptr + x_row + H + cols, q_up)

    scale_row = offs_m * (H_GROUPS * 2)
    tl.store(grad_input_s_ptr + scale_row + pid_g, scale_gate)
    tl.store(grad_input_s_ptr + scale_row + H_GROUPS + pid_g, scale_up)


@triton.jit
def _silu_y_t_quant_kernel(
    x_ptr,
    y_q_t_ptr,
    y_s_t_ptr,
    H: tl.constexpr,
    M: tl.constexpr,
    M_GROUPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_mg = tl.program_id(0)
    pid_hb = tl.program_id(1)

    offs_m = pid_mg * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_hb * BLOCK_H + tl.arange(0, BLOCK_H)

    x_offsets = offs_m[:, None] * (H * 2) + offs_h[None, :]

    gate = tl.load(x_ptr + x_offsets).to(tl.float32)
    up = tl.load(x_ptr + x_offsets + H).to(tl.float32)

    sigmoid = tl.sigmoid(gate)
    y = gate * sigmoid * up
    y = y.to(tl.bfloat16).to(tl.float32)

    absmax = tl.max(tl.abs(y), axis=0)
    scale = tl.maximum(absmax / 127.0, 1.0e-10)

    q_f = y / scale[None, :]
    q_f = tl.minimum(tl.maximum(q_f, -127.0), 127.0)
    q = q_f.to(tl.int8)

    out_offsets = offs_h[None, :] * M + offs_m[:, None]
    tl.store(y_q_t_ptr + out_offsets, q)
    tl.store(y_s_t_ptr + offs_h * M_GROUPS + pid_mg, scale)


def silu_dot_fwd_bwd_quant_fuse(
    x: torch.Tensor,
    grad_y: torch.Tensor,
    grad_input_q: torch.Tensor,
    grad_input_s: torch.Tensor,
    y_q_t: torch.Tensor,
    y_s_t: torch.Tensor,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert group_size == GROUP_SIZE
    M = x.shape[0]
    two_h = x.shape[1]
    H = two_h // 2

    assert two_h == H * 2
    assert H % GROUP_SIZE == 0
    assert M % GROUP_SIZE == 0
    assert M % _GRAD_BLOCK_M == 0
    assert H % _Y_BLOCK_H == 0

    h_groups = H // GROUP_SIZE
    m_groups = M // GROUP_SIZE

    grad_grid = (M // _GRAD_BLOCK_M, h_groups)
    _silu_grad_quant_kernel[grad_grid](
        x,
        grad_y,
        grad_input_q,
        grad_input_s,
        H,
        h_groups,
        BLOCK_M=_GRAD_BLOCK_M,
        BLOCK_N=GROUP_SIZE,
        num_warps=4,
        num_stages=2,
    )

    y_grid = (m_groups, H // _Y_BLOCK_H)
    _silu_y_t_quant_kernel[y_grid](
        x,
        y_q_t,
        y_s_t,
        H,
        M,
        m_groups,
        BLOCK_M=_Y_BLOCK_M,
        BLOCK_H=_Y_BLOCK_H,
        num_warps=4,
        num_stages=2,
    )

    return grad_input_q, grad_input_s, y_q_t, y_s_t
