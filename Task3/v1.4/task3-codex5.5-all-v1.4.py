import os
from typing import Tuple

import torch
import triton
import triton.language as tl


GROUP_SIZE = 128

_VENDOR_ALIASES = {
    "ascend": "ascend",
    "huawei": "ascend",
    "npu": "ascend",
    "mthreads": "mthreads",
    "moore": "mthreads",
    "musa": "mthreads",
    "tianshu": "tianshu",
    "tsingmicro": "tianshu",
    "txda": "tianshu",
    "aipu": "aipu",
    "pingtouge": "aipu",
    "hygon": "hygon",
    "haiguang": "hygon",
    "metax": "metax",
    "muxi": "metax",
    "cuda": "nvidia",
    "nvidia": "nvidia",
}


def _normalize_vendor(vendor):
    if vendor is None:
        return None
    value = str(vendor).strip().lower()
    for key, normalized in _VENDOR_ALIASES.items():
        if key in value:
            return normalized
    return None


def _detect_vendor(x):
    vendor = _normalize_vendor(os.environ.get("GEMS_VENDOR"))
    if vendor is not None:
        return vendor

    try:
        from flag_gems.runtime.backend.device import DeviceDetector

        detector = DeviceDetector()
        vendor_name = getattr(detector, "vendor_name", None)
        if callable(vendor_name):
            vendor_name = vendor_name()
        vendor = _normalize_vendor(vendor_name)
        if vendor is not None:
            return vendor
    except Exception:
        pass

    try:
        from flag_gems.runtime import DeviceDetector

        detector = DeviceDetector()
        vendor_name = getattr(detector, "vendor_name", None)
        if callable(vendor_name):
            vendor_name = vendor_name()
        vendor = _normalize_vendor(vendor_name)
        if vendor is not None:
            return vendor
    except Exception:
        pass

    vendor = _normalize_vendor(getattr(x.device, "type", None))
    if vendor is not None:
        return vendor
    return "nvidia"


@triton.jit
def _compact_grad_quant_kernel(
    x_ptr,
    grad_y_ptr,
    grad_input_q_ptr,
    grad_input_s_ptr,
    M,
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
def _compact_y_t_quant_kernel(
    x_ptr,
    y_q_t_ptr,
    y_s_t_ptr,
    H: tl.constexpr,
    M,
    M_GROUPS,
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

    y = (gate * tl.sigmoid(gate) * up).to(tl.bfloat16).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(y), axis=0) / 127.0, 1.0e-10)
    q = y / scale[None, :]
    q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)

    tl.store(y_q_t_ptr + offs_h[None, :] * M + offs_m[:, None], q)
    tl.store(y_s_t_ptr + offs_h * M_GROUPS + pid_mg, scale)


@triton.jit
def _stride_grad_quant_kernel(
    x_ptr,
    grad_y_ptr,
    grad_input_q_ptr,
    grad_input_s_ptr,
    H: tl.constexpr,
    stride_x_m: tl.constexpr,
    stride_x_h: tl.constexpr,
    stride_gy_m: tl.constexpr,
    stride_gy_h: tl.constexpr,
    stride_giq_m: tl.constexpr,
    stride_giq_h: tl.constexpr,
    stride_gis_m: tl.constexpr,
    stride_gis_g: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs_h[None, :] < H

    gate_ptrs = x_ptr + offs_m[:, None] * stride_x_m + offs_h[None, :] * stride_x_h
    up_ptrs = x_ptr + offs_m[:, None] * stride_x_m + (offs_h[None, :] + H) * stride_x_h
    gy_ptrs = grad_y_ptr + offs_m[:, None] * stride_gy_m + offs_h[None, :] * stride_gy_h

    gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptrs, mask=mask, other=0.0).to(tl.float32)
    grad_y = tl.load(gy_ptrs, mask=mask, other=0.0).to(tl.float32)

    sigmoid = tl.sigmoid(gate)
    silu = gate * sigmoid
    d_up = grad_y * silu
    d_gate = grad_y * up * sigmoid * (1.0 + gate * (1.0 - sigmoid))

    d_gate_bf = d_gate.to(tl.bfloat16).to(tl.float32)
    d_up_bf = d_up.to(tl.bfloat16).to(tl.float32)

    gate_scale = tl.maximum(tl.max(tl.abs(d_gate_bf), axis=1) / 127.0, 1.0e-10)
    up_scale = tl.maximum(tl.max(tl.abs(d_up_bf), axis=1) / 127.0, 1.0e-10)
    gate_q = d_gate_bf / gate_scale[:, None]
    up_q = d_up_bf / up_scale[:, None]
    gate_q = tl.minimum(tl.maximum(gate_q, -127.0), 127.0).to(tl.int8)
    up_q = tl.minimum(tl.maximum(up_q, -127.0), 127.0).to(tl.int8)

    gate_out_ptrs = grad_input_q_ptr + offs_m[:, None] * stride_giq_m + offs_h[None, :] * stride_giq_h
    up_out_ptrs = grad_input_q_ptr + offs_m[:, None] * stride_giq_m + (offs_h[None, :] + H) * stride_giq_h
    tl.store(gate_out_ptrs, gate_q, mask=mask)
    tl.store(up_out_ptrs, up_q, mask=mask)

    num_h_groups = H // 128
    scale_gate_ptrs = grad_input_s_ptr + offs_m * stride_gis_m + pid_h * stride_gis_g
    scale_up_ptrs = grad_input_s_ptr + offs_m * stride_gis_m + (pid_h + num_h_groups) * stride_gis_g
    tl.store(scale_gate_ptrs, gate_scale)
    tl.store(scale_up_ptrs, up_scale)


@triton.jit
def _stride_y_t_quant_kernel(
    x_ptr,
    y_q_t_ptr,
    y_s_t_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    stride_x_m: tl.constexpr,
    stride_x_h: tl.constexpr,
    stride_yqt_h: tl.constexpr,
    stride_yqt_m: tl.constexpr,
    stride_yst_h: tl.constexpr,
    stride_yst_g: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = (offs_m[:, None] < M) & (offs_h[None, :] < H)

    gate_ptrs = x_ptr + offs_m[:, None] * stride_x_m + offs_h[None, :] * stride_x_h
    up_ptrs = x_ptr + offs_m[:, None] * stride_x_m + (offs_h[None, :] + H) * stride_x_h

    gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptrs, mask=mask, other=0.0).to(tl.float32)

    y = (gate * tl.sigmoid(gate) * up).to(tl.bfloat16).to(tl.float32)
    y_scale = tl.maximum(tl.max(tl.abs(y), axis=0) / 127.0, 1.0e-10)
    y_q = y / y_scale[None, :]
    y_q = tl.minimum(tl.maximum(y_q, -127.0), 127.0).to(tl.int8)

    yqt_ptrs = y_q_t_ptr + offs_h[None, :] * stride_yqt_h + offs_m[:, None] * stride_yqt_m
    tl.store(yqt_ptrs, y_q, mask=mask)

    scale_ptrs = y_s_t_ptr + offs_h * stride_yst_h + pid_m * stride_yst_g
    tl.store(scale_ptrs, y_scale, mask=offs_h < H)


def _run_compact(
    x,
    grad_y,
    grad_input_q,
    grad_input_s,
    y_q_t,
    y_s_t,
    M,
    H,
    h_groups,
    m_groups,
    grad_block_m,
    y_block_h,
    num_stages,
):
    _compact_grad_quant_kernel[(M // grad_block_m, h_groups)](
        x,
        grad_y,
        grad_input_q,
        grad_input_s,
        M,
        H,
        h_groups,
        BLOCK_M=grad_block_m,
        BLOCK_N=GROUP_SIZE,
        num_warps=4,
        num_stages=num_stages,
    )
    _compact_y_t_quant_kernel[(m_groups, H // y_block_h)](
        x,
        y_q_t,
        y_s_t,
        H,
        M,
        m_groups,
        BLOCK_M=GROUP_SIZE,
        BLOCK_H=y_block_h,
        num_warps=4,
        num_stages=num_stages,
    )


def _run_stride_fast(x, grad_y, grad_input_q, grad_input_s, y_q_t, y_s_t, M, H):
    grad_block_m = 16
    grad_block_h = GROUP_SIZE
    y_block_m = GROUP_SIZE
    y_block_h = 32

    grad_grid = (triton.cdiv(M, grad_block_m), triton.cdiv(H, grad_block_h))
    _stride_grad_quant_kernel[grad_grid](
        x,
        grad_y,
        grad_input_q,
        grad_input_s,
        H=H,
        stride_x_m=x.stride(0),
        stride_x_h=x.stride(1),
        stride_gy_m=grad_y.stride(0),
        stride_gy_h=grad_y.stride(1),
        stride_giq_m=grad_input_q.stride(0),
        stride_giq_h=grad_input_q.stride(1),
        stride_gis_m=grad_input_s.stride(0),
        stride_gis_g=grad_input_s.stride(1),
        BLOCK_M=grad_block_m,
        BLOCK_H=grad_block_h,
        num_warps=4,
    )

    y_grid = (triton.cdiv(M, y_block_m), triton.cdiv(H, y_block_h))
    _stride_y_t_quant_kernel[y_grid](
        x,
        y_q_t,
        y_s_t,
        M=M,
        H=H,
        stride_x_m=x.stride(0),
        stride_x_h=x.stride(1),
        stride_yqt_h=y_q_t.stride(0),
        stride_yqt_m=y_q_t.stride(1),
        stride_yst_h=y_s_t.stride(0),
        stride_yst_g=y_s_t.stride(1),
        BLOCK_M=y_block_m,
        BLOCK_H=y_block_h,
        num_warps=4,
    )


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
    M, two_h = x.shape
    H = two_h // 2
    assert two_h == H * 2
    assert H % GROUP_SIZE == 0
    assert M % GROUP_SIZE == 0

    h_groups = H // GROUP_SIZE
    m_groups = M // GROUP_SIZE
    vendor = _detect_vendor(x)

    if vendor == "ascend":
        _run_compact(x, grad_y, grad_input_q, grad_input_s, y_q_t, y_s_t, M, H, h_groups, m_groups, 8, 32, 1)
    elif vendor == "mthreads":
        _run_compact(x, grad_y, grad_input_q, grad_input_s, y_q_t, y_s_t, M, H, h_groups, m_groups, 8, 16, 2)
    else:
        _run_stride_fast(x, grad_y, grad_input_q, grad_input_s, y_q_t, y_s_t, M, H)

    return grad_input_q, grad_input_s, y_q_t, y_s_t
