import os

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
    "tianshu": "tsingmicro",
    "tsingmicro": "tsingmicro",
    "txda": "tsingmicro",
    "hygon": "hygon",
    "haiguang": "hygon",
    "metax": "metax",
    "muxi": "metax",
    "aipu": "aipu",
    "pingtouge": "aipu",
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
    x_row = (offs_m * (H * 2))[:, None]
    gy_row = (offs_m * H)[:, None]
    cols = offs_n[None, :]

    gate = tl.load(x_ptr + x_row + cols).to(tl.float32)
    up = tl.load(x_ptr + x_row + H + cols).to(tl.float32)
    grad_y = tl.load(grad_y_ptr + gy_row + cols).to(tl.float32)

    sigmoid = tl.sigmoid(gate)
    silu_grad_factor = sigmoid * (1.0 + gate * (1.0 - sigmoid))

    d_up = (grad_y * gate * sigmoid).to(tl.bfloat16).to(tl.float32)
    d_gate = (grad_y * up * silu_grad_factor).to(tl.bfloat16).to(tl.float32)

    scale_gate = tl.maximum(tl.max(tl.abs(d_gate), axis=1) / 127.0, 1.0e-10)
    scale_up = tl.maximum(tl.max(tl.abs(d_up), axis=1) / 127.0, 1.0e-10)

    q_gate = tl.minimum(tl.maximum(d_gate * (1.0 / scale_gate)[:, None], -127.0), 127.0).to(tl.int8)
    q_up = tl.minimum(tl.maximum(d_up * (1.0 / scale_up)[:, None], -127.0), 127.0).to(tl.int8)

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

    y = (gate * tl.sigmoid(gate) * up).to(tl.bfloat16).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(y), axis=0) / 127.0, 1.0e-10)
    q = tl.minimum(tl.maximum(y * (1.0 / scale)[None, :], -127.0), 127.0).to(tl.int8)

    tl.store(y_q_t_ptr + offs_h[None, :] * M + offs_m[:, None], q)
    tl.store(y_s_t_ptr + offs_h * M_GROUPS + pid_mg, scale)


def _run_ascend(
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
):
    _silu_grad_quant_kernel[(M // 8, h_groups)](
        x,
        grad_y,
        grad_input_q,
        grad_input_s,
        H,
        h_groups,
        BLOCK_M=8,
        BLOCK_N=128,
        num_warps=4,
        num_stages=1,
    )
    _silu_y_t_quant_kernel[(m_groups, H // 16)](
        x,
        y_q_t,
        y_s_t,
        H,
        M,
        m_groups,
        BLOCK_M=128,
        BLOCK_H=16,
        num_warps=4,
        num_stages=1,
    )


def _run_mthreads(
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
):
    _silu_grad_quant_kernel[(M // 8, h_groups)](
        x,
        grad_y,
        grad_input_q,
        grad_input_s,
        H,
        h_groups,
        BLOCK_M=8,
        BLOCK_N=128,
        num_warps=4,
        num_stages=2,
    )
    _silu_y_t_quant_kernel[(m_groups, H // 16)](
        x,
        y_q_t,
        y_s_t,
        H,
        M,
        m_groups,
        BLOCK_M=128,
        BLOCK_H=16,
        num_warps=4,
        num_stages=2,
    )


def _run_fast(
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
):
    _silu_grad_quant_kernel[(M // 16, h_groups)](
        x,
        grad_y,
        grad_input_q,
        grad_input_s,
        H,
        h_groups,
        BLOCK_M=16,
        BLOCK_N=128,
        num_warps=4,
        num_stages=2,
    )
    _silu_y_t_quant_kernel[(m_groups, H // 32)](
        x,
        y_q_t,
        y_s_t,
        H,
        M,
        m_groups,
        BLOCK_M=128,
        BLOCK_H=32,
        num_warps=4,
        num_stages=2,
    )


def _run_tsingmicro(
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
):
    _run_fast(x, grad_y, grad_input_q, grad_input_s, y_q_t, y_s_t, M, H, h_groups, m_groups)


def _run_hygon(
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
):
    _run_fast(x, grad_y, grad_input_q, grad_input_s, y_q_t, y_s_t, M, H, h_groups, m_groups)


def _run_metax(
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
):
    _run_fast(x, grad_y, grad_input_q, grad_input_s, y_q_t, y_s_t, M, H, h_groups, m_groups)


def _run_aipu(
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
):
    _run_fast(x, grad_y, grad_input_q, grad_input_s, y_q_t, y_s_t, M, H, h_groups, m_groups)


def _run_nvidia(
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
):
    _run_fast(x, grad_y, grad_input_q, grad_input_s, y_q_t, y_s_t, M, H, h_groups, m_groups)


_DISPATCH = {
    "ascend": _run_ascend,
    "mthreads": _run_mthreads,
    "tsingmicro": _run_tsingmicro,
    "hygon": _run_hygon,
    "metax": _run_metax,
    "aipu": _run_aipu,
    "nvidia": _run_nvidia,
}


def silu_dot_fwd_bwd_quant_fuse(
    x: torch.Tensor,
    grad_y: torch.Tensor,
    grad_input_q: torch.Tensor,
    grad_input_s: torch.Tensor,
    y_q_t: torch.Tensor,
    y_s_t: torch.Tensor,
    group_size: int = 128,
):
    assert group_size == GROUP_SIZE
    M = x.shape[0]
    two_h = x.shape[1]
    H = two_h // 2
    assert two_h == H * 2
    assert H % GROUP_SIZE == 0
    assert M % GROUP_SIZE == 0

    h_groups = H // GROUP_SIZE
    m_groups = M // GROUP_SIZE
    runner = _DISPATCH.get(_detect_vendor(x), _run_fast)
    runner(x, grad_y, grad_input_q, grad_input_s, y_q_t, y_s_t, M, H, h_groups, m_groups)
    return grad_input_q, grad_input_s, y_q_t, y_s_t
