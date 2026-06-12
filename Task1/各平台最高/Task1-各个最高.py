# -*- coding: utf-8 -*-
# Final Task1 submission: platform dispatcher plus per-platform best kernels.

import os

import torch
import triton
import triton.language as tl


_VENDOR_ALIASES = {
    "ascend": "ascend",
    "huawei": "ascend",
    "npu": "ascend",
    "mthreads": "mthreads",
    "moore": "mthreads",
    "musa": "mthreads",
    "tsingmicro": "tsingmicro",
    "tianshu": "tsingmicro",
    "txda": "tsingmicro",
    "hygon": "hygon",
    "haiguang": "hygon",
    "metax": "metax",
    "muxi": "metax",
    "aipu": "aipu",
    "pingtouge": "aipu",
    "nvidia": "nvidia",
    "cuda": "nvidia",
}


def _normalize_vendor(vendor):
    if vendor is None:
        return None
    return _VENDOR_ALIASES.get(str(vendor).strip().lower())


def _detect_vendor(x_q: torch.Tensor):
    vendor = _normalize_vendor(os.environ.get("GEMS_VENDOR"))
    if vendor is not None:
        return vendor

    try:
        from flag_gems.runtime.backend.device import DeviceDetector

        vendor = _normalize_vendor(DeviceDetector().vendor_name)
        if vendor is not None:
            return vendor
    except Exception:
        pass

    device_type = _normalize_vendor(getattr(x_q.device, "type", None))
    if device_type is not None:
        return device_type
    return "nvidia"


def _max_m_per_expert(expert_offsets: torch.Tensor) -> int:
    expert_counts = expert_offsets[1:] - expert_offsets[:-1]
    return int(expert_counts.max().item())


@triton.jit
def _w4a8_group_gemm_moe_kernel_i8(
    x_q_ptr,
    x_scale_ptr,
    w_q4_packed_ptr,
    w_scale_ptr,
    w_zero_ptr,
    expert_offsets_ptr,
    out_ptr,
    N,
    K,
    GROUP_SIZE: tl.constexpr,
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
    BLOCK_PACKED_K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    expert_start = tl.load(expert_offsets_ptr + pid_e)
    expert_end = tl.load(expert_offsets_ptr + pid_e + 1)
    expert_m = expert_end - expert_start

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_pk = tl.arange(0, BLOCK_PACKED_K)

    rows = expert_start + offs_m
    mask_m = offs_m < expert_m
    mask_n = offs_n < N

    x_base = x_q_ptr + rows * stride_x_m
    w_base = w_q4_packed_ptr + pid_e * stride_w_e
    ws_base = w_scale_ptr + pid_e * stride_ws_e
    wz_base = w_zero_ptr + pid_e * stride_wz_e

    w_n_offsets = offs_n * stride_w_n
    ws_n_offsets = offs_n * stride_ws_n
    wz_n_offsets = offs_n * stride_wz_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for group_id in range(0, NUM_GROUPS):
        pk_offsets = group_id * BLOCK_PACKED_K + offs_pk
        k_even = group_id * GROUP_SIZE + offs_pk * 2
        k_odd = k_even + 1

        x_even = tl.load(
            x_base[:, None] + k_even[None, :] * stride_x_k,
            mask=mask_m[:, None],
            other=0,
        )
        x_odd = tl.load(
            x_base[:, None] + k_odd[None, :] * stride_x_k,
            mask=mask_m[:, None],
            other=0,
        )

        w_packed = tl.load(
            w_base + pk_offsets[:, None] * stride_w_k + w_n_offsets[None, :],
            mask=mask_n[None, :],
            other=0,
        )
        w_zero_vals = tl.load(
            wz_base + wz_n_offsets + group_id * stride_wz_g,
            mask=mask_n,
            other=0,
        )

        w_packed_i32 = w_packed.to(tl.int32)
        w_zero_i32 = w_zero_vals.to(tl.int32)

        w_low = ((w_packed_i32 & 0x0F) - w_zero_i32[None, :]).to(tl.int8)
        w_high = (((w_packed_i32 >> 4) & 0x0F) - w_zero_i32[None, :]).to(tl.int8)

        partial = tl.dot(x_even.to(tl.int8), w_low) + tl.dot(
            x_odd.to(tl.int8), w_high
        )
        w_scale_vals = tl.load(
            ws_base + ws_n_offsets + group_id * stride_ws_g,
            mask=mask_n,
            other=0.0,
        )
        acc += partial.to(tl.float32) * w_scale_vals[None, :]

    x_scale_vals = tl.load(x_scale_ptr + rows, mask=mask_m, other=0.0)
    acc *= x_scale_vals[:, None]

    out_ptrs = out_ptr + rows[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


@triton.jit
def _w4a8_group_gemm_moe_kernel_moore(
    x_q_ptr,
    x_scale_ptr,
    w_q4_packed_ptr,
    w_scale_ptr,
    w_zero_ptr,
    expert_offsets_ptr,
    out_ptr,
    N,
    K,
    GROUP_SIZE: tl.constexpr,
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
    NUM_GROUPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HALF_GROUP: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    expert_start = tl.load(expert_offsets_ptr + pid_e)
    expert_end = tl.load(expert_offsets_ptr + pid_e + 1)
    expert_m = expert_end - expert_start

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_hg = tl.arange(0, HALF_GROUP)

    rows = expert_start + offs_m
    mask_m = offs_m < expert_m
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    w_base = w_q4_packed_ptr + pid_e * stride_w_e
    ws_base = w_scale_ptr + pid_e * stride_ws_e
    wz_base = w_zero_ptr + pid_e * stride_wz_e

    offs_n_ws = offs_n * stride_ws_n
    offs_n_wz = offs_n * stride_wz_n
    offs_n_w = offs_n * stride_w_n

    row_x_base = rows * stride_x_m
    offs_hg_wk = offs_hg * stride_w_k
    offs_hg_even_xk = (offs_hg * 2) * stride_x_k
    offs_hg_odd_xk = (offs_hg * 2 + 1) * stride_x_k

    mask_n_2d = mask_n[None, :]
    mask_m_2d = mask_m[:, None]

    MASK_LOW: tl.constexpr = 0x0F

    for group_id in range(0, NUM_GROUPS):
        k_base_xk = (group_id * GROUP_SIZE) * stride_x_k
        packed_base_wk = (group_id * HALF_GROUP) * stride_w_k

        x_even_ptrs = x_q_ptr + row_x_base[:, None] + (
            k_base_xk + offs_hg_even_xk[None, :]
        )
        x_odd_ptrs = x_q_ptr + row_x_base[:, None] + (
            k_base_xk + offs_hg_odd_xk[None, :]
        )
        x_even = tl.load(x_even_ptrs, mask=mask_m_2d, other=0)
        x_odd = tl.load(x_odd_ptrs, mask=mask_m_2d, other=0)

        w_packed_ptrs = (
            w_base + (packed_base_wk + offs_hg_wk[:, None]) + offs_n_w[None, :]
        )
        w_packed = tl.load(w_packed_ptrs, mask=mask_n_2d, other=0)

        w_zero_vals = tl.load(
            wz_base + offs_n_wz + group_id * stride_wz_g,
            mask=mask_n,
            other=0,
        )

        w_packed_i16 = w_packed.to(tl.int16)
        w_zero_i16 = w_zero_vals.to(tl.int16)
        w_low_f = ((w_packed_i16 & MASK_LOW) - w_zero_i16[None, :]).to(tl.float16)
        w_high_f = (((w_packed_i16 >> 4) & MASK_LOW) - w_zero_i16[None, :]).to(
            tl.float16
        )

        x_even_f = x_even.to(tl.float16)
        x_odd_f = x_odd.to(tl.float16)

        partial = tl.dot(x_even_f, w_low_f) + tl.dot(x_odd_f, w_high_f)

        w_scale_vals = tl.load(
            ws_base + offs_n_ws + group_id * stride_ws_g,
            mask=mask_n,
            other=0.0,
        )
        acc += partial * w_scale_vals[None, :]

    x_scale_vals = tl.load(x_scale_ptr + rows, mask=mask_m, other=0.0)
    acc *= x_scale_vals[:, None]

    out_ptrs = out_ptr + rows[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    out_mask = mask_m_2d & mask_n[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


@triton.jit
def _w4a8_group_gemm_moe_kernel_ascend(
    x_q_ptr,
    x_scale_ptr,
    w_q4_packed_ptr,
    w_scale_ptr,
    w_zero_ptr,
    expert_offsets_ptr,
    out_ptr,
    N,
    K,
    E,
    max_m_per_expert,
    GROUP_SIZE: tl.constexpr,
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
    BLOCK_PACKED_K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    NUM_M_BLOCKS: tl.constexpr,
    NUM_N_BLOCKS: tl.constexpr,
):
    pid = tl.program_id(0)

    blocks_per_expert = NUM_M_BLOCKS * NUM_N_BLOCKS
    pid_e = pid // blocks_per_expert
    remainder = pid % blocks_per_expert
    pid_m = remainder // NUM_N_BLOCKS
    pid_n = remainder % NUM_N_BLOCKS

    valid_expert = pid_e.to(tl.float32) < E.to(tl.float32)
    expert_start = tl.load(expert_offsets_ptr + pid_e, mask=valid_expert, other=0)
    expert_end = tl.load(expert_offsets_ptr + pid_e + 1, mask=valid_expert, other=0)
    expert_m = expert_end - expert_start

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_pk = tl.arange(0, BLOCK_PACKED_K)

    mask_m = offs_m.to(tl.float32) < expert_m.to(tl.float32)
    mask_n = offs_n.to(tl.float32) < N.to(tl.float32)
    rows = expert_start + offs_m

    x_base = x_q_ptr + rows * stride_x_m
    w_base = w_q4_packed_ptr + pid_e * stride_w_e
    ws_base = w_scale_ptr + pid_e * stride_ws_e
    wz_base = w_zero_ptr + pid_e * stride_wz_e

    w_n_offsets = offs_n * stride_w_n
    ws_n_offsets = offs_n * stride_ws_n
    wz_n_offsets = offs_n * stride_wz_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for group_id in range(0, NUM_GROUPS):
        pk_offsets = group_id * BLOCK_PACKED_K + offs_pk
        k_even = group_id * GROUP_SIZE + offs_pk * 2
        k_odd = k_even + 1

        x_even = tl.load(
            x_base[:, None] + k_even[None, :] * stride_x_k,
            mask=mask_m[:, None],
            other=0,
        )
        x_odd = tl.load(
            x_base[:, None] + k_odd[None, :] * stride_x_k,
            mask=mask_m[:, None],
            other=0,
        )

        w_packed = tl.load(
            w_base + pk_offsets[:, None] * stride_w_k + w_n_offsets[None, :],
            mask=mask_n[None, :],
            other=0,
        )
        w_zero_vals = tl.load(
            wz_base + wz_n_offsets + group_id * stride_wz_g,
            mask=mask_n,
            other=0,
        )

        w_packed_i32 = w_packed.to(tl.int32)
        w_zero_i32 = w_zero_vals.to(tl.int32)

        w_low = ((w_packed_i32 & 0x0F) - w_zero_i32[None, :]).to(tl.int8)
        w_high = (((w_packed_i32 >> 4) & 0x0F) - w_zero_i32[None, :]).to(tl.int8)

        partial = tl.dot(x_even.to(tl.int8), w_low) + tl.dot(
            x_odd.to(tl.int8), w_high
        )
        w_scale_vals = tl.load(
            ws_base + ws_n_offsets + group_id * stride_ws_g,
            mask=mask_n,
            other=0.0,
        )
        acc += partial.to(tl.float32) * w_scale_vals[None, :]

    x_scale_vals = tl.load(x_scale_ptr + rows, mask=mask_m, other=0.0)
    acc *= x_scale_vals[:, None]

    out_ptrs = out_ptr + rows[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_stages=2, num_warps=8),
    ],
    key=["N", "K", "GROUP_SIZE", "max_m"],
)
@triton.jit
def _w4a8_group_gemm_moe_kernel_nvidia(
    x_q_ptr,
    x_scale_ptr,
    w_q4_packed_ptr,
    w_scale_ptr,
    w_zero_ptr,
    expert_offsets_ptr,
    out_ptr,
    N,
    K,
    max_m,
    GROUP_SIZE: tl.constexpr,
    stride_x_m,
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_PACKED_K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_mn = tl.program_id(1)

    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    expert_start = tl.load(expert_offsets_ptr + pid_e)
    expert_end = tl.load(expert_offsets_ptr + pid_e + 1)
    expert_m = expert_end - expert_start

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_pk = tl.arange(0, BLOCK_PACKED_K)

    mask_m = offs_m < expert_m
    mask_n = offs_n < N
    rows = expert_start + offs_m

    x_base = x_q_ptr + rows * stride_x_m
    w_base = w_q4_packed_ptr + pid_e * stride_w_e
    ws_base = w_scale_ptr + pid_e * stride_ws_e
    wz_base = w_zero_ptr + pid_e * stride_wz_e

    w_n_offsets = offs_n * stride_w_n
    ws_n_offsets = offs_n * stride_ws_n
    wz_n_offsets = offs_n * stride_wz_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for group_id in range(0, NUM_GROUPS):
        pk_start = group_id * BLOCK_PACKED_K
        k_even = group_id * GROUP_SIZE + offs_pk * 2
        k_odd = k_even + 1

        x_even = tl.load(
            x_base[:, None] + k_even[None, :],
            mask=mask_m[:, None],
            other=0,
            eviction_policy="evict_last",
        )
        x_odd = tl.load(
            x_base[:, None] + k_odd[None, :],
            mask=mask_m[:, None],
            other=0,
            eviction_policy="evict_last",
        )

        w_packed = tl.load(
            w_base + (pk_start + offs_pk)[:, None] * stride_w_k + w_n_offsets[None, :],
            mask=mask_n[None, :],
            other=0,
            eviction_policy="evict_first",
        )
        w_zero_vals = tl.load(
            wz_base + wz_n_offsets + group_id * stride_wz_g,
            mask=mask_n,
            other=0,
        )

        w_packed_i32 = w_packed.to(tl.int32)
        w_zero_i32 = w_zero_vals.to(tl.int32)

        w_low = ((w_packed_i32 & 0x0F) - w_zero_i32[None, :]).to(tl.int8)
        w_high = (((w_packed_i32 >> 4) & 0x0F) - w_zero_i32[None, :]).to(tl.int8)

        partial = tl.dot(x_even.to(tl.int8), w_low) + tl.dot(
            x_odd.to(tl.int8), w_high
        )
        w_scale_vals = tl.load(
            ws_base + ws_n_offsets + group_id * stride_ws_g,
            mask=mask_n,
            other=0.0,
        )
        acc += partial.to(tl.float32) * w_scale_vals[None, :]

    x_scale_vals = tl.load(x_scale_ptr + rows, mask=mask_m, other=0.0)
    acc *= x_scale_vals[:, None]

    out_ptrs = out_ptr + rows[:, None] * stride_out_m + offs_n[None, :]
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


def _select_block_m(max_m_per_expert: int) -> int:
    if max_m_per_expert <= 16:
        return 16
    if max_m_per_expert <= 32:
        return 32
    return 64


def _run_i8_fixed(
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    w_q4_packed: torch.Tensor,
    w_scale: torch.Tensor,
    w_zero: torch.Tensor,
    expert_offsets: torch.Tensor,
    out: torch.Tensor,
    group_size: int,
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
) -> torch.Tensor:
    _, K = x_q.shape
    E, N, _ = w_q4_packed.shape
    num_groups = K // group_size
    block_packed_k = group_size // 2
    max_m_per_expert = _max_m_per_expert(expert_offsets)
    if max_m_per_expert == 0:
        return out

    grid = (E, triton.cdiv(max_m_per_expert, block_m), triton.cdiv(N, block_n))
    _w4a8_group_gemm_moe_kernel_i8[grid](
        x_q,
        x_scale,
        w_q4_packed,
        w_scale,
        w_zero,
        expert_offsets,
        out,
        N=N,
        K=K,
        GROUP_SIZE=group_size,
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
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_PACKED_K=block_packed_k,
        NUM_GROUPS=num_groups,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def _run_ascend(
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    w_q4_packed: torch.Tensor,
    w_scale: torch.Tensor,
    w_zero: torch.Tensor,
    expert_offsets: torch.Tensor,
    out: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    import torch_npu  # noqa: F401

    _, K = x_q.shape
    E, N, _ = w_q4_packed.shape
    num_groups = K // group_size
    block_packed_k = group_size // 2
    max_m_per_expert = _max_m_per_expert(expert_offsets)
    if max_m_per_expert == 0:
        return out

    block_m = _select_block_m(max_m_per_expert)
    block_n = 64
    num_m_blocks = triton.cdiv(max_m_per_expert, block_m)
    num_n_blocks = triton.cdiv(N, block_n)
    total_blocks = E * num_m_blocks * num_n_blocks

    while total_blocks > 65535 and block_m < 256:
        block_m *= 2
        num_m_blocks = triton.cdiv(max_m_per_expert, block_m)
        total_blocks = E * num_m_blocks * num_n_blocks
    while total_blocks > 65535 and block_n < 256:
        block_n *= 2
        num_n_blocks = triton.cdiv(N, block_n)
        total_blocks = E * num_m_blocks * num_n_blocks

    _w4a8_group_gemm_moe_kernel_ascend[(total_blocks,)](
        x_q,
        x_scale,
        w_q4_packed,
        w_scale,
        w_zero,
        expert_offsets,
        out,
        N=N,
        K=K,
        E=E,
        max_m_per_expert=max_m_per_expert,
        GROUP_SIZE=group_size,
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
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_PACKED_K=block_packed_k,
        NUM_GROUPS=num_groups,
        NUM_M_BLOCKS=num_m_blocks,
        NUM_N_BLOCKS=num_n_blocks,
        num_warps=4,
        num_stages=2,
    )
    return out


def _run_mthreads(
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    w_q4_packed: torch.Tensor,
    w_scale: torch.Tensor,
    w_zero: torch.Tensor,
    expert_offsets: torch.Tensor,
    out: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    _, K = x_q.shape
    E, N, _ = w_q4_packed.shape
    num_groups = K // group_size
    half_group = group_size // 2
    max_m_per_expert = _max_m_per_expert(expert_offsets)
    if max_m_per_expert == 0:
        return out

    block_m = _select_block_m(max_m_per_expert)
    if N <= 64:
        block_n = 32
        num_warps = 4
    elif N <= 256:
        block_n = 64
        num_warps = 4
    else:
        block_n = 128
        num_warps = 8

    grid = (E, triton.cdiv(max_m_per_expert, block_m), triton.cdiv(N, block_n))
    _w4a8_group_gemm_moe_kernel_moore[grid](
        x_q,
        x_scale,
        w_q4_packed,
        w_scale,
        w_zero,
        expert_offsets,
        out,
        N=N,
        K=K,
        GROUP_SIZE=group_size,
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
        NUM_GROUPS=num_groups,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HALF_GROUP=half_group,
        num_warps=num_warps,
        num_stages=2,
    )
    return out


def _run_tsingmicro(
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    w_q4_packed: torch.Tensor,
    w_scale: torch.Tensor,
    w_zero: torch.Tensor,
    expert_offsets: torch.Tensor,
    out: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    M_total, _ = x_q.shape
    E, _, _ = w_q4_packed.shape
    avg_m = triton.cdiv(M_total, E)
    block_m = 16 if avg_m <= 32 else 32
    return _run_i8_fixed(
        x_q, x_scale, w_q4_packed, w_scale, w_zero, expert_offsets, out,
        group_size, block_m, 64, 4, 2
    )


def _run_hygon(
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    w_q4_packed: torch.Tensor,
    w_scale: torch.Tensor,
    w_zero: torch.Tensor,
    expert_offsets: torch.Tensor,
    out: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    block_m = _select_block_m(_max_m_per_expert(expert_offsets))
    return _run_i8_fixed(
        x_q, x_scale, w_q4_packed, w_scale, w_zero, expert_offsets, out,
        group_size, block_m, 64, 4, 2
    )


def _run_metax(
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    w_q4_packed: torch.Tensor,
    w_scale: torch.Tensor,
    w_zero: torch.Tensor,
    expert_offsets: torch.Tensor,
    out: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    block_m = _select_block_m(_max_m_per_expert(expert_offsets))
    return _run_i8_fixed(
        x_q, x_scale, w_q4_packed, w_scale, w_zero, expert_offsets, out,
        group_size, block_m, 64, 4, 2
    )


def _run_aipu(
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    w_q4_packed: torch.Tensor,
    w_scale: torch.Tensor,
    w_zero: torch.Tensor,
    expert_offsets: torch.Tensor,
    out: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    block_m = _select_block_m(_max_m_per_expert(expert_offsets))
    return _run_i8_fixed(
        x_q, x_scale, w_q4_packed, w_scale, w_zero, expert_offsets, out,
        group_size, block_m, 64, 4, 2
    )


def _run_nvidia(
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    w_q4_packed: torch.Tensor,
    w_scale: torch.Tensor,
    w_zero: torch.Tensor,
    expert_offsets: torch.Tensor,
    out: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    _, K = x_q.shape
    E, N, _ = w_q4_packed.shape
    max_m_per_expert = _max_m_per_expert(expert_offsets)
    if max_m_per_expert == 0:
        return out

    if x_q.stride(1) != 1 or out.stride(1) != 1:
        block_m = 16 if triton.cdiv(x_q.shape[0], E) <= 32 else 32
        return _run_i8_fixed(
            x_q, x_scale, w_q4_packed, w_scale, w_zero, expert_offsets, out,
            group_size, block_m, 64, 4, 2
        )

    num_groups = K // group_size
    block_packed_k = group_size // 2

    def grid(meta):
        tiles_m = triton.cdiv(max_m_per_expert, meta["BLOCK_M"])
        tiles_n = triton.cdiv(N, meta["BLOCK_N"])
        return (E, tiles_m * tiles_n)

    _w4a8_group_gemm_moe_kernel_nvidia[grid](
        x_q,
        x_scale,
        w_q4_packed,
        w_scale,
        w_zero,
        expert_offsets,
        out,
        N=N,
        K=K,
        max_m=max_m_per_expert,
        GROUP_SIZE=group_size,
        stride_x_m=x_q.stride(0),
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
        BLOCK_PACKED_K=block_packed_k,
        NUM_GROUPS=num_groups,
    )
    return out


def _run_muxi_original_dispatch(
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    w_q4_packed: torch.Tensor,
    w_scale: torch.Tensor,
    w_zero: torch.Tensor,
    expert_offsets: torch.Tensor,
    out: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    vendor = _detect_vendor(x_q)
    if vendor == "ascend":
        return _run_ascend(
            x_q, x_scale, w_q4_packed, w_scale, w_zero, expert_offsets, out,
            group_size
        )
    if vendor == "mthreads":
        return _run_mthreads(
            x_q, x_scale, w_q4_packed, w_scale, w_zero, expert_offsets, out,
            group_size
        )
    if vendor == "tsingmicro":
        return _run_tsingmicro(
            x_q, x_scale, w_q4_packed, w_scale, w_zero, expert_offsets, out,
            group_size
        )
    if vendor == "hygon":
        return _run_hygon(
            x_q, x_scale, w_q4_packed, w_scale, w_zero, expert_offsets, out,
            group_size
        )
    if vendor == "metax":
        return _run_metax(
            x_q, x_scale, w_q4_packed, w_scale, w_zero, expert_offsets, out,
            group_size
        )
    if vendor == "aipu":
        return _run_aipu(
            x_q, x_scale, w_q4_packed, w_scale, w_zero, expert_offsets, out,
            group_size
        )
    return _run_nvidia(
        x_q, x_scale, w_q4_packed, w_scale, w_zero, expert_offsets, out,
        group_size
    )


# ---- Per-platform best kernels from Task1/各平台最高 ----


# ---- Task1/各平台最高/task1-华为昇腾-11.22.py ----
import torch
import triton
import triton.language as tl


@triton.jit
def _ascend_best_kernel(
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


def _run_ascend_best(
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
    _ascend_best_kernel[(total_blocks,)](
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



# ---- Task1/各平台最高/task1-摩尔线程-2.44.py ----
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
def _mthreads_best_kernel(
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


def _run_mthreads_best(
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

    _mthreads_best_kernel[grid](
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


# ---- Task1/各平台最高/task1-天数-9.94.py ----
import torch
import triton
import triton.language as tl


@triton.jit
def _tsingmicro_best_kernel(
    x_q_ptr,
    x_scale_ptr,
    w_q4_packed_ptr,
    w_scale_ptr,
    w_zero_ptr,
    expert_offsets_ptr,
    out_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    num_tiles_m,
    num_tiles_n: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
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
    BLOCK_PACKED_K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    pid = tl.program_id(0)
    tiles_per_expert = num_tiles_m * num_tiles_n
    pid_e = pid // tiles_per_expert
    remainder = pid - pid_e * tiles_per_expert
    pid_m = remainder // num_tiles_n
    pid_n = remainder - pid_m * num_tiles_n

    expert_start = tl.load(expert_offsets_ptr + pid_e).to(tl.int32)
    expert_end = tl.load(expert_offsets_ptr + pid_e + 1).to(tl.int32)
    expert_m = expert_end - expert_start

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_pk = tl.arange(0, BLOCK_PACKED_K)

    mask_m = offs_m < expert_m
    mask_n = offs_n < N

    rows = expert_start + offs_m
    x_base = x_q_ptr + (rows * stride_x_m)[:, None]

    pid_e_i32 = pid_e.to(tl.int32)
    w_base = w_q4_packed_ptr + pid_e_i32 * stride_w_e
    ws_base = w_scale_ptr + pid_e_i32 * stride_ws_e
    wz_base = w_zero_ptr + pid_e_i32 * stride_wz_e

    w_n_offsets = (offs_n * stride_w_n)[None, :]
    ws_n_offsets = offs_n * stride_ws_n
    wz_n_offsets = offs_n * stride_wz_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for g in tl.static_range(0, NUM_GROUPS):
        pk_start = g * BLOCK_PACKED_K
        k_base = g * GROUP_SIZE

        x_even_k = k_base + offs_pk * 2
        x_odd_k = x_even_k + 1

        x_even_offsets = (x_even_k * stride_x_k)[None, :]
        x_odd_offsets = (x_odd_k * stride_x_k)[None, :]

        x_even = tl.load(
            x_base + x_even_offsets,
            mask=mask_m[:, None],
            other=0,
            eviction_policy="evict_first",
        )
        x_odd = tl.load(
            x_base + x_odd_offsets,
            mask=mask_m[:, None],
            other=0,
            eviction_policy="evict_first",
        )

        pk_offsets = pk_start + offs_pk
        w_pk_offsets = (pk_offsets * stride_w_k)[:, None]
        w_packed = tl.load(
            w_base + w_pk_offsets + w_n_offsets,
            mask=mask_n[None, :],
            other=0,
            eviction_policy="evict_first",
        )

        w_zero_vals = tl.load(
            wz_base + g * stride_wz_g + wz_n_offsets,
            mask=mask_n,
            other=0,
            eviction_policy="evict_last",
        )
        w_scale_vals = tl.load(
            ws_base + g * stride_ws_g + ws_n_offsets,
            mask=mask_n,
            other=0.0,
            eviction_policy="evict_last",
        )

        w_packed_i32 = w_packed.to(tl.int32)
        w_zero_i32 = w_zero_vals.to(tl.int32)[None, :]
        w_low = ((w_packed_i32 & 0x0F) - w_zero_i32).to(tl.int8)
        w_high = (((w_packed_i32 >> 4) & 0x0F) - w_zero_i32).to(tl.int8)

        partial = tl.dot(x_even.to(tl.int8), w_low)
        partial += tl.dot(x_odd.to(tl.int8), w_high)
        acc += partial.to(tl.float32) * w_scale_vals[None, :]

    x_scale_vals = tl.load(x_scale_ptr + rows, mask=mask_m, other=0.0)
    acc *= x_scale_vals[:, None]

    out_ptrs = out_ptr + (rows * stride_out_m)[:, None] + (offs_n * stride_out_n)[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def _run_tsingmicro_best(
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
    E, N, _ = w_q4_packed.shape
    num_groups = K // group_size
    block_packed_k = group_size // 2

    expert_counts = expert_offsets[1:] - expert_offsets[:-1]
    max_m_per_expert = int(expert_counts.max().item())
    if max_m_per_expert == 0:
        return out

    if max_m_per_expert <= 64:
        block_m = 16
    else:
        block_m = 32

    if N <= 512:
        block_n = 64
    else:
        block_n = 128

    num_tiles_m = (max_m_per_expert + block_m - 1) // block_m
    num_tiles_n = (N + block_n - 1) // block_n
    grid = (E * num_tiles_m * num_tiles_n,)

    _tsingmicro_best_kernel[grid](
        x_q,
        x_scale,
        w_q4_packed,
        w_scale,
        w_zero,
        expert_offsets,
        out,
        N=N,
        K=K,
        num_tiles_m=num_tiles_m,
        num_tiles_n=num_tiles_n,
        GROUP_SIZE=group_size,
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
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_PACKED_K=block_packed_k,
        NUM_GROUPS=num_groups,
        num_stages=3,
        num_warps=4,
    )
    return out


# ---- Task1/各平台最高/task1-平头哥-10.66.py ----
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
def _aipu_best_kernel(
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


def _run_aipu_best(
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

    _aipu_best_kernel[grid](
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

# ---- Task1/各平台最高/task1-海光-26.44.py ----
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
def _hygon_best_kernel(
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


def _run_hygon_best(
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

    _hygon_best_kernel[grid](
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

# ---- Task1/各平台最高/task1-国际通用芯片-21.56.py ----
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
def _nvidia_best_kernel(
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


def _run_nvidia_best(
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

    _nvidia_best_kernel[grid](
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

# ---- Final public entry ----

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
    vendor = _detect_vendor(x_q)
    if vendor == "ascend":
        return _run_ascend_best(
            x_q, x_scale, w_q4_packed, w_scale, w_zero, expert_offsets, out,
            group_size
        )
    if vendor == "mthreads":
        return _run_mthreads_best(
            x_q, x_scale, w_q4_packed, w_scale, w_zero, expert_offsets, out,
            group_size
        )
    if vendor == "tsingmicro":
        return _run_tsingmicro_best(
            x_q, x_scale, w_q4_packed, w_scale, w_zero, expert_offsets, out,
            group_size
        )
    if vendor == "hygon":
        return _run_hygon_best(
            x_q, x_scale, w_q4_packed, w_scale, w_zero, expert_offsets, out,
            group_size
        )
    if vendor == "metax":
        return _run_metax(
            x_q, x_scale, w_q4_packed, w_scale, w_zero, expert_offsets, out,
            group_size
        )
    if vendor == "aipu":
        return _run_aipu_best(
            x_q, x_scale, w_q4_packed, w_scale, w_zero, expert_offsets, out,
            group_size
        )
    return _run_nvidia_best(
        x_q, x_scale, w_q4_packed, w_scale, w_zero, expert_offsets, out,
        group_size
    )
