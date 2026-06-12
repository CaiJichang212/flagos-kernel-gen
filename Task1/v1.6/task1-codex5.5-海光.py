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
    return _run_hygon(
        x_q, x_scale, w_q4_packed, w_scale, w_zero, expert_offsets, out,
        group_size
    )
