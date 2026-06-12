# SPDX-License-Identifier: Apache-2.0
# Task 02: c128_256_512_compress (DeepSeek V4 KV compression)
#
# This submission intentionally keeps the public Python signature identical to
# the task statement and routes the numerical path through Triton kernels only.
# No torch fallback path is used.

import torch
import triton
import triton.language as tl


HEAD_DIM = 512
ROPE_HEAD_DIM = 64
NOPE_HEAD_DIM = HEAD_DIM - ROPE_HEAD_DIM
KV_BLOCK_SIZE = 64
TOKEN_STRIDE = 576
SCALE_DIM = 8
NUM_Q_GROUPS = NOPE_HEAD_DIM // 64
NUM_DIM_GROUPS = HEAD_DIM // 64


@triton.jit
def _zero_u8_kernel(
    out_ptr,
    n_cols: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    z = tl.zeros((BLOCK_N,), dtype=tl.uint8)
    tl.store(out_ptr + row * out_s0 + cols * out_s1, z, mask=mask)


@triton.jit
def _compress_64d_kernel(
    state_cache,
    token_to_req,
    positions,
    boundary_token_indices,
    block_table,
    compressed,
    partial_sumsq,
    state_s0: tl.constexpr,
    state_s1: tl.constexpr,
    state_s2: tl.constexpr,
    bt_s0: tl.constexpr,
    bt_s1: tl.constexpr,
    BLOCK_T: tl.constexpr,
    STATE_BLOCK_SIZE: tl.constexpr,
    CHUNK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    out_id = tl.program_id(0)
    group_id = tl.program_id(1)

    d = group_id * BLOCK_D + tl.arange(0, BLOCK_D)
    ct = tl.arange(0, CHUNK_T)

    boundary_token = tl.load(boundary_token_indices + out_id)
    boundary_pos = tl.load(positions + boundary_token)
    req_id = tl.load(token_to_req + boundary_token)
    first_pos = boundary_pos - BLOCK_T + 1

    # First pass: per-dimension max for numerically stable softmax.
    m = tl.full((BLOCK_D,), -float("inf"), dtype=tl.float32)
    for base_t in tl.static_range(0, BLOCK_T, CHUNK_T):
        pos = first_pos + base_t + ct
        local_block_idx = pos // STATE_BLOCK_SIZE
        local_block_off = pos - local_block_idx * STATE_BLOCK_SIZE
        global_block = tl.load(block_table + req_id * bt_s0 + local_block_idx * bt_s1)

        score_ptrs = (
            state_cache
            + global_block[:, None] * state_s0
            + local_block_off[:, None] * state_s1
            + (HEAD_DIM + d[None, :]) * state_s2
        )
        scores = tl.load(score_ptrs)
        m = tl.maximum(m, tl.max(scores, axis=0))

    # Second pass: denominator and value-weighted numerator.
    denom = tl.zeros((BLOCK_D,), dtype=tl.float32)
    numer = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for base_t in tl.static_range(0, BLOCK_T, CHUNK_T):
        pos = first_pos + base_t + ct
        local_block_idx = pos // STATE_BLOCK_SIZE
        local_block_off = pos - local_block_idx * STATE_BLOCK_SIZE
        global_block = tl.load(block_table + req_id * bt_s0 + local_block_idx * bt_s1)

        score_ptrs = (
            state_cache
            + global_block[:, None] * state_s0
            + local_block_off[:, None] * state_s1
            + (HEAD_DIM + d[None, :]) * state_s2
        )
        value_ptrs = (
            state_cache
            + global_block[:, None] * state_s0
            + local_block_off[:, None] * state_s1
            + d[None, :] * state_s2
        )
        scores = tl.load(score_ptrs)
        vals = tl.load(value_ptrs)
        p = tl.exp(scores - m[None, :])
        denom += tl.sum(p, axis=0)
        numer += tl.sum(vals * p, axis=0)

    comp = numer / denom
    tl.store(compressed + out_id * HEAD_DIM + d, comp)
    ss = tl.sum(comp * comp, axis=0)
    tl.store(partial_sumsq + out_id * NUM_DIM_GROUPS + group_id, ss)


@triton.jit
def _finalize_store_kernel(
    compressed,
    partial_sumsq,
    boundary_token_indices,
    positions,
    rms_norm_weight,
    cos_sin_cache,
    kv_slot_mapping,
    out_ptr,
    cos_s0: tl.constexpr,
    cos_s1: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    BLOCK_T: tl.constexpr,
    RMS_EPS: tl.constexpr,
):
    out_id = tl.program_id(0)

    offs8 = tl.arange(0, NUM_DIM_GROUPS)
    ss = tl.load(partial_sumsq + out_id * NUM_DIM_GROUPS + offs8)
    sumsq = tl.sum(ss, axis=0)
    rrms = tl.rsqrt(sumsq * (1.0 / 512.0) + RMS_EPS)

    boundary_token = tl.load(boundary_token_indices + out_id)
    boundary_pos = tl.load(positions + boundary_token)
    kv_slot = tl.load(kv_slot_mapping + boundary_token)
    page = kv_slot // KV_BLOCK_SIZE
    slot_off = kv_slot - page * KV_BLOCK_SIZE
    value_base = slot_off * TOKEN_STRIDE
    scale_base = KV_BLOCK_SIZE * TOKEN_STRIDE + slot_off * SCALE_DIM
    out_page_base = page * out_s0

    offs64 = tl.arange(0, 64)

    # First 448 dimensions: bf16-round then per-64 INT8 quantization.
    for qg in tl.static_range(0, NUM_Q_GROUPS):
        d = qg * 64 + offs64
        comp = tl.load(compressed + out_id * HEAD_DIM + d)
        w = tl.load(rms_norm_weight + d).to(tl.float32)
        norm = comp * rrms * w

        # Match: normed[:, :448].to(torch.bfloat16).to(torch.float32)
        nope = norm.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
        amax = tl.max(tl.abs(nope), axis=0)
        amax = tl.maximum(amax, 1.0e-4)
        exponent = tl.ceil(tl.log2(amax * (1.0 / 127.0)))
        inv_scale = tl.exp2(-exponent)
        qf = nope * inv_scale
        qf = tl.minimum(tl.maximum(qf, -127.0), 127.0)
        qi8 = qf.to(tl.int8)
        qbytes = qi8.to(tl.uint8, bitcast=True)

        tl.store(out_ptr + out_page_base + (value_base + qg * 64 + offs64) * out_s1, qbytes)

        scale = exponent + 127.0
        scale = tl.minimum(tl.maximum(scale, 0.0), 255.0).to(tl.uint8)
        tl.store(out_ptr + out_page_base + (scale_base + qg) * out_s1, scale)

    # Last 64 dimensions: GPT-J interleaved RoPE, then bf16 byte layout.
    pair = tl.arange(0, ROPE_HEAD_DIM // 2)
    even_d = NOPE_HEAD_DIM + pair * 2
    odd_d = even_d + 1

    even_comp = tl.load(compressed + out_id * HEAD_DIM + even_d)
    odd_comp = tl.load(compressed + out_id * HEAD_DIM + odd_d)
    even_w = tl.load(rms_norm_weight + even_d).to(tl.float32)
    odd_w = tl.load(rms_norm_weight + odd_d).to(tl.float32)
    rope_even = even_comp * rrms * even_w
    rope_odd = odd_comp * rrms * odd_w

    compressed_pos = (boundary_pos // BLOCK_T) * BLOCK_T
    cos_v = tl.load(cos_sin_cache + compressed_pos * cos_s0 + pair * cos_s1)
    sin_v = tl.load(cos_sin_cache + compressed_pos * cos_s0 + (pair + ROPE_HEAD_DIM // 2) * cos_s1)

    rot_even = rope_even * cos_v - rope_odd * sin_v
    rot_odd = rope_odd * cos_v + rope_even * sin_v

    # Store rotated bf16 as the exact uint8 view used by PyTorch on little-endian devices:
    # [even_lo, even_hi, odd_lo, odd_hi] for each GPT-J pair.
    rot_even_u16 = rot_even.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.uint16, bitcast=True)
    rot_odd_u16 = rot_odd.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.uint16, bitcast=True)

    rope_byte_base = value_base + NOPE_HEAD_DIM + pair * 4
    tl.store(
        out_ptr + out_page_base + (rope_byte_base + 0) * out_s1,
        (rot_even_u16 & 0x00FF).to(tl.uint8),
    )
    tl.store(
        out_ptr + out_page_base + (rope_byte_base + 1) * out_s1,
        ((rot_even_u16 >> 8) & 0x00FF).to(tl.uint8),
    )
    tl.store(
        out_ptr + out_page_base + (rope_byte_base + 2) * out_s1,
        (rot_odd_u16 & 0x00FF).to(tl.uint8),
    )
    tl.store(
        out_ptr + out_page_base + (rope_byte_base + 3) * out_s1,
        ((rot_odd_u16 >> 8) & 0x00FF).to(tl.uint8),
    )


def c128_256_512_compress(
    state_cache: torch.Tensor,
    token_to_req: torch.Tensor,
    positions: torch.Tensor,
    boundary_token_indices: torch.Tensor,
    block_table: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    block_size: int,
    compress_ratio: int,
    rms_norm_eps: float = 1.0e-6,
) -> torch.Tensor:
    """
    DeepSeek V4 KV-cache compressor for compress_ratio in {128, 256, 512}.

    The implementation uses three Triton stages:
      1. zero the uint8 output cache,
      2. compute 64-dim softmax-weighted compression blocks and RMS partial sums,
      3. apply RMSNorm, INT8 packing, RoPE bf16 packing, and paged scatter.
    """
    out = torch.empty_like(kv_cache)

    if out.numel() > 0:
        grid_zero = (out.shape[0], triton.cdiv(out.shape[1], 4096))
        _zero_u8_kernel[grid_zero](
            out,
            out.shape[1],
            out.stride(0),
            out.stride(1),
            BLOCK_N=4096,
        )

    num_outputs = boundary_token_indices.numel()
    if num_outputs == 0:
        return out

    if compress_ratio not in (128, 256, 512):
        raise ValueError("compress_ratio must be one of {128, 256, 512}")

    compressed = torch.empty((num_outputs, HEAD_DIM), device=state_cache.device, dtype=torch.float32)
    partial_sumsq = torch.empty((num_outputs, NUM_DIM_GROUPS), device=state_cache.device, dtype=torch.float32)

    grid_compress = (num_outputs, NUM_DIM_GROUPS)
    _compress_64d_kernel[grid_compress](
        state_cache,
        token_to_req,
        positions,
        boundary_token_indices,
        block_table,
        compressed,
        partial_sumsq,
        state_cache.stride(0),
        state_cache.stride(1),
        state_cache.stride(2),
        block_table.stride(0),
        block_table.stride(1),
        BLOCK_T=compress_ratio,
        STATE_BLOCK_SIZE=block_size,
        CHUNK_T=64,
        BLOCK_D=64,
        num_warps=8,
    )

    grid_finalize = (num_outputs,)
    _finalize_store_kernel[grid_finalize](
        compressed,
        partial_sumsq,
        boundary_token_indices,
        positions,
        rms_norm_weight,
        cos_sin_cache,
        kv_slot_mapping,
        out,
        cos_sin_cache.stride(0),
        cos_sin_cache.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_T=compress_ratio,
        RMS_EPS=float(rms_norm_eps),
        num_warps=8,
    )

    return out
