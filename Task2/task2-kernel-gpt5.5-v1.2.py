import torch
import triton
import triton.language as tl


HEAD_DIM = 512
ROPE_HEAD_DIM = 64
NOPE_HEAD_DIM = HEAD_DIM - ROPE_HEAD_DIM
KV_BLOCK_SIZE = 64
TOKEN_STRIDE = 576
SCALE_DIM = 8


@triton.jit
def _zero_2d_u8_kernel(
    out_ptr,
    n_cols,
    total,
    stride_out_0,
    stride_out_1,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    row = offs // n_cols
    col = offs - row * n_cols
    tl.store(
        out_ptr + row * stride_out_0 + col * stride_out_1,
        tl.full((BLOCK,), 0, tl.uint8),
        mask=mask,
    )


@triton.jit
def _compress_tile_kernel(
    state_cache_ptr,
    token_to_req_ptr,
    positions_ptr,
    boundary_token_indices_ptr,
    block_table_ptr,
    compressed_ptr,
    stride_state_block,
    stride_state_token,
    stride_state_dim,
    stride_token_to_req,
    stride_positions,
    stride_boundary,
    stride_block_req,
    stride_block_idx,
    stride_compressed_0,
    stride_compressed_1,
    BLOCK_SIZE_STATE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_out = tl.program_id(0)
    pid_d = tl.program_id(1)
    dims = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    boundary_token = tl.load(
        boundary_token_indices_ptr + pid_out * stride_boundary
    ).to(tl.int32)
    boundary_pos = tl.load(positions_ptr + boundary_token * stride_positions).to(
        tl.int32
    )
    req_id = tl.load(token_to_req_ptr + boundary_token * stride_token_to_req).to(
        tl.int32
    )

    max_score = tl.full((BLOCK_D,), -3.4028234663852886e38, tl.float32)
    for t in range(0, COMPRESS_RATIO):
        pos = boundary_pos - COMPRESS_RATIO + 1 + t
        local_block = pos // BLOCK_SIZE_STATE
        block_offset = pos - local_block * BLOCK_SIZE_STATE
        state_block = tl.load(
            block_table_ptr + req_id * stride_block_req + local_block * stride_block_idx
        ).to(tl.int32)
        base = (
            state_block.to(tl.int64) * stride_state_block
            + block_offset.to(tl.int64) * stride_state_token
        )
        score = tl.load(
            state_cache_ptr + base + (512 + dims).to(tl.int64) * stride_state_dim
        ).to(tl.float32)
        max_score = tl.maximum(max_score, score)

    denom = tl.full((BLOCK_D,), 0.0, tl.float32)
    acc = tl.full((BLOCK_D,), 0.0, tl.float32)
    for t in range(0, COMPRESS_RATIO):
        pos = boundary_pos - COMPRESS_RATIO + 1 + t
        local_block = pos // BLOCK_SIZE_STATE
        block_offset = pos - local_block * BLOCK_SIZE_STATE
        state_block = tl.load(
            block_table_ptr + req_id * stride_block_req + local_block * stride_block_idx
        ).to(tl.int32)
        base = (
            state_block.to(tl.int64) * stride_state_block
            + block_offset.to(tl.int64) * stride_state_token
        )
        value = tl.load(
            state_cache_ptr + base + dims.to(tl.int64) * stride_state_dim
        ).to(tl.float32)
        score = tl.load(
            state_cache_ptr + base + (512 + dims).to(tl.int64) * stride_state_dim
        ).to(tl.float32)
        weight = tl.exp(score - max_score)
        denom += weight
        acc += value * weight

    compressed = acc / denom
    tl.store(
        compressed_ptr + pid_out * stride_compressed_0 + dims * stride_compressed_1,
        compressed,
    )


@triton.jit
def _rms_kernel(
    compressed_ptr,
    rrms_ptr,
    rms_norm_eps,
    stride_compressed_0,
    stride_compressed_1,
    BLOCK_D: tl.constexpr,
):
    pid_out = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    vals = tl.load(
        compressed_ptr + pid_out * stride_compressed_0 + dims * stride_compressed_1
    ).to(tl.float32)
    ss = tl.sum(vals * vals, axis=0)
    rrms = tl.rsqrt(ss * 0.001953125 + rms_norm_eps)
    tl.store(rrms_ptr + pid_out, rrms)


@triton.jit
def _quant_nope_kernel(
    compressed_ptr,
    rrms_ptr,
    rms_norm_weight_ptr,
    boundary_token_indices_ptr,
    kv_slot_mapping_ptr,
    out_ptr,
    stride_compressed_0,
    stride_compressed_1,
    stride_weight,
    stride_boundary,
    stride_kv_slot,
    stride_out_0,
    stride_out_1,
    BLOCK_D: tl.constexpr,
):
    pid_out = tl.program_id(0)
    pid_group = tl.program_id(1)
    lane = tl.arange(0, BLOCK_D)
    dims = pid_group * BLOCK_D + lane

    rrms = tl.load(rrms_ptr + pid_out).to(tl.float32)
    vals = tl.load(
        compressed_ptr + pid_out * stride_compressed_0 + dims * stride_compressed_1
    ).to(tl.float32)
    weight = tl.load(rms_norm_weight_ptr + dims * stride_weight).to(tl.float32)

    nope = (vals * rrms * weight).to(tl.bfloat16).to(tl.float32)
    amax = tl.max(tl.abs(nope), axis=0)
    amax = tl.maximum(amax, 1.0e-4)
    exponent = tl.ceil(tl.log(amax * 0.007874015748031496) * 1.4426950408889634)
    inv_scale = tl.exp2(-exponent)

    q = nope * inv_scale
    q = tl.maximum(tl.minimum(q, 127.0), -127.0).to(tl.int8)

    boundary_token = tl.load(
        boundary_token_indices_ptr + pid_out * stride_boundary
    ).to(tl.int32)
    kv_slot = tl.load(kv_slot_mapping_ptr + boundary_token * stride_kv_slot).to(
        tl.int32
    )
    page = kv_slot // 64
    slot = kv_slot - page * 64
    value_base = slot * 576 + pid_group * BLOCK_D

    page_i64 = page.to(tl.int64)
    tl.store(
        out_ptr
        + page_i64 * stride_out_0
        + (value_base + lane).to(tl.int64) * stride_out_1,
        q,
    )

    scale = tl.maximum(tl.minimum(exponent + 127.0, 255.0), 0.0).to(tl.uint8)
    scale_col = 64 * 576 + slot * 8 + pid_group
    tl.store(
        out_ptr + page_i64 * stride_out_0 + scale_col.to(tl.int64) * stride_out_1,
        scale,
    )


@triton.jit
def _rope_and_scatter_kernel(
    compressed_ptr,
    rrms_ptr,
    rms_norm_weight_ptr,
    positions_ptr,
    boundary_token_indices_ptr,
    cos_sin_cache_ptr,
    kv_slot_mapping_ptr,
    out_ptr,
    stride_compressed_0,
    stride_compressed_1,
    stride_weight,
    stride_positions,
    stride_boundary,
    stride_cos_0,
    stride_cos_1,
    stride_kv_slot,
    stride_out_0,
    stride_out_1,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid_out = tl.program_id(0)
    pair = tl.arange(0, BLOCK_P)

    boundary_token = tl.load(
        boundary_token_indices_ptr + pid_out * stride_boundary
    ).to(tl.int32)
    boundary_pos = tl.load(positions_ptr + boundary_token * stride_positions).to(
        tl.int32
    )
    compressed_pos = (boundary_pos // COMPRESS_RATIO) * COMPRESS_RATIO

    rrms = tl.load(rrms_ptr + pid_out).to(tl.float32)
    even_dim = 448 + pair * 2
    odd_dim = even_dim + 1

    even = tl.load(
        compressed_ptr + pid_out * stride_compressed_0 + even_dim * stride_compressed_1
    ).to(tl.float32)
    odd = tl.load(
        compressed_ptr + pid_out * stride_compressed_0 + odd_dim * stride_compressed_1
    ).to(tl.float32)
    even_w = tl.load(rms_norm_weight_ptr + even_dim * stride_weight).to(tl.float32)
    odd_w = tl.load(rms_norm_weight_ptr + odd_dim * stride_weight).to(tl.float32)
    even = even * rrms * even_w
    odd = odd * rrms * odd_w

    cos_v = tl.load(
        cos_sin_cache_ptr
        + compressed_pos.to(tl.int64) * stride_cos_0
        + pair.to(tl.int64) * stride_cos_1
    ).to(tl.float32)
    sin_v = tl.load(
        cos_sin_cache_ptr
        + compressed_pos.to(tl.int64) * stride_cos_0
        + (32 + pair).to(tl.int64) * stride_cos_1
    ).to(tl.float32)

    rot_even = (even * cos_v - odd * sin_v).to(tl.bfloat16)
    rot_odd = (odd * cos_v + even * sin_v).to(tl.bfloat16)

    rot_even_i16 = rot_even.to(tl.int16, bitcast=True)
    rot_odd_i16 = rot_odd.to(tl.int16, bitcast=True)

    even_lo = (rot_even_i16 & 0xFF).to(tl.uint8)
    even_hi = ((rot_even_i16 >> 8) & 0xFF).to(tl.uint8)
    odd_lo = (rot_odd_i16 & 0xFF).to(tl.uint8)
    odd_hi = ((rot_odd_i16 >> 8) & 0xFF).to(tl.uint8)

    kv_slot = tl.load(kv_slot_mapping_ptr + boundary_token * stride_kv_slot).to(
        tl.int32
    )
    page = kv_slot // 64
    slot = kv_slot - page * 64
    page_i64 = page.to(tl.int64)
    base = slot * 576 + 448

    byte_off = base + pair * 4
    tl.store(
        out_ptr + page_i64 * stride_out_0 + byte_off.to(tl.int64) * stride_out_1,
        even_lo,
    )
    tl.store(
        out_ptr + page_i64 * stride_out_0 + (byte_off + 1).to(tl.int64) * stride_out_1,
        even_hi,
    )
    tl.store(
        out_ptr + page_i64 * stride_out_0 + (byte_off + 2).to(tl.int64) * stride_out_1,
        odd_lo,
    )
    tl.store(
        out_ptr + page_i64 * stride_out_0 + (byte_off + 3).to(tl.int64) * stride_out_1,
        odd_hi,
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
    out = torch.empty_like(kv_cache)
    total = out.numel()
    if total > 0:
        _zero_2d_u8_kernel[(triton.cdiv(total, 1024),)](
            out,
            out.shape[1],
            total,
            out.stride(0),
            out.stride(1),
            BLOCK=1024,
            num_warps=4,
        )

    num_outputs = boundary_token_indices.numel()
    if num_outputs == 0:
        return out

    compressed = torch.empty(
        (num_outputs, HEAD_DIM), device=state_cache.device, dtype=torch.float32
    )
    rrms = torch.empty((num_outputs,), device=state_cache.device, dtype=torch.float32)

    _compress_tile_kernel[(num_outputs, HEAD_DIM // 64)](
        state_cache,
        token_to_req,
        positions,
        boundary_token_indices,
        block_table,
        compressed,
        state_cache.stride(0),
        state_cache.stride(1),
        state_cache.stride(2),
        token_to_req.stride(0),
        positions.stride(0),
        boundary_token_indices.stride(0),
        block_table.stride(0),
        block_table.stride(1),
        compressed.stride(0),
        compressed.stride(1),
        BLOCK_SIZE_STATE=block_size,
        COMPRESS_RATIO=compress_ratio,
        BLOCK_D=64,
        num_warps=2,
    )
    _rms_kernel[(num_outputs,)](
        compressed,
        rrms,
        rms_norm_eps,
        compressed.stride(0),
        compressed.stride(1),
        BLOCK_D=HEAD_DIM,
        num_warps=8,
    )
    _quant_nope_kernel[(num_outputs, NOPE_HEAD_DIM // 64)](
        compressed,
        rrms,
        rms_norm_weight,
        boundary_token_indices,
        kv_slot_mapping,
        out,
        compressed.stride(0),
        compressed.stride(1),
        rms_norm_weight.stride(0),
        boundary_token_indices.stride(0),
        kv_slot_mapping.stride(0),
        out.stride(0),
        out.stride(1),
        BLOCK_D=64,
        num_warps=2,
    )
    _rope_and_scatter_kernel[(num_outputs,)](
        compressed,
        rrms,
        rms_norm_weight,
        positions,
        boundary_token_indices,
        cos_sin_cache,
        kv_slot_mapping,
        out,
        compressed.stride(0),
        compressed.stride(1),
        rms_norm_weight.stride(0),
        positions.stride(0),
        boundary_token_indices.stride(0),
        cos_sin_cache.stride(0),
        cos_sin_cache.stride(1),
        kv_slot_mapping.stride(0),
        out.stride(0),
        out.stride(1),
        COMPRESS_RATIO=compress_ratio,
        BLOCK_P=ROPE_HEAD_DIM // 2,
        num_warps=1,
    )
    return out
