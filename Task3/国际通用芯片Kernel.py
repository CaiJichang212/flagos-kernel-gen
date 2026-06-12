import torch
import triton
import triton.language as tl


@triton.jit
def _silu_dot_fwd_bwd_quant_fuse_kernel(
    x_ptr,
    grad_y_ptr,
    grad_input_q_ptr,
    grad_input_s_ptr,
    y_q_t_ptr,
    y_s_t_ptr,
    M,
    H,
    stride_x_m,
    stride_x_h,
    stride_gy_m,
    stride_gy_h,
    stride_giq_m,
    stride_giq_h,
    stride_gis_m,
    stride_gis_h,
    stride_yqt_h,
    stride_yqt_m,
    stride_yst_h,
    stride_yst_m,
    GROUP_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # Each program handles one row (m) and one group of H columns
    pid_m = tl.program_id(0)
    pid_group = tl.program_id(1)

    # Column offsets for this group
    h_offs = pid_group * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = h_offs < H

    # Load gate and up from x: x has shape [M, 2H], gate = x[:, :H], up = x[:, H:]
    gate_ptrs = x_ptr + pid_m * stride_x_m + h_offs * stride_x_h
    up_ptrs = x_ptr + pid_m * stride_x_m + (h_offs + H) * stride_x_h

    gate = tl.load(gate_ptrs, mask=mask_h, other=0.0).to(tl.float32)
    up = tl.load(up_ptrs, mask=mask_h, other=0.0).to(tl.float32)

    # Load grad_y
    gy_ptrs = grad_y_ptr + pid_m * stride_gy_m + h_offs * stride_gy_h
    grad_y = tl.load(gy_ptrs, mask=mask_h, other=0.0).to(tl.float32)

    # Recompute forward: y = silu(gate) * up
    # silu(gate) = gate * sigmoid(gate)
    sigmoid_gate = tl.sigmoid(gate)
    silu_gate = gate * sigmoid_gate
    y = silu_gate * up

    # Backward computation for SwiGLU:
    # d_up = grad_y * silu(gate)
    # d_gate = grad_y * up * (sigmoid(gate) + gate * sigmoid(gate) * (1 - sigmoid(gate)))
    #        = grad_y * up * sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
    d_up = grad_y * silu_gate
    d_gate = grad_y * up * sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))

    # --- Quantize grad_input [d_gate, d_up] per m-group (per-row, per-128-channel) ---
    # Each BLOCK_H is GROUP_SIZE (128), so each program block is one quantization group

    # Quantize d_gate group
    abs_d_gate = tl.abs(d_gate)
    max_d_gate = tl.max(abs_d_gate, axis=0)
    scale_d_gate = max_d_gate / 127.0
    scale_d_gate = tl.where(scale_d_gate == 0.0, 1.0, scale_d_gate)

    d_gate_q = d_gate / scale_d_gate
    # Clamp to [-127, 127] and round
    d_gate_q = tl.extra.cuda.libdevice.rint(d_gate_q)
    d_gate_q = tl.maximum(tl.minimum(d_gate_q, 127.0), -127.0)
    d_gate_q_i8 = d_gate_q.to(tl.int8)

    # Store d_gate quantized: grad_input_q[:, :H]
    giq_gate_ptrs = grad_input_q_ptr + pid_m * stride_giq_m + h_offs * stride_giq_h
    tl.store(giq_gate_ptrs, d_gate_q_i8, mask=mask_h)

    # Store scale for d_gate group
    # grad_input_s has shape [M, 2H/128], gate groups are at indices [0, H/128)
    gis_gate_ptr = grad_input_s_ptr + pid_m * stride_gis_m + pid_group * stride_gis_h
    tl.store(gis_gate_ptr, scale_d_gate)

    # Quantize d_up group
    abs_d_up = tl.abs(d_up)
    max_d_up = tl.max(abs_d_up, axis=0)
    scale_d_up = max_d_up / 127.0
    scale_d_up = tl.where(scale_d_up == 0.0, 1.0, scale_d_up)

    d_up_q = d_up / scale_d_up
    d_up_q = tl.extra.cuda.libdevice.rint(d_up_q)
    d_up_q = tl.maximum(tl.minimum(d_up_q, 127.0), -127.0)
    d_up_q_i8 = d_up_q.to(tl.int8)

    # Store d_up quantized: grad_input_q[:, H:]
    giq_up_ptrs = grad_input_q_ptr + pid_m * stride_giq_m + (h_offs + H) * stride_giq_h
    tl.store(giq_up_ptrs, d_up_q_i8, mask=mask_h)

    # Store scale for d_up group: offset by H/128 groups
    num_gate_groups = H // GROUP_SIZE
    gis_up_ptr = grad_input_s_ptr + pid_m * stride_gis_m + (pid_group + num_gate_groups) * stride_gis_h
    tl.store(gis_up_ptr, scale_d_up)

    # --- Quantize y transposed per K-group (per-128-token group) ---
    # y_q_t has shape [H, M], y_s_t has shape [H, M/128]
    # We need to store y[pid_m, h_offs] into y_q_t[h_offs, pid_m]
    # K-group quantization: groups of 128 tokens along M dimension
    # The scale is shared across 128 tokens for each h

    # We compute per-element: store y value and let a separate pass handle grouping
    # Actually, we need to handle the K-group (128 tokens) quantization.
    # Since each kernel instance handles one row, we need atomic max or a two-pass approach.
    # Instead, we'll compute the quantization scale across the K-group in a second kernel.
    # But the problem asks for a single fused kernel.

    # Alternative: We store y in bf16 to a temp buffer and quantize in groups.
    # But let's try: each program handles one (m, h_group). For K-group quant of y_t,
    # we need to group along M dimension (groups of 128 rows).
    # We can store y values into y_q_t after computing per-group scales.
    # Since we process one row at a time, we need cooperation across rows.

    # Approach: Use a two-phase within the same kernel launch isn't feasible for cross-row.
    # Let's use a separate small kernel for y quantization, or compute it inline
    # by having each block also handle the K-group it belongs to - but that requires
    # reading other rows' y values.

    # For the fused kernel, let's store y in a transposed manner and quantize per K-group.
    # Each thread block processes (pid_m, pid_group). For K-group quant, the group is
    # along M with size 128. So pid_m // 128 gives the K-group index.

    # We need to find the max across 128 tokens for each h channel.
    # This requires synchronization across 128 program instances - not possible in one kernel.

    # Practical approach: store y into y_q_t as fp32 temporarily, then quantize separately.
    # But the signature requires int8 output. So let's do a two-kernel approach within
    # the wrapper function.

    # For now, store y transposed as bf16-converted-to-int8 placeholder.
    # Actually, let's just store the raw y values transposed for now and handle
    # quantization in a second kernel launched from the wrapper.

    # Store y transposed (as float for the second kernel to quantize)
    # We'll use y_q_t as a temporary float storage - but it's int8.
    # We need an intermediate buffer or a second kernel.

    # Let's store y to a temporary location. Since we can't add buffers,
    # we'll launch a second kernel from the wrapper.
    # For this kernel, we just store y transposed in bf16 bits reinterpreted.

    # Actually, the cleanest approach: compute y, convert to bf16, store transposed
    # into a temp buffer, then second kernel quantizes. But we only have y_q_t (int8).

    # Let's just write y as bf16 reinterpreted as int8 pairs (2 bytes) - messy.
    # Better: use the wrapper to allocate a temp buffer.

    # For the main kernel, we'll store y values. The wrapper will handle the rest.
    # We'll pass a temp buffer pointer. But the signature is fixed.

    # SOLUTION: We'll handle y quantization in a separate kernel launched from wrapper.
    # This main kernel will store y transposed into a temp bf16 buffer.
    # But we don't have that buffer in the signature...

    # FINAL APPROACH: We'll do the y_q_t quantization inline by having each program
    # also be responsible for computing the scale for its K-group. This means we need
    # all 128 rows in the K-group to have written their values before we compute the scale.
    # We can't guarantee ordering across blocks.

    # The pragmatic solution: store y values to y_q_t and y_s_t using a block-level
    # approach where pid_m iterates over the K-group. But this changes the grid.

    # Let me reconsider the grid: we can make the grid (M // 128, H // 128) and have
    # each block process 128 rows x 128 columns, computing y for all of them,
    # then quantizing both grad_input per-row-group and y_t per-column(row)-group.

    # This is the right approach. Let me restructure.
    pass


@triton.jit
def _silu_dot_fwd_bwd_quant_fuse_kernel_v2(
    x_ptr,
    grad_y_ptr,
    grad_input_q_ptr,
    grad_input_s_ptr,
    y_q_t_ptr,
    y_s_t_ptr,
    M,
    H,
    stride_x_m,
    stride_gy_m,
    stride_giq_m,
    stride_gis_m,
    stride_yqt_h,
    stride_yst_h,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """
    Grid: (M // BLOCK_M, H // BLOCK_H) where BLOCK_M = BLOCK_H = GROUP_SIZE = 128
    Each block processes a tile of [BLOCK_M, BLOCK_H] for both gate and up channels.
    """
    pid_m = tl.program_id(0)  # which group of 128 rows
    pid_h = tl.program_id(1)  # which group of 128 columns

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)  # [BLOCK_H]

    mask_m = m_offs < M  # [BLOCK_M]
    mask_h = h_offs < H  # [BLOCK_H]
    mask_mh = mask_m[:, None] & mask_h[None, :]  # [BLOCK_M, BLOCK_H]

    # Compute 2D offsets for loading
    # x has shape [M, 2H], gate = x[:, :H], up = x[:, H:]
    x_gate_offs = m_offs[:, None] * stride_x_m + h_offs[None, :]  # [BLOCK_M, BLOCK_H]
    x_up_offs = m_offs[:, None] * stride_x_m + (h_offs[None, :] + H)

    gate = tl.load(x_ptr + x_gate_offs, mask=mask_mh, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + x_up_offs, mask=mask_mh, other=0.0).to(tl.float32)

    # Load grad_y [M, H]
    gy_offs = m_offs[:, None] * stride_gy_m + h_offs[None, :]
    grad_y = tl.load(grad_y_ptr + gy_offs, mask=mask_mh, other=0.0).to(tl.float32)

    # Recompute forward
    sigmoid_gate = tl.sigmoid(gate)
    silu_gate = gate * sigmoid_gate
    y = silu_gate * up  # [BLOCK_M, BLOCK_H]

    # Backward
    d_up = grad_y * silu_gate
    d_gate = grad_y * up * sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))

    # === Quantize grad_input per m-group (per-row, per-128-channel) ===
    # d_gate: [BLOCK_M, BLOCK_H], each row is one group
    abs_d_gate = tl.abs(d_gate)
    max_d_gate = tl.max(abs_d_gate, axis=1)  # [BLOCK_M]
    scale_d_gate = max_d_gate / 127.0
    scale_d_gate = tl.where(scale_d_gate == 0.0, 1.0, scale_d_gate)

    d_gate_q = d_gate / scale_d_gate[:, None]
    d_gate_q = tl.extra.cuda.libdevice.rint(d_gate_q)
    d_gate_q = tl.maximum(tl.minimum(d_gate_q, 127.0), -127.0)
    d_gate_q_i8 = d_gate_q.to(tl.int8)

    # Store d_gate quantized into grad_input_q[:, :H]
    giq_gate_offs = m_offs[:, None] * stride_giq_m + h_offs[None, :]
    tl.store(grad_input_q_ptr + giq_gate_offs, d_gate_q_i8, mask=mask_mh)

    # Store d_gate scale: grad_input_s[m, pid_h] (since each pid_h is one 128-channel group for gate)
    gis_gate_offs = m_offs * stride_gis_m + pid_h
    tl.store(grad_input_s_ptr + gis_gate_offs, scale_d_gate, mask=mask_m)

    # d_up quantization
    abs_d_up = tl.abs(d_up)
    max_d_up = tl.max(abs_d_up, axis=1)  # [BLOCK_M]
    scale_d_up = max_d_up / 127.0
    scale_d_up = tl.where(scale_d_up == 0.0, 1.0, scale_d_up)

    d_up_q = d_up / scale_d_up[:, None]
    d_up_q = tl.extra.cuda.libdevice.rint(d_up_q)
    d_up_q = tl.maximum(tl.minimum(d_up_q, 127.0), -127.0)
    d_up_q_i8 = d_up_q.to(tl.int8)

    # Store d_up quantized into grad_input_q[:, H:]
    giq_up_offs = m_offs[:, None] * stride_giq_m + (h_offs[None, :] + H)
    tl.store(grad_input_q_ptr + giq_up_offs, d_up_q_i8, mask=mask_mh)

    # Store d_up scale: offset by H // GROUP_SIZE groups
    num_gate_groups = H // GROUP_SIZE
    gis_up_offs = m_offs * stride_gis_m + (pid_h + num_gate_groups)
    tl.store(grad_input_s_ptr + gis_up_offs, scale_d_up, mask=mask_m)

    # === Quantize y transposed per K-group (per-128-token) ===
    # y has shape [BLOCK_M, BLOCK_H], K-group is along M dimension with size 128 = BLOCK_M
    # y_q_t has shape [H, M], y_s_t has shape [H, M // 128]
    # For each h channel, we quantize the 128 token values together

    abs_y = tl.abs(y)
    max_y = tl.max(abs_y, axis=0)  # [BLOCK_H] - max across M dimension (128 tokens)
    scale_y = max_y / 127.0
    scale_y = tl.where(scale_y == 0.0, 1.0, scale_y)

    y_q = y / scale_y[None, :]
    y_q = tl.extra.cuda.libdevice.rint(y_q)
    y_q = tl.maximum(tl.minimum(y_q, 127.0), -127.0)
    y_q_i8 = y_q.to(tl.int8)

    # Store y_q transposed: y_q_t[h, m] = y_q[m, h]
    yqt_offs = h_offs[:, None] * stride_yqt_h + m_offs[None, :]  # [BLOCK_H, BLOCK_M]
    mask_hm = mask_h[:, None] & mask_m[None, :]
    # Transpose y_q_i8 from [BLOCK_M, BLOCK_H] to [BLOCK_H, BLOCK_M]
    y_q_i8_t = tl.trans(y_q_i8)
    tl.store(y_q_t_ptr + yqt_offs, y_q_i8_t, mask=mask_hm)

    # Store y scale: y_s_t[h, pid_m] (one scale per K-group per channel)
    yst_offs = h_offs * stride_yst_h + pid_m
    tl.store(y_s_t_ptr + yst_offs, scale_y, mask=mask_h)


def silu_dot_fwd_bwd_quant_fuse(
    x: torch.Tensor,
    grad_y: torch.Tensor,
    grad_input_q: torch.Tensor,
    grad_input_s: torch.Tensor,
    y_q_t: torch.Tensor,
    y_s_t: torch.Tensor,
    group_size: int = 128,
):
    """
    Fused SiLU-dot recompute, backward, and INT8 quantization.

    Args:
        x: [M, 2H] bf16, packed [gate, up]
        grad_y: [M, H] bf16
        grad_input_q: [M, 2H] int8 output
        grad_input_s: [M, 2H//128] fp32 output
        y_q_t: [H, M] int8 output
        y_s_t: [H, M//128] fp32 output
        group_size: quantization group size, 128
    """
    M, two_H = x.shape
    H = two_H // 2

    assert group_size == 128
    assert M % group_size == 0, f"M={M} must be divisible by group_size={group_size}"
    assert H % group_size == 0, f"H={H} must be divisible by group_size={group_size}"

    BLOCK_M = group_size  # 128
    BLOCK_H = group_size  # 128

    grid = (M // BLOCK_M, H // BLOCK_H)

    stride_x_m = x.stride(0)
    stride_gy_m = grad_y.stride(0)
    stride_giq_m = grad_input_q.stride(0)
    stride_gis_m = grad_input_s.stride(0)
    stride_yqt_h = y_q_t.stride(0)
    stride_yst_h = y_s_t.stride(0)

    _silu_dot_fwd_bwd_quant_fuse_kernel_v2[grid](
        x,
        grad_y,
        grad_input_q,
        grad_input_s,
        y_q_t,
        y_s_t,
        M,
        H,
        stride_x_m,
        stride_gy_m,
        stride_giq_m,
        stride_gis_m,
        stride_yqt_h,
        stride_yst_h,
        GROUP_SIZE=group_size,
        BLOCK_M=BLOCK_M,
        BLOCK_H=BLOCK_H,
    )

    return grad_input_q, grad_input_s, y_q_t, y_s_t