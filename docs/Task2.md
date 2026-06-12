## Task 02: c128_256_512_compress（DeepSeek V4 KV 压缩）

DeepSeek V4 长上下文推理中的 KV cache 压缩算子。将连续 128/256/512 个 token 的 KV 状态通过 softmax 加权求和压缩为 1 个 entry，经过 RMSNorm 和 RoPE 后，前 448 维 INT8 量化存储，后 64 维 bf16 存储，写入分页 KV cache。

### 接口签名

```python
def c128_256_512_compress(
    state_cache: torch.Tensor,       # [num_blocks, block_size, 2*head_dim], fp32
    token_to_req: torch.Tensor,      # [num_tokens], int32
    positions: torch.Tensor,         # [num_tokens], int64
    boundary_token_indices: torch.Tensor,  # [num_outputs], int64
    block_table: torch.Tensor,       # [num_reqs, blocks_per_req], int32
    rms_norm_weight: torch.Tensor,   # [head_dim], bf16
    cos_sin_cache: torch.Tensor,     # [max_pos, rope_head_dim], fp32
    kv_slot_mapping: torch.Tensor,   # [num_tokens], int64
    kv_cache: torch.Tensor,          # [kv_blocks, block_stride], uint8
    block_size: int,                 # 8
    compress_ratio: int,             # 128/256/512
    rms_norm_eps: float = 1e-6,
) -> torch.Tensor
```

### 参数

- `head_dim`: 512（固定）
- `rope_head_dim`: 64（RoPE 应用到最后 64 维）
- `compress_ratio`: 128/256/512
- `block_size`: state_cache 分页大小（8）
- `kv_block_size`: KV cache 分页大小（64）

### 测试 Workload（12 configs）

| num_reqs | total_tokens | compress_ratio |
|----------|-------------|----------------|
| 1 | 8192 | 128 |
| 4 | 32768 | 128 |
| 8 | 65536 | 128 |
| 8 | 131072 | 128 |
| 1 | 8192 | 256 |
| 4 | 32768 | 256 |
| 8 | 65536 | 256 |
| 8 | 131072 | 256 |
| 1 | 8192 | 512 |
| 4 | 32768 | 512 |
| 8 | 65536 | 512 |
| 8 | 131072 | 512 |

### 正确性

- INT8 量化值 + scale bytes: `torch.equal`（精确匹配）
- RoPE bf16 部分: `assert_close(atol=1e-2, rtol=1e-2)`

### 参考实现

```python
import torch
import torch.nn.functional as F


HEAD_DIM = 512
ROPE_HEAD_DIM = 64
NOPE_HEAD_DIM = HEAD_DIM - ROPE_HEAD_DIM
KV_BLOCK_SIZE = 64
TOKEN_STRIDE = 576
SCALE_DIM = 8


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
    C128/C256/C512 DeepSeek-style KV compressor.
    Single-pass gather + batched softmax/norm/quant/rope + scatter.
    """
    device = state_cache.device
    out = torch.zeros_like(kv_cache)
    num_outputs = boundary_token_indices.numel()
    if num_outputs == 0:
        return out

    # 1. Compute flat gather indices
    boundary_pos = positions[boundary_token_indices]
    boundary_req = token_to_req[boundary_token_indices]

    offsets = torch.arange(compress_ratio, device=device)
    all_pos = (boundary_pos - compress_ratio + 1).unsqueeze(1) + offsets.unsqueeze(0)

    block_numbers_idx = all_pos // block_size
    block_offsets = all_pos % block_size
    req_expanded = boundary_req.unsqueeze(1).expand_as(block_numbers_idx)
    block_numbers = block_table[req_expanded, block_numbers_idx]

    flat_idx = (block_numbers * block_size + block_offsets).reshape(-1)

    # 2. Gather + softmax weighted sum
    sc_flat = state_cache.reshape(-1, 2 * HEAD_DIM)
    all_rows = sc_flat[flat_idx].reshape(num_outputs, compress_ratio, 2 * HEAD_DIM)

    kv_vals = all_rows[:, :, :HEAD_DIM]
    scores = all_rows[:, :, HEAD_DIM:]
    compressed = (kv_vals * F.softmax(scores, dim=1)).sum(dim=1)

    # 3. RMSNorm
    rrms = torch.rsqrt(compressed.square().mean(dim=-1, keepdim=True) + rms_norm_eps)
    rms_w = rms_norm_weight.to(torch.float32)
    normed = compressed * rrms * rms_w.unsqueeze(0)

    # 4. INT8 quantization (first 448 dims, per-64-block)
    nope_data = normed[:, :NOPE_HEAD_DIM].to(torch.bfloat16).to(torch.float32)
    num_qb = NOPE_HEAD_DIM // 64
    nope_blocks = nope_data.reshape(num_outputs, num_qb, 64)
    amax = nope_blocks.abs().amax(dim=-1).clamp(min=1.0e-4)
    exponent = torch.ceil(torch.log2(amax * (1.0 / 127.0)))
    inv_scale = torch.exp2(-exponent)
    q = (nope_blocks * inv_scale.unsqueeze(-1)).clamp(-127.0, 127.0).to(torch.int8)
    value_bytes = q.reshape(num_outputs, NOPE_HEAD_DIM).view(torch.uint8)
    scale_u8 = (exponent + 127).clamp(0, 255).to(torch.uint8)

    # 5. RoPE (GPT-J interleaved)
    rope_pairs = ROPE_HEAD_DIM // 2
    pairs = normed.reshape(num_outputs, HEAD_DIM // 2, 2)
    rope_even = pairs[:, -rope_pairs:, 0]
    rope_odd = pairs[:, -rope_pairs:, 1]

    compressed_pos = (boundary_pos // compress_ratio) * compress_ratio
    cos_v = cos_sin_cache[compressed_pos, :rope_pairs]
    sin_v = cos_sin_cache[compressed_pos, rope_pairs:]

    rot_even = rope_even * cos_v - rope_odd * sin_v
    rot_odd = rope_odd * cos_v + rope_even * sin_v
    rotated = torch.stack([rot_even, rot_odd], dim=-1).reshape(num_outputs, ROPE_HEAD_DIM)
    rope_bytes = rotated.to(torch.bfloat16).view(torch.uint8).reshape(num_outputs, ROPE_HEAD_DIM * 2)

    # 6. Scatter into KV cache
    kv_slots = kv_slot_mapping[boundary_token_indices]
    pages = (kv_slots // KV_BLOCK_SIZE).long()
    slot_offsets = kv_slots % KV_BLOCK_SIZE
    value_bases = slot_offsets * TOKEN_STRIDE
    scale_bases = KV_BLOCK_SIZE * TOKEN_STRIDE + slot_offsets * SCALE_DIM

    payload = torch.cat([value_bytes, rope_bytes], dim=1)
    col_offsets_payload = torch.arange(TOKEN_STRIDE, device=device)
    payload_cols = value_bases.unsqueeze(1) + col_offsets_payload.unsqueeze(0)
    page_idx = pages.unsqueeze(1).expand_as(payload_cols)
    out[page_idx, payload_cols.long()] = payload

    scale_padded = torch.zeros(num_outputs, SCALE_DIM, dtype=torch.uint8, device=device)
    scale_padded[:, :num_qb] = scale_u8
    col_offsets_s = torch.arange(SCALE_DIM, device=device)
    scale_cols = scale_bases.unsqueeze(1) + col_offsets_s.unsqueeze(0)
    out[pages.unsqueeze(1).expand_as(scale_cols), scale_cols.long()] = scale_padded

    return out
```
## 评分标准

### 1. 正确性测试（Correctness Test）
参赛代码须通过正确性校验，校验合格方可参与性能成绩核算及最终排名；未通过正确性测试的代码，不予纳入排名序列。

### 2. 加速比测试（Speedup Test）
以 Speedup（加速比）为核心性能指标，综合华为昇腾、摩尔、天数、平头哥、海光、沐曦、国际通用芯片 7 款芯片平台的平均加速比进行排名。加速比的基准（baseline）为官方给出的 reference implementation。

## 评测维度

1. 按正确性测试通过的芯片数量，从高到低排序（全芯片优先）
2. 正确性测试芯片通过数量相同时，按平均加速比从高到低排序
3. 正确性测试未通过的芯片不参与性能比较
4. 作弊无成绩，参赛代码核心计算逻辑必须完全基于 Triton 或 Triton-TLE 实现。严禁通过 try-catch 异常捕获、分支判断等方式，在 Triton 执行失败时兜底调用 PyTorch 内置算子；若代码实际执行路径未运行 Triton 自定义算子，全程仅使用 PyTorch 内置算子，不计成绩，不参与排名；若采用异常捕获、条件分支等手段规避 Triton 执行、fallback 至 Torch 原生算子，一经判定为作弊，直接取消参赛成绩与排名资格
5. 分数相同者，按提交时间排序，提交时间在前者获奖
