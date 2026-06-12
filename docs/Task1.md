## Task 01: w4a8_group_gemm_moe（W4A8 分组 GEMM MoE）

MoE 推理中的量化矩阵乘法。对每个 expert，将 INT4 打包权重解包为 INT8，与 INT8 激活做分组矩阵乘，按 group 应用 weight scale 后累加，最终乘以 per-token activation scale，输出 bf16。

### 接口签名

```python
def w4a8_group_gemm_moe(
    x_q: torch.Tensor,           # [M_total, K], int8
    x_scale: torch.Tensor,       # [M_total], fp32
    w_q4_packed: torch.Tensor,   # [E, N, K//2], uint8
    w_scale: torch.Tensor,       # [E, N, K//group_size], fp32
    w_zero: torch.Tensor,        # [E, N, K//group_size], int8
    expert_offsets: torch.Tensor,# [E+1], int32
    out: torch.Tensor,           # [M_total, N], bf16
    group_size: int,             # 64 or 128
) -> torch.Tensor
```

### 参数

- `E`: expert 数量（4～64）
- `M_total`: 所有 expert 的 token 总数
- `N`: expert 输出维度（512～2048）
- `K`: expert 输入维度（512～4096）
- `group_size`: INT4 量化分组大小（64 or 128）

### 测试 Workload（10 configs）

| E | M_total | N | K | group_size |
|---|---------|---|---|-----------|
| 4 | 32 | 512 | 512 | 128 |
| 8 | 64 | 1024 | 1024 | 128 |
| 8 | 128 | 1408 | 2048 | 128 |
| 8 | 256 | 2048 | 2048 | 128 |
| 16 | 128 | 1024 | 1536 | 64 |
| 16 | 256 | 1536 | 2048 | 128 |
| 32 | 256 | 1408 | 2048 | 128 |
| 32 | 512 | 2048 | 4096 | 128 |
| 64 | 512 | 1024 | 2048 | 128 |
| 64 | 1024 | 2048 | 4096 | 128 |

### 正确性

`gems_assert_close`（rtol=0.016, atol=1e-4*N，针对 bf16 输出）

### 参考实现

```python
import torch


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
    """
    W4A8 Group GEMM (MoE-style) référence implementation.

    Per-expert GEMM where weights are stored in INT4 (per-group asymmetric
    quantization) and activations are stored in INT8 (per-token symmetric
    quantization). The target deployment hardware only exposes INT8 tensor /
    matrix cores, so the INT4 weights must be unpacked / dequantized to INT8
    before being fed to the INT8 MMA pipeline.
    """
    M_total, K = x_q.shape
    E, N, K_half = w_q4_packed.shape
    assert K_half * 2 == K
    assert K % group_size == 0
    G = K // group_size

    # Unpack INT4 weights: low nibble = even-K, high nibble = odd-K
    low = (w_q4_packed & 0x0F).to(torch.int16)
    high = ((w_q4_packed >> 4) & 0x0F).to(torch.int16)
    w_int4 = torch.stack([low, high], dim=-1).reshape(E, N, K)

    # Fold per-group zero-points into INT8 operand
    w_zero_expanded = w_zero.to(torch.int16).repeat_interleave(group_size, dim=-1)
    w_signed = (w_int4 - w_zero_expanded).to(torch.int8)

    # Per-expert grouped GEMM
    for g in range(E):
        m_start = int(expert_offsets[g].item())
        m_end = int(expert_offsets[g + 1].item())
        if m_end <= m_start:
            continue

        Xq_g = x_q[m_start:m_end]
        xs_g = x_scale[m_start:m_end]
        Wq_g = w_signed[g]
        ws_g = w_scale[g]

        # INT8 * INT8 -> INT32 GEMM, split per group along K
        Xq_g_grp = Xq_g.reshape(-1, G, group_size).to(torch.float32)
        Wq_g_grp = Wq_g.reshape(N, G, group_size).to(torch.float32)

        # Per-group partial sums: einsum "mgk,ngk->mng"
        partial = torch.einsum("mgk,ngk->mng", Xq_g_grp, Wq_g_grp)

        # Apply per-group weight scale and reduce
        acc = (partial * ws_g.unsqueeze(0)).sum(dim=-1)

        # Apply per-token activation scale
        acc = acc * xs_g.unsqueeze(-1)

        out[m_start:m_end].copy_(acc.to(out.dtype))

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
