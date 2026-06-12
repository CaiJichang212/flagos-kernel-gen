## Task 03: silu_dot_fwd_bwd_quant_fuse（SwiGLU 反向融合量化）

MoE 训练反向传播中的融合算子。重算 `y = silu(gate) * up`，计算 SwiGLU 的梯度 `[d_gate, d_up]`，然后将梯度按 m-group（per-row per-128-channel）量化为 INT8，将 y 转置后按 K-group（per-128-token）量化为 INT8，供下游 INT8 GEMM 使用。

### 接口签名

```python
def silu_dot_fwd_bwd_quant_fuse(
    x: torch.Tensor,              # [M, 2H], bf16
    grad_y: torch.Tensor,         # [M, H], bf16
    grad_input_q: torch.Tensor,   # [M, 2H], int8 (output)
    grad_input_s: torch.Tensor,   # [M, 2H/128], fp32 (output)
    y_q_t: torch.Tensor,          # [H, M], int8 (output)
    y_s_t: torch.Tensor,          # [H, M/128], fp32 (output)
    group_size: int = 128,        # 固定为 128
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

### 参数

- `M = E * T`：flattened expert-token rows
- `H`：hidden dimension（2560 or 4096）
- `x`：打包的 `[gate, up]`
- `grad_y`：上游梯度（对 `y = silu(gate) * up` 的梯度）
- 输出：`grad_input_q/s`（m-group 量化）+ `y_q_t/s_t`（K-group 量化）

### 测试 Workload（12 configs）

| num_experts | tokens_per_expert | H |
|------------|------------------|---|
| 8 | 128 | 2560 |
| 8 | 256 | 2560 |
| 16 | 128 | 2560 |
| 16 | 256 | 2560 |
| 32 | 128 | 2560 |
| 32 | 256 | 2560 |
| 8 | 128 | 4096 |
| 8 | 256 | 4096 |
| 16 | 128 | 4096 |
| 16 | 256 | 4096 |
| 32 | 128 | 4096 |
| 32 | 256 | 4096 |

### 正确性

- FP32 scale: `assert_close(atol=1e-4, rtol=1e-5)`
- INT8 反量化值: `assert_close(atol=0.25, rtol=0.25)`

### 参考实现

```python
import torch


GROUP_SIZE = 128


def _quantize_rows(x):
    """Per-128-group symmetric INT8 quantization (m-group for 1d2d GEMM)."""
    M, hidden_dim = x.shape
    grouped = x.reshape(M, hidden_dim // GROUP_SIZE, GROUP_SIZE)
    absmax = grouped.abs().amax(dim=2)
    scales = torch.clamp(absmax / 127.0, min=1.0e-10)
    x_q = torch.clamp(
        grouped / scales[:, :, None], -127.0, 127.0,
    ).reshape(M, hidden_dim).to(torch.int8)
    return x_q, scales


def _quantize_transposed_k_group(y):
    """Per-128-token-group symmetric INT8 quantization on y.T (K-group for 1d1d GEMM)."""
    M, H = y.shape
    y_t = y.T.contiguous()
    grouped = y_t.reshape(H, M // GROUP_SIZE, GROUP_SIZE)
    absmax = grouped.abs().amax(dim=2)
    scales = torch.clamp(absmax / 127.0, min=1.0e-10)
    y_q_t = torch.clamp(
        grouped / scales[:, :, None], -127.0, 127.0,
    ).reshape(H, M).to(torch.int8)
    return y_q_t, scales


def silu_dot_fwd_bwd_quant_fuse(
    x: torch.Tensor,
    grad_y: torch.Tensor,
    grad_input_q: torch.Tensor,
    grad_input_s: torch.Tensor,
    y_q_t: torch.Tensor,
    y_s_t: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused SiLU-dot recompute, backward, and INT8 quant.

    Targets the MoE training backward path for SwiGLU-style expert FFN blocks.
    Recomputes y = silu(gate) * up, computes grad_input = [d_gate, d_up], then
    quantizes both for downstream INT8 GEMMs.
    """
    assert group_size == 128
    M, two_h = x.shape
    H = two_h // 2

    gate = x[:, :H].to(torch.float32)
    up = x[:, H:].to(torch.float32)
    grad_y_f32 = grad_y.to(torch.float32)

    # Forward recompute
    sigmoid = torch.sigmoid(gate)
    silu = gate * sigmoid
    y = silu * up

    # Backward
    d_up = grad_y_f32 * silu
    d_gate = grad_y_f32 * up * sigmoid * (1.0 + gate * (1.0 - sigmoid))
    grad_input = torch.cat([d_gate, d_up], dim=1).to(torch.bfloat16)

    # M-group INT8 quantization
    gi_q, gi_s = _quantize_rows(grad_input.to(torch.float32))
    grad_input_q.copy_(gi_q)
    grad_input_s.copy_(gi_s)

    # K-group INT8 quantization
    yq, ys = _quantize_transposed_k_group(y.to(torch.bfloat16).to(torch.float32))
    y_q_t.copy_(yq)
    y_s_t.copy_(ys)

    return grad_input_q, grad_input_s, y_q_t, y_s_t
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
