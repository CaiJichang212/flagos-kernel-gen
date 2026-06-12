# Task1 v1.6 Optimization Report

## Platform Targets

| 平台 | 历史最好 | 排行榜参考 | v1.6 目标 | v1.6 路线 |
| --- | ---: | ---: | ---: | --- |
| 华为昇腾 | 2.05 | 2.89 | 先恢复通过，争取 >=2.0 | 早期已通过的转置权重 INT8 dot |
| 天数 | 9.15 | 15.21 | >=10 | v1.5 固定 INT8 dot |
| 海光 | 20.92 | 27.89 | >=22 | v1.5 固定 INT8 dot |
| 摩尔线程 | 1.78 | 4.65 | >=2 | v1.5 float16 dot 兼容路线 |
| 沐曦 | 18.30 | 45.27 | >=20 | v1.5 固定 INT8 dot |
| 平头哥 | 7.55 | 23.18 | >=8 | v1.4 固定 INT8 dot |
| 国际通用芯片 | 16.47 | 36.20 | >=18 | v1.4 固定 INT8 dot |

## MCP Iterations

预检：`autotune_kernel` 在 NVIDIA/international 目标上未能生成初始 Triton 代码，`generate_kernel` 一次性生成请求超时，因此正式迭代改用历史代码作为输入。

| 轮次 | 工具 | 目标 | 结果 | 采用情况 |
| ---: | --- | --- | --- | --- |
| 1 | optimize_kernel | NVIDIA/international | 返回 autotune 候选 | 未采用，历史 v1.5 的 autotune 路线曾导致国际通用失败 |
| 2 | optimize_kernel | Huawei Ascend | 返回转置权重整理候选 | 参考，最终保留历史已通过的更保守版本 |
| 3 | optimize_kernel | Moore Threads | 返回更激进 float16 候选 | 未采用，含早退和更大 BLOCK_N，兼容风险高 |
| 4 | optimize_kernel | Haiguang/general INT8 | 返回 flatten M/N 候选 | 未采用，跨厂商稳定性不如历史固定配置 |
| 5 | specialize_kernel | Huawei Ascend | 返回 1D grid Ascend 候选 | 参考，未直接采用以避免 JIT 早退和 assert 风险 |

## Key Changes

- `all.py` 补充 `iluvatar`、`cambricon`、`kunlunxin` 等真实 vendor 名，避免国际通用平台误走 NVIDIA autotune 路线。
- `all.py` 默认和国际通用路径使用固定 INT8 dot，不再使用 v1.5 的 NVIDIA autotune 分支。
- `all.py` 昇腾路径切换到早期提交中曾通过昇腾的转置权重加载方式。
- 天数、海光、沐曦、摩尔平台文件强制调用对应平台路线，减少 vendor 检测失败风险。
