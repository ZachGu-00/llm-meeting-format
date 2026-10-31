# E1 面板与 E4 选模

已完整校验 E1：10/10；E4 可用模型：9。
选模状态：frozen。本报告不代表 E2 或 E4 实验已完成。

| 模型 | 结构 | E0 读数层级 | 主 C / 95% CI | 主样本 | 匹配 R0 / R1 C | E1 状态 | E4 可用 |
|---|---|---|---|---:|---|---|---|
| LiquidAI/LFM2.5-1.2B-Instruct | convolution_and_attention | R0_primary | 0.002423 [-0.07057, 0.0832] | 180 | 0.1022 / 0.1177 | complete_verified | 是 |
| Qwen/Qwen3.5-4B | linear_and_full_attention | R0_primary | -0.4231 [-0.6093, -0.2462] | 180 | -0.6361 / -0.5896 | complete_verified | 是 |
| Qwen/Qwen3.5-2B | linear_and_full_attention | R1_primary_R0_operational | -0.2809 [-0.4332, -0.1176] | 200 | -0.3805 / -0.3664 | complete_verified | 是 |
| Qwen/Qwen3.5-0.8B | linear_and_full_attention | R1_only | — | — | — / — | complete_verified | R1_only |
| openbmb/MiniCPM5-2B | attention | R0_primary | -0.2728 [-0.6532, 0.04438] | 200 | -0.4343 / -0.4216 | complete_verified | 是 |
| ibm-granite/granite-4.2-3b | attention | R0_primary | -0.1019 [-0.24, 0.01906] | 200 | -0.103 / 0.161 | complete_verified | 是 |
| HuggingFaceTB/SmolLM3-3B | attention | R1_primary_R0_operational [conditional-q] | -0.4939 [-0.6564, -0.3285] | 200 | -0.5743 / -0.082 | complete_verified | 是 |
| microsoft/Phi-4-mini-instruct | attention | R0_primary | -0.9499 [-1.369, -0.6155] | 200 | -1.119 / -0.3956 | complete_verified | 是 |
| google/gemma-4-E2B-it | attention | R1_primary_R0_operational | -0.2845 [-0.6245, 0.02068] | 200 | -0.478 / -0.2836 | complete_verified | 是 |
| mistralai/Ministral-3-3B-Instruct-2512-BF16 | attention | R0_primary [conditional-q] | 0.1022 [-0.1484, 0.3586] | 200 | -0.1326 / 0.09281 | complete_verified | 是 |

Qwen4/LFM 的主 C 来自未见的 180 条，匹配读数诊断来自 44 条；其他模型对应 200/64。R1-primary 且 R0 操作门通过的模型保留资格并带标记；R1-only 不进入 E4。

仅两个事先提出的 H1 假设做 Holm 校正；其他模型的 C 及类型解释属于探索性结果。

| 预设模型 | 假设 | 原 p | Holm p | 状态 |
|---|---|---:|---:|---|
| Qwen/Qwen3.5-4B | C < 0 | 9.999e-05 | 0.0002 | resolved |
| LiquidAI/LFM2.5-1.2B-Instruct | abs(C) < 0.1 | 0.006999 | 0.006999 | resolved |

E4 冻结名单：

- Qwen/Qwen3.5-4B
- LiquidAI/LFM2.5-1.2B-Instruct
- ibm-granite/granite-4.2-3b
- openbmb/MiniCPM5-2B
- google/gemma-4-E2B-it
- microsoft/Phi-4-mini-instruct

结构依据已保存且与运行 SHA 一致的官方 config；Qwen3.5 的 linear/full attention 混合层不会误标成纯注意力。
未完成、未同步、未定义 C 和运行完整性问题详列于同名 JSON；不会据此伪造完成状态。
