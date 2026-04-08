# Mainline Results Summary

**更新时间**: 2026-04-08（UTC）  
**说明**: 该文件只记录当前主线严格 125e 对照的统一数字。默认口径为 Kaggle 联合数据、`roi_mode=fov`、ROI 内 Dice、同随机种子、同配置 125e。

## 1. 主线结果摘要表（best 点）

| 指标 | Baseline-ROI 125e (Best) | Topo-ROI Fragment-Suppress 125e (Best) | Gap (Topo - Baseline) |
|---|---:|---:|---:|
| Epoch | 72 | 119 | — |
| Dice | 0.7841 | 0.7615 | -0.0226 |
| IoU | 0.6466 | 0.6175 | -0.0291 |
| Precision | 0.7882 | 0.7685 | -0.0197 |
| Recall | 0.7823 | 0.7572 | -0.0251 |
| CL-Break | 7.5 | 7.8 | +0.3 |
| Δβ₀ | 15.0 | 15.4 | +0.4 |

## 2. 主线结果摘要表（final / epoch 125）

| 指标 | Baseline-ROI 125e (Final) | Topo-ROI Fragment-Suppress 125e (Final) | Gap (Topo - Baseline) |
|---|---:|---:|---:|
| Epoch | 125 | 125 | — |
| Dice | 0.7814 | 0.7611 | -0.0203 |
| IoU | 0.6429 | 0.6170 | -0.0259 |
| Precision | 0.7978 | 0.7667 | -0.0311 |
| Recall | 0.7680 | 0.7582 | -0.0098 |
| CL-Break | 6.8 | 7.8 | +1.0 |
| Δβ₀ | 14.8 | 15.7 | +0.9 |

## 3. 证据来源

- Baseline 日志：`logs/20260405_baseline_roi_125e.csv`
- Topo 日志：`logs/20260405_topo_roi_fragment_suppress_125e.csv`
- 曲线图文件名：`baseline_roi_125e.png`、`topo_fragment_suppress_125e.png`
- 仓库当前对应图：`results/20260405_baseline_roi_125e.png`、`results/20260405_topo_roi_fragment_suppress_125e.png`

## 4. 一句话结论

当前主线结论：Fragment-Suppress 已修正 Standard topo loss 的方向性问题，但在严格同配置 125e 对照下，Dice 仍落后 Baseline 约 0.02，拓扑指标未形成稳定优势。
