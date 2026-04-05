# CURRENT STATUS

更新时间：2026-04-05（UTC）

## 1) 当前最强 Baseline

- 结论：`Baseline-ROI`（同口径 ROI 内 Dice）是当前最强基线。
- 证据（来自 `logs/training_baseline_roi_log.csv`）：
  - best `val_dice = 0.7841`（epoch 72）
  - 125e 末尾 `val_dice = 0.7814`（epoch 125）
  - 对应拓扑指标（best 点）：`val_cl_break = 7.5`，`val_delta_beta0 = 15.0`

## 2) 当前最佳 Topo

- 结论：`Topo-Fragment-Suppress`（`fragment_suppress`）是当前最佳拓扑路线。
- 证据（来自 `logs/fragment_suppress_125e.csv` 与 `experiments/FRAGMENT_SUPPRESS_125E_REPORT.md`）：
  - best `val_dice = 0.7618`（epoch 114）
  - 125e 末尾 `val_dice = 0.7610`（epoch 125）
  - best 点拓扑指标：`cl_break = 6.8`，`delta_beta0 = 15.6`

## 3) 当前一句话判断

- Topo 路线已经“可用且接近 baseline”，但尚未超过当前最强 baseline。
- 以 best 点对比：`0.7618 vs 0.7841`，Topo 仍落后 `0.0223` Dice。

## 4) 当前下一步唯一任务（指向 NEXT_ONE_THING）

- 唯一任务：在不改数据口径（Kaggle+FOV+ROI 内 Dice+125e）的前提下，只做 `fragment_suppress` 单变量收敛优化，目标把 Topo Dice 差距从 `0.0223` 缩小到 `<=0.0100`，且保持拓扑指标不退化。

## 30 秒口播版

- 我们的最强 baseline 是 `Baseline-ROI`，best Dice 0.7841。
- 我们的最佳 topo 是 `Fragment-Suppress`，best Dice 0.7618，拓扑指标和 baseline 基本同级。
- 现在只做一件事：固定口径、单变量优化 `fragment_suppress`，先把 Dice 差距压到 1 个点以内。
