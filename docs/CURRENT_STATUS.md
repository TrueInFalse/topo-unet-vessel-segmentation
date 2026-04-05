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
- 证据（来自 `logs/fragment_suppress_125e.csv` 与 `reports/FRAGMENT_SUPPRESS_125E_REPORT.md`）：
  - best `val_dice = 0.7618`（epoch 114）
  - 125e 末尾 `val_dice = 0.7610`（epoch 125）
  - best 点拓扑指标：`cl_break = 6.8`，`delta_beta0 = 15.6`

## 3) 当前一句话判断

- Topo 路线已经“可用且接近 baseline”，但尚未超过当前最强 baseline。
- 以 best 点对比：`0.7618 vs 0.7841`，Topo 仍落后 `0.0223` Dice。

## 4) 当前下一步唯一任务（指向 NEXT_ONE_THING）

- 唯一任务：在不改数据口径（Kaggle+FOV+ROI 内 Dice+125e）的前提下，只做 `fragment_suppress` 单变量收敛优化，目标把 Topo Dice 差距从 `0.0223` 缩小到 `<=0.0100`，且保持拓扑指标不退化。

## 5) 主线拓扑损失实现状态（已扶正）

- 当前默认拓扑损失实现：`topology_loss_fragment_suppress.py`
- `train_topo_roi.py` 已改为直接依赖该主线实现
- `topology_loss_ablation.py` 为历史消融保留文件，不再作为默认主线依赖

## 6) 入口一致性状态（已修正）

- `train_topo_roi.py` 已支持 `--config` 参数（默认读取 `config.yaml`）。
- `--epochs` 默认值已改为 `None`，仅显式传入时覆盖 YAML 的 `training.max_epochs`。
- `--loss-mode` 默认值已改为 `fragment_suppress`（其他值仅兼容提示，不作为主线分支）。
- 规则文档：`docs/TRAIN_TOPO_ENTRY_RULES.md`。

## 30 秒口播版

- 最强 baseline 是 `Baseline-ROI`，best Dice 0.7841。
- 最佳 topo 是 `Fragment-Suppress`，best Dice 0.7618，拓扑指标和 baseline 基本同级。
- 主线拓扑损失已扶正为 `topology_loss_fragment_suppress.py`；`topology_loss_ablation.py` 仅保留复盘。
