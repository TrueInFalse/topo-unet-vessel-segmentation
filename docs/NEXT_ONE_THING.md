# NEXT ONE THING
2026.4.5
## 唯一任务

在完全固定主线口径（Kaggle 联合数据、`roi_mode=fov`、ROI 内 Dice、125 epoch、同随机种子）下，
仅对 `fragment_suppress` 做 **单变量优化**，把 Topo 与最强 Baseline 的 Dice 差距从 `0.0223` 缩小到 `<= 0.0100`。

## 任务边界（今天不做新分支）

- 不改数据集
- 不改评估口径
- 不引入新 loss 形态
- 不并行开新实验线

## 完成判据（Pass/Fail）

- Pass：存在 Topo 运行满足
  - `val_dice >= 0.7741`
  - `cl_break <= 8.0`
  - `delta_beta0 <= 16.0`
- Fail：未达到上述阈值，则回滚到当前最佳 `fragment_suppress` 配置作为稳定主线

## 执行顺序（单线程）

1. 以当前最佳 `fragment_suppress` 配置为起点（best Dice 0.7618）。
2. 只改一个参数并完成 125e 训练。
3. 与 `Baseline-ROI best=0.7841` 做同表对比，决定保留/回滚。

## 为什么是这一个

- Baseline 已经够强（0.7841），主矛盾只剩 Topo 的最后 2.23 个 Dice 点。
- Topo 主线已明确为 `fragment_suppress`，继续分叉只会增加混乱。
