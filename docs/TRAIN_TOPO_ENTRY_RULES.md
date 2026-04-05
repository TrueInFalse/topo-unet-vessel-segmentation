# TRAIN TOPO ENTRY RULES

更新时间：2026-04-05（UTC）

## 1) 默认拓扑损失实现

- 当前默认主线实现：`topology_loss_fragment_suppress.py`
- 历史消融保留文件：`topology_loss_ablation.py`（不作为默认主线依赖）

## 2) 默认读取的配置文件

- 脚本：`train_topo_roi.py`
- 参数：`--config`
- 默认值：`config.yaml`

说明：不传 `--config` 时，脚本读取 `config.yaml`；传入后按显式路径读取。

## 3) epochs 优先级规则

优先级：

1. 命令行 `--epochs`（仅在显式传入时生效）
2. YAML 中的 `training.max_epochs`

规则细化：

- `--epochs` 默认值为 `None`
- 当 `--epochs` 为 `None` 时，不覆盖 YAML
- 仅当用户显式传入 `--epochs` 时，才覆盖 `training.max_epochs`

## 4) loss_mode 默认值

- 参数：`--loss-mode`
- 默认值：`fragment_suppress`
- 兼容可选值：`standard`、`main_component`、`fragment_suppress`
- 主线行为：训练时固定使用 `fragment_suppress`（非主线值仅作兼容提示）

## 5) 当前推荐运行命令

推荐主线（显式固定 125e 配置）：

```bash
python train_topo_roi.py --config config_125e.yaml
```

可选（临时覆盖轮数）：

```bash
python train_topo_roi.py --config config_125e.yaml --epochs 150
```

可选（临时切换 loss_mode 做对照入口兼容）：

```bash
python train_topo_roi.py --config config_125e.yaml --loss-mode standard
```

