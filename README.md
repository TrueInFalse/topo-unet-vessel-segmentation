# Retina PH Seg

基于 U-Net 与持续同调拓扑约束的视网膜血管分割项目。

> 面向毕业设计场景：在保证像素级精度的同时，尽量改善血管连通性与碎片化问题。

---

## 项目背景

视网膜血管分割是高血压、糖尿病等慢病筛查中的关键环节。传统像素级优化（如 Dice）通常能得到可接受的重叠指标，但在细血管区域容易出现断裂、碎片化和连通性不稳定。

本项目在 U-Net 框架下引入持续同调相关拓扑正则，目标是在可复现工程流程下比较 Baseline 与 Topo 路线的真实收益。

---

## 任务介绍

核心任务：二分类视网膜血管分割（前景=血管，背景=非血管）。

当前默认对照口径：

- ROI 对齐（训练/验证口径一致）
- Kaggle 联合数据模式
- 统一阈值与统一日志口径

---

## 当前状态（主线结论）

- 当前最强结果：`Baseline-ROI 125e`（`best val_dice = 0.7841`，epoch 72；125e 末尾约 `0.7814`）
- 当前最佳 Topo：`Fragment-Suppress`（125e，`best val_dice = 0.7618`，epoch 114）
- 当前结论：修正 PD→loss 方向后，Topo 已接近可用；但在严格 125e 对照下仍略低于 Baseline

一句话：当前主线是“强 Baseline + 可用但略落后的 Topo”。

---

## 当前主线

默认只看以下入口：

- 训练脚本：`train_baseline_roi.py`、`train_topo_roi.py`
- 配置文件：`config.yaml`（主配置母版）、`config_125e.yaml`（125e 主线配置）
- 主线拓扑模块：`topology_loss_fragment_suppress.py`（默认）
- 主线数据加载：`data_combined.py`

`train_topo_roi.py` 当前入口规则：

- 默认读取：`config.yaml`（可用 `--config` 指定）
- `--epochs` 默认 `None`，仅显式传入才覆盖 YAML
- `--loss-mode` 默认 `fragment_suppress`
- 详细规则：见 `docs/TRAIN_TOPO_ENTRY_RULES.md`

历史文件说明（仅复盘）：

- `legacy/topology_loss_ablation.py` 为历史消融保留文件，不作为默认主线依赖
- 其他历史入口已归档到 `legacy/` 或 `configs/archive/`

---

## 数据集说明

项目支持两种数据模式：

1. 纯 DRIVE 模式
- 训练：21–36
- 验证：37–40
- 测试：01–20
- ROI：原始 FOV mask

2. Kaggle 联合数据模式（当前主线默认）
- 来源：DRIVE + HRF + CHASE DB1 + STARE
- 训练集：`Training/`
- 验证集：`Test/`
- 无标签测试：`Unlabeled_test/`
- ROI：由配置与加载器统一生成

---

## 指标说明

像素级指标：

- Dice
- IoU
- Precision
- Recall

结构级指标：

- CL-Break（中心线碎片数，越低越好）
- Δβ₀（连通分量差异，越低越好）

---

## 快速开始（主线默认）

### 1) 环境准备

```bash
pip install torch torchvision
pip install segmentation-models-pytorch
pip install scikit-image scipy matplotlib tqdm pyyaml
pip install cripser
```

### 2) 配置文件

- 日常主配置：`config.yaml`
- 主线 125e：`config_125e.yaml`

### 3) 训练 Baseline-ROI

```bash
python train_baseline_roi.py
```

### 4) 训练 Topo-ROI（推荐显式指定 125e 配置）

```bash
python train_topo_roi.py --config config_125e.yaml
```

### 5) 评估（可选）

```bash
python evaluate.py --split val
```

---

## 仓库结构（主线化视图）

```text
.
├── README.md
├── config.yaml
├── config_125e.yaml
├── data_combined.py
├── model_unet.py
├── train_baseline_roi.py
├── train_topo_roi.py
├── topology_loss_fragment_suppress.py
├── utils_metrics.py
├── evaluate.py
├── docs/
│   ├── REPO_MAP.md
│   ├── MOVE_LOG.md
│   ├── CURRENT_STATUS.md
│   ├── NEXT_ONE_THING.md
│   ├── MAINLINE_FILES.md
│   ├── ENVIRONMENT_TEMPLATE.md
│   └── TRAIN_TOPO_ENTRY_RULES.md
├── reports/
├── configs/archive/
└── legacy/
    └── topology_loss_ablation.py
```

说明：

- `reports/` 存放阶段结论报告。
- `configs/archive/` 存放历史阶段配置。
- `legacy/` 存放历史脚本与旧入口（保留，不作为默认运行路径）。
