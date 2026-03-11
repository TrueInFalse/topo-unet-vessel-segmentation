
# Retina PH Seg
A retinal vessel segmentation project based on U-Net and persistent-homology-inspired topology regularization.

> 本项目面向本科毕业设计，研究目标是：在视网膜血管分割任务中，引入拓扑约束，缓解传统像素级损失导致的血管断裂、碎片化与连通性不足问题。

---

## 当前状态

**Last Updated:** 2026-03-11

- 当前主线：`Baseline-ROI` 与 `Topology-ROI` 对照实验
- 当前最稳结论：Baseline 已较稳定；Topology 分支已跑通并进入审计/消融阶段，但**尚未形成“稳定优于 Baseline”** 的最终结论
- 当前数据模式：支持 **纯 DRIVE** 与 **Kaggle 联合数据集**
- 当前重点问题：
  - README 与代码/实验现状存在不同步风险
  - Topology 分支的收益需要在统一口径下继续验证
  - ROI、评估口径、训练脚本分支仍需持续收束
- 建议读者先看：
  - `README.md`
  - `0307structure.txt`
  - `experiments/` 或阶段性报告文档
  - 最新训练脚本与配置文件

---

## 项目简介

视网膜血管形态是高血压、糖尿病等慢病早期筛查的重要生物标志。  
传统 U-Net 在该任务上通常能够获得较好的像素级重叠指标，但仍容易出现以下问题：

- 细小血管断裂
- 中心线碎片化
- 局部连通性不一致
- 拓扑结构不稳定

为此，本项目尝试将**持续同调（Persistent Homology）相关思想**引入分割训练过程，在不改变 U-Net 主体框架的前提下，为模型增加结构先验约束。

---

## 研究目标

本项目的目标不是单纯“把 Dice 做高”，而是同时关注：

1. 像素级分割质量  
   - Dice
   - IoU
   - Precision / Recall

2. 结构级拓扑质量  
   - CL-Break（中心线碎片数）
   - Δβ₀（Betti 数误差 / 连通分量差异）

3. 工程可复现性  
   - 配置可切换
   - 训练/评估流程可重跑
   - 实验结论有日志和文档支撑

---

## 方法概览

### 1. Baseline
- 主体模型：`smp.Unet + ResNet34 encoder`
- 训练目标：以像素级损失为主
- 用途：提供稳定对照基线

### 2. Topology branch
- 在 U-Net 输出概率图基础上，引入拓扑正则化模块
- 核心关注 0 维拓扑特征（连通分量）
- 训练时采用 λ 调度策略，逐步提高拓扑约束权重
- 当前阶段重点是：
  - 验证拓扑损失是否真正起作用
  - 验证其收益是否在统一 ROI / 统一数据口径下成立
  - 排除“伪改进”与实验口径污染

---

## 数据集与数据模式

本项目支持两种数据模式，通过配置切换。

### 模式 A：纯 DRIVE
- 数据集：DRIVE
- 训练集：16 张（ID 21–36）
- 验证集：4 张（ID 37–40）
- 测试集：20 张（ID 01–20，无 GT）
- ROI：使用原始 FOV 掩码

适用场景：
- 小规模、干净对照
- 纯 DRIVE 基线验证
- ROI 口径审计

### 模式 B：Kaggle 联合数据集
- 数据来源：DRIVE + HRF + CHASE DB1 + STARE
- 训练集：`Training/`
- 验证集：`Test/`
- 无标签测试：`Unlabeled_test/`
- ROI：由代码根据当前配置处理

适用场景：
- 扩大训练数据规模
- 观察 Baseline 与 Topology 在更大数据规模下的趋势
- 进行更贴近最终论文展示的实验

---

## 目前较稳的项目结论

截至当前版本，可以较稳地说：

- 项目已经完成可运行的 Baseline 训练、评估、可视化闭环
- 项目已经完成 Topology 分支的接入与多轮修正
- 早期某些拓扑路线曾被审计为“形式存在但效果失效”
- 当前路线已进入更严格的 ROI / 数据 / 参数口径核验阶段
- 因此，本项目现阶段最稳的定位是：

> **一个已经具备稳定 Baseline，并正在严肃审计拓扑增强有效性的医学图像分割毕业设计项目。**

---

## 仓库结构

### 核心脚本
- `config.yaml`：主配置文件
- `config_20e.yaml` / `config_125e.yaml`：特定实验配置
- `data_drive.py`：DRIVE 数据加载
- `data_combined.py`：联合数据集加载
- `model_unet.py`：U-Net 模型定义
- `train_baseline.py`：Baseline 训练
- `train_with_topology.py`：Topology 训练主脚本
- `train_baseline_roi.py`：ROI 对齐 Baseline 实验
- `train_topo_roi.py`：ROI 对齐 Topology 实验
- `topology_loss.py`：拓扑损失
- `topology_loss_ablation.py`：拓扑损失/参数消融
- `evaluate.py`：评估脚本
- `visualize_results.py`：可视化脚本
- `utils_metrics.py`：指标计算

### 结果与辅助目录
- `audit_results/`：审计结果
- `experiments/`：实验记录/阶段性产物
- `test/`：测试相关内容
- `0307structure.txt`：结构梳理
- `回收站/`：历史废弃内容暂存

---

## 环境依赖

建议环境：
- Python 3.12+
- PyTorch 2.x
- CUDA 可用
- Ubuntu / AutoDL 环境优先

核心依赖示例：
```bash
pip install torch torchvision
pip install segmentation-models-pytorch
pip install scikit-image scipy matplotlib tqdm pyyaml
````

如启用 Kaggle 联合数据集下载：

```bash
pip install kagglehub
```

---

## 数据准备

### 1. DRIVE

将 DRIVE 数据集按如下结构放在项目根目录：

```text
DRIVE/
├── training/
│   ├── images/
│   ├── 1st_manual/
│   └── mask/
└── test/
    ├── images/
    └── mask/
```

### 2. Kaggle 联合数据集

本地目录建议为：

```text
data/combined/
├── Training/
│   ├── images/
│   └── masks/
├── Test/
│   ├── images/
│   └── masks/
└── Unlabeled_test/
```

---

## 快速开始

### 1. 训练 Baseline

```bash
python train_baseline.py
```

### 2. 训练 Topology 版本

```bash
python train_with_topology.py
```

### 3. 评估

```bash
python evaluate.py --split val
python evaluate.py --split test
```

### 4. 可视化

```bash
python visualize_results.py
```

---

## 评估指标

项目持续关注两类指标：

### 像素级指标

* Dice
* IoU
* Precision
* Recall

### 结构级指标

* CL-Break：中心线碎片数，越低越好
* Δβ₀：连通分量差异，越低越好

> 注意：项目强调 ROI 内评估；若更换数据模式或 ROI 生成策略，结论必须在统一口径下重新比较。

---

## 当前已知问题与限制

1. README 可能落后于最新代码与实验状态
2. 不同训练脚本之间仍存在历史分支与阶段性遗留
3. Topology 分支的最终收益仍需更严格对照验证
4. 不同数据模式下的 ROI 生成与评估口径需要持续统一
5. 当前仓库中仍保留部分历史文件、阶段性实验脚本与审计产物，后续会继续收束

---

## 文档约定

从现在开始，建议采用以下职责划分：

* `README.md`：项目总览、快速开始、当前状态摘要
* `docs/CURRENT_STATUS.md`：当前实验现状与结论
* `docs/DECISIONS.md`：技术路线变化与关键决策
* `docs/ENVIRONMENT.md`：环境、代理、推送、运维说明
* `docs/ARCHIVE/`：历史修复记录与已过期文档

---

## 面向答辩/复试的项目定位

这个项目最值得强调的不是“单次涨点”，而是：

* 从混乱原型逐步收束到可复现实验系统
* 从单纯像素分割推进到结构约束分割
* 对拓扑模块进行过失败诊断、路线修正与重新验证
* 既有工程工作量，也有方法理解与实验审计过程

---

## License

仅用于课程/毕业设计研究与学习交流。
