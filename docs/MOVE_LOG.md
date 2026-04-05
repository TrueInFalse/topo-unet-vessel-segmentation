# MOVE LOG

## 阶段 1：文档/报告/配置轻量整理

执行时间：2026-04-05（UTC）
执行类型：第一阶段轻量整理（仅移动文件，不改代码逻辑）

### 1) 迁移记录（old path -> new path）

| old path | new path | status |
|---|---|---|
| `CURRENT_STATUS.md` | `docs/CURRENT_STATUS.md` | `MOVED` |
| `NEXT_ONE_THING.md` | `docs/NEXT_ONE_THING.md` | `MOVED` |
| `MAINLINE_FILES.md` | `docs/MAINLINE_FILES.md` | `MOVED` |
| `0307structure.txt` | `docs/0307structure.txt` | `MOVED` |
| `experiments/FRAGMENT_SUPPRESS_125E_REPORT.md` | `reports/FRAGMENT_SUPPRESS_125E_REPORT.md` | `MOVED` |
| `experiments/roi_aligned_20e/ROI_ALIGNED_20E_REPORT.md` | `reports/ROI_ALIGNED_20E_REPORT.md` | `MOVED` |
| `experiments/topo_shape_ablation/TOPO_SHAPE_ABLATION_20E_REPORT.md` | `reports/TOPO_SHAPE_ABLATION_20E_REPORT.md` | `MOVED` |
| `artifacts/roi_audit/ROI_AUDIT_REPORT.md` | `reports/ROI_AUDIT_REPORT.md` | `MOVED` |
| `artifacts/roi_audit/TOPO_ROI_EVIDENCE.md` | `reports/TOPO_ROI_EVIDENCE.md` | `MOVED` |
| `config_20e.yaml` | `configs/archive/config_20e.yaml` | `MOVED` |

### 2) 未找到文件

- 无（本阶段目标文件均存在并已移动）。

### 3) 路径风险

- 风险项：1 个
- 无风险项：9 个

风险明细：

- `artifacts/roi_audit/ROI_AUDIT_REPORT.md -> reports/ROI_AUDIT_REPORT.md`
  - 检测到直接引用：`test/roi_audit_final.py:14, 330, 332`
  - 处理策略：按限制未自动修复，仅记录风险。

---

## 阶段 2：主文件夹轻量收束（legacy）

执行时间：2026-04-05（UTC）
执行类型：主文件夹轻量收束（仅移动旧入口文件，不改训练逻辑）

### 1) 迁移记录（old path -> new path）

| old path | new path | status |
|---|---|---|
| `data_drive.py` | `legacy/data_drive.py` | `MOVED` |
| `topology_loss.py` | `legacy/topology_loss.py` | `MOVED` |
| `train_baseline.py` | `legacy/train_baseline.py` | `MOVED` |
| `train_with_topology.py` | `legacy/train_with_topology.py` | `MOVED` |
| `visualize_results.py` | `legacy/visualize_results.py` | `MOVED` |
| `项目结构260315.png` | `legacy/项目结构260315.png` | `MOVED` |

### 2) 未找到文件

- 无（本阶段目标文件均存在并已移动）。

### 3) 路径风险

检查范围：`*.py`、`*.sh`、`*.yaml`、`*.yml`

- 风险项：5 个
- 无风险项：1 个（`项目结构260315.png`）

风险明细（未自动修复）：

1. `data_drive.py -> legacy/data_drive.py`
- 相关引用示例：`config.yaml:28`、`evaluate.py:11`、`legacy/train_baseline.py:11`、`legacy/train_with_topology.py:11`、`test/verify_data.py:11`

2. `topology_loss.py -> legacy/topology_loss.py`
- 相关引用示例：`legacy/train_with_topology.py:11`

3. `train_baseline.py -> legacy/train_baseline.py`
- 相关引用示例：`model_unet.py:12`、`utils_metrics.py:12`、`legacy/train_with_topology.py:585,650`

4. `train_with_topology.py -> legacy/train_with_topology.py`
- 相关引用示例：`config.yaml:28,86`、`test/run_topo_roi_debug.sh:36,53`、`test/test_end_to_end.py:17,21`

5. `visualize_results.py -> legacy/visualize_results.py`
- 相关引用示例：`legacy/train_baseline.py:532`

说明：

- 以上风险包含“注释/文案引用”和“脚本调用引用”两类。
- 按你的限制，本次未改 import、未改路径引用，仅记录风险。

---

## 阶段 3：历史消融文件归档（ablation -> legacy）

执行时间：2026-04-05（UTC）
执行类型：最小路径收束（仅移动文件与同步文档）

### 1) 迁移记录（old path -> new path）

| old path | new path | status |
|---|---|---|
| `topology_loss_ablation.py` | `legacy/topology_loss_ablation.py` | `MOVED` |

### 2) 未找到文件

- 无（目标文件存在并已移动）。

### 3) 路径风险

检查范围：`*.py`、`*.sh`、`*.yaml`、`*.yml`

- 风险项：0 个
- 无风险项：1 个

说明：

- 未检测到代码对 `topology_loss_ablation.py` 的直接引用。
- 按限制未自动改 import 或脚本路径，仅同步必要文档路径。

---

## 统一约束执行确认

- 未删除任何文件。
- 未修改训练逻辑。
- 未修改 import。
- 未做实验。
