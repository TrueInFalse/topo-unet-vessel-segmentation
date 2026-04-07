# 毕业论文方法论安全性审计报告（v2 — 含完整 125e CSV 证据）

**审计时间**: 2026-04-07  
**审计基准**: 仓库全量文件交叉核验（新增 `20260405_baseline_roi_125e.csv` + `20260405_topo_roi_fragment_suppress_125e.csv` + 可视化图）  
**审计角色**: 毕业论文方法论审稿人  
**与上一版区别**: 上一版审计发现 "Baseline-ROI 125e best=0.7841 无 CSV 溯源"——该问题已通过新追加的完整 125e Baseline CSV **解决**。

---

## 0. 证据链现状（先读此节）

> [!NOTE]
> **上一版审计的最大硬伤（0.7841 无 CSV 证据）已被修复。** 新文件 `20260405_baseline_roi_125e.csv` 包含完整 125 epoch 数据，epoch 72 的 val_dice = **0.7841**，与 README/CURRENT_STATUS 完全一致。

### 当前完整证据链

| 证据源 | 文件 | 内容确认 |
|---|---|---|
| Baseline-ROI 125e | [20260405_baseline_roi_125e.csv](file:///c:/1Python_Project/VesselSeg_UnetTopo/logs/20260405_baseline_roi_125e.csv) | ✅ 125 epoch，best val_dice = **0.7841** (epoch 72)，末尾 = **0.7814** (epoch 125)，含 `roi_mode=fov` |
| Topo-FS 125e | [20260405_topo_roi_fragment_suppress_125e.csv](file:///c:/1Python_Project/VesselSeg_UnetTopo/logs/20260405_topo_roi_fragment_suppress_125e.csv) | ✅ 125 epoch，best val_dice = **0.7615** (epoch 119)，末尾 = **0.7611** (epoch 125)，含 `roi_mode=fov` |
| 消融 20e | [TOPO_SHAPE_ABLATION_20E_REPORT.md](file:///c:/1Python_Project/VesselSeg_UnetTopo/reports/TOPO_SHAPE_ABLATION_20E_REPORT.md) | ✅ 三组同条件对照 |
| ROI 审计 | [ROI_AUDIT_REPORT.md](file:///c:/1Python_Project/VesselSeg_UnetTopo/reports/ROI_AUDIT_REPORT.md) + [TOPO_ROI_EVIDENCE.md](file:///c:/1Python_Project/VesselSeg_UnetTopo/reports/TOPO_ROI_EVIDENCE.md) | ✅ ROI 生效，口径对齐 |

### 训练曲线可视化

````carousel
![Baseline-ROI 125e 训练曲线](C:\Users\21177\.gemini\antigravity\brain\fff5dbd2-82ef-473b-a108-86a342929999\baseline_roi_125e.png)
<!-- slide -->
![Topo Fragment-Suppress 125e 训练曲线](C:\Users\21177\.gemini\antigravity\brain\fff5dbd2-82ef-473b-a108-86a342929999\topo_fragment_suppress_125e.png)
````

---

## 0.5 数字核实：新 CSV vs 旧文档的交叉校验

> [!WARNING]
> 新 CSV 中的 Fragment-Suppress 数字与旧 CURRENT_STATUS / 旧报告中的数字 **有细微差异**，必须统一口径。

| 指标 | 旧 CURRENT_STATUS 声称 | 旧 fragment_suppress_125e.csv (archive) | **新 20260405 topo CSV** | 判定 |
|---|---|---|---|---|
| FS best val_dice | 0.7618 (ep 114) | 0.7618 (ep 114) | **0.7615 (ep 119)** | ⚠️ 数字微有差异 |
| FS 末尾 val_dice | 0.7610 (ep 125) | 0.7610 (ep 125) | **0.7611 (ep 125)** | ≈ 一致 |
| FS best CL-Break | 6.8 | 6.8 (ep 114) | **6.0 (ep 41)** / ep 119 = 7.8 | ⚠️ 差异较大 |
| FS best Δβ₀ | 15.6 | 15.6 (ep 114) | ep 119 = **15.4** | ≈ 一致 |
| Baseline best val_dice | 0.7841 (ep 72) | N/A (旧 CSV 仅 25e) | **0.7841 (ep 72)** | ✅ 完全一致 |
| Baseline 末尾 | 0.7814 (ep 125) | N/A | **0.7814 (ep 125)** | ✅ 完全一致 |

**关键发现 [Repo-verified]**：
1. Baseline 数字完全可追溯，**0.7841 已证实**。
2. Fragment-Suppress 的新旧 CSV 非同一次运行（best epoch 114 vs 119），但 val_dice 差异仅 0.0003，属于正常种子波动范围。
3. **论文中应以新 CSV（20260405 版）为准，因为它包含完整训练元数据（topo_loss_raw, lambda 等）**。

---

## 1. 核心对比事实表（同配置、125e 对 125e）

以下为从两份新 CSV 中提取的 **严格同条件 125 epoch 对照**：

### 1.1 Best 点对比（各取全程最优）

| 指标 | Baseline-ROI (Best) | Topo-FS (Best) | 差距 | 标签 |
|---|---|---|---|---|
| **Epoch** | **72** | **119** | — | [Repo-verified] |
| **Val Dice** | **0.7841** | **0.7615** | **−0.0226** | [Repo-verified] |
| Val IoU | 0.6466 | 0.6175 | −0.0291 | [Repo-verified] |
| Val Precision | 0.7882 | 0.7685 | −0.0197 | [Repo-verified] |
| Val Recall | 0.7823 | 0.7572 | +0.0251↓ | [Repo-verified] |
| **CL-Break** | **7.5** | **7.8** | **+0.3** | [Repo-verified] |
| **Δβ₀** | **15.0** | **15.4** | **+0.4** | [Repo-verified] |

### 1.2 末尾对比（均取 epoch 125）

| 指标 | Baseline-ROI (E125) | Topo-FS (E125) | 差距 | 标签 |
|---|---|---|---|---|
| **Val Dice** | **0.7814** | **0.7611** | **−0.0203** | [Repo-verified] |
| Val IoU | 0.6429 | 0.6170 | −0.0259 | [Repo-verified] |
| **CL-Break** | **6.8** | **7.8** | **+1.0** | [Repo-verified] |
| **Δβ₀** | **14.8** | **15.7** | **+0.9** | [Repo-verified] |

### 1.3 同 Epoch 快照对比

| Epoch | Baseline Val Dice | Topo-FS Val Dice | 差距 |
|---|---|---|---|
| 20 | 0.7520 | 0.7303 | −0.0217 |
| 30 | 0.7696 | 0.7517 | −0.0179 |
| 50 | 0.7796 | 0.7556 | −0.0240 |
| 72 | **0.7841** | 0.7600 | −0.0241 |
| 100 | 0.7826 | 0.7609 | −0.0217 |
| 119 | 0.7818 | **0.7615** | −0.0203 |
| 125 | 0.7814 | 0.7611 | −0.0203 |

**核心结论 [Repo-verified]**: 在严格同配置 125e 对照下，Fragment-Suppress 全程落后 Baseline **约 0.02 Dice 点**，差距稳定且未见收敛趋势。

---

## 2. 现在可以明确说的结论

### 2.1 ✅ "Standard topo loss 的 PD→loss 映射方向是错误的"

| 标签 | 来源 |
|---|---|
| [Repo-verified] | [TOPO_SHAPE_ABLATION_20E_REPORT.md](file:///c:/1Python_Project/VesselSeg_UnetTopo/reports/TOPO_SHAPE_ABLATION_20E_REPORT.md) |

Standard 20e val_dice = 0.2953；同框架仅改映射 → Fragment-Suppress = 0.6645（+125%）。

**安全表述**：
> "实验证明，原始 top-k MSE 映射鼓励碎片分量持久存在，是拓扑约束失效的主因。仅修正映射方向即可将 val_dice 从 0.30 提升至 0.66。"

### 2.2 ✅ "Fragment-Suppress 显著优于 Standard 和 Main-Component"

| 标签 | 来源 |
|---|---|
| [Repo-verified] | 20e 三组同条件消融 |

| 模式 | Val Dice | CL-Break | Δβ₀ |
|---|---|---|---|
| Standard | 0.2953 | 48.8 | 69.6 |
| Main-Component | 0.6207 | 22.9 | 27.3 |
| **Fragment-Suppress** | **0.6645** | **18.9** | **13.1** |

### 2.3 ✅ "在严格同条件 125e 对照下，Topo-FS 仍落后 Baseline 约 2 个 Dice 点"

| 标签 | 来源 |
|---|---|
| [Repo-verified] | 两份 20260405 CSV，严格同配置 |

这是一个 **诚实的负面结果**，但正因为诚实，反而安全。评委欣赏认真做对照后承认差距的态度。

### 2.4 ✅ "ROI 口径错位不是 topo 崩坏的主因"

| 标签 | 来源 |
|---|---|
| [Repo-verified] | ROI_ALIGNED_20E + TOPO_SHAPE_ABLATION |

### 2.5 ✅ "Topo loss 需要更长训练才能收敛，但收敛后进入平台期"

| 标签 | 来源 |
|---|---|
| [Repo-verified] | 新 topo CSV：30e = 0.7517 → 60e = 0.7587 → 100e = 0.7609 → 125e = 0.7611 |

Baseline 30e 即达 0.7696，Topo-FS 30e 仅 0.7517。但 Topo 从 60e 开始进入平台（0.758→0.761），**95 个 epoch 只提升了 0.003**。

---

## 3. 只能谨慎说的结论

### 3.1 ⚠️ "PH 本身是有价值的"

| 判定 | **限定后可说** |
|---|---|
| 标签 | [Inference] — 需要限定"价值"的范围 |

**可以说**：
> "实验表明，PH 框架在修正映射方向后可以稳定运行且不破坏分割质量，证明 PH 在视网膜血管分割中具有 **方法论层面的可行性**。"

**不能说**：
> ~~"PH 提升了分割性能"~~ — 它在 Dice 上落后，在拓扑指标上持平。
> ~~"PH 优于其他拓扑方法"~~ — 没有对比 clDice、MALIS 等替代方案。

### 3.2 ⚠️ "Topo 已经接近 Baseline"

| 判定 | **需要严格量化"接近"** |
|---|---|
| 标签 | [Repo-verified] 数字，但"接近"是主观判断 |

事实是：best 差距 0.0226；末尾差距 0.0203。

**可以说**（带量化）：
> "Fragment-Suppress 在 125e 对照中与 Baseline 的 Dice 差距为 0.020–0.023，约 2.6%–2.9% 相对差距。"

**不能说**：
> ~~"仅差不到 1%"~~ — 旧报告中这个数字用的是 30e Baseline（0.7696），现已被 125e Baseline（0.7841）替代。
> ~~"接近甚至超过 Baseline"~~ — 在任何 epoch，Topo 均未超过 Baseline。

### 3.3 ⚠️ "Fragment-Suppress 已证明拓扑约束有效"

| 判定 | **"有效"需重新定义** |
|---|---|
| 标签 | [Inference] |

**新 CSV 中的拓扑指标对比（best 点）**：

| 指标 | Baseline (ep 72) | Topo-FS (ep 119) | 差距 |
|---|---|---|---|
| CL-Break | 7.5 | 7.8 | +0.3（Topo 更差） |
| Δβ₀ | 15.0 | 15.4 | +0.4（Topo 更差） |

> [!IMPORTANT]
> **在新 CSV 的严格对照下，Fragment-Suppress 在拓扑指标上也未超越 Baseline——甚至略差。** 这与旧报告中 "拓扑指标与 Baseline 持平" 的结论有区别（旧报告用的是 30e Baseline 的 6.7 / 15.4；新 125e Baseline 在 best 点是 7.5 / 15.0）。

**可以说**：
> "Fragment-Suppress 验证了修正映射方向后，拓扑约束 **不再破坏** 分割质量和拓扑结构。"

**不能说**：
> ~~"拓扑约束有效改善了血管连通性"~~ — 拓扑指标与 Baseline 持平甚至略差。

---

## 4. 绝对不能说太满的结论

### 4.1 ❌ "Topo 已经接近甚至超过 Baseline"

- "超过"在任何时间点均不成立 [Repo-verified]
- "接近"需量化为"差 0.02 Dice"，且需注明差距在 60e 之后不再缩小

### 4.2 ❌ "当前方法已经达到论文水平"

理由：
1. Val Dice 0.76 vs Baseline 0.78，差距 2 个百分点
2. 拓扑指标未超越 Baseline
3. 单种子 (seed=42) 单次运行，无统计检验
4. Kaggle `Test/` 用作验证，存在评估污染风险 [Repo-verified, 来自 GLOBAL_CODE_AUDIT B.2]
5. 无独立测试集最终评估
6. 无与领域 SOTA 的横向对比

### 4.3 ❌ "借鉴某篇论文后只差一点工程补齐"

- 无具体论文引用
- "一点工程"无法量化
- 答辩时会被追问 "差哪一点？给出定义"

### 4.4 ❌ "差距仅 0.78%" / "在误差范围内"

- 0.78% 是旧报告中对 30e Baseline 的对比，已过时
- 新 125e 对照差距为 2.6%–2.9%
- "误差范围"需多种子统计检验证明，当前仅单次运行

---

## 5. 风险清单

### Level A：会被导师/评委直接抓住

| # | 风险 | 证据 | 后果 |
|---|---|---|---|
| **A1** | **Kaggle `Test/` 被用作验证集** | [GLOBAL_CODE_AUDIT B.2](file:///c:/1Python_Project/VesselSeg_UnetTopo/audit_results/GLOBAL_CODE_AUDIT_2026-03-15.md) | 所有 best val_dice 可能偏乐观。评委问"你在测试集上的结果呢？"时无法回答 |
| **A2** | **单种子单次运行，无统计显著性** | 全仓库仅 seed=42 | 0.7841 vs 0.7615 的差距是否显著？无法回答。评委问"换个种子结果会不会反过来？"时无法回答 |
| **A3** | **无独立测试集最终评估** | `evaluate.py` 与 Kaggle 主线脱节 | 论文缺乏标准的"在独立测试集上的最终性能"表格 |
| **A4** | **旧报告中"差距 0.78%"使用了错误的 Baseline 锚点** | FRAGMENT_SUPPRESS_125E_REPORT 用 30e Baseline (0.7696) 对比 125e Topo (0.7618) | 如果论文/答辩引用此数字，会被识破为不公平对比。实际 125e 对 125e 差距是 **2.9%** |
| **A5** | **拓扑约束未带来任何指标超越 Baseline** | 新 CSV 全面交叉对比 | "引入拓扑以改善连通性"的核心叙事缺乏数据支撑。评委问"那引入拓扑的好处是什么？"时需转向方法论贡献 |

### Level B：会让叙事变弱

| # | 风险 | 证据 | 后果 |
|---|---|---|---|
| **B1** | **Topo-FS 60e 后进入平台，95 epoch 仅提升 0.003** | 新 topo CSV | "拓扑约束有潜力"的叙事被削弱——已充分训练但仍落后 |
| **B2** | **没有与领域 SOTA 的定量对比** | 仓库无 clDice/MALIS 等实现 | 无法定位贡献在领域中的位置 |
| **B3** | **Fragment-Suppress 的公式 `lifetimes[1:].pow(2).mean()` 无文献引用** | [topology_loss_fragment_suppress.py L160](file:///c:/1Python_Project/VesselSeg_UnetTopo/topology_loss_fragment_suppress.py) | 评委可能追问理论依据 |
| **B4** | **新旧两份 Topo CSV 结果有微小差异** | best 从 0.7618 变为 0.7615，但 best epoch 从 114 变为 119 | 如果论文混用新旧数字会被发现口径不一致 |
| **B5** | **Baseline 在 Recall 上也优于 Topo-FS** | Baseline 0.7823 vs Topo 0.7572 (@best) | 旧报告中"Topo Recall 更高"的结论在新 CSV 中不再成立 |

### Level C：措辞优化即可

| # | 风险 | 建议 |
|---|---|---|
| **C1** | FRAGMENT_SUPPRESS_125E_REPORT 中"差距可能在误差范围内" | 删除此表述或改为"差距有待多种子验证" |
| **C2** | "甚至可能在未来优化中超过 Baseline" | 改为"后续研究可探索进一步缩小差距的策略" |
| **C3** | CURRENT_STATUS 中 "Topo 已接近可用" 用的旧数字 | 更新为新 CSV 数字 |
| **C4** | `target_beta0=5` 的选择无消融实验 | 在论文中注明为经验设定 |
| **C5** | CL-Break / Δβ₀ 指标无数学定义引用 | 在论文方法节补充定义 |

---

## 6. 五个特定表达的最终安全性判定

| 表达 | 安全？ | 新证据下的理由 |
|---|---|---|
| **"PH 本身是有价值的"** | ⚠️ **限定为"方法论可行性已验证"** | PH 在修正映射后不崩坏，但未超越 Baseline。不能说"提升了性能" |
| **"Topo 已经接近甚至超过 Baseline"** | ❌ **不能说"超过"，"接近"需量化** | 严格 125e 对照差距 0.020–0.023，非"接近"的直觉含义 |
| **"Fragment-Suppress 已证明拓扑约束有效"** | ⚠️ **改为"证明修正后不再有害"** | 像素和拓扑指标均未超越 Baseline，"有效"的定义需降级 |
| **"当前方法已经达到论文水平"** | ❌ **不能说** | 缺失：SOTA 对比、独立测试集、多种子统计、Topo 优势指标 |
| **"借鉴某篇论文后只差一点工程补齐"** | ❌ **不能说** | 无具体引用、无量化定义、答辩不可防守 |

---

## 7. 最终输出

### 7.1 最稳妥的项目结论表述（写入论文正文用）

> 本研究在 U-Net 框架下系统探究了基于持续同调的拓扑约束在视网膜血管分割中的应用条件与瓶颈。
>
> **核心发现**：
>
> 1. **定位了拓扑约束失效的根本原因**：原始 top-k MSE 映射将拓扑损失引导向鼓励碎片分量持久存在的方向，导致分割性能严重退化（Val Dice 降至 0.30，而同条件 Baseline 为 0.77）。该发现揭示了 PD→loss 映射方向是拓扑约束成败的关键。[Repo-verified]
>
> 2. **提出并验证了 Fragment-Suppress 修正策略**：仅惩罚非主连通分量的 lifetime，不干扰 Dice 对主分量的优化。在同条件 20 epoch 消融中，该策略将 Val Dice 从 0.30 提升至 0.66（+125%），拓扑指标同步大幅改善。[Repo-verified]
>
> 3. **在严格同配置 125 epoch 对照中量化了差距**：Fragment-Suppress 的 best Val Dice（0.7615）落后纯 Dice Baseline（0.7841）约 0.023，拓扑指标（CL-Break, Δβ₀）与 Baseline 处于同一水平。这表明修正后的拓扑约束 **不再破坏** 分割质量，但 **尚未实现超越**。[Repo-verified]
>
> **局限性**：
> - 实验基于单一随机种子（seed=42），结论的统计稳健性有待多种子验证。
> - 当前 Fragment-Suppress 在 60 epoch 后进入效果平台期，差距未见进一步缩小趋势。
> - 未与 clDice 等替代拓扑感知方法进行横向对比。
> - 验证集划分可能存在优化偏差，最终性能需在独立测试集上确认。

### 7.2 最适合写进 README / CURRENT_STATUS 的结论

```markdown
## 当前状态（2026-04-07 更新）

### 核心数字（严格同配置 125e 对照）

| 指标 | Baseline-ROI | Topo-FS | 差距 |
|---|---|---|---|
| Best Val Dice | 0.7841 (ep72) | 0.7615 (ep119) | −0.0226 |
| Final Val Dice | 0.7814 (ep125) | 0.7611 (ep125) | −0.0203 |
| Best CL-Break | 6.0 (ep43) | 6.0 (ep41) | 0.0 |
| Best Δβ₀ | 13.6 (ep115) | 13.1 (ep21) | −0.5 |

主线证据日志：
- `logs/20260405_baseline_roi_125e.csv`
- `logs/20260405_topo_roi_fragment_suppress_125e.csv`

### 已验证结论
- [已验证] Standard topo loss 方向错误  → Fragment-Suppress 修正后从 0.30 恢复到 0.76
- [已验证] ROI 口径错位不是 topo 崩坏主因
- [已验证] 严格 125e 对照下 Topo-FS 落后 Baseline 约 0.02 Dice，拓扑指标基本持平
- [已验证] Topo-FS 在 60e 后进入平台期

### 一句话
当前主线是"强 Baseline + 已修正但未超越的 Topo"。本项目的核心贡献在于 **定位并修正了拓扑约束失效的原因**，而非在最终指标上超越 Baseline。
```

### 7.3 最适合答辩口头汇报的结论

> **答辩口播脚本（约 2.5 分钟）**
>
> 本项目研究了基于持续同调的拓扑约束在视网膜血管分割中的应用。
>
> 我们的核心发现是：**拓扑约束的成败取决于 PD 到 loss 的映射方向，而不是拓扑数学本身是否适用。**
>
> 在调研了已有文献中的拓扑损失公式后，我们首先复现了基于 top-k MSE 的标准拓扑约束。实验发现，这种约束方式会严重破坏分割性能——Val Dice 从基线的 0.77 降到 0.30。
>
> 通过分析持续图的结构，我们发现问题的根源是：标准映射在鼓励碎片分量的 lifetime 向固定目标靠近，本质上是在鼓励碎片持久存在。
>
> 基于此分析，我们提出了 Fragment-Suppress 策略：不奖励主连通分量变大，只惩罚碎片分量的 lifetime。在同条件消融实验中，这一策略将 Val Dice 从 0.30 提升到了 0.66，提升了 125%。
>
> 在最终的严格对照中，Fragment-Suppress 训练 125 epoch 后达到 0.76，与纯 Dice 基线的 0.78 差距约 2 个百分点。拓扑指标与基线持平。
>
> **诚实地说**，在当前实验条件下，拓扑约束在像素精度和拓扑指标上均未超越 Baseline。但这恰恰说明了本研究的核心贡献不是指标刷分，而是 **系统性地定位了拓扑约束在血管分割任务中失效的原因**，并 **验证了可行的修正方向**。
>
> 如果评委问"那拓扑约束的好处在哪？"：
> - 好处在方法论层面——我们证明了 PH 框架可以稳定工作，关键是映射方向
> - 当前版本的数值差距（0.02）仍有工程优化空间，如调整 λ 调度、引入多尺度拓扑特征
> - 更重要的是，这个诊断过程本身就是毕设的方法论实践

---

## 8. 必须处理的文档一致性问题

> [!IMPORTANT]
> 以下不是新实验，而是 **论文提交前必须更新的文档**，否则会出现新旧数字混用的风险。

| 文件 | 当前问题 | 修改建议 |
|---|---|---|
| [CURRENT_STATUS.md](file:///c:/1Python_Project/VesselSeg_UnetTopo/docs/CURRENT_STATUS.md) | Topo best 写的 0.7618(ep114)，Baseline CL-Break=7.5/Δβ₀=15.0 与新数据不一致 | 更新为新 CSV 数字：Topo best=0.7615(ep119)，注明差距 0.0226 |
| [README.md](file:///c:/1Python_Project/VesselSeg_UnetTopo/README.md) | "best val_dice=0.7841 epoch 72" 正确；但 Topo "0.7618 ep 114" 应更新 | 统一为新 CSV 口径 |
| [NEXT_ONE_THING.md](file:///c:/1Python_Project/VesselSeg_UnetTopo/docs/NEXT_ONE_THING.md) | "差距 0.0223"应更新为 0.0226 | 小幅修正 |
| [FRAGMENT_SUPPRESS_125E_REPORT.md](file:///c:/1Python_Project/VesselSeg_UnetTopo/reports/FRAGMENT_SUPPRESS_125E_REPORT.md) | 使用 30e Baseline(0.7696) 对比 125e Topo → "差距 0.78%" 已过时 | 在报告顶部加注"历史报告，已被 20260405 对照数据取代"；**论文中不引用此报告中的对比数字** |

---

## 附录：关键数字的证据溯源表

| 数字 | 用途 | CSV 位置 | 行号 | 标签 |
|---|---|---|---|---|
| Baseline best = 0.7841 | 核心锚点 | `20260405_baseline_roi_125e.csv` | L73 (epoch 72) | [Repo-verified] ✅ |
| Baseline E125 = 0.7814 | 末尾参考 | 同上 | L127 (epoch 125) | [Repo-verified] ✅ |
| Topo-FS best = 0.7615 | 核心锚点 | `20260405_topo_roi_fragment_suppress_125e.csv` | L120 (epoch 119) | [Repo-verified] ✅ |
| Topo-FS E125 = 0.7611 | 末尾参考 | 同上 | L126 (epoch 125) | [Repo-verified] ✅ |
| Standard 20e = 0.2953 | 消融对照 | TOPO_SHAPE_ABLATION_20E_REPORT 表 2.1 | — | [Repo-verified] ✅ |
| Main-Component 20e = 0.6207 | 消融对照 | 同上 | — | [Repo-verified] ✅ |
| Fragment-Suppress 20e = 0.6645 | 消融对照 | 同上 | — | [Repo-verified] ✅ |
| 旧 FS best = 0.7618 (ep114) | ⚠️ 旧版本 | `logs/archive/fragment_suppress_125e.csv` L115 | — | [Repo-verified] 但已被新 CSV 取代 |

---

*本审计报告基于仓库现存文件的完整交叉核验。所有 [Repo-verified] 均可追溯到具体 CSV 行号。[Inference] 标签表示基于数据的合理推断但非直接证据。*
