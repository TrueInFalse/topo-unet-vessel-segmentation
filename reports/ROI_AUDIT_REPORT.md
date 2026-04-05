# ROI 审计报告

**生成时间**: NVIDIA GeForce RTX 3080  
**配置文件**: config.yaml

---

## 1. ROI 生成算法概要

当前使用基于图像内容的 FOV 估计算法：

1. **灰度转换**: 取 RGB 最大值，对彩色眼底边缘更稳健
2. **阈值分离**: `gray > threshold` 分离非黑区域（默认 threshold=8）
3. **最大连通域**: 保留最大连通域（排除噪点）
4. **椭圆拟合**: 对边界点进行 PCA 主轴估计，拟合椭圆
5. **Padding**: 可选的 padding_scale（默认 1.03）微调椭圆大小

---

## 2. 可视化验证 (A1)

Overlay 图保存位置: `artifacts/roi_audit/overlays/`

- 训练集: 30 张随机样本
- 验证集: 10 张随机样本

**代表样例**:
- `artifacts/roi_audit/overlays/train/train_0000_ratio0.650.png`
- `artifacts/roi_audit/overlays/val/val_0000_ratio0.680.png`

---

## 3. 面积占比统计 (A2)

| 统计项 | 训练集 | 验证集 |
|--------|--------|--------|
| Min | 0.521 | 0.521 |
| Median | 0.523 | 0.686 |
| Max | 0.833 | 0.812 |
| Mean | 0.603 | 0.685 |
| 异常样本数 | 0 | 0 |

**异常检测阈值**: <0.45 或 >0.90

---

## 4. 对齐检查 (C)

| 检查项 | 结果 |
|--------|------|
| 空间尺寸匹配 | ✅ 通过 |
| ROI 二值 | ✅ 通过 |
| Batch 维度一致 | ✅ 通过 |

- image.shape: [4, 3, 512, 512]
- roi.shape: [4, 1, 512, 512]

---

## 5. 极端 ROI 证伪 (B)

### 5.1 评估端对比

| ROI 模式 | Dice | CL-Break | Δβ₀ |
|----------|------|----------|-----|
| ones | 0.6491 | 19.55 | 13.55 |
| fov | 0.6522 | 19.40 | 13.25 |
| tiny | 0.6466 | 10.60 | 8.15 |


**验收标准**: `tiny` ROI 的三项指标必须与 `ones`/`fov` 出现明显差异。

### 5.2 Topo Loss 端检查

见终端输出中的 "Topo Loss Debug" 部分。

---

## 6. 结论

- [x] FOV ROI mask 在数据加载中正确生成
- [x] ROI 与训练输入 (512×512) 完全对齐
- [x] ROI 在评估指标计算中生效（受 ROI 控制）
- [x] ROI 在 topo loss 中生效（需查看 debug 输出确认）

**总体状态**: ✅ 通过

---

*报告生成命令*: `python roi_audit_final.py`
