"""
拓扑损失主线实现：Fragment-Suppress

说明：
- 本文件是当前默认主线拓扑损失实现
- 数学行为与历史消融容器中的 fragment_suppress 分支保持一致
- 仅保留主线所需逻辑，不再包含 standard/main_component 分支
"""

import torch
import torch.nn as nn
import cripser


class TopologicalRegularizerFragmentSuppress(nn.Module):
    """
    拓扑正则化器（主线 Fragment-Suppress 版）

    Args:
        target_beta0: 历史兼容参数（本模式不直接使用）
        max_death: 历史兼容参数（本模式不直接使用）
        loss_scale: 损失缩放因子
        fragment_penalty_factor: 碎片惩罚系数
        loss_mode: 兼容参数，主线固定为 fragment_suppress
        target_lifetime: 历史兼容参数（本模式不直接使用）
        main_boost_factor: 历史兼容参数（本模式不直接使用）
    """

    def __init__(
        self,
        target_beta0: int = 5,
        max_death: float = 0.5,
        loss_scale: float = 100.0,
        fragment_penalty_factor: float = 1.0,
        loss_mode: str = 'fragment_suppress',
        target_lifetime: float = 0.5,
        main_boost_factor: float = 1.0,
    ):
        super().__init__()
        self.target_beta0 = target_beta0
        self.max_death = max_death
        self.loss_scale = loss_scale
        self.fragment_penalty_factor = fragment_penalty_factor

        # 仅为参数兼容保留，不参与主线计算
        self.loss_mode = 'fragment_suppress'
        self.target_lifetime = target_lifetime
        self.main_boost_factor = main_boost_factor

        if loss_mode != 'fragment_suppress':
            print(f"[兼容提示] 主线仅支持 fragment_suppress，忽略 loss_mode={loss_mode}")

        print(
            f"[拓扑损失-主线] fragment_suppress, "
            f"loss_scale={loss_scale}, fragment_penalty_factor={fragment_penalty_factor}"
        )

    def forward(
        self,
        prob_map: torch.Tensor,
        roi_mask: torch.Tensor = None,
        epoch: int = None,
    ) -> torch.Tensor:
        """
        计算拓扑正则损失（Fragment-Suppress 主线公式）

        Args:
            prob_map: [B, 1, H, W] 概率图（已sigmoid）
            roi_mask: [B, 1, H, W] ROI掩码

        Returns:
            loss: 标量Tensor
        """
        batch_size = prob_map.shape[0]
        device = prob_map.device

        if prob_map.dim() == 4:
            prob_map = prob_map.squeeze(1)

        if roi_mask is not None and roi_mask.dim() == 4:
            roi_mask = roi_mask.squeeze(1)

        losses = []
        pd_stats = []  # 与主线训练日志兼容

        for i in range(batch_size):
            prob = prob_map[i]

            # 应用ROI
            if roi_mask is not None:
                roi = roi_mask[i]
                prob = prob * roi + (1.0 - roi) * 0.0

            # 子水平集filtration
            filtration = 1.0 - prob
            if filtration.dim() > 2:
                filtration = filtration.squeeze()

            # 计算持续同调
            pd = cripser.compute_ph_torch(
                filtration,
                maxdim=0,
                filtration="V"
            )

            # 提取0维birth/death
            dim0_mask = pd[:, 0] == 0
            births = pd[dim0_mask, 1]
            deaths = pd[dim0_mask, 2]

            # 过滤inf
            finite_mask = torch.isfinite(deaths)
            if finite_mask.sum() == 0:
                losses.append(torch.tensor(0.0, device=device))
                pd_stats.append({
                    'num_finite': 0,
                    'max_lifetime': 0.0,
                    'top5_lifetimes': [],
                    'fragments_mean': 0.0,
                })
                continue

            births_finite = births[finite_mask]
            deaths_finite = deaths[finite_mask]
            lifetimes = deaths_finite - births_finite

            # 按lifetime降序排序
            sorted_lifetimes, _ = torch.sort(lifetimes, descending=True)

            # 收集PD统计信息（保持与历史接口一致）
            num_finite = len(sorted_lifetimes)
            max_lifetime = sorted_lifetimes[0].item() if num_finite > 0 else 0.0
            top5 = [sorted_lifetimes[i].item() if i < num_finite else 0.0 for i in range(min(5, num_finite))]
            fragments_mean = sorted_lifetimes[1:].mean().item() if num_finite > 1 else 0.0
            pd_stats.append({
                'num_finite': num_finite,
                'max_lifetime': max_lifetime,
                'top5_lifetimes': top5,
                'fragments_mean': fragments_mean,
            })

            loss_i = self._fragment_suppress_loss(sorted_lifetimes)
            losses.append(loss_i)

        self.last_pd_stats = pd_stats

        # 与历史实现一致：batch 均值后乘 loss_scale
        return torch.stack(losses).mean() * self.loss_scale

    def _fragment_suppress_loss(self, sorted_lifetimes: torch.Tensor) -> torch.Tensor:
        """
        碎片抑制主线公式（与历史实现保持一致）:
        - 不奖励主连通分量变大
        - 仅惩罚除最长条外的 finite lifetimes 过大
        """
        if len(sorted_lifetimes) <= 1:
            return torch.tensor(0.0, device=sorted_lifetimes.device)

        fragment_lifetimes = sorted_lifetimes[1:]
        loss = fragment_lifetimes.pow(2).mean() * self.fragment_penalty_factor
        return loss

    def get_last_pd_stats(self):
        """获取最近一次 forward 的PD统计信息"""
        return getattr(self, 'last_pd_stats', [])


# 兼容别名
CubicalRipserLoss = TopologicalRegularizerFragmentSuppress
