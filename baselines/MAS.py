from typing import Dict
import torch


class MAS:
    """
    Memory Aware Synapses (MAS) for continual learning.

    - omega_acc: 在一个任务期间累积的梯度重要性（未归一化）
    - omega:     归一化后的重要性 Ω_{ij}
    - theta_star: 上一任务收敛后的参数 θ^*
    """

    def __init__(self, mapping, lam: float = 1.0):
        self.mapping = mapping
        self.device = mapping.device
        self.lam = lam

        self.omega: Dict[str, torch.Tensor] = {}
        self.theta_star: Dict[str, torch.Tensor] = {}

        # 累积器 & 计数
        self.omega_acc: Dict[str, torch.Tensor] = {}
        self.acc_steps: int = 0

    @torch.no_grad()
    def init_if_needed(self):
        """在第一次调用前初始化结构。"""
        if self.omega:
            return
        for name, p in self.mapping.model.named_parameters():
            if not p.requires_grad:
                continue
            z = torch.zeros_like(p, device=self.device)
            self.omega[name] = z.clone()
            self.omega_acc[name] = z.clone()
            self.theta_star[name] = p.detach().clone()
        self.acc_steps = 0

    def reset_accumulator(self):
        """在新任务开始前清空累积器。"""
        self.init_if_needed()
        for name in self.omega_acc:
            self.omega_acc[name].zero_()
        self.acc_steps = 0

    def accumulate_importance_from_grad(self):
        """
        在一次 backward 之后调用：
        - 用当前梯度 |∂(||F||^2)/∂θ| 累积到 omega_acc
        """
        self.init_if_needed()
        self.acc_steps += 1
        for name, p in self.mapping.model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            self.omega_acc[name] += p.grad.detach().abs()

    @torch.no_grad()
    def finalize_importance(self):
        """
        在任务结束时调用：
        - 将 omega_acc / acc_steps 累积到 omega（可以累加多任务）
        - 更新 theta_star 为当前参数
        """
        self.init_if_needed()
        if self.acc_steps == 0:
            return

        for name in self.omega:
            self.omega[name] += self.omega_acc[name] / float(self.acc_steps)

        for name, p in self.mapping.model.named_parameters():
            if not p.requires_grad:
                continue
            self.theta_star[name] = p.detach().clone()

        # 清空累积器，准备下一任务
        self.reset_accumulator()

    def mas_loss(self) -> torch.Tensor:
        """
        计算 MAS 正则项：
            λ * Σ Ω_ij (θ_ij - θ*_ij)^2
        """
        if not self.omega or not self.theta_star:
            return torch.tensor(0.0, device=self.device)

        reg = torch.tensor(0.0, device=self.device)
        for name, p in self.mapping.model.named_parameters():
            if name not in self.omega or name not in self.theta_star:
                continue
            if not p.requires_grad:
                continue
            omega = self.omega[name]
            theta_star = self.theta_star[name]
            reg += (omega * (p - theta_star).pow(2)).sum()

        return self.lam * reg
