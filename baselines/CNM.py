import torch
import copy
import random
from typing import List, Tuple

class CNMTeacher:
    """
    对应论文中的 f(x; θ^{t-1})。
    在持续学习过程中，它代表"上一时刻"或"近期"的模型，用于提供离面点的符号约束。
    """
    def __init__(self, mapping):
        self.mapping = mapping
        self.device = mapping.device
        self.teacher = None

    def update(self):
        """
        将当前 Student 模型参数复制给 Teacher。
        对应论文：θ^{t-1} <- θ^t
        """
        # 深拷贝当前模型结构和参数
        ModelCls = type(self.mapping.model)
        self.teacher = ModelCls(self.mapping.config, self.mapping.bounding_box).to(self.device)
        self.teacher.load_state_dict(self.mapping.model.state_dict())
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def has_teacher(self) -> bool:
        return self.teacher is not None

    @torch.no_grad()
    def query_sdf(self, pts: torch.Tensor) -> torch.Tensor:
        if self.teacher is None:
            return torch.zeros(pts.shape[0], device=self.device)
        # embed=True 意味着使用位置编码
        return self.teacher.query_sdf(pts, embed=True).squeeze(-1)


class CNMRealBuffer:
    """
    对应论文中的 "Sample Buffer B"。
    存储过去观测中的真实表面点 (Zero level-set samples)。
    """
    def __init__(self, device, max_points=200000):
        self.device = device
        self.max_points = max_points
        # 存储世界坐标系下的点 (N, 3)
        self.points = torch.empty((0, 3), device=self.device)

    def add_observation(self, batch, num_samples=500):
        """
        从当前帧观测中下采样真实表面点并存入 Buffer。
        batch: 包含 'c2w', 'depth', 'direction' (camera space)
        """
        c2w = batch['c2w'][0].to(self.device)       # [4, 4]
        depth = batch['depth'][0].to(self.device)   # [H, W]
        direction = batch['direction'][0].to(self.device) # [H, W, 3]

        H, W = depth.shape
        
        # 1. 随机下采样像素
        # 过滤掉无效深度 (假设 depth > 0 为有效)
        valid_mask = (depth > 0).reshape(-1)
        valid_indices = torch.nonzero(valid_mask).squeeze()
        
        if valid_indices.numel() < num_samples:
            indices = valid_indices
        else:
            # 随机选 num_samples 个
            perm = torch.randperm(valid_indices.numel(), device=self.device)[:num_samples]
            indices = valid_indices[perm]

        # 2. 反投影到世界坐标系 (Back-projection)
        # indices 是 flatten 后的索引
        h_idx = indices // W
        w_idx = indices % W

        d_cam = direction[h_idx, w_idx, :] # [N, 3]
        d_val = depth[h_idx, w_idx].unsqueeze(-1) # [N, 1]

        # Camera -> World
        # rays_o = c2w[:3, -1]
        # rays_d_world = d_cam @ R.T
        # P_world = rays_o + rays_d_world * depth
        
        rays_d_world = torch.sum(d_cam[..., None, :] * c2w[:3, :3], -1)
        rays_o = c2w[:3, -1].expand_as(rays_d_world)
        
        pts_world = rays_o + rays_d_world * d_val # [N, 3]

        # 3. 存入 Buffer (FIFO)
        self.points = torch.cat([self.points, pts_world], dim=0)
        if self.points.shape[0] > self.max_points:
            # 移除最旧的
            self.points = self.points[-self.max_points:]

    def sample(self, batch_size):
        """
        随机采样历史表面点。
        """
        if self.points.shape[0] == 0:
            return None
        
        n = min(batch_size, self.points.shape[0])
        idx = torch.randint(0, self.points.shape[0], (n,), device=self.device)
        return self.points[idx]


class CNMLearner:
    """
    CNM 算法核心控制器。
    """
    def __init__(self, mapping):
        self.mapping = mapping
        self.device = mapping.device
        
        cfg = mapping.config['training'].get('cnm', {})
        
        # 参数
        self.buffer_size = cfg.get('buffer_size', 200000)
        self.pts_per_frame = cfg.get('pts_per_frame', 200) # 每帧存多少点
        self.batch_size_zero = cfg.get('batch_size_zero', 1024) # 每次BA回放多少表面点
        self.batch_size_off = cfg.get('batch_size_off', 2048)   # 每次BA采样多少离面点
        self.alpha = cfg.get('alpha', 100.0) # 符号正则强度
        self.weight_zero = cfg.get('weight_zero', 1.0) # 表面回放权重
        self.weight_sign = cfg.get('weight_sign', 0.1) # 符号正则权重

        # 组件
        self.teacher = CNMTeacher(mapping)
        self.buffer = CNMRealBuffer(self.device, max_points=self.buffer_size)
        self.update_freq = cfg.get('update_freq', 20) 
        self.epsilon = cfg.get('epsilon', 0.05) # 增大死区
        self.frame_count = 0

    def step_end_of_frame(self, batch):
        """
        在每一帧处理结束时调用：
        1. 将当前帧的观测存入 Buffer。
        2. 更新 Teacher (θ^{t-1})。
        """
        # 1. 存入真实表面点
        self.buffer.add_observation(batch, num_samples=self.pts_per_frame)
        
        # 2. 更新 Teacher (论文 imply 是流式的，所以每帧或每几帧更新)
        # 这里简化为每帧更新，确保 teacher 总是"上一时刻"的状态
        self.frame_count += 1
        if self.frame_count % self.update_freq == 0:
            self.teacher.update()



    def compute_loss(self):
        """
        计算 CNM 的两个 Loss 项：
        1. Zero level-set loss (on Buffer samples)
        2. Sign regularization loss (on Random Off-surface samples)
        """
        total_loss = torch.tensor(0.0, device=self.device)
        
        # --- 1. Zero Level-Set Replay (Buffer) ---
        # ...existing code...
        pts_zero = self.buffer.sample(self.batch_size_zero)
        if pts_zero is not None:
            pred_sdf = self.mapping.model.query_sdf(pts_zero, embed=True)
            loss_zero = torch.abs(pred_sdf).mean()
            total_loss += self.weight_zero * loss_zero

        # --- 2. Off-surface Sign Regularization (Random) ---
        if self.teacher.has_teacher():
            # ...existing code...
            bb = self.mapping.bounding_box
            rand_pts = torch.rand(self.batch_size_off, 3, device=self.device)
            rand_pts = rand_pts * (bb[:, 1] - bb[:, 0]) + bb[:, 0]

            with torch.no_grad():
                sdf_teacher = self.teacher.query_sdf(rand_pts)
            
            sdf_student = self.mapping.model.query_sdf(rand_pts, embed=True).squeeze(-1)

            # 【改进】软间隔符号正则化
            # 定义一个死区 epsilon，在这个区间内不强制符号一致性
            epsilon = self.epsilon
            # 计算指数项的输入
            exponent = torch.where(
                sdf_teacher > 0,
                -self.alpha * sdf_student,
                self.alpha * sdf_student
            )
            
            raw_loss = torch.exp(exponent)
            
            # 【关键】应用掩码：只在 Teacher 确信远离表面时施加约束
            # 如果 |sdf_teacher| < epsilon，则 mask = 0，忽略该点的 Loss
            mask = (torch.abs(sdf_teacher) > epsilon).float()
            
            # 计算加权平均 (避免分母为0)
            loss_sign = (raw_loss * mask).sum() / (mask.sum() + 1e-6)
            
            total_loss += self.weight_sign * loss_sign

        return total_loss