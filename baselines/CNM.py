import torch
import random
from typing import List, Tuple


class CNMTeacher:
    """
    CNM-style teacher: 保存上一阶段的网络参数，作为“旧知识”的紧凑记忆。
    对应论文里的 f(x; θ^{t-1}).
    """

    def __init__(self, mapping):
        """
        mapping: 当前 agent 的 Mapping 实例，用来访问 model / config / bounding_box / device。
        """
        self.mapping = mapping
        self.device = mapping.device
        self.teacher = None  # type: torch.nn.Module | None

    def update(self):
        """
        在“任务边界”或每 frames_per_task 帧调用一次，把当前 model 拷贝为 teacher。
        """
        ModelCls = type(self.mapping.model)
        teacher = ModelCls(self.mapping.config, self.mapping.bounding_box).to(self.device)
        teacher.load_state_dict(self.mapping.model.state_dict())
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        self.teacher = teacher

    def has_teacher(self) -> bool:
        return self.teacher is not None

    @torch.no_grad()
    def query_sdf(self, pts: torch.Tensor) -> torch.Tensor:
        if self.teacher is None:
            raise RuntimeError("CNMTeacher has no teacher model yet. Call update() first.")
        return self.teacher.query_sdf(pts, embed=True).squeeze(-1)




class CNMReplayBuffer:
    """
    CNM 专用的 off-surface replay buffer.
    存储 (points, sdf_teacher) 对，用于 function replay。
    与 UDON 的 replay 完全独立。
    """

    def __init__(self, device: torch.device,
                 max_points: int = 200000,
                 sample_batch_size: int = 2048):
        self.device = device
        self.max_points = max_points
        self.sample_batch_size = sample_batch_size

        self.points: torch.Tensor | None = None   # (N, 3)
        self.sdfs: torch.Tensor | None = None     # (N, C) or (N,)
        self.num_points: int = 0
        self._write_ptr: int = 0
        self.sdf_dim: int | None = None          # 通道数 C

    @torch.no_grad()
    def add_samples(self, pts: torch.Tensor, sdfs: torch.Tensor):
        """
        向 buffer 里添加一批新样本。
        pts:  (B,3)
        sdfs: (B,) 或 (B,C)
        """
        pts = pts.detach().to(self.device)
        sdfs = sdfs.detach().to(self.device)

        if sdfs.ndim == 1:
            # (B,) -> (B,1)
            sdfs = sdfs.unsqueeze(-1)

        B, C = sdfs.shape
        if self.points is None:
            # 第一次初始化
            self.points = torch.empty((self.max_points, 3),
                                      device=self.device,
                                      dtype=pts.dtype)
            self.sdfs = torch.empty((self.max_points, C),
                                    device=self.device,
                                    dtype=sdfs.dtype)
            self.num_points = 0
            self._write_ptr = 0
            self.sdf_dim = C
        else:
            # 后续写入时，通道数必须一致
            if C != self.sdf_dim:
                raise RuntimeError(
                    f"CNMReplayBuffer: sdf dim mismatch. "
                    f"buffer has C={self.sdf_dim}, new samples have C={C}"
                )

        # 如果一次要写的比缓冲区还大，只保留后面的 max_points 个
        if B >= self.max_points:
            pts = pts[-self.max_points:]
            sdfs = sdfs[-self.max_points:]
            B = pts.shape[0]

        end = self._write_ptr + B
        if end <= self.max_points:
            self.points[self._write_ptr:end] = pts
            self.sdfs[self._write_ptr:end] = sdfs
        else:
            first = self.max_points - self._write_ptr
            self.points[self._write_ptr:] = pts[:first]
            self.sdfs[self._write_ptr:] = sdfs[:first]

            second = B - first
            if second > 0:
                self.points[:second] = pts[first:]
                self.sdfs[:second] = sdfs[first:]

        self._write_ptr = (self._write_ptr + B) % self.max_points
        self.num_points = min(self.max_points, self.num_points + B)

    def has_data(self) -> bool:
        return self.num_points > 0

    def sample(self, batch_size: int | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从 buffer 中随机采样一个 batch。
        返回:
          pts:  (B,3)
          sdfs: (B,C)  若 C==1，可以在外部 squeeze 到 (B,)
        """
        if not self.has_data():
            raise RuntimeError("CNMReplayBuffer is empty, cannot sample.")

        if batch_size is None:
            batch_size = self.sample_batch_size

        batch_size = min(batch_size, self.num_points)
        indices = torch.randint(
            low=0,
            high=self.num_points,
            size=(batch_size,),
            device=self.device
        )
        pts_batch = self.points[indices]
        sdfs_batch = self.sdfs[indices]
        return pts_batch, sdfs_batch



class CNMOffsurfaceReplay:
    """
    CNM-style off-surface 伪样本回放：
      - 通过 teacher 在 bounding box 内生成 3D 点及其 SDF（pseudo-label）；
      - 把这些 (x, sdf_teacher) 存进 CNM 专用 buffer；
      - 训练时从 buffer 采样，对当前模型加符号正则（ψ_s）+ 值一致性约束，
        并在 near-surface 区域施加强约束。
    """

    def __init__(self, mapping, replay_buffer: CNMReplayBuffer):
        self.mapping = mapping
        self.device = mapping.device
        self.buffer = replay_buffer

        cnm_cfg = mapping.config['training'].get('cnm', {})
        # 每次从 buffer 采多少伪样本
        self.batch_size = cnm_cfg.get('num_offsurface_samples', 2048)
        # 总体 CNM loss 权重
        self.weight = cnm_cfg.get('offsurface_weight', 0.1)
        # 每次 teacher 更新时，往 buffer 里灌多少伪样本
        self.num_samples_per_teacher = cnm_cfg.get('num_samples_per_teacher', 50000)
        # ψ_s 中的 α 系数（符号正则的硬度）
        self.alpha = cnm_cfg.get('alpha', 1.0)
        # near-surface 阈值：|sdf_old| < near_threshold 认为在当前表面附近
        self.near_threshold = cnm_cfg.get('near_threshold', 0.05)
        # near-surface 区域里 value consistency 的额外权重
        self.value_weight_near = cnm_cfg.get('value_weight_near', 1.0)
        # far 区域 value consistency 的权重（通常较小）
        self.value_weight_far = cnm_cfg.get('value_weight_far', 0.1)

    @torch.no_grad()
    def _sample_points_in_bbox(self, num: int) -> torch.Tensor:
        """
        在当前 bounding box 内采样 3D 点，形状 (num,3)。

        这里按 stratified / near-surface 思路：
          1) 在 bbox 内先采 2*num 个粗点；
          2) 用“当前模型”的 SDF 做近似，选出 |sdf_current| 较小的 num 个点，
             作为 near-surface 候选点；
          3) 这些点再交给 teacher 提供更稳定的 sdf_old。
        """
        bb = self.mapping.bounding_box  # shape (3,2)
        mins = bb[:, 0]
        maxs = bb[:, 1]

        # 先采 2*num 个粗点
        coarse_num = num * 2
        pts_coarse = torch.rand(coarse_num, 3, device=self.device) * (maxs - mins)[None, :] + mins[None, :]

        # 用当前模型估计 SDF，选出 |SDF| 最小的 num 个点（near-surface）
        with torch.no_grad():
            sdf_coarse = self.mapping.model.query_sdf(pts_coarse, embed=True)
            if sdf_coarse.ndim > 1:
                sdf_coarse = sdf_coarse.mean(dim=-1)
            sdf_abs = sdf_coarse.abs()
            # 按 |sdf| 排序，取最小的 num 个
            _, idx = torch.topk(-sdf_abs, k=num)  # 取负号再 topk 相当于取最小
            pts_near = pts_coarse[idx]

        return pts_near  # (num,3)，更接近当前表面

    @torch.no_grad()
    def populate_buffer_from_teacher(self, teacher: CNMTeacher):
        """
        在 teacher 更新后调用一次，用 teacher 在 near-surface 区域生成伪样本并存入 buffer。
        """
        if not teacher.has_teacher():
            return

        # 在 near-surface 区域采样 num_samples_per_teacher 个点
        pts = self._sample_points_in_bbox(self.num_samples_per_teacher)  # (N,3)
        # 用 teacher 给 SDF（可能多通道）
        sdfs = teacher.query_sdf(pts)
        self.buffer.add_samples(pts, sdfs)

    def compute_loss(self, teacher: CNMTeacher) -> torch.Tensor:
        """
        计算 CNM 式的 off-surface loss：
          - 符号正则 ψ_s（exp(±α f)），保持 inside/outside 不翻；
          - 值一致性 |sdf_new - sdf_old|，在 near-surface 区域权重更大；
          - 两项都仅在 buffer 中采样的伪样本上计算。
        """
        if (not teacher.has_teacher()) or self.weight <= 0.0:
            return torch.tensor(0.0, device=self.device)

        if not self.buffer.has_data():
            return torch.tensor(0.0, device=self.device)

        # 从 CNM 专用 buffer 采一个 batch 伪样本 (x, sdf_old_all)
        pts, sdf_old_all = self.buffer.sample(self.batch_size)  # (B, C)

        # 将多通道 SDF 压成标量：这里取通道平均
        sdf_old_scalar = sdf_old_all.mean(dim=-1)  # (B,)

        # 当前网络预测：sdf_new_all = f(x; θ)，可能也是多通道 -> 压成标量
        sdf_new_all = self.mapping.model.query_sdf(pts, embed=True)
        if sdf_new_all.ndim == 1:
            sdf_new_scalar = sdf_new_all
        else:
            sdf_new_scalar = sdf_new_all.mean(dim=-1).squeeze()  # (B,)

        # ========= 1) 符号正则 ψ_s =========
        target_sign = torch.sign(sdf_old_scalar)
        alpha = self.alpha
        pos_mask = (target_sign > 0).float()
        neg_mask = (target_sign < 0).float()

        # outside 期望: 想要 sdf_new_scalar > 0 → ψ_s = exp(-α * sdf_new)
        loss_pos = pos_mask * torch.exp(-alpha * sdf_new_scalar)
        # inside  期望: 想要 sdf_new_scalar < 0 → ψ_s = exp( α * sdf_new)
        loss_neg = neg_mask * torch.exp(alpha * sdf_new_scalar)

        loss_symbolic = (loss_pos + loss_neg).mean()

        # ========= 2) 值一致性（value consistency） =========
        # near-surface 掩码：|sdf_old| < near_threshold
        near_mask = (sdf_old_scalar.abs() < self.near_threshold).float()
        far_mask = 1.0 - near_mask

        # L1 value consistency
        value_diff = (sdf_new_scalar - sdf_old_scalar).abs()

        # near 区域给高权重，far 区域给低权重
        loss_value_near = (near_mask * value_diff).sum() / (near_mask.sum() + 1e-8)
        loss_value_far = (far_mask * value_diff).sum() / (far_mask.sum() + 1e-8)

        loss_value = self.value_weight_near * loss_value_near + \
                     self.value_weight_far * loss_value_far

        # ========= 3) 总 CNM loss =========
        loss_total = loss_symbolic + loss_value
        return self.weight * loss_total