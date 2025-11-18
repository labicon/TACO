
from typing import List, Tuple, Optional
import random
import torch

class KRKeyframeReplay:
    """
    KR baseline: keyframe replay of last K keyframes, following iMAP-style replay.

    - 不修改原来的 KeyFrameDatabase 结构；
    - 只是在外面维护一个“最近 K 个 keyframe 的 frame_id 列表”，
      并在 global_BA 中额外采样这些 KF 的射线，形成一个附加 loss。
    """

    def __init__(self, mapping, K: int = 10):
        """
        Args:
            mapping: Mapping 实例（包含 keyframeDatabase, config 等）
            K:      最多保留的关键帧个数（默认 10）
        """
        self.mapping = mapping
        self.device = mapping.device
        self.K = K
        # 存储最近 K 个 keyframe 的 frame_id（对应 keyframeDatabase.frame_ids 的元素）
        self.replay_kf_ids: List[int] = []

    def register_keyframe(self, frame_id: int):
        """
        在每次 add_keyframe 时调用，用于维护“最近 K 个 keyframe”的列表。
        frame_id: 当前被加入 keyframeDatabase 的帧 id
        """
        self.replay_kf_ids.append(frame_id)
        # 只保留最近 K 个
        if len(self.replay_kf_ids) > self.K:
            self.replay_kf_ids = self.replay_kf_ids[-self.K:]

    def _sample_rays_from_kfs(self, num_rays: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从最近 K 个 keyframe 中采样 num_rays 条射线。
        返回:
            rays: (N, 7) = [dir, rgb, depth]
            ids:  (N,)   = 对应的 frame_id（原始帧 id）
        若 keyframe 不足，则返回空 tensor。
        """
        kf_db = self.mapping.keyframeDatabase
        if len(self.replay_kf_ids) == 0 or len(kf_db.frame_ids) == 0:
            return torch.empty(0, 7), torch.empty(0, dtype=torch.long)

        # 收集这些 KF 在数据库里的下标
        # keyframeDatabase.frame_ids 存的是所有 KF 对应的 frame_id
        # 我们只关心 replay_kf_ids 这个子集
        candidate_indices = []
        for idx, fid in enumerate(kf_db.frame_ids):
            if fid in self.replay_kf_ids:
                candidate_indices.append(idx)
        if len(candidate_indices) == 0:
            return torch.empty(0, 7), torch.empty(0, dtype=torch.long)

        # 简单做法：从这些 candidate KF 里 sample rays
        # KeyFrameDatabase 目前只支持 global sampling，我们这里复用它：
        rays, ids = kf_db.sample_global_rays(num_rays)
        # ids 是 KF 索引，我们需要把它映射回 frame_id
        if rays.shape[0] == 0:
            return rays, ids

        # 过滤掉不在 replay_kf_ids 中的射线
        kf_indices = ids.tolist()
        keep_mask = [kf_db.frame_ids[kid] in self.replay_kf_ids for kid in kf_indices]
        keep_mask = torch.tensor(keep_mask, dtype=torch.bool)
        rays = rays[keep_mask]
        ids = ids[keep_mask]
        # 映射 ids -> frame_id（真实帧 id）
        if rays.shape[0] > 0:
            frame_ids = torch.tensor([kf_db.frame_ids[int(k)] for k in ids], dtype=torch.long)
        else:
            frame_ids = torch.empty(0, dtype=torch.long)
        return rays, frame_ids

    def sample_rays(self, num_rays: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对外接口：从 KR 的最近 K 个 keyframe 中采样射线。
        """
        return self._sample_rays_from_kfs(num_rays)

    def sample_replay_rays(self, num_rays_total: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        兼容 Mapping.global_BA 中调用的接口。
        与 sample_rays 功能相同，但直接返回 (rays, pose_ids)，
        方便在 global_BA 里拼 poses_all 索引。

        Args:
            num_rays_total: 需要从最近 K 个 KF 中采样的总射线数量

        Returns:
            rays: [N, 7]
            ids_pose: [N]，对应 poses_all 的索引（通过 frame_id // keyframe_every 得到）
        """
        rays, frame_ids = self.sample_rays(num_rays_total)
        if rays.shape[0] == 0:
            return rays, torch.empty(0, dtype=torch.long)

        keyframe_every = self.mapping.config['mapping']['keyframe_every']
        ids_pose = (frame_ids // keyframe_every).clamp(min=0)
        return rays, ids_pose.to(torch.long)

    def compute_loss(
        self,
        poses_all: torch.Tensor,
        base_rays: torch.Tensor,
        base_ids_all: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 KR 的额外 loss。
        思路：
            - 从最近 K 个 KF 中额外采样一批 rays；
            - 用同样的 model.forward 计算 loss；
            - 将这部分 loss 作为 regularizer 返回。
        参数:
            poses_all: (N_pose, 4,4) BA 中用到的所有 pose（和 global_BA 里一致）
            base_rays: (N0, 7) 当前 BA 已经组装好的 rays（方向、RGB、深度）
            base_ids_all: (N0,) 与 base_rays 对应的 pose index（已经是 [0..N_pose) 的索引）
        返回:
            kr_loss: 一个标量张量，可直接加到总 loss 上。
        """
        # 从 KR keyframes 中取一批射线
        kr_cfg = self.mapping.config['training'].get('kr', {})
        kr_num = kr_cfg.get('num_replay_rays', 256)
        rays_kr, frame_ids_kr = self.sample_rays(kr_num)
        if rays_kr.shape[0] == 0:
            return torch.tensor(0.0, device=self.device)

        rays_kr = rays_kr.to(self.device)
        # frame_ids_kr 是原始帧 id，需要映射到 poses_all 的索引
        # global_BA 中 poses_all 的构成是：所有 KF pose + 当前帧 pose
        # 假设所有 KF pose 的顺序是 0, keyframe_every, 2*keyframe_every, ...
        keyframe_every = self.mapping.config['mapping']['keyframe_every']
        # 简单映射：index = frame_id // keyframe_every
        ids_kr = (frame_ids_kr // keyframe_every).clamp(min=0, max=poses_all.shape[0]-1)

        rays_d_cam = rays_kr[..., :3]
        target_s = rays_kr[..., 3:6]
        target_d = rays_kr[..., 6:7]

        # 和 global_BA 一样的射线变换
        rays_d = torch.sum(
            rays_d_cam[..., None, None, :] * poses_all[ids_kr, None, :3, :3],
            -1
        )
        rays_o = poses_all[ids_kr, None, :3, -1].repeat(1, rays_d.shape[1], 1).reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)

        ret_kr = self.mapping.model.forward(rays_o, rays_d, target_s, target_d)
        loss_kr = self.mapping.get_loss_from_ret(ret_kr)
        # 权重
        weight = kr_cfg.get('weight', 1.0)
        return weight * loss_kr