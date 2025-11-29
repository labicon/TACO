import torch
import random
from typing import List, Dict, Tuple

class KRKeyframeReplay:
    """
    Standalone KR baseline:
    维护一个独立的 Replay Buffer。
    - save_every: 每隔多少帧保存一次 (Interval)
    - buffer_size: Buffer 最大容量 (Max Size)
    - 淘汰策略: 当 Buffer 满时，计算所有帧的重建 Loss，淘汰 Loss 最低（模型最熟悉）的一帧。
    """

    def __init__(self, mapping, buffer_size: int = 10, save_every: int = 50):
        self.mapping = mapping
        self.device = mapping.device
        self.buffer_size = buffer_size
        self.save_every = save_every
        
        # 独立 Buffer：存储字典列表 [{'rgb':..., 'depth':..., 'frame_id':...}, ...]
        # 数据存储在 CPU 上以节省显存
        self.buffer: List[Dict] = [] 

    def _compute_frame_loss(self, frame_data: Dict, num_samples: int = 1024) -> float:
        """
        计算单帧的重建误差 (RGB + Depth)。
        为了速度，只采样 num_samples 条射线。
        """
        frame_id = frame_data['frame_id']
        
        # 1. 获取 Pose (从 mapping 中获取当前估计的 Pose)
        # 注意：est_c2w_data 存储在 GPU 上
        if frame_id not in self.mapping.est_c2w_data:
            # 极端情况：如果 Pose 还没存（不太可能），返回 0 让其被淘汰
            return 0.0
        
        c2w = self.mapping.est_c2w_data[frame_id].to(self.device)

        # 2. 随机采样像素
        H, W, _ = frame_data['rgb'].shape[1:] # [1, H, W, 3]
        indices = torch.randint(0, H * W, (num_samples,))
        h_idx = indices // W
        w_idx = indices % W

        # 3. 准备数据 (CPU -> GPU)
        target_rgb = frame_data['rgb'][0, h_idx, w_idx, :].to(self.device)
        target_depth = frame_data['depth'][0, h_idx, w_idx].to(self.device).unsqueeze(-1)
        directions = frame_data['direction'][0, h_idx, w_idx, :].to(self.device)

        # 4. 转换射线 (Camera -> World)
        # rays_o: [N, 3]
        rays_o = c2w[None, :3, -1].repeat(num_samples, 1)
        # rays_d: [N, 3]
        rays_d = torch.sum(directions[..., None, :] * c2w[:3, :3], -1)

        # 5. 前向传播 (No Grad, Eval Mode 逻辑)
        with torch.no_grad():
            # 调用模型 forward
            ret = self.mapping.model(rays_o, rays_d, target_rgb, target_depth)
        
        # 6. 计算 Loss (L1 Loss)
        loss_rgb = torch.abs(ret['rgb'] - target_rgb).mean()
        loss_depth = torch.abs(ret['depth'] - target_depth).mean()
        
        total_loss = loss_rgb + loss_depth
        return total_loss.item()

    def add_frame(self, batch, frame_id: int):
        """
        尝试将当前帧添加到 KR 的独立 Buffer 中。
        """
        # 1. 间隔检查 (Interval Check)
        if frame_id % self.save_every != 0:
            return

        # 2. 关键帧对齐检查
        kf_every = self.mapping.config['mapping']['keyframe_every']
        if frame_id % kf_every != 0:
            return 

        # 3. 提取数据并转存到 CPU
        frame_data = {
            'rgb': batch['rgb'].detach().cpu(),
            'depth': batch['depth'].detach().cpu(),
            'direction': batch['direction'].detach().cpu(),
            'frame_id': frame_id
        }
        
        # 4. 容量检查与淘汰机制
        if len(self.buffer) >= self.buffer_size:
            # Buffer 已满，寻找 Loss 最小的帧进行淘汰
            min_loss = float('inf')
            remove_idx = -1
            
            # print(f"KR Buffer Full ({len(self.buffer)}), evaluating losses for eviction...")
            
            for i, existing_frame in enumerate(self.buffer):
                loss = self._compute_frame_loss(existing_frame)
                if loss < min_loss:
                    min_loss = loss
                    remove_idx = i
            
            if remove_idx != -1:
                # print(f"KR Eviction: Removing frame {self.buffer[remove_idx]['frame_id']} (Loss: {min_loss:.4f})")
                self.buffer.pop(remove_idx)
            else:
                # 理论上不应发生，如果发生则回退到 FIFO
                self.buffer.pop(0)

        # 5. 加入新帧
        self.buffer.append(frame_data)

    def sample_replay_rays(self, num_rays_total: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从 Buffer 中随机采样射线。
        """
        if len(self.buffer) == 0:
            return torch.empty(0, 7).to(self.device), torch.empty(0, dtype=torch.long).to(self.device)

        rays_list = []
        ids_list = []
        
        # 简单的均匀采样策略
        num_rays_per_frame = max(1, num_rays_total // len(self.buffer))
        kf_every = self.mapping.config['mapping']['keyframe_every']

        current_count = 0
        buffer_indices = list(range(len(self.buffer)))
        random.shuffle(buffer_indices)

        for idx in buffer_indices:
            if current_count >= num_rays_total:
                break
                
            frame_data = self.buffer[idx]
            n_samples = min(num_rays_per_frame, num_rays_total - current_count)
            
            H, W, _ = frame_data['rgb'].shape[1:]
            
            indices = torch.randint(0, H * W, (n_samples,))
            h_idx = indices // W
            w_idx = indices % W
            
            rgb = frame_data['rgb'][0, h_idx, w_idx, :].to(self.device)
            depth = frame_data['depth'][0, h_idx, w_idx].to(self.device).unsqueeze(-1)
            direction = frame_data['direction'][0, h_idx, w_idx, :].to(self.device)
            
            rays_batch = torch.cat([direction, rgb, depth], dim=-1)
            rays_list.append(rays_batch)
            
            fid = frame_data['frame_id']
            pose_idx = fid // kf_every
            ids_batch = torch.full((n_samples,), pose_idx, dtype=torch.long, device=self.device)
            ids_list.append(ids_batch)
            
            current_count += n_samples

        if len(rays_list) > 0:
            rays = torch.cat(rays_list, dim=0)
            ids_pose = torch.cat(ids_list, dim=0)
            return rays, ids_pose
        else:
            return torch.empty(0, 7).to(self.device), torch.empty(0, dtype=torch.long).to(self.device)