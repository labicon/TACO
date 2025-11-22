import torch
import random
import copy

class MAS_original:
    """
    Original (Offline) Memory Aware Synapses (MAS).
    严格遵循原始论文设定：
    1. 维护一个独立的 Replay Buffer (image_buffer) 存储当前任务的数据。
    2. 在 Mapping 过程中不计算梯度。
    3. 在任务边界 (Task Boundary) 冻结模型，遍历 Buffer 计算梯度并更新重要性矩阵 Omega。
    """
    def __init__(self, mapping, config):
        self.mapping = mapping
        self.device = mapping.device
        self.config = config
        self.lam = config.get('lambda', 1.0)
        self.buffer_size = config.get('buffer_size', 50) # 每个任务最多存多少帧

        # 重要性矩阵 Omega 和 参考参数 Theta*
        self.omega = {}
        self.theta_star = {}
        
        # 独立的 Replay Buffer，用于存储当前任务的帧
        # 结构: List[Dict] (batch data)
        self.image_buffer = []
        self.frames_seen_in_task = 0

    def init_if_needed(self):
        """初始化 Omega 和 Theta* 结构"""
        if self.omega:
            return
        for name, p in self.mapping.model.named_parameters():
            if p.requires_grad:
                self.omega[name] = torch.zeros_like(p, device=self.device)
                self.theta_star[name] = p.detach().clone()

    def add_to_buffer(self, batch):
        """
        将当前帧加入 Buffer。
        如果 Buffer 满了，使用蓄水池采样 (Reservoir Sampling) 替换旧帧，
        保证 Buffer 中的数据能代表整个任务的分布。
        """
        # 深拷贝并转到 CPU 以节省显存
        batch_cpu = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_cpu[k] = v.cpu().clone()
            else:
                batch_cpu[k] = copy.deepcopy(v)

        if len(self.image_buffer) < self.buffer_size:
            self.image_buffer.append(batch_cpu)
        else:
            # 蓄水池采样：以 buffer_size / N 的概率保留新数据
            idx = random.randint(0, self.frames_seen_in_task)
            if idx < self.buffer_size:
                self.image_buffer[idx] = batch_cpu
        
        self.frames_seen_in_task += 1

    def calculate_importance(self):
        """
        【核心逻辑】离线计算重要性。
        在任务结束时调用，遍历 Buffer，计算梯度，更新 Omega。
        """
        if not self.image_buffer:
            return

        print(f"Agent {self.mapping.agent_id}: MAS_original computing importance on {len(self.image_buffer)} buffered frames...")
        
        self.init_if_needed()
        
        # 1. 切换到 Eval 模式，防止 BN 层更新 (虽然 NICE-SLAM 少用 BN)
        self.mapping.model.eval()
        self.mapping.map_optimizer.zero_grad()
        
        # 临时累积器，用于当前任务
        omega_current = {n: torch.zeros_like(p) for n, p in self.mapping.model.named_parameters() if p.requires_grad}
        
        num_samples = 0
        sample_pixels = self.mapping.config['mapping']['sample']

        # 2. 遍历 Buffer 回放数据
        for batch in self.image_buffer:
            # 准备数据 (转回 GPU)
            c2w = batch['c2w'][0].to(self.device)
            H, W = self.mapping.dataset_info['H'], self.mapping.dataset_info['W']
            
            # 随机采样像素
            idx = random.sample(range(H * W), sample_pixels)
            idx_h = [i // W for i in idx]
            idx_w = [i % W for i in idx]
            
            rays_d_cam = batch['direction'].squeeze(0)[idx_h, idx_w, :].to(self.device)
            
            # 转世界坐标
            rays_o = c2w[None, :3, -1].repeat(sample_pixels, 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)
            
            # Forward (不需要 target，只需要输出)
            ret = self.mapping.model.forward(rays_o, rays_d, target_rgb=None, target_d=None)
            
            # 3. 计算 Proxy Loss: 输出函数的 L2 范数平方
            # MAS 论文公式: || F(x; \theta) ||^2
            loss = 0.0
            if 'rgb' in ret: loss += ret['rgb'].pow(2).sum()
            if 'depth' in ret: loss += ret['depth'].pow(2).sum()
            if 'sdf' in ret: loss += ret['sdf'].pow(2).sum()
            
            # 平均化 (Scale invariant)
            loss = loss / sample_pixels
            
            loss.backward()
            
            # 4. 累积梯度绝对值
            for name, p in self.mapping.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    omega_current[name] += p.grad.abs()
            
            self.mapping.map_optimizer.zero_grad()
            num_samples += 1

        # 平均化当前任务的重要性
        if num_samples > 0:
            for name in omega_current:
                omega_current[name] /= num_samples

        # 5. 更新全局 Omega (累加)
        for name in self.omega:
            self.omega[name] += omega_current[name]

        # 6. 更新锚点参数 Theta*
        for name, p in self.mapping.model.named_parameters():
            if p.requires_grad:
                self.theta_star[name] = p.detach().clone()

        # 7. 清空 Buffer，准备下一个任务
        self.image_buffer = []
        self.frames_seen_in_task = 0
        
        # 恢复 Train 模式
        self.mapping.model.train()
        print(f"Agent {self.mapping.agent_id}: MAS_original importance updated.")

    def mas_loss(self):
        """
        计算正则化 Loss: λ * Σ Ω_ij (θ_ij - θ*_ij)^2
        """
        if not self.omega or not self.theta_star:
            return torch.tensor(0.0, device=self.device)
        
        loss = 0.0
        for name, p in self.mapping.model.named_parameters():
            if name in self.omega and p.requires_grad:
                loss += (self.omega[name] * (p - self.theta_star[name]).pow(2)).sum()
        
        return self.lam * loss