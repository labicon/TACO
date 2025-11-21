import torch
import torch.nn as nn
import copy

class EWC:
    def __init__(self, agent, config):
        self.agent = agent
        self.config = config
        self.device = agent.device
        
        # Hyperparameters
        self.ewc_lambda = config.get('ewc_lambda', 0.1)
        
        # State
        self.fisher_matrix = {}
        self.optimal_params = {}
        
        print(f"Agent {self.agent.agent_id}: EWC initialized with lambda={self.ewc_lambda}")

    def update_fisher(self):
        '''
        Compute the Fisher Information Matrix using ONLY the CURRENT frame data.
        In strict incremental learning, we cannot access past keyframes.
        '''
        # 获取当前帧 ID
        # 注意：在 run() 中调用时，通常是在处理完某一帧之后
        # 我们需要获取当前这一帧的数据来计算梯度
        
        # 假设 agent.est_c2w_data 中存有当前帧的 Pose
        if len(self.agent.est_c2w_data) == 0:
            return

        # 获取最新的一帧 ID
        cur_frame_id = max(self.agent.est_c2w_data.keys())
        
        # 准备数据：我们需要重新渲染当前帧的一些光线来计算梯度
        # 由于我们不能保存历史图像数据，我们只能利用模型当前的知识或者假设当前帧数据还在内存中
        # 在 main.py 的 run 循环中，batch 数据是传进来的，但 EWC 类里拿不到 batch。
        # 
        # 妥协方案：
        # 既然是增量式 SLAM，当前帧的 Pose 和 图像信息通常是已知的。
        # 但为了代码解耦，我们这里采用一种“自监督”式的近似，或者利用 agent 中可能缓存的当前帧信息。
        # 
        # 更严格的实现：
        # 我们利用 agent.keyframeDatabase 中的 *最后一帧* (Just added keyframe)。
        # 即使 enable_replay=False，KeyFrameDatabase 依然会暂存当前帧用于 Tracking/Mapping，
        # 只是它不会保留之前的帧。
        
        self.agent.model.train() 
        self.agent.map_optimizer.zero_grad()
        
        # 初始化 Fisher 矩阵 (如果是第一次，全0；如果是后续，可以选择累加或者移动平均)
        # 标准 EWC 是在任务切换时计算一个新的 F，并可能与旧 F 进行加权。
        # 这里简化为：每次更新都重新计算当前状态的 F，或者累积。
        # 为了防止 F 无限增大，通常采用移动平均: F_new = alpha * F_old + (1-alpha) * F_curr
        # 或者在 SLAM 中，我们通常只保留针对“上一时刻”的约束。
        
        # 这里我们采用：覆盖式更新 (假设每次调用都是一个新的 Task 结束)
        # 或者累加式。鉴于 SLAM 帧数多，建议使用移动平均或仅在关键帧时累加。
        
        # 如果字典为空，初始化
        if not self.fisher_matrix:
            self.fisher_matrix = {name: torch.zeros_like(p) for name, p in self.agent.model.named_parameters() if p.requires_grad}

        # 采样数量：只从当前这一帧采样
        num_rays_to_sample = self.agent.config['mapping']['sample']
        
        # 尝试从 KF Database 获取最后一帧的数据
        # 即使 replay=False，add_keyframe 也会把当前帧加进去，只是之后可能会被覆盖或不被采样
        # 我们需要一种方法获取当前帧的光线。
        
        # 方案：直接生成随机光线，利用当前 Pose。
        # 由于没有 GT 颜色（EWC 类没存），我们计算 log(p(x|theta)) 的梯度。
        # 在回归任务中，这等价于 MSE Loss 的梯度。
        # 我们使用模型预测值作为 target (自监督)，或者如果能拿到 GT 更好。
        # 鉴于代码结构，最稳妥的方式是：利用 KeyFrameDatabase 的 sample_global_rays 
        # 但限制它只采样 *最近加入的那一帧*。
        
        # 临时修改 KF database 的采样逻辑比较麻烦。
        # 我们手动构建当前帧的光线。
        
        c2w = self.agent.est_c2w_data[cur_frame_id].to(self.device)
        H, W = self.agent.dataset_info['H'], self.agent.dataset_info['W']
        
        # 随机采样像素
        import random
        idx = torch.tensor(random.sample(range(H * W), num_rays_to_sample)).to(self.device)
        y, x = torch.div(idx, W, rounding_mode='floor'), idx % W
        
        # 计算相机坐标系光线 (简化版，假设针孔模型参数在 config 或 dataset_info)
        # 这里需要 dataset 的 intrinsics。
        # 为了避免复杂的内参传递，我们利用 agent.keyframeDatabase (如果它存了当前帧)
        # 如果 enable_replay=False，KF database 可能只有当前这一帧，或者空的。
        
        # 【修正逻辑】：
        # 如果 enable_replay=False，main.py 里把 rays_to_save 设为了 0。
        # 这意味着 KF database 里确实没有光线数据！
        # 所以我们必须在 EWC update 时，显式地传入 batch 数据，或者让 main.py 传递当前帧数据。
        
        # 由于修改接口比较大，我们采用一个 trick：
        # 在 main.py 调用 update_fisher 时，我们其实是有 batch 数据的。
        # 我们修改 update_fisher 接收 batch 参数。
        pass 

    # 重写 update_fisher，接收 batch
    def update_fisher_with_batch(self, batch):
        self.agent.model.train() 
        self.agent.map_optimizer.zero_grad()
        
        if not self.fisher_matrix:
            self.fisher_matrix = {name: torch.zeros_like(p) for name, p in self.agent.model.named_parameters() if p.requires_grad}
            
        # 解析当前帧数据
        c2w = batch['c2w'][0].to(self.device)
        rays_d_cam = batch['direction'].reshape(-1, 3).to(self.device)
        target_s = batch['rgb'].reshape(-1, 3).to(self.device)
        target_d = batch['depth'].reshape(-1, 1).to(self.device)
        
        # 随机采样
        num_rays = rays_d_cam.shape[0]
        sample_size = self.agent.config['mapping']['sample']
        indices = torch.randperm(num_rays)[:sample_size]
        
        rays_d_cam = rays_d_cam[indices]
        target_s = target_s[indices]
        target_d = target_d[indices]
        
        # 转世界坐标
        rays_o = c2w[None, :3, -1].repeat(sample_size, 1)
        rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)
        
        # Forward
        ret = self.agent.model.forward(rays_o, rays_d, target_s, target_d)
        
        # 计算 Loss (MSE) -> 梯度
        loss = self.agent.get_loss_from_ret(ret, sdf=False, fs=False, smooth=False) 
        loss.backward()

        # 累积 Fisher (移动平均，alpha=0.95 保留历史，0.05 更新当前)
        # 或者直接累加（但在持续学习中通常会归一化）
        # 这里采用简单的累加策略，但在实际应用中可能需要衰减
        for name, param in self.agent.model.named_parameters():
            if param.grad is not None:
                # F_new = F_old + grad^2
                # 为了避免数值爆炸，通常会除以采样数 N，或者使用移动平均
                # 这里我们假设每次 update 都是针对一个新的 task 增量
                self.fisher_matrix[name] += param.grad.data.pow(2)
        
        # 更新最优参数为当前参数
        self.optimal_params = {name: p.clone().detach() for name, p in self.agent.model.named_parameters() if p.requires_grad}
        
        print(f"Agent {self.agent.agent_id}: EWC Fisher matrix updated using CURRENT frame.")
        self.agent.map_optimizer.zero_grad()

    def compute_loss(self):
        '''
        Calculate the EWC regularization loss.
        L = (lambda / 2) * sum( F_i * (theta_i - theta_i*)^2 )
        '''
        if not self.fisher_matrix or not self.optimal_params:
            return torch.tensor(0.0).to(self.device)
        
        loss = 0.0
        for name, param in self.agent.model.named_parameters():
            if name in self.fisher_matrix and name in self.optimal_params:
                fisher = self.fisher_matrix[name]
                optimal = self.optimal_params[name]
                loss += (fisher * (param - optimal).pow(2)).sum()
        
        return self.ewc_lambda * loss