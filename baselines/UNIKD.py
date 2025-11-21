import torch
import torch.nn as nn
import copy
import numpy as np
import random
from scipy.spatial.transform import Rotation as R

class UNIKD:
    def __init__(self, agent, config):
        self.agent = agent
        self.config = config
        self.device = agent.device
        self.teacher_model = None
        
        # Hyperparameters
        self.uncertainty_threshold = config.get('uncertainty_threshold', 1.0)
        self.n_random_poses = config.get('n_random_poses', 10)
        self.eta = config.get('eta', 0.1) # Margin for loss
        self.beta_min = config.get('beta_min', 0.05) # Min uncertainty
        
        # Bounds for random inquirer
        self.pose_bounds = {
            'x': [float('inf'), float('-inf')],
            'y': [float('inf'), float('-inf')],
            'z': [float('inf'), float('-inf')],
            'yaw': [float('inf'), float('-inf')],
            'pitch': [float('inf'), float('-inf')],
            'roll': [float('inf'), float('-inf')]
        }

    def update_teacher(self):
        """
        将当前 Student 模型克隆为 Teacher。
        通常在任务边界或关键帧窗口滑动时调用。
        """
        if self.teacher_model is None:
            self.teacher_model = copy.deepcopy(self.agent.model)
        else:
            self.teacher_model.load_state_dict(self.agent.model.state_dict())
        
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        print(f"Agent {self.agent.agent_id}: UNIKD Teacher updated.")

    def update_pose_bounds(self, c2w_matrix):
        """
        更新历史相机参数的边界，用于 Random Inquirer。
        c2w_matrix: [4, 4] tensor or numpy
        """
        if isinstance(c2w_matrix, torch.Tensor):
            c2w_matrix = c2w_matrix.cpu().numpy()
            
        # Translation
        trans = c2w_matrix[:3, 3]
        for i, axis in enumerate(['x', 'y', 'z']):
            self.pose_bounds[axis][0] = min(self.pose_bounds[axis][0], trans[i])
            self.pose_bounds[axis][1] = max(self.pose_bounds[axis][1], trans[i])
            
        # Rotation (Matrix -> Euler)
        rot = c2w_matrix[:3, :3]
        r = R.from_matrix(rot)
        euler = r.as_euler('xyz', degrees=True) # yaw pitch roll approximation
        for i, axis in enumerate(['yaw', 'pitch', 'roll']):
            self.pose_bounds[axis][0] = min(self.pose_bounds[axis][0], euler[i])
            self.pose_bounds[axis][1] = max(self.pose_bounds[axis][1], euler[i])

    def sample_random_poses(self, n_poses):
        """
        Random Inquirer: 在历史 Pose 范围内随机采样。
        """
        poses = []
        for _ in range(n_poses):
            # Sample translation
            tx = random.uniform(self.pose_bounds['x'][0], self.pose_bounds['x'][1])
            ty = random.uniform(self.pose_bounds['y'][0], self.pose_bounds['y'][1])
            tz = random.uniform(self.pose_bounds['z'][0], self.pose_bounds['z'][1])
            
            # Sample rotation
            ry = random.uniform(self.pose_bounds['yaw'][0], self.pose_bounds['yaw'][1])
            rp = random.uniform(self.pose_bounds['pitch'][0], self.pose_bounds['pitch'][1])
            rr = random.uniform(self.pose_bounds['roll'][0], self.pose_bounds['roll'][1])
            
            r = R.from_euler('xyz', [ry, rp, rr], degrees=True)
            rot_mat = r.as_matrix()
            
            pose = np.eye(4)
            pose[:3, :3] = rot_mat
            pose[:3, 3] = [tx, ty, tz]
            poses.append(torch.from_numpy(pose).float().to(self.device))
            
        return torch.stack(poses)

    def get_distillation_batch(self, batch_size):
        """
        生成蒸馏用的 Batch。
        1. 随机采样 Pose
        2. Teacher 渲染
        3. 不确定性过滤
        """
        if self.teacher_model is None:
            return None, None, None

        # 1. Random Inquirer
        random_poses = self.sample_random_poses(self.n_random_poses)
        
        valid_rays_o = []
        valid_rays_d = []
        valid_teacher_rgb = []
        valid_teacher_unc = []
        
        H, W = self.agent.dataset_info['H'], self.agent.dataset_info['W']
        # 简单的相机内参假设 (FOV 90)，实际应从 dataset 读取 K
        fx = fy = W / 2.0 
        cx, cy = W / 2.0, H / 2.0
        
        # 2. & 3. Filter loop
        for i in range(self.n_random_poses):
            c2w = random_poses[i]
            
            # 随机采样该 Pose 下的一些光线进行测试 (Probe)
            # 为了效率，我们不渲染全图，只采样一部分光线来判断该 Pose 是否靠谱
            probe_size = 64 
            i_j = torch.randint(0, W, (probe_size, 2)).to(self.device)
            i, j = i_j[:, 0], i_j[:, 1]
            
            dirs = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1)
            rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
            rays_o = c2w[:3, -1].expand(rays_d.shape)
            
            with torch.no_grad():
                # 假设 Teacher 模型输出包含 'uncertainty'
                # 【修正】这里必须使用 target_rgb 而不是 target_s
                ret = self.teacher_model(rays_o, rays_d, target_rgb=None, target_d=None)
                
                if 'uncertainty' in ret:
                    uncertainty = ret['uncertainty']
                else:
                    # Fallback: 如果模型没改，给一个全 0 (即完全确信)
                    uncertainty = torch.zeros_like(ret['rgb'][..., 0:1])
                    # Uncertainty Filter Condition (Eq. 5)
            mean_uncertainty = uncertainty.mean()
            
            if mean_uncertainty < self.uncertainty_threshold:
                # 这个 Pose 是 Teacher 熟悉的，加入训练集
                valid_rays_o.append(rays_o)
                valid_rays_d.append(rays_d)
                valid_teacher_rgb.append(ret['rgb'])
                valid_teacher_unc.append(uncertainty)
        
        if len(valid_rays_o) == 0:
            return None, None, None

        # Concat
        rays_o = torch.cat(valid_rays_o, dim=0)
        rays_d = torch.cat(valid_rays_d, dim=0)
        teacher_rgb = torch.cat(valid_teacher_rgb, dim=0)
        teacher_unc = torch.cat(valid_teacher_unc, dim=0)
        
        # 如果数据太多，随机选 batch_size 个
        if rays_o.shape[0] > batch_size:
            idx = torch.randperm(rays_o.shape[0])[:batch_size]
            return rays_o[idx], rays_d[idx], {'rgb': teacher_rgb[idx], 'uncertainty': teacher_unc[idx]}
        
        return rays_o, rays_d, {'rgb': teacher_rgb, 'uncertainty': teacher_unc}


    def unikd_loss(self, pred_rgb, target_rgb, pred_uncertainty):
        """
        核心 Loss 公式 (Eq. 4 & Eq. 6)
        L = ||c - c*||^2 / 2 + ||c - c*||^2 / (2*beta^2) + log(beta) + eta
        """
        # 保证 beta > 0
        beta = pred_uncertainty + self.beta_min
        
        mse = (pred_rgb - target_rgb).pow(2).sum(dim=-1, keepdim=True) # [N, 1]
        
        loss = 0.5 * mse + 0.5 * mse / (beta.pow(2)) + torch.log(beta) + self.eta
        
        return loss.mean()