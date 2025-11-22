import os
#os.environ['TCNN_CUDA_ARCHITECTURES'] = '86'
import shutil

# Package imports
import torch
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
import argparse
import json
import copy
import shutil 
import psutil

from torch.utils.tensorboard import SummaryWriter


from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from collections import deque

import networkx as nx 
import matplotlib.pyplot as plt

# Local imports
import config
from model.scene_rep import JointEncoding
from model.keyframe import KeyFrameDatabase
from model.decoder_NICESLAM import NICE
from datasets.dataset import get_dataset
from utils import coordinates, extract_mesh, colormap_image
from tools.eval_ate import pose_evaluation
from optimization.utils import at_to_transform_matrix, qt_to_transform_matrix, matrix_to_axis_angle, matrix_to_quaternion
from baselines.CNM import CNMTeacher, CNMOffsurfaceReplay
from baselines.KR import KRKeyframeReplay
from baselines.MAS import MAS
from baselines.MAS_original import MAS_original
from baselines.EWC import EWC 
from baselines.UNIKD import UNIKD

import sys

from torch.nn.utils import parameters_to_vector as p2v
import copy


class Mapping():
    def __init__(self, config, id, dataset_info):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent_id = id 
        self.dataset_info = dataset_info 

        self.create_bounds()
        self.create_pose_data()
        self.get_pose_representation()
        self.keyframeDatabase = self.create_kf_database(config)
        self.model = JointEncoding(config, self.bounding_box).to(self.device)
        self.fix_decoder = config['multi_agents']['fix_decoder']
        self.create_optimizer()


        # add tf for every agent
        log_dir = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], f'agent_{self.agent_id}', 'logs')
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"Agent {self.agent_id} TensorBoard logs will be saved to: {log_dir}")

      
        self.dist_algorithm = config['multi_agents']['distributed_algorithm']
        self.track_uncertainty = config['multi_agents']['track_uncertainty']

        
        if self.track_uncertainty:
            embed_fn_params_vec = p2v(self.model.embed_fn.parameters())
            self.uncertainty_tensor = torch.zeros(embed_fn_params_vec.size()).to(self.device)
            self.W_i = torch.zeros(self.uncertainty_tensor.size()).to(self.device)
           
        self.total_loss = []
        self.obj_loss = []
        self.lag_loss = []
        self.aug_loss = []

        # initialize dual variable 
        theta_i = p2v(self.model.parameters())
        if self.config['edge_based'] == False:
            self.p_i = torch.zeros(theta_i.size()).to(self.device) # combination of dual variables
        else:
            theta_i_size = p2v(self.model.parameters()).size()
            # Change p_i to a dictionary p_ij to store dual variables per edge 
            self.p_ij = {} # Key: neighbor_id, Value: dual variable tensor
        # a list to hold neighbor model parameters, and uncertainty tensor (optional)
        self.neighbors = []
        # step size in the gradient ascent of the dual variable
        self.rho = config['multi_agents']['rho']

        # for DSGD/DSGT
        self.ds_mat = None # doubly stochastic matrix for DSGD/DSGT
        self.num_params = sum( p.numel() for p in self.model.parameters() )
        self.alpha = config['multi_agents']['alpha']
        base_zeros = [
            torch.zeros_like(p, requires_grad=False, device=self.device)
            for p in self.model.parameters()
        ]
        self.g_dsgt = copy.deepcopy(base_zeros)
        self.y_dsgt = copy.deepcopy(base_zeros) 

        self.com_perIter = 0 # communication cost in MB per communication iteration 
        self.com_total = 0 # total accumulated communication cost in MB 

        self.gt_pose = config['tracking']['gt_pose']
        print(f'If agent{self.agent_id} uses gt pose: {self.gt_pose}')


        # --- 新增：时序共识 (Incremental UDON) 参数 ---
        self.temporal_consensus_config = self.config['training'].get('temporal_consensus', {})
        self.temporal_consensus_enabled = self.temporal_consensus_config.get('enabled', False)
        self.uncert_decay = self.temporal_consensus_config.get('uncertainty_decay', 1.0)
        if self.temporal_consensus_enabled:
            self.K = self.temporal_consensus_config.get('K', 1)  # 快照窗口大小
            # 使用maxlen的deque可以自动维护一个固定大小的滑动窗口
            # 队列中存储 (参数, 不确定性) 的元组
            self.temporal_snapshots = deque(maxlen=self.K)
            print(f"Agent {self.agent_id} has Temporal Consensus enabled with window size K={self.K}.")

        # EWC parameters
        self.ewc_enabled = config['training'].get('ewc_enabled', False)
        self.ewc_enabled = config['training'].get('ewc_enabled', False)
        if self.ewc_enabled:
            ewc_cfg = self.config # EWC config is usually mixed in training
            self.ewc = EWC(self, ewc_cfg)


        # --- CNM: Continual Neural Mapping 基线 ---
        self.cnm_enabled = self.config['training'].get('cnm_enabled', False)
        if self.cnm_enabled:
            cnm_cfg = self.config['training'].get('cnm', {})
            # 初始化 teacher
            self.cnm_teacher = CNMTeacher(self)
            # 初始化 CNM 自己的 function replay buffer
            from baselines.CNM import CNMReplayBuffer  # 也可以在文件头一起 import
            buffer_size = cnm_cfg.get('buffer_size', 200000)
            batch_size = cnm_cfg.get('num_offsurface_samples', 2048)
            self.cnm_buffer = CNMReplayBuffer(
                device=self.device,
                max_points=buffer_size,
                sample_batch_size=batch_size,
            )
            # 初始化 off-surface replay 控制器
            self.cnm_replay = CNMOffsurfaceReplay(self, self.cnm_buffer)
            print(f"Agent {self.agent_id} has CNM enabled.")

        # --- KR: Keyframe Replay baseline (replay last K keyframes) ---
        self.kr_enabled = self.config['training'].get('kr_enabled', False)
        if self.kr_enabled:
            kr_cfg = self.config['training'].get('kr', {})
            kr_K = kr_cfg.get('K', 10)
            self.kr_replay = KRKeyframeReplay(self, K=kr_K)
            print(f"Agent {self.agent_id} has KR enabled with K={kr_K}.")

        # --- MAS: Memory Aware Synapses baseline ---
        self.mas_enabled = self.config['training'].get('mas_enabled', False)
        if self.mas_enabled:
            mas_cfg = self.config['training'].get('mas', {})
            mas_lambda = mas_cfg.get('lambda', 1.0)
            self.mas = MAS(self, lam=mas_lambda)
            print(f"Agent {self.agent_id} has MAS enabled with lambda={mas_lambda}.")

         # --- 【新增】MAS Original: Offline/Post-hoc baseline ---
        self.mas_original_enabled = self.config['training'].get('mas_original_enabled', False)
        if self.mas_original_enabled:
            mas_orig_cfg = self.config['training'].get('mas_original', {})
            self.mas_original = MAS_original(self, mas_orig_cfg)
            print(f"Agent {self.agent_id} has MAS_original (Offline) enabled.")
            

        # --- 【新增】UNIKD 初始化 ---
        self.unikd_enabled = self.config['training'].get('unikd_enabled', False)
        if self.unikd_enabled:
            unikd_cfg = self.config['training'].get('unikd', {})
            self.unikd = UNIKD(self, unikd_cfg)
            print(f"Agent {self.agent_id} has UNIKD enabled.")

        # Replay parameter
        self.enable_replay = self.config['mapping'].get('enable_replay', True)

    def seed_everything(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # --- 新增：更新历史快照的方法 ---
    def update_temporal_snapshot(self):
        """
        捕获当前模型状态，并将其作为新的历史快照添加到滑动窗口中。
        """
        if not self.temporal_consensus_enabled:
            return
        
        #print(f"Agent {self.agent_id}: Capturing new temporal snapshot.")
        
        current_params = p2v(self.model.parameters()).clone().detach()
        current_uncertainty = None
        if self.track_uncertainty:
            """# 确保不确定性张量存在
            if not hasattr(self, 'uncertainty_tensor'):
                 embed_fn_params_vec = p2v(self.model.embed_fn.parameters())
                 self.uncertainty_tensor = torch.zeros(embed_fn_params_vec.size()).to(self.device)"""
            current_uncertainty = self.uncertainty_tensor.clone().detach()
        
        self.temporal_snapshots.append((current_params, current_uncertainty))
    # --- 结束新增 ---

    def extra_persistent_mem_bytes(self):
        """估算‘额外持久存储’的运行期占用（不含模型本体/激活/缓存池）"""
        bpe = 4  # 以 fp32 计
        bytes_uncert = int(self.uncertainty_tensor.numel()) * bpe if hasattr(self, 'uncertainty_tensor') else 0
        bytes_Wi = int(self.W_i.numel()) * bpe if hasattr(self, 'W_i') else 0

        # 时序快照（K<=1 时至多一份）
        bytes_snapshot = 0
        if getattr(self, 'temporal_consensus_enabled', False) and hasattr(self, 'temporal_snapshots') and len(self.temporal_snapshots) > 0:
            snap_params, snap_uncert = self.temporal_snapshots[-1]
            bytes_snapshot += int(snap_params.numel()) * bpe
            if snap_uncert is not None:
                bytes_snapshot += int(snap_uncert.numel()) * bpe

        # 伪邻居对偶变量（负数ID），存在则计入（大小≈|theta|）
        bytes_dual_pseudo = 0
        if hasattr(self, 'p_ij'):
            for k, v in self.p_ij.items():
                if isinstance(k, int) and k < 0 and hasattr(v, 'numel'):
                    bytes_dual_pseudo += int(v.numel()) * bpe

        # 关键帧重放：关闭则为 0
        bytes_replay = 0

        # 常驻教师网络：当前无单独教师副本
        bytes_teacher = 0

        total = bytes_uncert + bytes_Wi + bytes_snapshot + bytes_dual_pseudo + bytes_replay + bytes_teacher
        return {
            'total': total,
            'uncertainty_tensor': bytes_uncert,
            'W_i': bytes_Wi,
            'temporal_snapshot': bytes_snapshot,
            'dual_pseudo': bytes_dual_pseudo,
            'replay': bytes_replay,
            'teacher': bytes_teacher,
        }
        
    def log_extra_persistent_mem(self, step: int):
        info = self.extra_persistent_mem_bytes()
        to_mb = lambda b: b / (1024**2)
        # 打印
        # print(f"[ExtraPersistentMem] step={step} | total={to_mb(info['total']):.2f} MB "
        #       f"(uncert={to_mb(info['uncertainty_tensor']):.2f}, Wi={to_mb(info['W_i']):.2f}, "
        #       f"snapshot={to_mb(info['temporal_snapshot']):.2f}, dual_pseudo={to_mb(info['dual_pseudo']):.2f}, "
        #       f"replay={to_mb(info['replay']):.2f}, teacher={to_mb(info['teacher']):.2f})")
        # 写入 TensorBoard
        self.writer.add_scalar('ExtraMem/Total_MB', to_mb(info['total']), step)
        for k in ['uncertainty_tensor','W_i','temporal_snapshot','dual_pseudo','replay','teacher']:
            self.writer.add_scalar(f'ExtraMem/{k}_MB', to_mb(info[k]), step)
        # 追加到文件
        out_dir = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], f'agent_{self.agent_id}')
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, 'extra_persistent_mem.csv')
        if not os.path.exists(path):
            with open(path, 'w') as f:
                f.write('step,total_mb,uncertainty_mb,Wi_mb,snapshot_mb,dual_pseudo_mb,replay_mb,teacher_mb\n')
        with open(path, 'a') as f:
            f.write(f"{step},{to_mb(info['total']):.2f},{to_mb(info['uncertainty_tensor']):.2f},"
                    f"{to_mb(info['W_i']):.2f},{to_mb(info['temporal_snapshot']):.2f},"
                    f"{to_mb(info['dual_pseudo']):.2f},{to_mb(info['replay']):.2f},{to_mb(info['teacher']):.2f}\n")

        
    def get_pose_representation(self):
        '''
        Get the pose representation axis-angle or quaternion
        '''
        if self.config['training']['rot_rep'] == 'axis_angle':
            self.matrix_to_tensor = matrix_to_axis_angle
            self.matrix_from_tensor = at_to_transform_matrix
            print('Using axis-angle as rotation representation, identity init would cause inf')
        
        elif self.config['training']['rot_rep'] == "quat":
            print("Using quaternion as rotation representation")
            self.matrix_to_tensor = matrix_to_quaternion
            self.matrix_from_tensor = qt_to_transform_matrix
        else:
            raise NotImplementedError
        

    def log_memory(self, step: int):
        # 记录 CPU 内存（RSS）
        process = psutil.Process(os.getpid())
        rss_mb = process.memory_info().rss / (1024**2)
        self.writer.add_scalar('Memory/CPU_RSS_MB', rss_mb, step)

        # 记录 GPU 显存（本进程）
        alloc_mb = reserved_mb = max_alloc_mb = 0.0
        if torch.cuda.is_available():
            dev = self.device
            try:
                torch.cuda.synchronize(dev)
            except Exception:
                pass
            alloc_mb = torch.cuda.memory_allocated(dev) / (1024**2)
            reserved_mb = torch.cuda.memory_reserved(dev) / (1024**2)
            max_alloc_mb = torch.cuda.max_memory_allocated(dev) / (1024**2)
            self.writer.add_scalar('Memory/GPU_Allocated_MB', alloc_mb, step)
            self.writer.add_scalar('Memory/GPU_Reserved_MB', reserved_mb, step)
            self.writer.add_scalar('Memory/GPU_MaxAllocated_MB', max_alloc_mb, step)

        # 追加到 CSV（每个 agent 各一份）
        out_dir = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], f'agent_{self.agent_id}')
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, 'mem_usage.csv')
        header_needed = not os.path.exists(csv_path)
        with open(csv_path, 'a') as f:
            if header_needed:
                f.write('step,cpu_rss_mb,gpu_alloc_mb,gpu_reserved_mb,gpu_max_alloc_mb\n')
            f.write(f'{step},{rss_mb:.2f},{alloc_mb:.2f},{reserved_mb:.2f},{max_alloc_mb:.2f}\n')


    def create_pose_data(self):
        '''
        Create the pose data
        '''
        self.est_c2w_data = {}
        self.est_c2w_data_rel = {}
    

    def create_bounds(self):
        '''
        Get the pre-defined bounds for the scene
        '''
        self.bounding_box = torch.from_numpy(np.array(self.config['mapping']['bound'])).to(torch.float32).to(self.device)
        self.marching_cube_bound = torch.from_numpy(np.array(self.config['mapping']['marching_cubes_bound'])).to(torch.float32).to(self.device)


    def create_kf_database(self, config):  
        '''
        Create the keyframe database
        '''
        num_kf = int(self.dataset_info['num_frames'] // self.config['mapping']['keyframe_every'] + 1)  
        print('#kf:', num_kf)
        print('#Pixels to save:', self.dataset_info['num_rays_to_save'])

        # 关键：关闭 replay 时，KF 数据库存到 CPU，且不预留射线缓冲
        use_replay = self.config['mapping'].get('enable_replay', True)
        dev = self.device if use_replay else torch.device('cpu')
        rays_to_save = self.dataset_info['num_rays_to_save'] if use_replay else 0

        return KeyFrameDatabase(
            config,
            self.dataset_info['H'],
            self.dataset_info['W'],
            num_kf,
            rays_to_save,
            dev
        )

    def save_state_dict(self, save_path):
        torch.save(self.model.state_dict(), save_path)
    

    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path))


    def load_decoder(self, load_path):
        dict = torch.load(load_path, weights_only=True)
        model_dict = dict['model']
        del model_dict['embedpos_fn.params']
        del model_dict['embed_fn.params']

        if self.unikd_enabled:
                    own_state = self.model.state_dict()
                    for name, param in list(model_dict.items()):
                        if name in own_state:
                            if param.shape != own_state[name].shape:
                                # 检测是否是 ColorNet 的输出层 (3 -> 4)
                                # 权重形状通常是 [out_channels, in_channels]
                                if param.ndim == 2 and param.shape[0] == 3 and own_state[name].shape[0] == 4 and param.shape[1] == own_state[name].shape[1]:
                                    print(f"Agent {self.agent_id}: Adapting {name} from {param.shape} to {own_state[name].shape} for UNIKD.")
                                    
                                    # 1. 创建新的参数，使用当前模型的初始化值 (这样第4维就是随机初始化的，不会是0)
                                    new_param = own_state[name].clone() 
                                    
                                    # 2. 将预训练的 RGB 权重 (前3行) 复制过去
                                    new_param[:3, :] = param 
                                    
                                    # 3. 更新字典，替换掉旧的参数
                                    model_dict[name] = new_param

        self.model.load_state_dict(model_dict, strict=False) # load from a partial state_dict missing some keys, use strict=False
    

    def save_ckpt(self, save_path):
        '''
        Save the model parameters and the estimated pose
        '''
        save_dict = {'pose': self.est_c2w_data,
                     'pose_rel': self.est_c2w_data_rel,
                     'total_loss': self.total_loss,
                     'obj_loss': self.obj_loss,
                     'lag_loss': self.lag_loss,
                     'aug_loss': self.aug_loss,
                     'model': self.model.state_dict()}
        torch.save(save_dict, save_path)
        print('Save the checkpoint')


    def load_ckpt(self, load_path):
        '''
        Load the model parameters and the estimated pose
        '''
        dict = torch.load(load_path)
        self.model.load_state_dict(dict['model'])
        self.est_c2w_data = dict['pose']
        self.est_c2w_data_rel = dict['pose_rel']


    def select_samples(self, H, W, samples):
        '''
        randomly select samples from the image
        '''
        #indice = torch.randint(H*W, (samples,))
        indice = random.sample(range(H * W), int(samples))
        indice = torch.tensor(indice)
        return indice


    def get_loss_from_ret(self, ret, rgb=True, sdf=True, depth=True, fs=True, smooth=False):
        '''
        Get the training loss
        '''
        loss = 0
        if rgb:
            loss += self.config['training']['rgb_weight'] * ret['rgb_loss']
        if depth:
            loss += self.config['training']['depth_weight'] * ret['depth_loss']
        if sdf:
            loss += self.config['training']['sdf_weight'] * ret["sdf_loss"]
        if fs:
            loss +=  self.config['training']['fs_weight'] * ret["fs_loss"]
        
        if smooth and self.config['training']['smooth_weight']>0:
            loss += self.config['training']['smooth_weight'] * self.smoothness(self.config['training']['smooth_pts'], 
                                                                                  self.config['training']['smooth_vox'], 
                                                                                  margin=self.config['training']['smooth_margin'])
        
        return loss             


    def first_frame_mapping(self, batch, n_iters=100):
        '''
        First frame mapping
        Params:
            batch['c2w']: [1, 4, 4]
            batch['rgb']: [1, H, W, 3]
            batch['depth']: [1, H, W, 1]
            batch['direction']: [1, H, W, 3]
        Returns:
            ret: dict
            loss: float
        
        '''
        print(f'Agent {self.agent_id} First frame mapping...')
        c2w = batch['c2w'][0].to(self.device)
        self.est_c2w_data[0] = c2w
        self.est_c2w_data_rel[0] = c2w

        self.model.train()

        # Training
        for i in range(n_iters):
            self.map_optimizer.zero_grad()
            indice = self.select_samples(self.dataset_info['H'], self.dataset_info['W'], self.config['mapping']['sample'])
            
            indice_h, indice_w = indice % (self.dataset_info['H']), indice // (self.dataset_info['H'])
            rays_d_cam = batch['direction'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_s = batch['rgb'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_d = batch['depth'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1)

            rays_o = c2w[None, :3, -1].repeat(self.config['mapping']['sample'], 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)

            # Forward
            ret = self.model.forward(rays_o, rays_d, target_s, target_d)

            # 1. 先清空梯度
            self.map_optimizer.zero_grad()

            # 2. 【新增】计算结构梯度 (Proxy Loss)
            if self.track_uncertainty:
                mas_proxy_loss = 0.0
                if 'rgb' in ret: mas_proxy_loss += ret['rgb'].pow(2).sum()
                if 'depth' in ret: mas_proxy_loss += ret['depth'].pow(2).sum()
                if 'sdf' in ret: mas_proxy_loss += ret['sdf'].pow(2).sum()
                mas_proxy_loss = mas_proxy_loss / ret['rgb'].shape[0]
                
                mas_proxy_loss.backward(retain_graph=True)
                
                grid_grad = self.model.embed_fn.params.grad
                if grid_grad is not None:
                    grad_mag = torch.abs(grid_grad)
                    # 使用和 global_BA 一样的更新策略
                    self.uncertainty_tensor *= self.uncert_decay
                    self.uncertainty_tensor += grad_mag
                
                # 清空梯度，准备算真正的 Loss
                self.map_optimizer.zero_grad()

            loss = self.get_loss_from_ret(ret)
            loss.backward()
            
            """            
            if self.track_uncertainty:

                grid_grad = self.model.embed_fn.params.grad
                
                if grid_grad is not None and grid_grad.numel() > 0:
                    grid_has_grad = (torch.abs(grid_grad) > 0).to(torch.int32)
                    self.uncertainty_tensor *= self.uncert_decay
                    self.uncertainty_tensor += grid_has_grad
            """
                

            self.map_optimizer.step()

        # First frame will always be a keyframe
        self.keyframeDatabase.add_keyframe(batch, filter_depth=self.config['mapping']['filter_depth'])
        if self.config['mapping']['first_mesh']:
            self.save_mesh(0)
        
        print(f'Agent {self.agent_id} First frame mapping done')
        return ret, loss

    def smoothness(self, sample_points=256, voxel_size=0.1, margin=0.05, color=False):
        '''
        Smoothness loss of feature grid
        '''
        volume = self.bounding_box[:, 1] - self.bounding_box[:, 0]

        grid_size = (sample_points-1) * voxel_size
        offset_max = self.bounding_box[:, 1]-self.bounding_box[:, 0] - grid_size - 2 * margin

        offset = torch.rand(3).to(offset_max) * offset_max + margin
        coords = coordinates(sample_points - 1, 'cpu', flatten=False).float().to(volume)
        pts = (coords + torch.rand((1,1,1,3)).to(volume)) * voxel_size + self.bounding_box[:, 0] + offset

        if self.config['grid']['tcnn_encoding']:
            pts_tcnn = (pts - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])
        

        sdf = self.model.query_sdf(pts_tcnn, embed=True)
        tv_x = torch.pow(sdf[1:,...]-sdf[:-1,...], 2).sum()
        tv_y = torch.pow(sdf[:,1:,...]-sdf[:,:-1,...], 2).sum()
        tv_z = torch.pow(sdf[:,:,1:,...]-sdf[:,:,:-1,...], 2).sum()

        loss = (tv_x + tv_y + tv_z)/ (sample_points**3)

        return loss
    

    def get_gamma_t(self, t: int) -> float:
        """
        时序共识折扣系数 γ_t，默认常数；如需调度可在此实现 schedule。
        t 可以用 cur_frame_id 或内层迭代 i。
        """
        if not self.temporal_consensus_enabled:
            return 1.0
        return float(self.temporal_consensus_config.get('gamma', 1.0))
    
    def scaling_AUQ_CADMM(self, k, uncertainty_i, uncertainty_j):

        uncertainty = uncertainty_i + uncertainty_j
        a_1 = self.rho/1000
        b_1 = self.rho

        # scale to a_1 and b_1: uncertainty_scaled = p*uncertainty + q
        p = (b_1-a_1)/(torch.max(uncertainty) - torch.min(uncertainty)) 
        q = a_1 - p*torch.min(uncertainty)
        return p, q

    def communicate(self,input):
        if self.dist_algorithm == 'AUQ_CADMM':
            if self.config['edge_based'] == True:
                neighbor_id = input[0]
                model_j = input[1]
                uncertainty_j = input[2]
                theta_j = p2v(model_j.parameters()).detach()
                # The list now acts as a collection of "flags" for successful communication
                # It stores [neighbor_id, theta_j, uncertainty_j]
                self.neighbors.append( [neighbor_id, theta_j, uncertainty_j] )
            else:
                model_j = input[0]  
                theta_j = p2v(model_j.parameters()).detach()
                uncertainty_j = input[1].detach()
                step = input[2]
                self.neighbors.append( [theta_j, uncertainty_j] )

        elif self.dist_algorithm in ('CADMM', 'MACIM'):
            if self.config['edge_based']:
                neighbor_id = input[0]
                model_j = input[1]
                theta_j = p2v(model_j.parameters()).detach()
                self.neighbors.append( [neighbor_id, theta_j] )
            else:
                model_j = input[0]  
                theta_j = p2v(model_j.parameters()).detach()
                self.neighbors.append( [theta_j] )

        elif self.dist_algorithm == 'DSGD':
            model_j = input[0]  
            
            j = input[1]
            self.neighbors.append( [model_j.parameters(), j] )

        elif self.dist_algorithm == 'DSGT':
            model_j = input[0]  
            
            y_dsgt_j = input[1]
            j = input[2]
            self.neighbors.append( [model_j.parameters(), y_dsgt_j, j] )

    def dual_update(self, theta_i_k):
        if self.config['edge_based']:
            for neighbor in self.neighbors:
                neighbor_id = neighbor[0]
                theta_j_k = neighbor[1]
                # 确保 p_ij 中有该邻居的条目
                if neighbor_id in self.p_ij:
                    self.p_ij[neighbor_id] += self.rho * (theta_i_k - theta_j_k)
        else:
            for neighbor in self.neighbors:
                theta_j_k = neighbor[0]
                self.p_i += self.rho * (theta_i_k - theta_j_k)    


    def dual_update_AUQ_CADMM(self, theta_i_k, uncertainty_i, k):
        padding_size = theta_i_k.size(0) - uncertainty_i.size(0)
        if self.config['edge_based'] == True:
            for neighbor in self.neighbors:
                neighbor_id = neighbor[0]
                theta_j_k = neighbor[1]
                uncertainty_j = neighbor[2]
                
                p, q = self.scaling_AUQ_CADMM(k, uncertainty_i, uncertainty_j)
                # 非对称：伪邻居（负数 ID）时对 W_i 乘以 gamma_t
                gamma_t = 1.0
                if isinstance(neighbor_id, int) and neighbor_id < 0:
                    gamma_t = self.get_gamma_t(k)

                W_i = gamma_t * (p*uncertainty_i + q)
                W_i = torch.nn.functional.pad(W_i, (0, padding_size), "constant", self.rho)
                W_j = p*uncertainty_j + q
                W_j = torch.nn.functional.pad(W_j, (0, padding_size), "constant", self.rho)

                denominator = W_i + W_j
                epsilon = 1e-8
                update_term = 2*W_i * torch.div(W_j*theta_i_k - W_j*theta_j_k, denominator + epsilon)

                # Update the specific dual variable for this neighbor
                self.p_ij[neighbor_id] += update_term


        else:
            for neighbor in self.neighbors:
                theta_j_k = neighbor[0]
                uncertainty_j = neighbor[1]
                p, q = self.scaling_AUQ_CADMM(k, uncertainty_i, uncertainty_j)
                W_i = p*uncertainty_i + q
                W_i = torch.nn.functional.pad(W_i, (0,padding_size), "constant", self.rho) 
                W_j = p*uncertainty_j + q
                W_j = torch.nn.functional.pad(W_j, (0,padding_size), "constant", self.rho)

                denominator = W_i + W_j
                epsilon = 1e-8
                update_term = 2*W_i * torch.div( W_j*theta_i_k - W_j*theta_j_k, denominator + epsilon)
                self.p_i += update_term



    def primal_update(self, theta_i_k, loss):
        theta_i = p2v(self.model.parameters())
        if self.config['edge_based']:
            lag_loss = torch.tensor(0, dtype=torch.float64).to(self.device)
            aug_loss = torch.tensor(0, dtype=torch.float64).to(self.device)
            for neighbor in self.neighbors:
                neighbor_id = neighbor[0]
                theta_j_k = neighbor[1]
                if neighbor_id in self.p_ij:
                    lag_loss += torch.dot(theta_i, self.p_ij[neighbor_id])
                    aug_loss += self.rho * torch.norm(theta_i - (theta_i_k + theta_j_k) / 2)**2
        else:
            lag_loss = torch.dot(theta_i, self.p_i)
            aug_loss = torch.tensor(0, dtype=torch.float64).to(self.device)
            for neighbor in self.neighbors:
                theta_j_k = neighbor[0]
                aug_loss += self.rho * torch.norm(theta_i - (theta_i_k+theta_j_k)/2)**2

        total_loss = loss + lag_loss + aug_loss 
        return total_loss, lag_loss.item(), aug_loss.item()
    

    def primal_update_AUQ_CADMM(self, theta_i_k, loss, uncertainty_i, k):
        theta_i = p2v(self.model.parameters())
        #  Both lag_loss and aug_loss are accumulated inside the loop
        
        aug_loss = torch.tensor(0, dtype=torch.float64).to(self.device)
        padding_size = theta_i.size(0) - uncertainty_i.size(0)
        
        if self.config['edge_based'] == True:
            lag_loss = torch.tensor(0, dtype=torch.float64).to(self.device)
            # The loop now iterates only over neighbors with a success "flag"
            for neighbor in self.neighbors:
                neighbor_id = neighbor[0]
                theta_j_k = neighbor[1]
                uncertainty_j = neighbor[2]

                # Add the lagrangian term for this specific neighbor
                lag_loss += torch.dot(theta_i, self.p_ij[neighbor_id])

                p, q = self.scaling_AUQ_CADMM(k, uncertainty_i, uncertainty_j)
                gamma_t = 1.0
                if isinstance(neighbor_id, int) and neighbor_id < 0:
                    gamma_t = self.get_gamma_t(k)

                W_i = gamma_t * (p*uncertainty_i + q)
                W_i = torch.nn.functional.pad(W_i, (0, padding_size), "constant", self.rho)
                W_j = p*uncertainty_j + q
                W_j = torch.nn.functional.pad(W_j, (0, padding_size), "constant", self.rho)
                
                denominator = W_i + W_j
                epsilon = 1e-8
                consensus_theta = torch.div( W_i*theta_i_k + W_j*theta_j_k, denominator + epsilon)
                difference = theta_i - consensus_theta
                
                W_i_clamped = torch.clamp(W_i, min=0.)
                weighted_norm = torch.dot(difference*W_i_clamped, difference)
                
                # Add the augmented term for this specific neighbor
                aug_loss += weighted_norm
        else:
            lag_loss = torch.dot(theta_i, self.p_i) #TODO: uncomment? comment?
            aug_loss = torch.tensor(0, dtype=torch.float64).to(self.device)
            padding_size = theta_i.size(0) - uncertainty_i.size(0)
            for neighbor in self.neighbors:
                theta_j_k = neighbor[0]     
                uncertainty_j = neighbor[1]
                p, q = self.scaling_AUQ_CADMM(k, uncertainty_i, uncertainty_j)
                W_i = p*uncertainty_i + q
                W_i = torch.nn.functional.pad(W_i, (0,padding_size), "constant", self.rho) 
                W_j = p*uncertainty_j + q
                W_j = torch.nn.functional.pad(W_j, (0,padding_size), "constant", self.rho) 
                denominator = W_i + W_j
                epsilon = 1e-8
                consensus_theta = torch.div( W_i*theta_i_k + W_j*theta_j_k, denominator + epsilon)
                difference = theta_i - consensus_theta
                
                W_i_clamped = torch.clamp(W_i, min=0.)
                weighted_norm = torch.dot(difference*W_i_clamped, difference)
                aug_loss += weighted_norm
        loss += lag_loss + aug_loss
        return loss, lag_loss.item(), aug_loss.item()

    # MACIM loss function, serve as a regularization term
    # It is not used in the paper
    def MACIM_cc_loss(self, loss):
        theta_i = p2v(self.model.parameters())
        for neighbor in self.neighbors:
            theta_j = neighbor[0]
            difference = self.rho * torch.norm(theta_i - theta_j)**2
            loss += difference
        return loss

    def DSGD_update(self):
        rid = self.agent_id
        deg_i = len(self.neighbors)
        w = 1/(deg_i+1)
        with torch.no_grad():
            for param_i in self.model.parameters():
                #param_i.multiply_(self.ds_mat[rid, rid]) 
                param_i.multiply_(w) 
                param_i.add_(-self.alpha * param_i.grad)  # Gradient descent update
                param_i.grad.zero_()  # Reset the gradient

            for model_j, j in self.neighbors:
                for param_i, param_j in zip(self.model.parameters(), model_j):
                    #param_i.add_(self.ds_mat[rid, j] * param_j)
                    param_i.add_(w * param_j)


    def DSGT_update(self):
        rid = self.agent_id
        deg_i = len(self.neighbors)
        w = 1/(deg_i+1)
        with torch.no_grad():
            for p, param_i in enumerate(self.model.parameters()):
                param_i.multiply_(w) 
                param_i.add_(-w*self.alpha*self.y_dsgt[p])  # Gradient descent update
                self.y_dsgt[p].multiply_(w) 
                self.y_dsgt[p].add_(param_i.grad - self.g_dsgt[p]) 
                self.g_dsgt[p] = param_i.grad.clone()
                param_i.grad.zero_() 

            for model_j, y_j, j in self.neighbors:
                for p, (param_i, param_j) in enumerate(zip(self.model.parameters(), model_j)):
                    param_i.add_(w*param_j - w*self.alpha*y_j[p])
                    self.y_dsgt[p].add_(w*y_j[p])
               

    def global_BA(self, batch, cur_frame_id, dist_algorithm):
        '''
        Global bundle adjustment that includes all the keyframes and the current frame
        Params:
            batch['c2w']: ground truth camera pose [1, 4, 4]
            batch['rgb']: rgb image [1, H, W, 3]
            batch['depth']: depth image [1, H, W]
            batch['direction']: view direction [1, H, W, 3]
            cur_frame_id: current frame id
            dist_algorithm: algorithm used for multi-agent learning
        '''

        # --- 清理旧的伪邻居（负数ID）的对偶量与邻居项 ---
        if hasattr(self, 'p_ij'):
            for k in list(self.p_ij.keys()):
                if isinstance(k, int) and k < 0:
                    del self.p_ij[k]
        # 若 neighbors 里可能残留上轮注入的伪邻居（当通信频率与 BA 频率解耦时）
        self.neighbors = [
            n for n in self.neighbors
            if not (isinstance(n[0], int) and n[0] < 0)
        ]

        # --- 注入当前窗口的伪邻居，并显式重置 p_ij ---
        inject_this_frame = False
        if self.temporal_consensus_enabled and self.temporal_snapshots:
            tc_inject_every = self.temporal_consensus_config.get('inject_every', 0)
            if tc_inject_every == 0 or (cur_frame_id % tc_inject_every == 0):
                inject_this_frame = True

        if inject_this_frame:
            for i, (snapshot_params_cpu, snapshot_uncertainty_cpu) in enumerate(self.temporal_snapshots):
                pseudo_neighbor_id = -(i + 1)
                snapshot_params = snapshot_params_cpu.to(self.device)
                
                # 根据算法模式决定是否传递不确定性
                snapshot_uncertainty = None
                if dist_algorithm == 'AUQ_CADMM' and snapshot_uncertainty_cpu is not None:
                    snapshot_uncertainty = snapshot_uncertainty_cpu.to(self.device)

                # 初始化对偶变量并添加到邻居列表
                if self.config['edge_based']:
                    self.p_ij[pseudo_neighbor_id] = torch.zeros_like(snapshot_params)
                
                # CADMM 路径下，uncertainty 为 None
                self.neighbors.append([pseudo_neighbor_id, snapshot_params, snapshot_uncertainty])


        # all the KF poses: 0, 5, 10, ...
        poses = torch.stack([self.est_c2w_data[i] for i in range(0, cur_frame_id, self.config['mapping']['keyframe_every'])])
        poses_fixed = torch.nn.parameter.Parameter(poses).to(self.device)
        current_pose = self.est_c2w_data[cur_frame_id][None,...]
        poses_all = torch.cat([poses_fixed, current_pose], dim=0)

        # Set up optimizer
        self.map_optimizer.zero_grad()
        
        current_rays = torch.cat([batch['direction'], batch['rgb'], batch['depth'][..., None]], dim=-1)
        current_rays = current_rays.reshape(-1, current_rays.shape[-1]) 

        theta_i_k = p2v(self.model.parameters()).detach()


        if dist_algorithm == 'CADMM':
            self.dual_update(theta_i_k) 
        elif dist_algorithm == 'AUQ_CADMM':
            self.dual_update_AUQ_CADMM(theta_i_k, self.uncertainty_tensor, cur_frame_id)

        mean_total_loss = 0
        mean_obj_loss = 0
        mean_lag_loss = 0
        mean_aug_loss = 0
        for i in range(self.config['mapping']['iters']):
            
            # Sample rays with real frame ids
            # rays [bs, 7]
            # frame_ids [bs]
            """
            if not self.enable_replay:
                # If replay is explicitly disabled, we don't sample from past keyframes.
                rays, ids = torch.tensor([]), torch.tensor([])
            else:
                # Original experience replay logic: only runs if EWC is off AND replay is on.
                rays, ids = self.keyframeDatabase.sample_global_rays(self.config['mapping']['sample'])

            #TODO: Checkpoint...
            sample_size = self.config['mapping']['sample']
            if self.enable_replay and not self.ewc_enabled and len(self.keyframeDatabase.frame_ids) > 0:
                 sample_size = max(self.config['mapping']['sample'] // len(self.keyframeDatabase.frame_ids), self.config['mapping']['min_pixels_cur'])

            idx_cur = random.sample(range(0, self.dataset_info['H'] * self.dataset_info['W']), sample_size)
            current_rays_batch = current_rays[idx_cur, :]

            rays = torch.cat([rays, current_rays_batch], dim=0) # N, 7
            if self.ewc_enabled:
                ids_all = -torch.ones((len(idx_cur))).to(torch.int64)
            else:
                ids_all = torch.cat([ids//self.config['mapping']['keyframe_every'], -torch.ones((len(idx_cur)))]).to(torch.int64)
            """
            # --- 【新增】UNIKD 交替训练逻辑 ---
            is_distillation_step = False
            loss = 0
            
            # 奇数步且 Teacher 存在时，执行蒸馏
            if self.unikd_enabled and (i % 2 == 1) and (self.unikd.teacher_model is not None):
                rays_o, rays_d, teacher_out = self.unikd.get_distillation_batch(self.config['mapping']['sample'])
                if rays_o is not None:
                    is_distillation_step = True
                    
                    # Forward Student
                    ret = self.model.forward(rays_o, rays_d, target_rgb=None, target_d=None)
                    
                    # Distillation Loss
                    pred_unc = ret.get('uncertainty', torch.zeros_like(ret['rgb'][..., 0:1]))
                    loss = self.unikd.unikd_loss(ret['rgb'], teacher_out['rgb'], pred_unc)
                    
                    # 清空梯度 (因为下面逻辑里有 zero_grad，这里先清一下保险)
                    self.map_optimizer.zero_grad()

            # 如果不是蒸馏步，执行正常的监督学习 (Supervised Step)
            if not is_distillation_step:
                rays_list = []
                ids_pose_list = []

                # 1) 先看 KR：如果 kr_enabled=True，则只用 KR-replay（不走原 replay）
                if getattr(self, 'kr_enabled', False) and len(self.keyframeDatabase.frame_ids) > 0:
                    kr_cfg = self.config['training'].get('kr', {})
                    num_replay_rays = kr_cfg.get('num_replay_rays', self.config['mapping']['sample'])
                    rays_kr, ids_pose_kr = self.kr_replay.sample_replay_rays(num_replay_rays)
                    if rays_kr.shape[0] > 0:
                        rays_list.append(rays_kr)
                        ids_pose_list.append(ids_pose_kr.to(torch.int64))

                # 2) 如果没开 KR，但 enable_replay=True，则走原来的 global replay
                elif self.enable_replay and len(self.keyframeDatabase.frame_ids) > 0:
                    rays_rep, ids_rep = self.keyframeDatabase.sample_global_rays(self.config['mapping']['sample'])
                    if rays_rep.shape[0] > 0:
                        rays_list.append(rays_rep)
                        ids_pose_list.append((ids_rep // self.config['mapping']['keyframe_every']).to(torch.int64))

                # 3) 当前帧采样 pixels（无论是否有 replay，都要有当前帧）
                sample_size = self.config['mapping']['sample']
                if self.enable_replay and not self.ewc_enabled and len(self.keyframeDatabase.frame_ids) > 0:
                    sample_size = max(
                        self.config['mapping']['sample'] // max(len(self.keyframeDatabase.frame_ids), 1),
                        self.config['mapping']['min_pixels_cur']
                    )

                idx_cur = random.sample(
                    range(0, self.dataset_info['H'] * self.dataset_info['W']),
                    sample_size
                )
                idx_cur = torch.tensor(idx_cur, dtype=torch.long)
                current_rays_batch = current_rays[idx_cur, :]  # [Nc,7]

                rays_list.append(current_rays_batch)
                ids_cur = -torch.ones((current_rays_batch.shape[0],), dtype=torch.int64)
                ids_pose_list.append(ids_cur)

                # 拼接所有 rays（至少包含当前帧）
                rays = torch.cat(rays_list, dim=0)  # [N,7]
                ids_all = torch.cat(ids_pose_list, dim=0)  # [N]

                rays_d_cam = rays[..., :3].to(self.device)
                target_s = rays[..., 3:6].to(self.device)
                target_d = rays[..., 6:7].to(self.device)

                # [N, Bs, 1, 3] * [N, 1, 3, 3] = (N, Bs, 3)
                rays_d = torch.sum(rays_d_cam[..., None, None, :] * poses_all[ids_all, None, :3, :3], -1)
                rays_o = poses_all[ids_all, None, :3, -1].repeat(1, rays_d.shape[1], 1).reshape(-1, 3)
                rays_d = rays_d.reshape(-1, 3)

                ret = self.model.forward(rays_o, rays_d, target_s, target_d)

                self.map_optimizer.zero_grad()

                # 开关：如果是 Temporal 模式且开启了不确定性追踪，或者是 MAS 模式
                calc_structural_grad = (self.track_uncertainty and self.temporal_consensus_enabled) or \
                                    getattr(self, 'mas_enabled', False)

                if calc_structural_grad:
                    # 1. 构建 Proxy Loss (只看输出幅度，不看 GT 误差)
                    mas_proxy_loss = 0.0
                    if 'rgb' in ret: mas_proxy_loss += ret['rgb'].pow(2).sum()
                    if 'depth' in ret: mas_proxy_loss += ret['depth'].pow(2).sum()
                    if 'sdf' in ret: mas_proxy_loss += ret['sdf'].pow(2).sum()
                    
                    mas_proxy_loss = mas_proxy_loss / ret['rgb'].shape[0] # Mean
                    
                    # 2. 反向传播获取“结构梯度”
                    mas_proxy_loss.backward(retain_graph=True)

                    # 3. 如果开启了 MAS，累积给 MAS
                    if getattr(self, 'mas_enabled', False):
                        self.mas.accumulate_importance_from_grad()

                    # 4. 【关键】如果开启了 Temporal，更新 W (Uncertainty)
                    if self.track_uncertainty and self.temporal_consensus_enabled:
                        # 获取梯度
                        grid_grad = self.model.embed_fn.params.grad

                        if grid_grad is not None:
                            # 获取幅度
                            grad_mag = torch.abs(grid_grad)
                            self.uncertainty_tensor*= self.uncert_decay
                            self.uncertainty_tensor += grad_mag 
                            
                    # 5. 清空梯度，准备计算真正的 Loss
                    self.map_optimizer.zero_grad()

                    # 【修改】Loss 计算
                if self.unikd_enabled:
                    # 使用 UNIKD 的不确定性加权 Loss
                    pred_unc = ret.get('uncertainty', torch.zeros_like(ret['rgb'][..., 0:1]))
                    loss = self.unikd.unikd_loss(ret['rgb'], target_s, pred_unc)
                    # 加上其他 Loss (Depth, SDF, Smoothness)
                    loss += self.get_loss_from_ret(ret, rgb=False) 
                else:
                    # 原有 Loss
                    loss = self.get_loss_from_ret(ret, smooth=True)

                # Add EWC loss if enabled
                if self.ewc_enabled:
                    loss += self.ewc.compute_loss()

                # --- CNM: off-surface function replay loss ---
                if getattr(self, 'cnm_enabled', False):
                    # 从 CNM 专用 buffer 取一批 (x, sdf_teacher)，对当前模型加符号 & 数值一致性约束
                    loss += self.cnm_replay.compute_loss(self.cnm_teacher)

                # --- MAS loss: parameter importance regularization ---
                if getattr(self, 'mas_enabled', False):
                    loss += self.mas.mas_loss()

                # --- MAS Original (Offline) loss ---
                if getattr(self, 'mas_original_enabled', False):
                    loss += self.mas_original.mas_loss()

                loss.backward(retain_graph=True)
                mean_obj_loss += loss.item() #item() method extracts the loss’s value as a Python float.
    
                """
                if self.track_uncertainty:
                    if self.config['grid']['enc'] == 'tensor':
                        # For TensorCP, iterate through its parameters to get gradients
                        grads = []
                        for p in self.model.embed_fn.parameters():
                            if p.grad is not None:
                                grads.append(p.grad.view(-1))
                        if grads:
                            grid_grad = torch.cat(grads)
                        else:
                            grid_grad = torch.tensor([], device=self.device)
                    else:
                        # Original code for tcnn encoders
                        grid_grad = self.model.embed_fn.params.grad
                    
                    if grid_grad is not None and grid_grad.numel() > 0:
                        grid_has_grad = (torch.abs(grid_grad) > 0).to(torch.int32)
                        self.uncertainty_tensor += grid_has_grad
                        #set tf 
                        if len(self.keyframeDatabase.frame_ids) > 0:
                            current_step = self.keyframeDatabase.frame_ids[-1]
                            
                            uncert_log = self.uncertainty_tensor.detach().cpu().float()

                            self.writer.add_scalar('Uncertainty/Mean', uncert_log.mean(), current_step)
                            self.writer.add_scalar('Uncertainty/Std', uncert_log.std(), current_step)
                            self.writer.add_scalar('Uncertainty/Max', uncert_log.max(), current_step)
                            self.writer.add_scalar('Uncertainty/Min', uncert_log.min(), current_step)
                        else:
                            print("no frame ids in keyframe database, cannot log uncertainty")

                """


                if dist_algorithm == 'CADMM':
                    loss, lag_loss, aug_loss = self.primal_update(theta_i_k, loss)
                    loss.backward(retain_graph=True)
                    self.map_optimizer.step()
                    mean_lag_loss += lag_loss
                    mean_aug_loss += aug_loss

                elif dist_algorithm == 'AUQ_CADMM':
                    loss, lag_loss, aug_loss  = self.primal_update_AUQ_CADMM(theta_i_k, loss, self.uncertainty_tensor, cur_frame_id)
                    loss.backward(retain_graph=True)
                    self.map_optimizer.step()
                    mean_lag_loss += lag_loss
                    mean_aug_loss += aug_loss

                elif dist_algorithm == 'MACIM':
                    loss = self.MACIM_cc_loss(loss)
                    loss.backward(retain_graph=True)
                    self.map_optimizer.step()

                elif dist_algorithm == 'DSGD':
                    self.DSGD_update()
                    break # DSDG does one update per mapping iteration 

                elif dist_algorithm == 'DSGT':
                    self.DSGT_update()
                    break # DSDT does one update per mapping iteration 


                mean_total_loss += loss.item()


        # save loss info 
        mean_total_loss /= self.config['mapping']['iters']
        mean_obj_loss /= self.config['mapping']['iters']
        mean_lag_loss /= self.config['mapping']['iters']
        mean_aug_loss /= self.config['mapping']['iters']
        self.total_loss.append( mean_total_loss )
        self.obj_loss.append(mean_obj_loss)
        self.lag_loss.append(mean_lag_loss)
        self.aug_loss.append(mean_aug_loss)
        # set tf
        self.writer.add_scalar('Loss/Total', mean_total_loss, cur_frame_id)
        self.writer.add_scalar('Loss/Objective', mean_obj_loss, cur_frame_id)
        self.writer.add_scalar('Loss/Lagrangian', mean_lag_loss, cur_frame_id)
        self.writer.add_scalar('Loss/Augmented', mean_aug_loss, cur_frame_id)




    def tracking_render(self, batch, frame_id):
        '''
            just save ground truth pose
        '''
        c2w_gt = batch['c2w'][0].to(self.device)
        self.est_c2w_data[frame_id] = c2w_gt
         # --- 【新增】UNIKD 更新 Pose 边界 ---
        if self.unikd_enabled:
            self.unikd.update_pose_bounds(c2w_gt)


    def create_optimizer(self):
        '''
        Create optimizer for mapping
        '''
        if self.fix_decoder:
            #TODO: pretrain
            trainable_parameters = [{'params': self.model.embed_fn.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed']}]
        else:
            # Optimizer for BA
            trainable_parameters = [{'params': self.model.decoder.parameters(), 'weight_decay': 1e-6, 'lr': self.config['mapping']['lr_decoder']},
                                    {'params': self.model.embed_fn.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed']}]

        if not self.config['grid']['oneGrid']:
            trainable_parameters.append({'params': self.model.embed_fn_color.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed_color']})
        
        self.map_optimizer = optim.Adam(trainable_parameters, betas=(0.9, 0.99))
        
    
    def save_mesh(self, i, voxel_size=0.05):
        mesh_savepath = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], f'agent_{self.agent_id}', 'mesh_track{}.ply'.format(i))
        if self.config['mesh']['render_color']:
            color_func = self.model.render_surface_color
        else:
            color_func = self.model.query_color
        extract_mesh(self.model.query_sdf, 
                        self.config, 
                        self.bounding_box, 
                        color_func=color_func, 
                        marching_cube_bound=self.marching_cube_bound, 
                        voxel_size=voxel_size, 
                        mesh_savepath=mesh_savepath)    

        if self.track_uncertainty == True:
            uncertainty_savepath = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], f'agent_{self.agent_id}', 'uncertain_track{}.pt'.format(i))
            torch.save(self.uncertainty_tensor, uncertainty_savepath)


    def run(self, i, batch):
        """
            @param i: current step
            @param batch:
        """
        # First frame mapping
        if i == 0:
            self.first_frame_mapping(batch, self.config['mapping']['first_iters'])
            if self.temporal_consensus_enabled:
                print(f"Agent {self.agent_id}: Initial snapshot at frame 0")
                self.update_temporal_snapshot()
            return 
            
        
        # Tracking + Mapping
        self.tracking_render(batch, i)

        if i%self.config['mapping']['map_every']==0:
            self.global_BA(batch, i, self.dist_algorithm)
            
        # Add keyframe
        if i % self.config['mapping']['keyframe_every'] == 0:
            if self.config['mapping'].get('enable_replay', True):
                self.keyframeDatabase.add_keyframe(batch, filter_depth=self.config['mapping']['filter_depth'])
                if getattr(self, 'kr_enabled', False):
                    self.kr_replay.register_keyframe(i)
            #print(f'\nAgent {self.agent_id} add keyframe:{i}')
            if self.ewc_enabled:
                self.ewc.update_fisher_with_batch(batch)

        # --- 【新增】MAS Original: 将当前帧加入 Buffer ---
        # 注意：这里只存数据，不计算梯度
        if getattr(self, 'mas_original_enabled', False):
            self.mas_original.add_to_buffer(batch)

        # --- 【新增】UNIKD 在任务边界更新 Teacher ---
        if self.unikd_enabled:
            unikd_cfg = self.config['training'].get('unikd', {})
            frames_per_task = unikd_cfg.get('frames_per_task', 100)
            if i > 0 and (i % frames_per_task) == 0:
                self.unikd.update_teacher()
        
         # --- CNM: 在任务边界更新 teacher 并填充 CNM buffer ---
        if getattr(self, 'cnm_enabled', False):
            cnm_cfg = self.config['training'].get('cnm', {})
            frames_per_task_cnm = cnm_cfg.get('frames_per_task', 500)
            # 在每个“任务结束”处更新 CNM teacher，并根据 teacher 生成伪样本填入 buffer
            if i > 0 and (i % frames_per_task_cnm) == 0:
                # 1) 冻结当前模型为 teacher（θ^{t-1}）
                self.cnm_teacher.update()
                # 2) 用 teacher 在 bbox 内生成一大批 (x, sdf_teacher)，写入 CNM buffer
                self.cnm_replay.populate_buffer_from_teacher(self.cnm_teacher)

        # --- 新增：周期性地创建历史快照 ---
        if self.temporal_consensus_enabled:
            frames_per_task = self.temporal_consensus_config.get('frames_per_task', 500)
            # 在每个任务结束时更新快照
            if i > 0 and i % frames_per_task == 0:
                self.update_temporal_snapshot()

        # --- MAS: 在任务边界归一化重要性 Ω_{ij} 并更新 θ* ---
        if getattr(self, 'mas_enabled', False):
            mas_cfg = self.config['training'].get('mas', {})
            frames_per_task_mas = mas_cfg.get('frames_per_task', 500)
            if i > 0 and (i % frames_per_task_mas) == 0:
                print(f"Agent {self.agent_id}: MAS finalize importance at frame {i}")
                self.mas.finalize_importance()

        # --- 【新增】MAS Original: 在任务边界离线计算重要性 (Offline) ---
        if getattr(self, 'mas_original_enabled', False):
            mas_orig_cfg = self.config['training'].get('mas_original', {})
            frames_per_task_orig = mas_orig_cfg.get('frames_per_task', 100)
            if i > 0 and (i % frames_per_task_orig) == 0:
                # 触发离线计算：遍历 Buffer -> 算梯度 -> 更新 Omega -> 清空 Buffer
                self.mas_original.calculate_importance()

        if i % self.config['mesh']['vis']==0:
            self.save_mesh(i, voxel_size=self.config['mesh']['voxel_eval'])


        if i == (self.dataset_info['num_frames']-1):
            model_savepath = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], f'agent_{self.agent_id}', 'checkpoint{}.pt'.format(i)) 
            self.save_ckpt(model_savepath)
            self.save_mesh(i, voxel_size=self.config['mesh']['voxel_final'])

        # 按频率记录内存（默认每步，也可在配置中改小频率）
        log_every = self.config['mapping'].get('mem_log_every', 1)
        if (i % log_every) == 0:
            self.log_memory(i)
            self.log_extra_persistent_mem(i)
        

def create_agent_graph(cfg, dataset):

    """
        @param cfg:
        @param dataset:
        @return G: created graph
        @return frames_per_agent:
    """
    num_agents = cfg['multi_agents']['num_agents']
    frames_per_agent = len(dataset) // num_agents
    dataset_info = {'num_frames':frames_per_agent, 'num_rays_to_save':dataset.num_rays_to_save, 'H':dataset.H, 'W':dataset.W }
    
    # Use first agent as model
    print(f'\nCreating agent 0 (template)')
    agent_template = Mapping(cfg, 0, dataset_info)
    if cfg['multi_agents']['fix_decoder']:
        agent_template.load_decoder(load_path=cfg['data']['load_path'])
    print(f'agent_0 fix decoder: {agent_template.fix_decoder}')

    # Temporarily remove non-copyable writer attribute
    writer_template = agent_template.writer
    agent_template.writer = None

    agents = [agent_template]

    # deep copy every agent
    for i in range(1, num_agents):
        print(f'\nCreating agent {i} by copying template')
        agent_i = copy.deepcopy(agent_template)
        agent_i.agent_id = i # 必须更新每个智能体的ID
        print(f'agent_{i} fix decoder: {agent_i.fix_decoder}')
        agents.append(agent_i)

    agent_template.writer = writer_template 
    for i in range(1, num_agents):
        agent_i = agents[i]
        log_dir = os.path.join(cfg['data']['output'], cfg['data']['exp_name'], f'agent_{agent_i.agent_id}', 'logs')
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        agent_i.writer = SummaryWriter(log_dir=log_dir)
        print(f"Agent {agent_i.agent_id} TensorBoard logs will be saved to: {log_dir}")

    if cfg['multi_agents']['complete_graph']:
        G = nx.complete_graph(num_agents)
        for i in range(num_agents):
        
            attrs = {i:{"agent": agents[i]}}
            nx.set_node_attributes(G, attrs) 
        nx.set_edge_attributes(G, 1, "weight")
    else:
        G = nx.Graph()
        node_list = []
        for i in range(num_agents):
            node_list.append( [ i, {"agent": agents[i]} ] )
        G.add_nodes_from(node_list) 
        G.add_edges_from(cfg['multi_agents']['edges_list'], weight=1)

    # plot graph
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()  

    # create doubly stochastic matrix for DSGD and DSGT 
    N = G.number_of_nodes()
    W = torch.zeros((N, N))
    L = nx.laplacian_matrix(G)
    degs = [L[i, i] for i in range(N)]
    for i in range(N):
        for j in range(N):
            if G.has_edge(i, j) and i != j:
                W[i, j] = 1.0 / (max(degs[i], degs[j]) + 1.0) # metropolis weights
    for i in range(N):
        W[i, i] = 1.0 - torch.sum(W[i, :])

    if cfg['edge_based']:
        theta_i_size = None
        if G.number_of_nodes() > 0:

            some_agent = G.nodes[0]['agent']
            theta_i_size = p2v(some_agent.model.parameters()).size()

        for i, nbrs in G.adj.items():
            agent_i = G.nodes[i]['agent']
            agent_i.ds_mat = W
            if theta_i_size is not None:
                for j in nbrs:
                
                    agent_i.p_ij[j] = torch.zeros(theta_i_size).to(agent_i.device)
    else:
        for i, nbrs in G.adj.items():
            agent_i = G.nodes[i]['agent']
            agent_i.ds_mat = W

    return G, frames_per_agent

def get_data_memory(dataset, cfg, frames_per_agent):
    num_agents = cfg['multi_agents']['num_agents']
    output_path = os.path.join(cfg['data']['output'], cfg['data']['exp_name'])
    rgb = dataset[0]['rgb']
    depth = dataset[0]['depth']
    rgb_memory = torch.numel(rgb)*rgb.element_size() / 1e6 # bytes to megabytes
    depth_memory = torch.numel(depth)*depth.element_size() / 1e6 # bytes to megabytes
    single_size= f'size of a rgb img and a depth img: {rgb_memory + depth_memory} MB\n'
    total_size = f'total size of all images shared for centralized training: {(rgb_memory + depth_memory)*frames_per_agent*(num_agents-1)} MB\n'
    
    # Save to a text file
    print("Save Memory Info")
    with open(os.path.join(output_path, 'memory_sizes.txt'), 'w') as file:
        file.write(single_size)
        file.write(total_size)


def get_model_memory(model, fix_decoder=False, grid_enc_type='tcnn'):
    if fix_decoder:
        if grid_enc_type == 'tensor':
            model_tensor = p2v(model.embed_fn.parameters())
        else:  
            model_tensor = model.embed_fn.params
    else:
        model_tensor = p2v(model.parameters())
    model_size = torch.numel(model_tensor)*model_tensor.element_size() / 1e6 # bytes to megabytes
    return model_size


def train_multi_agent(cfg):
    dataset = get_dataset(cfg)

    G, frames_per_agent = create_agent_graph(cfg, dataset)

    get_data_memory(dataset, cfg, frames_per_agent)
    
    edges_for_dropout = cfg['multi_agents']['edges_for_dropout']
    com_history = {}
    fix_decoder = cfg['multi_agents']['fix_decoder']
    for step in trange(0, frames_per_agent, smoothing=0):

        # commnuication
        if step % cfg['mapping']['map_every'] == 0:

            # communication dropout
            for i, j, p in edges_for_dropout:
                G.edges[i,j]['weight'] = random.choices([0, 1], weights=[p, 1-p])[0] # 0 forcom dropout

            for i, nbrs in G.adj.items():
                #print(f'\nAgent {i} Communicating')
                agent_i = G.nodes[i]['agent']
                agent_i.neighbors = [] # clear communication buffer, only save the latest weights
                agent_i.com_perIter = 0
                for j, edge_attr in nbrs.items():
                    # save com history 
                    if i < j: # only save (i,j), don't save (j,i)
                        edge = (i, j) 
                        if edge not in com_history:
                            com_history[edge] = []
                        com_history[edge].append(edge_attr['weight'])
                    # send data 
                    if edge_attr['weight'] == 1:
                        agent_j = G.nodes[j]['agent']
                        grid_enc_type = cfg['grid']['enc']
                        if cfg['multi_agents']['distributed_algorithm'] == 'AUQ_CADMM':
                            if cfg['edge_based']:
                                agent_i.communicate([j, agent_j.model, agent_j.uncertainty_tensor])
                                model_size = get_model_memory(agent_j.model, fix_decoder, grid_enc_type)*2
                            else:
                                agent_i.communicate([agent_j.model, agent_j.uncertainty_tensor, step])
                                model_size = get_model_memory(agent_j.model, fix_decoder, grid_enc_type)*2

                        elif cfg['multi_agents']['distributed_algorithm'] in ('CADMM', 'MACIM'):
                            agent_i.communicate([agent_j.model],)
                            model_size = get_model_memory(agent_j.model, fix_decoder, grid_enc_type)

                        elif cfg['multi_agents']['distributed_algorithm'] == 'DSGD':
                            agent_i.communicate([agent_j.model, j])
                            model_size = get_model_memory(agent_j.model, fix_decoder, grid_enc_type)

                        elif cfg['multi_agents']['distributed_algorithm'] == 'DSGT':
                            agent_i.communicate([agent_j.model, agent_j.y_dsgt, j])
                            model_size = get_model_memory(agent_j.model, fix_decoder, grid_enc_type)*2
                        agent_i.com_perIter += model_size
                        agent_i.com_total += model_size
             
        # update
        for i, nbrs in G.adj.items():
            agent_i = G.nodes[i]['agent']
            batch_i = dataset[i*frames_per_agent+step] 
            batch_i["frame_id"] = step
            for key in list(batch_i.keys())[1:]:
                batch_i[key] = batch_i[key].unsqueeze(0)

            agent_i.run(step, batch_i)


    # write communication info
    output_path = os.path.join(cfg['data']['output'], cfg['data']['exp_name'])
    for i, nbrs in G.adj.items():
        agent_i = G.nodes[i]['agent']
        com_perIter = f'Agent {i} message received per communication iteration: {agent_i.com_perIter} MB\n'
        com_total = f'Agent {i} total message received: {agent_i.com_total} MB\n'
        with open(os.path.join(output_path, 'memory_sizes.txt'), 'a') as file: # mode 'a' for append mode, so you can add new content without deleting the previous one
            file.write(com_perIter)
            file.write(com_total)

    data_to_save = {'edge_weight_history': {str(edge): weights for edge, weights in com_history.items()}}
    with open(os.path.join(output_path, 'graph_data.json'), 'w') as f:
        json.dump(data_to_save, f, indent=4)     
    print("Agent Communication Info Saved")
    
    print("Closing TensorBoard writers...")
    for i in G.nodes():
        agent_i = G.nodes[i]['agent']
        agent_i.writer.close()



if __name__ == '__main__':

    print('Start running...')
    parser = argparse.ArgumentParser(
        description='Arguments for running the NICE-SLAM/iMAP*.'
    )
    parser.add_argument('--config', type=str, help='Path to config file.')
    
    args = parser.parse_args()

    cfg = config.load_config(args.config)

    if cfg['multi_agents']['distributed_algorithm'] == 'AUQ_CADMM':
        cfg['multi_agents']['track_uncertainty'] = True


    print("Saving config and script...")
    save_path = os.path.join(cfg["data"]["output"], cfg['data']['exp_name'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path, 'config.json'),"w", encoding='utf-8') as f:
        f.write(json.dumps(cfg, indent=4))


    # multi-agent training 
    train_multi_agent(cfg)
