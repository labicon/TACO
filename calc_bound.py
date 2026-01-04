import numpy as np
import argparse
import os

def calculate_bounds(traj_path, padding=2.5):
    """
    根据相机轨迹计算场景边界
    padding: 延伸范围（米），因为相机看的是周围的环境，所以边界要比相机轨迹大一圈
    """
    if not os.path.exists(traj_path):
        print(f"Error: File {traj_path} not found.")
        return

    poses = []
    with open(traj_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # 读取 4x4 矩阵
            val = list(map(float, line.strip().split()))
            pose = np.array(val).reshape(4, 4)
            poses.append(pose)
    
    poses = np.array(poses)
    # 提取平移部分 (x, y, z)
    translations = poses[:, :3, 3]
    
    min_xyz = translations.min(axis=0)
    max_xyz = translations.max(axis=0)
    
    print(f"Camera Trajectory Range:")
    print(f"  X: {min_xyz[0]:.2f} to {max_xyz[0]:.2f}")
    print(f"  Y: {min_xyz[1]:.2f} to {max_xyz[1]:.2f}")
    print(f"  Z: {min_xyz[2]:.2f} to {max_xyz[2]:.2f}")
    print("-" * 30)
    
    # 添加 Padding
    bound_min = min_xyz - padding
    bound_max = max_xyz + padding
    
    # 格式化输出
    def fmt(val): return round(val, 2)
    
    bound_str = f"[[{fmt(bound_min[0])},{fmt(bound_max[0])}],[{fmt(bound_min[1])},{fmt(bound_max[1])}],[{fmt(bound_min[2])},{fmt(bound_max[2])}]]"
    
    # Marching cubes 可以稍微紧一点，或者保持一致
    mc_padding = padding - 0.5 # 稍微紧一点
    mc_min = min_xyz - mc_padding
    mc_max = max_xyz + mc_padding
    mc_bound_str = f"[[{fmt(mc_min[0])},{fmt(mc_max[0])}],[{fmt(mc_min[1])},{fmt(mc_max[1])}],[{fmt(mc_min[2])},{fmt(mc_max[2])}]]"

    print("Copy these lines to your config file:")
    print(f"  bound: {bound_str}")
    print(f"  marching_cubes_bound: {mc_bound_str}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to traj.txt")
    parser.add_argument("--padding", type=float, default=2.0, help="Padding in meters (default 2.0)")
    args = parser.parse_args()
    
    calculate_bounds(args.path, args.padding)