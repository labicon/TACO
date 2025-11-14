INPUT_MESH=output/scannet/scene0000_00/Incremental-gamma1-inject2-frames_per_task100_decay/agent_0/mesh_track5577.ply
VIRT_CAM_PATH=eval_data/scannet/scene0000_00/virtual_cameras
python cull_mesh.py --config configs/scannet/scene0000.yaml --input_mesh $INPUT_MESH --remove_occlusion --gt_pose


INPUT_MESH=output/scannet/scene0000_00/CADMM/agent_0/mesh_track5577.ply
VIRT_CAM_PATH=eval_data/scannet/scene0000_00/virtual_cameras
python cull_mesh.py --config configs/scannet/scene0000.yaml --input_mesh $INPUT_MESH --remove_occlusion --gt_pose
