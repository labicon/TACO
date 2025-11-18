# INPUT_MESH=output/scannet/scene0000_00/Full_CNM_frames_per_task100_offsurface2048_weight0.001_buffer200k_alpha12_near0.05_vnear1.0_vfar0.1/agent_0/mesh_track5577.ply
# VIRT_CAM_PATH=eval_data/scannet/scene0000_00/virtual_cameras
# python cull_mesh.py --config configs/scannet/scene0000.yaml --input_mesh $INPUT_MESH --remove_occlusion --gt_pose


# INPUT_MESH=output/Replica/room1/CNM_frames_per_task100_offsurface2048_weight0.001_buffer200k_alpha8/agent_0/mesh_track1999.ply
# VIRT_CAM_PATH=eval_data/Replica/room1/virtual_cameras
# nohup python cull_mesh.py --config configs/Replica/room1.yaml --input_mesh $INPUT_MESH --remove_occlusion --virtual_cameras --virt_cam_path $VIRT_CAM_PATH --gt_pose > eval_replica_room1.log 2>&1 &


# # INPUT_MESH=output/Azure/Apartment/Incremental-gamma0.2-inject1-frames_per_task1/agent_0/mesh_track12594.ply
# # VIRT_CAM_PATH=eval_data/Azure/Apartment/virtual_cameras
# # python cull_mesh.py --config configs/Azure/apartment.yaml --input_mesh $INPUT_MESH --remove_occlusion --gt_pose


# # INPUT_MESH=output/Azure/Apartment/Incremental-gamma1-inject1-frames_per_task1/agent_0/mesh_track12594.ply
# # VIRT_CAM_PATH=eval_data/Azure/Apartment/virtual_cameras
# # python cull_mesh.py --config configs/Azure/apartment.yaml --input_mesh $INPUT_MESH --remove_occlusion --gt_pose

# # INPUT_MESH=output/Replica/room1/CNM/agent_0/mesh_track1999.ply
# # VIRT_CAM_PATH=eval_data/Replica/room1/virtual_cameras
# # python cull_mesh.py --config configs/Replica/room1.yaml --input_mesh $INPUT_MESH --remove_occlusion --virtual_cameras --virt_cam_path $VIRT_CAM_PATH --gt_pose

INPUT_MESH=output/Replica/room1/MAS/agent_0/mesh_track1999.ply
VIRT_CAM_PATH=eval_data/Replica/room1/virtual_cameras
nohup python cull_mesh.py --config configs/Replica/room1.yaml --input_mesh $INPUT_MESH --remove_occlusion --virtual_cameras --virt_cam_path $VIRT_CAM_PATH --gt_pose > eval_replica_room1.log 2>&1 &

# INPUT_MESH=output/Replica/room1/Full_CNM_frames_per_task500_offsurface256_weight0.02_buffer20000_alpha12_near0.05_vnear1.0_vfar0.1/agent_0/mesh_track1999.ply
# VIRT_CAM_PATH=eval_data/Replica/room1/virtual_cameras
# nohup python cull_mesh.py --config configs/Replica/room1.yaml --input_mesh $INPUT_MESH --remove_occlusion --virtual_cameras --virt_cam_path $VIRT_CAM_PATH --gt_pose > eval_replica_room1.log 2>&1 &

# INPUT_MESH=output/Replica/room1/Full_CNM_frames_per_task500_offsurface256_weight0.02_buffer20000_alpha14_near0.05_vnear1.0_vfar0.1/agent_0/mesh_track1999.ply
# VIRT_CAM_PATH=eval_data/Replica/room1/virtual_cameras
# nohup python cull_mesh.py --config configs/Replica/room1.yaml --input_mesh $INPUT_MESH --remove_occlusion --virtual_cameras --virt_cam_path $VIRT_CAM_PATH --gt_pose > eval_replica_room1.log 2>&1 &
