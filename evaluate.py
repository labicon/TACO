import argparse
import os
import subprocess
import sys


def default_culled_mesh_path(input_mesh, remove_occlusion, virtual_cameras):
    if virtual_cameras:
        suffix = "virt_cams"
    elif remove_occlusion:
        suffix = "occlusion"
    else:
        suffix = "frustum"
    root, ext = os.path.splitext(input_mesh)
    return f"{root}_cull_{suffix}{ext}"


def run_command(command):
    print(" ".join(command))
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Cull a reconstructed mesh and evaluate reconstruction metrics."
    )
    parser.add_argument("--config", required=True, help="Scene config used for the run.")
    parser.add_argument("--input_mesh", required=True, help="Reconstructed mesh to evaluate.")
    parser.add_argument("--gt_mesh", required=True, help="Ground-truth mesh.")
    parser.add_argument("--output_mesh", help="Path for the culled reconstructed mesh.")
    parser.add_argument("--ckpt_path", help="Checkpoint with estimated poses. Omit when using --gt_pose.")
    parser.add_argument("--gt_pose", action="store_true", help="Use dataset ground-truth poses for culling.")
    parser.add_argument("--skip", type=int, default=2, help="Frame stride used during culling.")
    parser.add_argument("--eps", type=float, default=0.03, help="Depth tolerance used during occlusion culling.")
    parser.add_argument("--th_obs", type=int, default=0, help="Observation threshold used during culling.")
    parser.add_argument("--no_occlusion", action="store_true", help="Use frustum-only culling.")
    parser.add_argument("--virtual_cameras", action="store_true", help="Include virtual camera views during culling.")
    parser.add_argument("--virt_cam_path", help="Directory containing virtual camera poses.")
    parser.add_argument("--metric_2d", action="store_true", help="Also run the slower 2D depth metric.")
    parser.add_argument(
        "--dataset_type",
        default="Replica",
        choices=["Replica", "RGBD"],
        help="Dataset type used by the optional 2D metric.",
    )
    args = parser.parse_args()

    remove_occlusion = not args.no_occlusion
    output_mesh = args.output_mesh or default_culled_mesh_path(
        args.input_mesh, remove_occlusion, args.virtual_cameras
    )

    cull_command = [
        sys.executable,
        "cull_mesh.py",
        "--config",
        args.config,
        "--input_mesh",
        args.input_mesh,
        "--output_mesh",
        output_mesh,
        "--skip",
        str(args.skip),
        "--eps",
        str(args.eps),
        "--th_obs",
        str(args.th_obs),
    ]
    if remove_occlusion:
        cull_command.append("--remove_occlusion")
    if args.virtual_cameras:
        cull_command.append("--virtual_cameras")
    if args.virt_cam_path:
        cull_command.extend(["--virt_cam_path", args.virt_cam_path])
    if args.gt_pose:
        cull_command.append("--gt_pose")
    elif args.ckpt_path:
        cull_command.extend(["--ckpt_path", args.ckpt_path])
    else:
        raise SystemExit("Provide either --gt_pose or --ckpt_path for culling poses.")

    run_command(cull_command)

    eval_command = [
        sys.executable,
        "eval_recon.py",
        "--rec_mesh",
        output_mesh,
        "--gt_mesh",
        args.gt_mesh,
        "--metric_3d",
    ]
    if args.metric_2d:
        eval_command.extend(["--metric_2d", "--dataset_type", args.dataset_type])

    run_command(eval_command)


if __name__ == "__main__":
    main()
