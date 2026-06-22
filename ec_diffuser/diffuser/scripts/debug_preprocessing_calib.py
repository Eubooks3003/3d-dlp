#!/usr/bin/env python
"""
Extract and print calibration values from the preprocessing pipeline
to compare with live wrapper values.

Usage:
    python scripts/debug_preprocessing_calib.py --hdf5 /path/to/stack_d1_rgbd_pcd.hdf5
"""

import os
import sys
import json
import argparse
import numpy as np
import h5py

# Add mimicgen/robosuite
import mimicgen.envs.robosuite
try:
    import robosuite_task_zoo
except ImportError:
    pass
import robosuite


def compute_K_from_fovy(fovy_deg, W, H):
    """Same as in mimicgen_ply_all_tasks.py"""
    fovy = np.deg2rad(float(fovy_deg))
    fy = (H / 2.0) / np.tan(fovy / 2.0)
    fx = fy * (W / H)
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float32)


def mujoco_near_far_meters(sim):
    """Same as in mimicgen_ply_all_tasks.py"""
    extent = float(sim.model.stat.extent)
    znear = float(sim.model.vis.map.znear) * extent
    zfar  = float(sim.model.vis.map.zfar)  * extent
    return znear, zfar


def build_env(env_args_json: str):
    """Same as in mimicgen_ply_all_tasks.py"""
    meta = json.loads(env_args_json)
    env_name = meta["env_name"]
    kwargs = dict(meta.get("env_kwargs", {}))
    kwargs["has_renderer"] = False
    kwargs["has_offscreen_renderer"] = False
    kwargs["use_camera_obs"] = False
    kwargs["camera_names"] = []
    kwargs["camera_depths"] = []
    return robosuite.make(env_name, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=str, required=True)
    parser.add_argument("--cams", nargs="+", default=["agentview", "sideview"])
    parser.add_argument("--H", type=int, default=256)
    parser.add_argument("--W", type=int, default=256)
    args = parser.parse_args()

    with h5py.File(args.hdf5, "r") as h5:
        env_args_json = h5["data"].attrs["env_args"]

        # Build env to get camera calibration
        env = build_env(env_args_json)
        sim = env.sim
        sim.forward()

        near_m, far_m = mujoco_near_far_meters(sim)

        print("=" * 60)
        print("PREPROCESSING CALIBRATION VALUES")
        print("=" * 60)
        print(f"near_m: {near_m:.6f}")
        print(f"far_m: {far_m:.6f}")
        print(f"extent: {sim.model.stat.extent:.6f}")
        print()

        for cam in args.cams:
            try:
                cam_id = sim.model.camera_name2id(cam)
            except:
                print(f"Camera {cam} not found, skipping")
                continue

            fovy = float(sim.model.cam_fovy[cam_id])
            K = compute_K_from_fovy(fovy, W=args.W, H=args.H)
            pos = np.array(sim.data.cam_xpos[cam_id], dtype=np.float32)
            R = np.array(sim.data.cam_xmat[cam_id], dtype=np.float32).reshape(3, 3)

            print(f"=== {cam} ===")
            print(f"  fovy: {fovy:.4f} deg")
            print(f"  H x W: {args.H} x {args.W}")
            print(f"  K:")
            print(f"    [[{K[0,0]:.4f}, {K[0,1]:.4f}, {K[0,2]:.4f}],")
            print(f"     [{K[1,0]:.4f}, {K[1,1]:.4f}, {K[1,2]:.4f}],")
            print(f"     [{K[2,0]:.4f}, {K[2,1]:.4f}, {K[2,2]:.4f}]]")
            print(f"  R:")
            print(f"    [[{R[0,0]:.6f}, {R[0,1]:.6f}, {R[0,2]:.6f}],")
            print(f"     [{R[1,0]:.6f}, {R[1,1]:.6f}, {R[1,2]:.6f}],")
            print(f"     [{R[2,0]:.6f}, {R[2,1]:.6f}, {R[2,2]:.6f}]]")
            print(f"  pos: [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]")
            print()

        # Also print depth values from first frame
        print("=" * 60)
        print("DEPTH VALUES FROM HDF5 (demo_0, frame 0)")
        print("=" * 60)
        obs = h5["data/demo_0/obs"]
        for cam in args.cams:
            depth_key = f"{cam}_depth"
            if depth_key in obs:
                depth = np.array(obs[depth_key][0])
                print(f"{cam}_depth: range=[{depth.min():.4f}, {depth.max():.4f}], mean={depth.mean():.4f}")

        env.close()


if __name__ == "__main__":
    main()
