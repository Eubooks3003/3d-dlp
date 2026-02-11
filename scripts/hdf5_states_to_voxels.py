#!/usr/bin/env python3
"""
Unified pipeline: HDF5 states -> RGBD (in-memory) -> fused point cloud -> voxels.

Replays MimicGen HDF5 trajectories in a robosuite environment to extract RGBD
observations, backprojects depth to 3D using camera intrinsics/extrinsics,
fuses multi-view point clouds, and voxelizes them directly.

No intermediate RGBD or PLY files are saved (except in debug mode).

Usage:
    # Debug (1 demo, 1 frame, saves PLY for inspection):
    python scripts/hdf5_states_to_voxels.py \\
        --input /path/to/source.hdf5 \\
        --output-dir /tmp/voxel_debug \\
        --debug-one --use-task-bounds

    # Full processing:
    python scripts/hdf5_states_to_voxels.py \\
        --input /path/to/source.hdf5 \\
        --output-dir /path/to/task_d0/core/voxel_cache \\
        --use-task-bounds

Requires: mimicgen, robosuite, robomimic, open3d, torch
"""

import os
import sys
import json
import argparse
import numpy as np
from copy import deepcopy
from tqdm import tqdm

import h5py
import torch
import open3d as o3d

# Add lpwm-dev root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.voxelize_ds_wrapper import VoxelGridXYZ

import mimicgen.envs.robosuite  # noqa: registers MimicGen envs
try:
    import robosuite_task_zoo  # noqa: registers Kitchen/Hammer etc.
except ImportError:
    pass

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils


# ==================== Constants ====================

DEFAULT_CAMS = ["agentview", "sideview"]

DEFAULT_CROP = {
    "xmin": -1.1, "xmax": 0.9,
    "ymin": -0.5, "ymax": 0.5,
    "zmin": -0.2, "zmax": 2.5,
}

# Per-task PLY crop bounds — looser than voxel bounds to avoid cutting off the scene.
# Points outside these are discarded before voxelization.
# Falls back to DEFAULT_CROP if a task is not listed here.
TASK_SPECIFIC_CROP = {  
    "stack_three": {"xmin": -0.7, "xmax": 0.9, "ymin": -0.5, "ymax": 0.5,"zmin": -0.2, "zmax": 2.5,},
    "stack": {"xmin": -0.7, "xmax": 0.9, "ymin": -0.5, "ymax": 0.5,"zmin": -0.2, "zmax": 2.5,}, 
}

# Per-task voxel grid bounds — tighter, defines the voxel grid extent.
TASK_SPECIFIC_BOUNDS = {
    "hammer_cleanup": {"xmin": -0.4, "xmax": 0.2, "ymin": -0.4, "ymax": 0.4, "zmin": 0.7, "zmax": 1.4},
    "nut_assembly": {"xmin": -0.5, "xmax": 0.4, "ymin": -0.5, "ymax": 0.4, "zmin": 0.1, "zmax": 1.6},
    "pick_place": {"xmin": -0.5, "xmax": 0.3, "ymin": -0.5, "ymax": 1.0, "zmin": 0.0, "zmax": 1.7},
    "square": {"xmin": -0.5, "xmax": 0.4, "ymin": -0.5, "ymax": 0.4, "zmin": 0.0, "zmax": 1.6},
    "stack_three": {"xmin": -0.5, "xmax": 0.4, "ymin": -0.4, "ymax": 0.4, "zmin": 0.2, "zmax": 1.6},
    "threading": {"xmin": -0.5, "xmax": 0.4, "ymin": -0.4, "ymax": 0.4, "zmin": 0.2, "zmax": 1.6},
    "kitchen": {"xmin": -1.1, "xmax": 0.3,"ymin": -0.4, "ymax": 0.4,"zmin": 0.7, "zmax": 1.4,},
    "coffee": {"xmin": -0.5, "xmax": 0.4, "ymin": -0.4, "ymax": 0.4, "zmin": 0.2, "zmax": 1.6},
    "coffee_preparation": {"xmin": -0.5, "xmax": 0.4, "ymin": -0.5, "ymax": 0.4, "zmin": 0.2, "zmax": 1.6},
    "mug_cleanup": {"xmin": -0.5, "xmax": 0.4, "ymin": -0.4, "ymax": 0.5, "zmin": 0.2, "zmax": 1.6},
    "three_piece_assembly": {"xmin": -0.7, "xmax": 0.4, "ymin": -0.4, "ymax": 0.5, "zmin": 0.2, "zmax": 1.6},
    "stack":     {"xmin": -0.7, "xmax": 0.4, "ymin": -0.5, "ymax": 0.5, "zmin": -0.2, "zmax": 1.6}
}


# ==================== Geometry Utilities ====================

def squeeze_hw(d):
    """Squeeze depth array to 2D [H, W]."""
    d = np.asarray(d)
    if d.ndim == 3:
        d = np.squeeze(d)
    if d.ndim != 2:
        d = d.reshape(d.shape[-2], d.shape[-1])
    return d


def compute_K_from_fovy(fovy_deg, W, H):
    """Compute camera intrinsic matrix from vertical field of view."""
    fovy = np.deg2rad(float(fovy_deg))
    fy = (H / 2.0) / np.tan(fovy / 2.0)
    fx = fy * (W / H)
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float32)


def backproject(depth_z, K, pixel_stride):
    """Backproject depth map to 3D camera-frame points."""
    z = squeeze_hw(depth_z).astype(np.float32)
    H, W = z.shape
    vv = np.arange(0, H, pixel_stride, dtype=np.int32)
    uu = np.arange(0, W, pixel_stride, dtype=np.int32)
    U, V = np.meshgrid(uu, vv)

    Z = z[V, U]
    valid = np.isfinite(Z) & (Z > 0)
    if not np.any(valid):
        return np.zeros((0, 3), np.float32), np.zeros((0, 2), np.int32)

    Uv = U[valid].astype(np.float32)
    Vv = V[valid].astype(np.float32)
    Zv = Z[valid].astype(np.float32)

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    X = (Uv - cx) / fx * Zv
    Y = (Vv - cy) / fy * Zv
    pts_cam = np.stack([X, Y, Zv], axis=-1).astype(np.float32)
    idxs = np.stack([V[valid], U[valid]], axis=-1).astype(np.int32)
    return pts_cam, idxs


def maybe_crop(xyz, rgb, bounds):
    """Crop point cloud to bounding box."""
    if bounds is None:
        return xyz, rgb
    b = bounds
    m = (
        (xyz[:, 0] >= b["xmin"]) & (xyz[:, 0] <= b["xmax"]) &
        (xyz[:, 1] >= b["ymin"]) & (xyz[:, 1] <= b["ymax"]) &
        (xyz[:, 2] >= b["zmin"]) & (xyz[:, 2] <= b["zmax"])
    )
    if not np.any(m):
        return xyz, rgb
    return xyz[m], rgb[m]


# ==================== Depth Conversion ====================

def depth_to_z(depth_raw, mode, near_m, far_m):
    """Convert raw depth buffer to metric depth."""
    d = squeeze_hw(depth_raw).astype(np.float32)
    dmax = float(np.nanmax(d)) if np.isfinite(d).any() else 0.0

    if mode == "as_is":
        return d
    if dmax > 1.05:
        return d

    d = np.clip(d, 0.0, 1.0)
    n, f = float(near_m), float(far_m)

    if mode == "linear":
        return n + d * (f - n)
    if mode == "opengl_inv_simple":
        return (n * f) / (f - d * (f - n) + 1e-12)
    if mode == "opengl_ndc":
        z_ndc = 2.0 * d - 1.0
        return (2.0 * n * f) / (f + n - z_ndc * (f - n) + 1e-12)
    raise ValueError(f"Unknown depth mode: {mode}")


# ==================== Camera / Env Utilities ====================

def get_sim(env):
    """Get MuJoCo sim from a possibly-wrapped robomimic env."""
    for path in ["sim", "env.sim", "env.env.sim", "base_env.sim"]:
        cur = env
        try:
            for part in path.split("."):
                cur = getattr(cur, part)
            _ = cur.model  # verify it's a sim
            return cur
        except (AttributeError, Exception):
            continue
    raise RuntimeError("Cannot find MuJoCo sim in env")


def available_cameras(env):
    """List available camera names from the env."""
    for path in ("env.sim.model.camera_names", "sim.model.camera_names",
                 "env.env.sim.model.camera_names"):
        try:
            cur = env
            for part in path.split("."):
                cur = getattr(cur, part)
            return list(cur)
        except Exception:
            pass
    return []


def choose_cameras(env_name, available, requested):
    """Choose 2 cameras based on env type and availability."""
    avail = list(available)
    avail_set = set(avail)
    req = [c for c in (requested or []) if c in avail_set]

    if env_name.startswith("PickPlace"):
        want = ["agentview", "frontview"]
    else:
        want = ["agentview", "sideview"]
        if "sideview" not in avail_set and "frontview" in avail_set:
            want = ["agentview", "frontview"]

    out = []
    for c in req:
        if c not in out:
            out.append(c)
        if len(out) == 2:
            return out
    for c in want:
        if c in avail_set and c not in out:
            out.append(c)
        if len(out) == 2:
            return out
    for c in avail:
        if c not in out:
            out.append(c)
        if len(out) == 2:
            return out
    return out


def mujoco_near_far_meters(sim):
    """Get near/far clipping planes in meters from MuJoCo sim."""
    extent = float(sim.model.stat.extent)
    znear = float(sim.model.vis.map.znear) * extent
    zfar = float(sim.model.vis.map.zfar) * extent
    return znear, zfar


# ==================== Point Cloud Construction ====================

def build_points_for_cam(sim, obs_dict, cam, K, near_m, far_m,
                         depth_mode, pixel_stride, crop_bounds,
                         max_points, rng):
    """
    Build world-frame point cloud for one camera from a single-frame obs dict.

    obs_dict: dict with keys like '{cam}_image' (H,W,3) and '{cam}_depth' (H,W,1).
    Returns (pts_world, rgb) or (None, None).
    """
    depth_key = f"{cam}_depth"
    img_key = f"{cam}_image"
    if depth_key not in obs_dict or img_key not in obs_dict:
        return None, None

    depth_raw = np.asarray(obs_dict[depth_key])
    img = np.asarray(obs_dict[img_key])

    z = depth_to_z(depth_raw, mode=depth_mode, near_m=near_m, far_m=far_m)
    pts_cam, idxs = backproject(z, K, pixel_stride=pixel_stride)
    if pts_cam.shape[0] == 0:
        return None, None

    v, u = idxs[:, 0], idxs[:, 1]
    rgb = img[v, u, :].astype(np.float32)
    if rgb.max() > 1.5:
        rgb /= 255.0

    # Camera -> world using MuJoCo cam params
    cam_id = sim.model.camera_name2id(cam)
    pos = np.array(sim.data.cam_xpos[cam_id], dtype=np.float32)
    R = np.array(sim.data.cam_xmat[cam_id], dtype=np.float32).reshape(3, 3)

    # CV -> OpenGL camera coords (flip y and z)
    pts_gl = pts_cam.copy()
    pts_gl[:, 1] *= -1.0
    pts_gl[:, 2] *= -1.0

    # Camera -> world
    pts_world = (R @ pts_gl.T).T + pos
    pts_world, rgb = maybe_crop(pts_world, rgb, crop_bounds)

    if pts_world.shape[0] == 0:
        return None, None

    if max_points > 0 and pts_world.shape[0] > max_points:
        sel = rng.choice(pts_world.shape[0], size=max_points, replace=False)
        pts_world, rgb = pts_world[sel], rgb[sel]

    return pts_world, rgb


def fuse_cameras(sim, obs_dict, cams, K_by_cam, convention, crop_bounds,
                 pixel_stride, max_points, rng):
    """Fuse point clouds from multiple cameras into one."""
    all_xyz, all_rgb = [], []

    for cam in cams:
        pts, rgb = build_points_for_cam(
            sim, obs_dict, cam, K_by_cam[cam],
            convention["near_m"], convention["far_m"],
            convention["depth_mode"], pixel_stride,
            crop_bounds, max_points, rng,
        )
        if pts is not None:
            all_xyz.append(pts)
            all_rgb.append(rgb)

    if not all_xyz:
        return None, None

    return np.concatenate(all_xyz, axis=0), np.concatenate(all_rgb, axis=0)


# ==================== Auto Convention Picking ====================

def score_fusion(ptsA, ptsB, max_n=20000):
    """Score how well two point clouds overlap (lower = better)."""
    if ptsA.shape[0] == 0 or ptsB.shape[0] == 0:
        return 1e9

    rng = np.random.default_rng(0)
    if ptsA.shape[0] > max_n:
        ptsA = ptsA[rng.choice(ptsA.shape[0], max_n, replace=False)]
    if ptsB.shape[0] > max_n:
        ptsB = ptsB[rng.choice(ptsB.shape[0], max_n, replace=False)]

    from scipy.spatial import cKDTree
    tA = cKDTree(ptsA)
    tB = cKDTree(ptsB)
    dAB = tB.query(ptsA, k=1)[0].mean()
    dBA = tA.query(ptsB, k=1)[0].mean()

    allp = np.concatenate([ptsA, ptsB], axis=0)
    diag = float(np.linalg.norm(allp.max(0) - allp.min(0)))

    size_pen = 0.0
    if diag < 0.2:
        size_pen += (0.2 - diag) * 50.0
    if diag > 8.0:
        size_pen += (diag - 8.0) * 5.0

    return 0.5 * (dAB + dBA) + size_pen


def auto_pick_convention(sim, obs_dict, cams, H, W, pixel_stride,
                         crop_bounds, max_points, seed):
    """
    Auto-pick depth conversion mode by testing combinations on a single frame
    and choosing the one where two camera views overlap best.
    """
    rng = np.random.default_rng(seed)
    near_m, far_m = mujoco_near_far_meters(sim)

    K_by_cam = {}
    for cam in cams:
        cam_id = sim.model.camera_name2id(cam)
        fovy = float(sim.model.cam_fovy[cam_id])
        K_by_cam[cam] = compute_K_from_fovy(fovy, W=W, H=H)

    depth_modes = ["linear", "opengl_ndc", "opengl_inv_simple"]
    best = None
    best_score = 1e18

    camA, camB = cams[0], cams[1]

    for dm in depth_modes:
        ptsA, _ = build_points_for_cam(
            sim, obs_dict, camA, K_by_cam[camA], near_m, far_m, dm,
            pixel_stride, crop_bounds, max_points, rng,
        )
        ptsB, _ = build_points_for_cam(
            sim, obs_dict, camB, K_by_cam[camB], near_m, far_m, dm,
            pixel_stride, crop_bounds, max_points, rng,
        )

        if ptsA is None or ptsB is None:
            continue

        sc = score_fusion(ptsA, ptsB)
        if sc < best_score:
            best_score = sc
            best = dict(depth_mode=dm, near_m=near_m, far_m=far_m)

    if best is None:
        raise RuntimeError("Auto-pick failed: could not build points for both cameras.")

    print(f"  [AUTO] depth_mode={best['depth_mode']} "
          f"near={best['near_m']:.4f} far={best['far_m']:.4f} "
          f"score={best_score:.4f}")
    return best, K_by_cam


# ==================== Voxelization Utilities ====================

def safe_tensor(arr, dtype=torch.float32):
    """Convert numpy array to torch tensor (AARCH compatible)."""
    return torch.tensor(arr.tolist(), dtype=dtype)


def bounds_dict_to_tensors(bounds_dict):
    """Convert bounds dict to (pmin, pmax) torch tensors."""
    pmin = torch.tensor(
        [bounds_dict["xmin"], bounds_dict["ymin"], bounds_dict["zmin"]],
        dtype=torch.float32,
    )
    pmax = torch.tensor(
        [bounds_dict["xmax"], bounds_dict["ymax"], bounds_dict["zmax"]],
        dtype=torch.float32,
    )
    return pmin, pmax


def voxelize_points(xyz, rgb, grid_whd, bounds, voxel_mode="avg_rgb"):
    """
    Voxelize a point cloud.

    Returns: (voxel_grid [C,D,H,W], meta_dict)
    """
    pts_xyz = safe_tensor(xyz)
    colors = safe_tensor(rgb) if (rgb is not None and voxel_mode == "avg_rgb") else None

    vg = VoxelGridXYZ(pts_xyz, colors, grid_whd=grid_whd, bounds=bounds, mode=voxel_mode)
    vox = vg.to_dense().cpu()
    md = vg.meta_dict()
    md_cpu = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in md.items()}
    return vox, md_cpu


# ==================== Compressed Voxel I/O ====================

def save_voxel_compressed(path, vox):
    """
    Save a sparse voxel grid efficiently.

    Stores only occupied voxel coords + values in float16.
    At 2% occupancy this is ~50x smaller than dense float32.
    """
    occupied = vox.abs().sum(dim=0) > 0  # [D, H, W]
    coords_long = occupied.nonzero(as_tuple=False)  # [N, 3] int64
    values = vox[:, coords_long[:, 0], coords_long[:, 1], coords_long[:, 2]].T.half()  # [N, C]
    coords = coords_long.to(torch.int16)  # store as int16 to save space

    torch.save({
        "coords": coords,
        "values": values,
        "shape": list(vox.shape),
        "compressed": True,
    }, path)


# ==================== PLY Writing (debug) ====================

def write_ply(path, xyz, rgb):
    """Write a colored point cloud to PLY file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
    if not o3d.io.write_point_cloud(path, pcd):
        raise RuntimeError(f"Failed to write {path}")


# ==================== Resume Support ====================

def count_cached_frames(demo_dir):
    """Count how many voxel files exist in a demo directory."""
    if not os.path.isdir(demo_dir):
        return 0
    return len([f for f in os.listdir(demo_dir) if f.endswith("_voxels.pt")])


# ==================== Main Processing ====================

def process_hdf5(args):
    # 1. Get env metadata from input HDF5
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.input)
    env_name = env_meta["env_name"]

    # 2. Probe available cameras
    probe_env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=[],
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        reward_shaping=False,
    )
    avail_cams = available_cameras(probe_env)
    try:
        probe_env.close()
    except Exception:
        pass

    # 3. Choose cameras
    chosen_cams = choose_cameras(env_name, avail_cams, args.cams)
    print(f"env: {env_name}")
    print(f"available cameras: {avail_cams}")
    print(f"chosen cameras: {chosen_cams}")

    if len(chosen_cams) < 2:
        raise RuntimeError(f"Need at least 2 cameras, got: {chosen_cams}")

    # 4. Create env with depth enabled
    env_meta_depth = deepcopy(env_meta)
    env_meta_depth["env_kwargs"] = dict(env_meta_depth.get("env_kwargs", {}))
    env_meta_depth["env_kwargs"]["camera_depths"] = [True] * len(chosen_cams)

    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta_depth,
        camera_names=chosen_cams,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        reward_shaping=False,
    )

    is_robosuite = EnvUtils.is_robosuite_env(env_meta)

    # 5. Open input HDF5 and enumerate demos
    f = h5py.File(args.input, "r")
    demos = sorted(list(f["data"].keys()), key=lambda x: int(x[5:]))

    if args.start_demo is not None or args.end_demo is not None:
        start = args.start_demo if args.start_demo is not None else 0
        end = args.end_demo if args.end_demo is not None else len(demos)
        demos = demos[start:end]
        print(f"Demo range: {start} to {end} ({len(demos)} demos)")
    elif args.n is not None:
        demos = demos[:args.n]

    if args.debug_one:
        demos = demos[:1]
        print("[DEBUG] Processing 1 demo, 1 frame only")

    # 6. Setup crop bounds (PLY-level, before voxelization)
    task_name_lower = env_name.lower().replace(" ", "_")
    if args.no_crop:
        crop_bounds = None
    elif args.use_task_bounds:
        crop_bounds = DEFAULT_CROP  # fallback
        for key in TASK_SPECIFIC_CROP:
            if key in task_name_lower:
                crop_bounds = TASK_SPECIFIC_CROP[key]
                print(f"PLY crop: task-specific '{key}'")
                break
        else:
            print("PLY crop: DEFAULT_CROP (no task-specific crop for this env)")
    else:
        crop_bounds = DEFAULT_CROP
        print("PLY crop: DEFAULT_CROP")

    # 7. Setup voxel bounds (tighter, defines the voxel grid)
    task_bounds = None
    if args.use_task_bounds:
        for key in TASK_SPECIFIC_BOUNDS:
            if key in task_name_lower:
                task_bounds = TASK_SPECIFIC_BOUNDS[key]
                print(f"Matched task-specific voxel bounds: '{key}'")
                break

    if task_bounds is not None:
        voxel_bounds = bounds_dict_to_tensors(task_bounds)
        bounds_mode = "task_specific"
        print(f"Voxel bounds: task-specific")
        print(f"  x: [{task_bounds['xmin']}, {task_bounds['xmax']}]")
        print(f"  y: [{task_bounds['ymin']}, {task_bounds['ymax']}]")
        print(f"  z: [{task_bounds['zmin']}, {task_bounds['zmax']}]")
    elif args.fixed_bounds:
        voxel_bounds = bounds_dict_to_tensors(DEFAULT_CROP)
        bounds_mode = "fixed_mimicgen_crop"
        print("Voxel bounds: fixed (MIMICGEN_CROP_BOUNDS)")
    else:
        voxel_bounds = None
        bounds_mode = "per_item"
        print("Voxel bounds: per-item (computed from each point cloud)")

    grid_whd = tuple(args.grid_whd)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    convention = None
    K_by_cam = None
    total_frames = 0

    if args.frame_stride > 1:
        print(f"Frame stride: {args.frame_stride} (saving every {args.frame_stride}th frame)")

    print(f"\nProcessing {len(demos)} demos from {args.input}")
    print(f"Output: {output_dir}\n")

    for ep_idx, ep in enumerate(tqdm(demos, desc=f"{env_name} demos")):
        states = f[f"data/{ep}/states"][()]
        initial_state = dict(states=states[0])
        if is_robosuite:
            initial_state["model"] = f[f"data/{ep}"].attrs["model_file"]

        actions = f[f"data/{ep}/actions"][()]
        traj_len = actions.shape[0]

        if args.debug_one:
            traj_len = 1
        elif args.max_frames_per_demo > 0:
            traj_len = min(traj_len, args.max_frames_per_demo)

        # Output subdirs: output_dir/voxel/demo_X and output_dir/ply/demo_X
        voxel_demo_dir = os.path.join(output_dir, "voxel", ep)
        ply_demo_dir = os.path.join(output_dir, "ply", ep)

        # Resume: skip fully-cached demos
        expected_frames = len(range(0, traj_len, args.frame_stride))
        if not args.force and not args.debug_one:
            cached = count_cached_frames(voxel_demo_dir)
            if cached >= expected_frames:
                tqdm.write(f"  {ep}: {cached} frames cached, skipping")
                total_frames += cached
                continue

        # Reset env to initial state
        env.reset()
        obs = env.reset_to(initial_state)

        # Get sim reference (for camera params)
        sim = get_sim(env)

        # Auto-pick depth convention on first frame
        if convention is None:
            print("Auto-picking depth convention...")
            convention, K_by_cam = auto_pick_convention(
                sim, obs, chosen_cams,
                H=args.camera_height, W=args.camera_width,
                pixel_stride=args.pixel_stride,
                crop_bounds=crop_bounds,
                max_points=min(args.max_points, 80000) if args.max_points > 0 else -1,
                seed=args.seed,
            )

        os.makedirs(voxel_demo_dir, exist_ok=True)
        if args.debug_one or args.save_ply:
            os.makedirs(ply_demo_dir, exist_ok=True)

        # Build list of frames to actually voxelize (respecting stride)
        frame_stride = args.frame_stride
        save_frames = set(range(0, traj_len, frame_stride))

        for t in tqdm(range(traj_len), desc=f"  {ep}", leave=False):
            # Always step the env (to keep sim in sync), but only
            # voxelize + save on frames that match the stride.
            if t not in save_frames:
                if t < actions.shape[0]:
                    obs, _, _, _ = env.step(actions[t])
                continue

            vox_path = os.path.join(voxel_demo_dir, f"frame{t}_voxels.pt")

            # Skip if already cached (frame-level resume)
            if not args.force and not args.debug_one and os.path.exists(vox_path):
                total_frames += 1
                # Still need to step env to advance
                if t < actions.shape[0]:
                    obs, _, _, _ = env.step(actions[t])
                continue

            # Build fused point cloud from current obs
            xyz, rgb = fuse_cameras(
                sim, obs, chosen_cams, K_by_cam, convention,
                crop_bounds, args.pixel_stride, args.max_points, rng,
            )

            if xyz is not None and xyz.shape[0] > 0:
                # Save fused PLY (debug or explicit)
                if args.debug_one or args.save_ply:
                    fused_ply = os.path.join(ply_demo_dir, f"frame{t:06d}_fused.ply")
                    write_ply(fused_ply, xyz, rgb)
                    print(f"  [PLY] {fused_ply} (N={xyz.shape[0]})")

                # Determine voxel bounds for this frame
                if voxel_bounds is not None:
                    bounds = voxel_bounds
                else:
                    pts_t = safe_tensor(xyz)
                    bounds = (pts_t.amin(dim=0), pts_t.amax(dim=0))

                # Voxelize
                vox, meta = voxelize_points(xyz, rgb, grid_whd, bounds, args.voxel_mode)

                # Save voxel + meta
                meta_path = os.path.join(voxel_demo_dir, f"frame{t}_meta.pt")
                if args.compress:
                    save_voxel_compressed(vox_path, vox)
                else:
                    torch.save(vox, vox_path)
                torch.save(meta, meta_path)

                total_frames += 1

                # Debug stats + wandb
                if args.debug_one:
                    n_occupied = (vox.abs().sum(dim=0) > 0).sum().item()
                    n_total = vox.shape[1] * vox.shape[2] * vox.shape[3]
                    print(f"\n  [VOXEL] shape: {tuple(vox.shape)}")
                    print(f"  [VOXEL] occupied: {n_occupied}/{n_total} "
                          f"({100 * n_occupied / n_total:.1f}%)")
                    print(f"  [VOXEL] saved: {vox_path}")
                    print(f"  [POINT CLOUD] N={xyz.shape[0]}")
                    print(f"  [POINT CLOUD] xyz range: "
                          f"x=[{xyz[:,0].min():.3f}, {xyz[:,0].max():.3f}] "
                          f"y=[{xyz[:,1].min():.3f}, {xyz[:,1].max():.3f}] "
                          f"z=[{xyz[:,2].min():.3f}, {xyz[:,2].max():.3f}]")
                    print(f"  [META] pmin={meta['pmin'].tolist()}")
                    print(f"  [META] pmax={meta['pmax'].tolist()}")
                    print(f"  [META] voxel_size={meta['voxel_size'].tolist()}")
                    vox_bytes = os.path.getsize(vox_path)
                    dense_bytes = vox.numel() * 4  # float32
                    print(f"  [STORAGE] {vox_bytes / 1024:.0f}KB on disk "
                          f"(vs {dense_bytes / 1024:.0f}KB dense, "
                          f"{dense_bytes / vox_bytes:.0f}x compression)"
                          if args.compress else
                          f"  [STORAGE] {vox_bytes / 1024:.0f}KB on disk (use --compress for ~{dense_bytes / (n_occupied * 10 + 256):.0f}x reduction)")

                    # Log to wandb
                    try:
                        import wandb
                        from eval.eval_vox import log_rgb_voxels

                        wandb.init(
                            project=args.wandb_project,
                            name=f"debug-{env_name}-{ep}-t{t}",
                        )
                        print(f"  [WANDB] Logging to project: {args.wandb_project}")

                        log_rgb_voxels(
                            name=f"debug/{env_name}",
                            rgb_vol=vox,
                            alpha_vol=None,
                            KPx=None,
                            step=0,
                            mode="splat",
                            topk=60000,
                            alpha_thresh=0.05,
                            pad=2.0,
                            show_axes=True,
                        )
                        wandb.finish()
                        print(f"  [WANDB] Done — check {args.wandb_project} for visualization")
                    except ImportError:
                        print("  [WANDB] wandb not installed, skipping visualization. "
                              "Install with: pip install wandb")
            else:
                tqdm.write(f"  [WARN] {ep} t={t}: empty point cloud, skipping")

            # Step env to get next obs
            if t < actions.shape[0]:
                obs, _, _, _ = env.step(actions[t])

    # Write manifest
    manifest = {
        "input": os.path.abspath(args.input),
        "env_name": env_name,
        "cameras": chosen_cams,
        "convention": convention,
        "total_frames": total_frames,
        "grid_whd": list(grid_whd),
        "voxel_mode": args.voxel_mode,
        "bounds_mode": bounds_mode,
        "frame_stride": args.frame_stride,
        "crop_bounds": crop_bounds,
        "complete": True,
    }
    if task_bounds is not None:
        manifest["task_bounds"] = task_bounds
    elif args.fixed_bounds:
        manifest["fixed_bounds"] = DEFAULT_CROP

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as mf:
        json.dump(manifest, mf, indent=2)

    f.close()
    try:
        env.close()
    except Exception:
        pass

    print(f"\nDone! Processed {total_frames} frames from {len(demos)} demos")
    print(f"Output: {output_dir}")

    if args.debug_one:
        print(f"\n[DEBUG] Inspect the PLY in {ply_demo_dir}/")
        print(f"[DEBUG] Inspect the voxel in {voxel_demo_dir}/")
        print("[DEBUG] Re-run without --debug-one for full processing.")


def main():
    parser = argparse.ArgumentParser(
        description="Unified pipeline: HDF5 states -> RGBD -> point cloud -> voxels",
    )

    # Input/output
    parser.add_argument("--input", type=str, required=True,
                        help="Input HDF5 file (MimicGen source with states/actions)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for voxels (and PLY in debug mode)")

    # Demo selection
    parser.add_argument("--n", type=int, default=None,
                        help="Process first N demos")
    parser.add_argument("--start-demo", type=int, default=None,
                        help="Start demo index (inclusive, 0-indexed)")
    parser.add_argument("--end-demo", type=int, default=None,
                        help="End demo index (exclusive, 0-indexed)")
    parser.add_argument("--max-frames-per-demo", type=int, default=-1,
                        help="Max frames per demo (-1 = all)")
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="Only voxelize every Nth frame (default: 1 = all frames). "
                             "Saves with original frame numbers.")

    # Camera
    parser.add_argument("--cams", nargs="+", default=DEFAULT_CAMS,
                        help="Preferred camera names (default: agentview sideview)")
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)

    # Point cloud
    parser.add_argument("--pixel-stride", type=int, default=1,
                        help="Subsampling stride for depth backprojection (default: 1)")
    parser.add_argument("--max-points", type=int, default=200000,
                        help="Max points per camera view (default: 200000)")
    parser.add_argument("--no-crop", action="store_true",
                        help="Disable workspace cropping")
    parser.add_argument("--seed", type=int, default=0)

    # Voxelization
    parser.add_argument("--grid-whd", type=int, nargs=3, default=[128, 128, 128],
                        help="Voxel grid dimensions (default: 128 128 128)")
    parser.add_argument("--voxel-mode", type=str, default="avg_rgb",
                        choices=["occupancy", "density", "avg_rgb"],
                        help="Voxelization mode (default: avg_rgb)")
    parser.add_argument("--fixed-bounds", action="store_true",
                        help="Use fixed MIMICGEN_CROP_BOUNDS for voxelization")
    parser.add_argument("--use-task-bounds", action="store_true",
                        help="Use task-specific bounds when available")

    # Debug & control
    parser.add_argument("--debug-one", action="store_true",
                        help="Debug: process 1 demo, 1 frame, save PLY + voxels with stats")
    parser.add_argument("--save-ply", action="store_true",
                        help="Save fused PLY files for all frames")
    parser.add_argument("--compress", action="store_true",
                        help="Save voxels as sparse float16 (~50x smaller for typical occupancy)")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess even if output already exists")
    parser.add_argument("--wandb-project", type=str, default="mimicgen-voxel-debug",
                        help="Wandb project for debug visualizations (default: mimicgen-voxel-debug)")

    args = parser.parse_args()
    process_hdf5(args)


if __name__ == "__main__":
    main()
