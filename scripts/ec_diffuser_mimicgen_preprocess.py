#!/usr/bin/env python3
import os, re, argparse, pickle
import numpy as np
import h5py
import torch
import open3d as o3d

import sys                                                                                                                   
import numpy.core as _core                                                                                                   
sys.modules['numpy._core'] = _core                                                                                           
sys.modules['numpy._core.multiarray'] = _core.multiarray    
from voxel_models import DLP
from utils.util_func import get_config
from utils.log_utils import load_checkpoint

from datasets.voxelize_ds_wrapper import VoxelGridXYZ
from robomimic_converter import RobomimicAbsoluteActionConverter


# ----------------------------
# Gripper state extraction
# ----------------------------
def quat_to_rot6d(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (w,x,y,z) or (x,y,z,w) to 6D rotation representation.
    6D = first two columns of rotation matrix, flattened.

    Args:
        quat: [..., 4] quaternion array
    Returns:
        rot6d: [..., 6] rotation representation
    """
    # Robomimic uses (x,y,z,w) convention
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    # Rotation matrix from quaternion
    # First column
    r00 = 1 - 2*(y*y + z*z)
    r10 = 2*(x*y + w*z)
    r20 = 2*(x*z - w*y)

    # Second column
    r01 = 2*(x*y - w*z)
    r11 = 1 - 2*(x*x + z*z)
    r21 = 2*(y*z + w*x)

    # Stack first two columns as 6D representation
    rot6d = np.stack([r00, r10, r20, r01, r11, r21], axis=-1)
    return rot6d.astype(np.float32)


def extract_gripper_state(h5, demo: str) -> np.ndarray:
    """
    Extract gripper state from H5 file for a demo.

    Gripper state format: [pos(3), rot_6d(6), gripper_open(1)] = 10 dims

    Args:
        h5: open h5py.File
        demo: demo name like "demo_0"

    Returns:
        gripper_state: [T, 10] array
    """
    obs_group = h5[f"data/{demo}/obs"]

    # End-effector position (3D)
    eef_pos_key = None
    for key in ["robot0_eef_pos", "eef_pos"]:
        if key in obs_group:
            eef_pos_key = key
            break
    if eef_pos_key is None:
        raise RuntimeError(f"Cannot find eef_pos in {demo}/obs. Available keys: {list(obs_group.keys())}")
    eef_pos = np.asarray(obs_group[eef_pos_key], dtype=np.float32)  # [T, 3]

    # End-effector quaternion (4D) -> convert to 6D rotation
    eef_quat_key = None
    for key in ["robot0_eef_quat", "eef_quat"]:
        if key in obs_group:
            eef_quat_key = key
            break
    if eef_quat_key is None:
        raise RuntimeError(f"Cannot find eef_quat in {demo}/obs. Available keys: {list(obs_group.keys())}")
    eef_quat = np.asarray(obs_group[eef_quat_key], dtype=np.float32)  # [T, 4]
    eef_rot6d = quat_to_rot6d(eef_quat)  # [T, 6]

    # Gripper state (openness) - typically from gripper joint positions
    gripper_key = None
    for key in ["robot0_gripper_qpos", "gripper_qpos"]:
        if key in obs_group:
            gripper_key = key
            break

    if gripper_key is not None:
        gripper_qpos = np.asarray(obs_group[gripper_key], dtype=np.float32)  # [T, 2] typically
        # Take mean of two finger positions as gripper openness (normalized)
        # MimicGen gripper: 0 = closed, 0.04 = open (for Panda)
        gripper_open = gripper_qpos.mean(axis=-1, keepdims=True)  # [T, 1]
        # Normalize to roughly [-1, 1] range (0.02 is roughly middle)
        gripper_open = (gripper_open - 0.02) / 0.02
    else:
        # Fallback: try to get from actions (last dim is often gripper command)
        print(f"[warn] No gripper_qpos found for {demo}, using zeros")
        gripper_open = np.zeros((eef_pos.shape[0], 1), dtype=np.float32)

    # Concatenate: [pos(3), rot6d(6), gripper_open(1)] = 10D
    gripper_state = np.concatenate([eef_pos, eef_rot6d, gripper_open], axis=-1)

    return gripper_state


# ----------------------------
# TODataset-equivalent processing
# ----------------------------
def read_ply_points(path: str, include_rgb: bool) -> np.ndarray:
    pc = o3d.io.read_point_cloud(path)
    xyz = np.asarray(pc.points, dtype=np.float32)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise RuntimeError(f"Bad xyz array from {path}: shape={xyz.shape}")

    if include_rgb:
        if len(pc.colors) == 0:
            raise RuntimeError(f"include_rgb=True but PLY has no colors: {path}")
        rgb = np.asarray(pc.colors, dtype=np.float32)
        if rgb.shape != xyz.shape:
            raise RuntimeError(f"PLY colors shape mismatch in {path}: xyz={xyz.shape}, rgb={rgb.shape}")
        pts = np.concatenate([xyz, rgb], axis=-1)  # [N,6]
        return pts

    return xyz  # [N,3]


def center_scale_unit_cube(pts: np.ndarray) -> np.ndarray:
    out = pts.copy()
    xyz = out[:, :3]
    if xyz.shape[0] == 0:
        return out
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    scale = (maxs - mins).max() + 1e-8
    out[:, :3] = (xyz - center[None, :]) / scale * 2.0
    return out


def downsample(pts: np.ndarray, max_points: int) -> np.ndarray:
    if pts.shape[0] <= max_points:
        return pts
    idx = np.random.choice(pts.shape[0], max_points, replace=False)
    return pts[idx]


def preprocess_ply(
    ply_path: str,
    *,
    max_points: int,
    normalize_to_unit_cube_flag: bool,
    include_rgb: bool,
) -> torch.Tensor:
    pts = read_ply_points(ply_path, include_rgb=include_rgb)
    if normalize_to_unit_cube_flag:
        pts = center_scale_unit_cube(pts)
    pts = downsample(pts, max_points=max_points)
    return torch.from_numpy(pts).float()  # CPU


# ----------------------------
# Voxelization (wrapper-faithful)
# ----------------------------
def voxelize_points_via_wrapper(
    pts_t: torch.Tensor,           # [N,3] or [N,6], CPU or GPU
    *,
    grid_dhw,                      # (D,H,W)
    voxel_mode: str,               # "occupancy"|"density"|"moments"|"avg_rgb"
    bounds_pm,                     # None for per-item; else (pmin_tuple, pmax_tuple)
) -> torch.Tensor:
    D, H, W = map(int, grid_dhw)

    if pts_t.ndim != 2 or pts_t.shape[1] not in (3, 6):
        raise RuntimeError(f"pts_t must be [N,3] or [N,6], got {tuple(pts_t.shape)}")

    xyz = pts_t[:, :3]
    colors = None
    if voxel_mode == "avg_rgb":
        if pts_t.shape[1] != 6:
            raise RuntimeError("voxel_mode='avg_rgb' requires include_rgb=True / points [N,6].")
        colors = pts_t[:, 3:6]

    # Wrapper expects grid_whd=(W,H,D)
    vg = VoxelGridXYZ(
        points_xyz=xyz,
        colors=colors,
        grid_whd=(W, H, D),
        bounds=bounds_pm,     # None => per-item bounds (amin/amax)
        mode=voxel_mode,
    )
    vox = vg.to_dense()       # [C,D,H,W]
    return vox


# ----------------------------
# DLP token packing (K=24)
# ----------------------------
def build_dlp_from_cfg(cfg, device):
    model = DLP(
        cdim=cfg["ch"],
        image_size=cfg["voxel_grid_whd"][0],
        normalize_rgb=cfg["normalize_rgb"],
        n_kp_per_patch=cfg["n_kp_per_patch"],
        patch_size=cfg["patch_size"],
        anchor_s=cfg["anchor_s"],
        n_kp_enc=cfg["n_kp_enc"],
        n_kp_prior=cfg["n_kp_prior"],
        pad_mode=cfg["pad_mode"],
        dropout=cfg["dropout"],
        features_dist=cfg.get("features_dist", "gauss"),
        learned_feature_dim=cfg["learned_feature_dim"],
        learned_bg_feature_dim=cfg.get("learned_bg_feature_dim", cfg["learned_feature_dim"]),
        n_fg_categories=cfg.get("n_fg_categories", 8),
        n_fg_classes=cfg.get("n_fg_classes", 4),
        n_bg_categories=cfg.get("n_bg_categories", 4),
        n_bg_classes=cfg.get("n_bg_classes", 4),
        scale_std=cfg["scale_std"],
        offset_std=cfg["offset_std"],
        obj_on_alpha=cfg["obj_on_alpha"],
        obj_on_beta=cfg["obj_on_beta"],
        obj_res_from_fc=cfg["obj_res_from_fc"],
        obj_ch_mult_prior=cfg.get("obj_ch_mult_prior", cfg["obj_ch_mult"]),
        obj_ch_mult=cfg["obj_ch_mult"],
        obj_base_ch=cfg["obj_base_ch"],
        obj_final_cnn_ch=cfg["obj_final_cnn_ch"],
        bg_res_from_fc=cfg["bg_res_from_fc"],
        bg_ch_mult=cfg["bg_ch_mult"],
        bg_base_ch=cfg["bg_base_ch"],
        bg_final_cnn_ch=cfg["bg_final_cnn_ch"],
        use_resblock=cfg["use_resblock"],
        num_res_blocks=cfg["num_res_blocks"],
        cnn_mid_blocks=cfg.get("cnn_mid_blocks", False),
        mlp_hidden_dim=cfg.get("mlp_hidden_dim", 256),
        pint_enc_layers=cfg["pint_enc_layers"],
        pint_enc_heads=cfg["pint_enc_heads"],
        timestep_horizon=1,
        separate_depth_features=cfg.get("separate_depth_features", False),
        depth_feature_dim=cfg.get("depth_feature_dim", 0),
        split_loss=cfg.get("split_loss", False),
        depth_loss_ratio=cfg.get("depth_loss_ratio", 1.0),
    ).to(device)
    model.eval()
    return model


def pack_tokens_k24(out: dict) -> torch.Tensor:
    # Expect the voxel DLP conventions you’re using: z,z_scale,z_depth,obj_on,z_features are (B,1,24,*)
    z       = out["z"][:, 0]          # (B,24,3)
    z_scale = out["z_scale"][:, 0]    # (B,24,3)
    z_depth = out["z_depth"][:, 0]    # (B,24,3)
    obj_on  = out["obj_on"][:, 0]     # (B,24,1) or (B,24)
    z_feat  = out["z_features"][:, 0] # (B,24,F)

    if obj_on.dim() == 2:
        obj_on = obj_on.unsqueeze(-1) # (B,24,1)

    toks = torch.cat([z, z_scale, z_depth, obj_on, z_feat], dim=-1)  # (B,24, 3+3+3+1+F)
    return toks


# ----------------------------
# Bounds mode handling (matching VoxelizedDataset)
# ----------------------------
def parse_fixed_bounds_pm(bounds_str: str):
    # expects "xmin,xmax,ymin,ymax,zmin,zmax"
    xmin, xmax, ymin, ymax, zmin, zmax = map(float, bounds_str.split(","))
    pmin = (xmin, ymin, zmin)
    pmax = (xmax, ymax, zmax)

    print("pmin: ", pmin)
    print("pmax: ", pmax)
    return (pmin, pmax)


def compute_global_bounds_pm(
    ply_map, demos, *,
    include_rgb, normalize_to_unit_cube_flag, max_points,
    max_frames=None,
):
    """
    Mimics VoxelizedDataset._compute_global_bounds() but over (demo,t) PLY items.
    IMPORTANT: bounds are computed on the SAME preprocessed points used later.
    """
    pmin_g = None
    pmax_g = None
    total = 0

    for demo in demos:
        ts = sorted(ply_map[demo].keys())
        for t in ts:
            if max_frames is not None and total >= max_frames:
                break
            pts_t = preprocess_ply(
                ply_map[demo][t],
                max_points=max_points,
                normalize_to_unit_cube_flag=normalize_to_unit_cube_flag,
                include_rgb=include_rgb,
            )
            xyz = pts_t[:, :3]
            if xyz.shape[0] == 0:
                raise RuntimeError(f"Empty point cloud after preprocessing: {ply_map[demo][t]}")

            pmin = xyz.amin(dim=0)
            pmax = xyz.amax(dim=0)

            if pmin_g is None:
                pmin_g = pmin.clone()
                pmax_g = pmax.clone()
            else:
                pmin_g = torch.minimum(pmin_g, pmin)
                pmax_g = torch.maximum(pmax_g, pmax)

            total += 1

        if max_frames is not None and total >= max_frames:
            break

    if pmin_g is None:
        raise RuntimeError("Global bounds scan found no frames.")

    return (tuple(pmin_g.cpu().tolist()), tuple(pmax_g.cpu().tolist()))


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--h5", required=True)
    ap.add_argument("--ply-dir", required=True)
    ap.add_argument("--ply-pattern", default=r"frame(?P<t>\d+)_fused_envcalib\.ply")
    ap.add_argument("--max-demos", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=None)

    # PC preprocessing (TODataset knobs)
    ap.add_argument("--max-points", type=int, default=4096)
    ap.add_argument("--normalize-to-unit-cube", action="store_true", default=True,
                    help="Normalize point cloud to unit cube (default: True, matching preprocessing pipeline)")
    ap.add_argument("--no-normalize", dest="normalize_to_unit_cube", action="store_false",
                    help="Disable unit cube normalization")
    ap.add_argument("--include-rgb", action="store_true", default=True,
                    help="Include RGB in point cloud (default: True)")
    ap.add_argument("--no-rgb", dest="include_rgb", action="store_false",
                    help="Exclude RGB from point cloud")

    # voxelization (VoxelizedDataset knobs)
    ap.add_argument("--grid-dhw", default="64,64,64")
    ap.add_argument("--voxel-mode", default="avg_rgb", choices=["occupancy","density","moments","avg_rgb"])
    ap.add_argument("--bounds-mode", default="per_item", choices=["per_item","global","fixed"])
    ap.add_argument("--fixed-bounds", default="-1,1,-1,1,-1,1")  # only used when bounds-mode=fixed

    # DLP
    ap.add_argument("--dlp-cfg", required=True)
    ap.add_argument("--dlp-ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8)

    # action mode
    ap.add_argument("--action-mode", default="absolute", choices=["absolute", "relative"],
                    help="'absolute': convert to absolute pose (default), 'relative': use raw delta actions from H5")

    # output
    ap.add_argument("--out-pkl", required=True)

    args = ap.parse_args()

    D, H, W = map(int, args.grid_dhw.split(","))
    device = torch.device(args.device if ("cuda" in args.device and torch.cuda.is_available()) else "cpu")

    # Build PLY index - supports two structures:
    # 1. Nested: demo_X/frameYYYYYY_fused_envcalib.ply
    # 2. Flat:   demo_X_frameYYYYYY_fused.ply
    ply_map = {}
    frame_pat = re.compile(args.ply_pattern)

    # First try nested structure (demo_* subdirectories)
    nested_found = False
    for demo_dir in sorted(os.listdir(args.ply_dir)):
        demo_path = os.path.join(args.ply_dir, demo_dir)
        if not os.path.isdir(demo_path):
            continue
        if not demo_dir.startswith("demo_"):
            continue

        demo = demo_dir  # e.g., "demo_0"

        # Scan PLY files in this demo directory
        for fn in os.listdir(demo_path):
            m = frame_pat.match(fn)
            if not m:
                continue
            t = int(m.group("t"))
            ply_map.setdefault(demo, {})[t] = os.path.join(demo_path, fn)
            nested_found = True

    # If no nested structure found, try flat structure (demo_X_frameYYYYYY_fused.ply)
    if not nested_found:
        print("[ply] No nested structure found, trying flat file naming...")
        flat_pat = re.compile(r"(?P<demo>demo_\d+)_frame(?P<t>\d+)_fused\.ply")

        for fn in sorted(os.listdir(args.ply_dir)):
            if not fn.endswith(".ply"):
                continue
            m = flat_pat.match(fn)
            if not m:
                continue
            demo = m.group("demo")  # e.g., "demo_0"
            t = int(m.group("t"))
            ply_map.setdefault(demo, {})[t] = os.path.join(args.ply_dir, fn)

    demos = sorted(ply_map.keys(), key=lambda s: int(s.split("_")[-1]))
    if args.max_demos is not None:
        demos = demos[:args.max_demos]
    if len(demos) == 0:
        raise RuntimeError(
            f"No demos found in ply-dir with given ply-pattern.\n"
            f"  Tried nested structure: {args.ply_dir}/demo_X/frame000000_fused_envcalib.ply\n"
            f"  Tried flat structure:   {args.ply_dir}/demo_X_frame000000_fused.ply\n"
            f"  Pattern used: {args.ply_pattern}"
        )

    print(f"[ply] demos found: {len(demos)}")
    if demos:
        sample_demo = demos[0]
        sample_frames = sorted(ply_map[sample_demo].keys())
        print(f"[ply] sample: {sample_demo} has {len(sample_frames)} frames (t={sample_frames[0]}..{sample_frames[-1]})")

    # Bounds mode (match VoxelizedDataset behavior)
    bounds_pm = None
    if args.bounds_mode == "fixed":
        bounds_pm = parse_fixed_bounds_pm(args.fixed_bounds)
    elif args.bounds_mode == "global":
        print("[bounds] scanning global bounds over preprocessed points...")
        bounds_pm = compute_global_bounds_pm(
            ply_map, demos,
            include_rgb=True,
            normalize_to_unit_cube_flag=args.normalize_to_unit_cube,
            max_points=args.max_points,
            max_frames=args.max_frames,
        )


        print(f"[bounds] global pmin={bounds_pm[0]} pmax={bounds_pm[1]}")
    else:
        print("bounds None")
        bounds_pm = None  # per_item

    # Load DLP
    cfg = get_config(args.dlp_cfg)
    model = build_dlp_from_cfg(cfg, device)
    _ = load_checkpoint(args.dlp_ckpt, model, None, None, map_location=device)

    expected_c = int(cfg.get("ch", 3))

    # Per-demo episode building
    ep_obs, ep_act, ep_gripper, path_lengths = [], [], [], []
    total_written = 0

    # Action converter (only needed for absolute mode)
    action_converter = None
    if args.action_mode == "absolute":
        action_converter = RobomimicAbsoluteActionConverter(args.h5)
        print(f"[actions] Using ABSOLUTE action mode (converting from relative)")
    else:
        print(f"[actions] Using RELATIVE action mode (raw delta actions from H5)")

    init_states = []         # (E, state_dim)
    demo_indices = []        # original H5 demo idx for each kept episode



    with h5py.File(args.h5, "r") as h5:
        for demo in demos:
            act_key = f"data/{demo}/actions"
            if act_key not in h5:
                raise RuntimeError(f"Missing actions in H5 for {demo}: {act_key}")
            demo_idx = int(demo.split("_")[-1])

            # Get actions based on mode
            if args.action_mode == "absolute":
                actions = action_converter.convert_idx(demo_idx).astype(np.float32)
            else:
                # Relative mode: use raw actions directly from H5
                actions = np.asarray(h5[act_key], dtype=np.float32)


            T = actions.shape[0]


            # --- skip demo if PLY coverage doesn't match timesteps ---
            avail = ply_map.get(demo, {})
            if not avail:
                print(f"[warn] {demo}: no PLY frames found; skipping demo")
                continue

            missing = [tt for tt in range(T) if tt not in avail]
            if missing:
                print(f"[warn] {demo}: missing {len(missing)}/{T} PLY frames "
                    f"(first few: {missing[:10]}); skipping demo")
                continue

            # Collect initial state for this demo (needed for env.reset_to)
            states_keys = [
                f"data/{demo}/states/states",  # common robomimic layout
                f"data/{demo}/states",         # alternate layout
            ]
            states_key = None
            for k in states_keys:
                if k in h5:
                    states_key = k
                    break
            if states_key is None:
                raise RuntimeError(
                    f"Missing states for {demo}. Tried: {states_keys}. "
                    "You need this to use env.reset_to({'states': init_state})."
                )

            init_state = np.asarray(h5[states_key][0], dtype=np.float64)
            init_states.append(init_state)
            demo_indices.append(demo_idx)

            # Extract gripper state for this demo
            gripper_state_full = extract_gripper_state(h5, demo)  # [T_full, 10]

            obs_steps = []
            act_steps = []
            gripper_steps = []

            t0 = 0
            while t0 < T:
                if args.max_frames is not None and total_written >= args.max_frames:
                    break

                Bcap = min(args.batch, T - t0)
                if args.max_frames is not None:
                    Bcap = min(Bcap, args.max_frames - total_written)
                if Bcap <= 0:
                    break

                ts = list(range(t0, t0 + Bcap))
                t0 += Bcap

                # preprocess + voxelize each frame in batch
                vox_list = []
                valid_ts = []
                for tt in ts:
                    if tt not in ply_map.get(demo, {}):
                        raise RuntimeError(f"Missing PLY for {demo} t={tt} (pattern/index mismatch).")
                    ply_path = ply_map[demo][tt]

                    pts_t = preprocess_ply(
                        ply_path,
                        max_points=args.max_points,
                        normalize_to_unit_cube_flag=args.normalize_to_unit_cube,
                        include_rgb=True,
                    ).to(device)

                    vox_t = voxelize_points_via_wrapper(
                        pts_t,
                        grid_dhw=(D, H, W),
                        voxel_mode="avg_rgb",
                        bounds_pm=bounds_pm,
                    )  # [C,D,H,W] on device
                    vox_list.append(vox_t)
                    valid_ts.append(tt)

                vox = torch.stack(vox_list, dim=0)  # [B,C,D,H,W]
                if vox.shape[1] != expected_c:
                    raise RuntimeError(f"Channel mismatch: got C={vox.shape[1]} expected C={expected_c}")

                with torch.no_grad():
                    out = model(vox, deterministic=True, warmup=False, with_loss=False)
                    toks = pack_tokens_k24(out)  # [B,24,Dtok]

                toks_np = toks.detach().cpu().numpy().astype(np.float32)
                obs_steps.append(toks_np)
                act_steps.append(actions[valid_ts].astype(np.float32))
                gripper_steps.append(gripper_state_full[valid_ts].astype(np.float32))

                total_written += toks_np.shape[0]

            if len(obs_steps) == 0:
                raise RuntimeError(f"No frames written for demo {demo}.")

            obs_ep = np.concatenate(obs_steps, axis=0)  # [L,K,Dtok]
            act_ep = np.concatenate(act_steps, axis=0)  # [L,A]
            gripper_ep = np.concatenate(gripper_steps, axis=0)  # [L, 10]
            L = obs_ep.shape[0]

            ep_obs.append(obs_ep)
            ep_act.append(act_ep)
            ep_gripper.append(gripper_ep)
            path_lengths.append(L)

            print(f"[demo] {demo}: wrote {L} frames (TOTAL={total_written})")

            if args.max_frames is not None and total_written >= args.max_frames:
                break

    # Pack into EC-Diffuser style buffer dict
    E = len(ep_obs)
    Tmax = int(max(path_lengths))
    K = int(ep_obs[0].shape[1])
    Dtok = int(ep_obs[0].shape[2])
    A = int(ep_act[0].shape[1])
    G = int(ep_gripper[0].shape[1])  # gripper_dim = 10

    observations  = np.zeros((E, Tmax, K, Dtok), dtype=np.float32)
    actions       = np.zeros((E, Tmax, A),       dtype=np.float32)
    gripper_state = np.zeros((E, Tmax, G),       dtype=np.float32)
    rewards       = np.zeros((E, Tmax, 1),       dtype=np.float32)
    terminals     = np.zeros((E, Tmax, 1),       dtype=np.float32)
    timeouts      = np.zeros((E, Tmax, 1),       dtype=np.float32)
    goals         = np.zeros((E, Tmax, K, Dtok), dtype=np.float32)

    for e in range(E):
        L = int(path_lengths[e])
        observations[e, :L]  = ep_obs[e]
        actions[e, :L]       = ep_act[e]
        gripper_state[e, :L] = ep_gripper[e]

        g = ep_obs[e][L - 1]          # [K,Dtok]
        goals[e, :L] = g[None, :, :]

        terminals[e, L - 1, 0] = 1.0

        if L < Tmax:
            observations[e, L:]  = observations[e, L - 1:L]
            goals[e, L:]         = goals[e, L - 1:L]
            actions[e, L:]       = 0.0
            gripper_state[e, L:] = gripper_state[e, L - 1:L]  # repeat last gripper state
            timeouts[e, L:, 0]   = 1.0

    path_lengths = np.asarray(path_lengths, dtype=np.int32)
    info_goals_reached = np.ones((E,), dtype=np.float32)
    info_goal_success_frac = np.ones((E,), dtype=np.float32)

    paths_dict = {
        "meta": {
            "K": int(K),
            "Dtok": int(Dtok),
            "A": int(A),
            "G": int(G),  # gripper_dim
            "Tmax": int(Tmax),
            "repr": "DLP_tokens_k24",
            "voxel_mode": args.voxel_mode,
            "bounds_mode": args.bounds_mode,
            "action_mode": args.action_mode,  # "absolute" or "relative"
            "demo_indices": np.asarray(demo_indices, dtype=np.int32),
            "gripper_state_format": "pos(3)_rot6d(6)_open(1)",  # document the format
        },
        "observations": observations,
        "actions": actions,
        "gripper_state": gripper_state,  # [E, Tmax, 10] - eef_pos(3) + rot6d(6) + gripper_open(1)
        "rewards": rewards,
        "terminals": terminals,
        "timeouts": timeouts,
        "goals": goals,
        "path_lengths": path_lengths,
        "info_goals_reached": info_goals_reached,
        "info_goal_success_frac": info_goal_success_frac,
        "init_states": np.asarray(init_states, dtype=np.float64),
    }

    os.makedirs(os.path.dirname(args.out_pkl) or ".", exist_ok=True)
    with open(args.out_pkl, "wb") as f:
        pickle.dump(paths_dict, f)

    print(f"\nWrote: {args.out_pkl}")
    print(f"E={E}, Tmax={Tmax}, K={K}, Dtok={Dtok}, A={A}, G={G} (gripper_dim), action_mode={args.action_mode}")
    print(f"gripper_state shape: {gripper_state.shape} (format: pos(3) + rot6d(6) + gripper_open(1))")


if __name__ == "__main__":
    main()
