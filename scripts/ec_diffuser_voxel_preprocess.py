#!/usr/bin/env python3
"""
Preprocess pre-voxelized MimicGen demos for EC-Diffuser.

Reads from voxel cache structure:
  {voxel_cache_dir}/demo_0/frame0_voxels.pt
  {voxel_cache_dir}/demo_0/frame0_meta.pt
  {voxel_cache_dir}/demo_0/frame0_extras.pt (optional)
  ...

Runs DLP encoder to get tokens, then saves EC-Diffuser compatible pickle.

Debug mode (--debug): Process one frame and visualize GT vs reconstructed in wandb.
"""
import os
import re
import argparse
import pickle
import numpy as np
import h5py
import torch

import sys
import numpy.core as _core
sys.modules['numpy._core'] = _core
sys.modules['numpy._core.multiarray'] = _core.multiarray

from voxel_models import DLP
from utils.util_func import get_config
from utils.log_utils import load_checkpoint


# ----------------------------
# Gripper state extraction
# ----------------------------
def quat_to_rot6d(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (x,y,z,w) to 6D rotation representation.
    6D = first two columns of rotation matrix, flattened.
    """
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    # Rotation matrix from quaternion
    r00 = 1 - 2*(y*y + z*z)
    r10 = 2*(x*y + w*z)
    r20 = 2*(x*z - w*y)
    r01 = 2*(x*y - w*z)
    r11 = 1 - 2*(x*x + z*z)
    r21 = 2*(y*z + w*x)

    rot6d = np.stack([r00, r10, r20, r01, r11, r21], axis=-1)
    return rot6d.astype(np.float32)


def extract_gripper_state(h5, demo: str) -> np.ndarray:
    """
    Extract gripper state from H5 file for a demo.
    Gripper state format: [pos(3), rot_6d(6), gripper_open(1)] = 10 dims
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
    eef_pos = np.asarray(obs_group[eef_pos_key], dtype=np.float32)

    # End-effector quaternion -> 6D rotation
    eef_quat_key = None
    for key in ["robot0_eef_quat", "eef_quat"]:
        if key in obs_group:
            eef_quat_key = key
            break
    if eef_quat_key is None:
        raise RuntimeError(f"Cannot find eef_quat in {demo}/obs. Available keys: {list(obs_group.keys())}")
    eef_quat = np.asarray(obs_group[eef_quat_key], dtype=np.float32)
    eef_rot6d = quat_to_rot6d(eef_quat)

    # Gripper state
    gripper_key = None
    for key in ["robot0_gripper_qpos", "gripper_qpos"]:
        if key in obs_group:
            gripper_key = key
            break

    if gripper_key is not None:
        gripper_qpos = np.asarray(obs_group[gripper_key], dtype=np.float32)
        gripper_open = gripper_qpos.mean(axis=-1, keepdims=True)
        gripper_open = (gripper_open - 0.02) / 0.02
    else:
        print(f"[warn] No gripper_qpos found for {demo}, using zeros")
        gripper_open = np.zeros((eef_pos.shape[0], 1), dtype=np.float32)

    gripper_state = np.concatenate([eef_pos, eef_rot6d, gripper_open], axis=-1)
    return gripper_state


# ----------------------------
# DLP model building
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
    """Pack DLP output into K=24 tokens."""
    z       = out["z"][:, 0]          # (B,24,3)
    z_scale = out["z_scale"][:, 0]    # (B,24,3)
    z_depth = out["z_depth"][:, 0]    # (B,24,3)
    obj_on  = out["obj_on"][:, 0]     # (B,24,1) or (B,24)
    z_feat  = out["z_features"][:, 0] # (B,24,F)

    if obj_on.dim() == 2:
        obj_on = obj_on.unsqueeze(-1)

    toks = torch.cat([z, z_scale, z_depth, obj_on, z_feat], dim=-1)
    return toks


# ----------------------------
# Debug visualization
# ----------------------------
def run_debug_mode(model, voxel_path: str, device: torch.device, wandb_project: str = "ec-diffuser-debug"):
    """
    Run a single frame through encode->decode and visualize in wandb.

    Shows:
    - Ground truth voxel grid (input)
    - Reconstructed voxel grid (from DLP decode)
    - Keypoint locations
    """
    import wandb

    # Initialize wandb
    wandb.init(project=wandb_project, name=f"debug_{os.path.basename(voxel_path)}")

    # Load voxel
    print(f"[debug] Loading voxel: {voxel_path}")
    vox_gt = load_voxel(voxel_path, device)  # [C, D, H, W]
    vox_input = vox_gt.unsqueeze(0)  # [1, C, D, H, W]

    print(f"[debug] Input shape: {vox_input.shape}")

    # Run full forward pass with reconstruction
    with torch.no_grad():
        out = model(vox_input, deterministic=True, warmup=False, with_loss=True)

    # Get reconstruction
    vox_rec = out.get("rec", out.get("reconstruction", None))
    if vox_rec is None:
        # Try to decode from latents
        print("[debug] No 'rec' in output, attempting manual decode...")
        vox_rec = model.decode(out)

    vox_rec = vox_rec[0]  # [C, D, H, W]
    vox_gt = vox_gt  # [C, D, H, W]

    print(f"[debug] GT shape: {vox_gt.shape}, Rec shape: {vox_rec.shape}")

    # Get keypoint info
    z = out["z"][0, 0]  # [K, 3] - keypoint positions
    z_scale = out["z_scale"][0, 0]  # [K, 3]
    obj_on = out["obj_on"][0, 0]  # [K] or [K, 1]
    if obj_on.dim() == 2:
        obj_on = obj_on.squeeze(-1)

    K = z.shape[0]
    print(f"[debug] Keypoints: K={K}")
    print(f"[debug] obj_on stats: min={obj_on.min():.3f}, max={obj_on.max():.3f}, mean={obj_on.mean():.3f}")

    # Prepare visualizations
    # For voxels, we'll create slice visualizations (middle slices along each axis)
    C, D, H, W = vox_gt.shape

    # Convert to numpy for visualization (workaround for PyTorch without NumPy support)
    def to_numpy(t):
        """Convert tensor to numpy, handling PyTorch builds without NumPy support."""
        try:
            return t.cpu().numpy()
        except RuntimeError:
            # Fallback: convert via Python list
            return np.array(t.cpu().tolist())

    vox_gt_np = to_numpy(vox_gt)
    vox_rec_np = to_numpy(vox_rec)
    z_np = to_numpy(z)
    obj_on_np = to_numpy(obj_on)

    # Create occupancy masks (sum across channels for RGB, or just use channel 0)
    if C == 1:
        occ_gt = vox_gt_np[0]  # [D, H, W]
        occ_rec = vox_rec_np[0]
    else:
        # For RGB voxels, compute occupancy as any non-zero
        occ_gt = np.abs(vox_gt_np).sum(axis=0) > 0.01  # [D, H, W]
        occ_rec = np.abs(vox_rec_np).sum(axis=0) > 0.01

    # Log metrics
    wandb.log({
        "input_channels": C,
        "grid_size": D,
        "num_keypoints": K,
        "obj_on_mean": float(obj_on.mean()),
        "obj_on_max": float(obj_on.max()),
        "gt_occupancy_ratio": float(occ_gt.sum() / occ_gt.size),
        "rec_occupancy_ratio": float(occ_rec.sum() / occ_rec.size),
    })

    # Create slice images (XY, XZ, YZ planes through center)
    mid_d, mid_h, mid_w = D // 2, H // 2, W // 2

    # XY slice (looking down Z axis)
    slice_xy_gt = occ_gt[mid_d, :, :]
    slice_xy_rec = occ_rec[mid_d, :, :]

    # XZ slice (looking down Y axis)
    slice_xz_gt = occ_gt[:, mid_h, :]
    slice_xz_rec = occ_rec[:, mid_h, :]

    # YZ slice (looking down X axis)
    slice_yz_gt = occ_gt[:, :, mid_w]
    slice_yz_rec = occ_rec[:, :, mid_w]

    # Log slice images
    wandb.log({
        "slices/xy_gt": wandb.Image(slice_xy_gt.astype(np.float32), caption="GT XY slice"),
        "slices/xy_rec": wandb.Image(slice_xy_rec.astype(np.float32), caption="Rec XY slice"),
        "slices/xz_gt": wandb.Image(slice_xz_gt.astype(np.float32), caption="GT XZ slice"),
        "slices/xz_rec": wandb.Image(slice_xz_rec.astype(np.float32), caption="Rec XZ slice"),
        "slices/yz_gt": wandb.Image(slice_yz_gt.astype(np.float32), caption="GT YZ slice"),
        "slices/yz_rec": wandb.Image(slice_yz_rec.astype(np.float32), caption="Rec YZ slice"),
    })

    # If RGB voxels, also log RGB slices
    if C >= 3:
        # RGB XY slice
        rgb_xy_gt = vox_gt_np[:3, mid_d, :, :].transpose(1, 2, 0)  # [H, W, 3]
        rgb_xy_rec = vox_rec_np[:3, mid_d, :, :].transpose(1, 2, 0)

        # Normalize to [0, 1] for visualization
        rgb_xy_gt = np.clip(rgb_xy_gt, 0, 1)
        rgb_xy_rec = np.clip(rgb_xy_rec, 0, 1)

        wandb.log({
            "rgb_slices/xy_gt": wandb.Image(rgb_xy_gt, caption="GT RGB XY slice"),
            "rgb_slices/xy_rec": wandb.Image(rgb_xy_rec, caption="Rec RGB XY slice"),
        })

    # Create 3D point cloud visualization of keypoints
    # Scale keypoints from [-1, 1] to voxel grid coordinates
    kp_voxel = (z_np + 1) / 2 * np.array([W, H, D])  # [K, 3] in voxel coords

    # Create point cloud for wandb (active keypoints only)
    active_mask = obj_on_np > 0.5
    kp_active = kp_voxel[active_mask]

    if len(kp_active) > 0:
        # Create colored point cloud based on obj_on values
        colors = np.zeros((len(kp_active), 3))
        colors[:, 0] = 1.0  # Red for keypoints

        wandb.log({
            "keypoints/positions": wandb.Table(
                columns=["x", "y", "z", "obj_on"],
                data=[[float(z_np[i, 0]), float(z_np[i, 1]), float(z_np[i, 2]), float(obj_on_np[i])]
                      for i in range(K)]
            ),
            "keypoints/num_active": int(active_mask.sum()),
        })

    # Log voxel grids as 3D objects if possible
    try:
        # Get occupied voxel coordinates
        gt_coords = np.argwhere(occ_gt > 0.5)
        rec_coords = np.argwhere(occ_rec > 0.5)

        if len(gt_coords) > 0:
            wandb.log({
                "point_clouds/gt_voxels": wandb.Object3D(gt_coords.astype(np.float32)),
            })
        if len(rec_coords) > 0:
            wandb.log({
                "point_clouds/rec_voxels": wandb.Object3D(rec_coords.astype(np.float32)),
            })
        if len(kp_active) > 0:
            wandb.log({
                "point_clouds/keypoints": wandb.Object3D(kp_active.astype(np.float32)),
            })
    except Exception as e:
        print(f"[debug] Could not log 3D objects: {e}")

    # Compute reconstruction error
    mse = float(((vox_gt - vox_rec) ** 2).mean())
    mae = float((vox_gt - vox_rec).abs().mean())

    # IoU for occupancy
    intersection = ((occ_gt > 0.5) & (occ_rec > 0.5)).sum()
    union = ((occ_gt > 0.5) | (occ_rec > 0.5)).sum()
    iou = float(intersection / (union + 1e-8))

    wandb.log({
        "metrics/mse": mse,
        "metrics/mae": mae,
        "metrics/iou": iou,
    })

    print(f"[debug] Reconstruction metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  IoU: {iou:.4f}")
    print(f"  Active keypoints: {int(active_mask.sum())}/{K}")

    # Log tokens
    toks = pack_tokens_k24(out)
    print(f"[debug] Token shape: {toks.shape}")
    wandb.log({"tokens/shape": str(list(toks.shape))})

    wandb.finish()
    print(f"[debug] Done! Check wandb project '{wandb_project}' for visualizations.")


# ----------------------------
# Voxel cache loading
# ----------------------------
def scan_voxel_cache(voxel_cache_dir: str):
    """
    Scan voxel cache directory for demos and frames.

    Expected structure:
        {voxel_cache_dir}/demo_0/frame0_voxels.pt
        {voxel_cache_dir}/demo_0/frame1_voxels.pt
        ...
        {voxel_cache_dir}/demo_1/frame0_voxels.pt
        ...

    Returns:
        voxel_map: {demo_name: {frame_idx: voxel_path}}
    """
    voxel_map = {}
    frame_pat = re.compile(r"frame(\d+)_voxels\.pt")

    for demo_dir in sorted(os.listdir(voxel_cache_dir)):
        demo_path = os.path.join(voxel_cache_dir, demo_dir)
        if not os.path.isdir(demo_path):
            continue
        if not demo_dir.startswith("demo_"):
            continue

        demo = demo_dir
        for fn in os.listdir(demo_path):
            m = frame_pat.match(fn)
            if not m:
                continue
            frame_idx = int(m.group(1))
            voxel_map.setdefault(demo, {})[frame_idx] = os.path.join(demo_path, fn)

    return voxel_map


def load_voxel(voxel_path: str, device: torch.device) -> torch.Tensor:
    """Load a voxel tensor from .pt file."""
    vox = torch.load(voxel_path, map_location=device)
    if not isinstance(vox, torch.Tensor):
        raise RuntimeError(f"Expected tensor in {voxel_path}, got {type(vox)}")
    return vox


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Preprocess pre-voxelized MimicGen demos for EC-Diffuser"
    )

    # Data paths
    ap.add_argument("--h5", required=True,
                    help="Path to MimicGen H5 file (for actions and gripper state)")
    ap.add_argument("--voxel-cache-dir", required=True,
                    help="Path to voxel cache directory (e.g., .../voxel_cache_new)")
    ap.add_argument("--max-demos", type=int, default=None,
                    help="Limit number of demos to process")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Limit total frames to process")

    # DLP model
    ap.add_argument("--dlp-cfg", required=True,
                    help="Path to DLP config JSON")
    ap.add_argument("--dlp-ckpt", required=True,
                    help="Path to DLP checkpoint")
    ap.add_argument("--device", default="cuda",
                    help="Device to run DLP on")
    ap.add_argument("--batch", type=int, default=8,
                    help="Batch size for DLP inference")

    # Action mode
    ap.add_argument("--action-mode", default="absolute", choices=["absolute", "relative"],
                    help="'absolute': convert to absolute pose, 'relative': use raw delta actions")

    # Output
    ap.add_argument("--out-pkl", required=True,
                    help="Output pickle file path")

    # Debug mode
    ap.add_argument("--debug", action="store_true",
                    help="Debug mode: process one frame and visualize GT vs reconstructed in wandb")
    ap.add_argument("--debug-demo", type=int, default=0,
                    help="Demo index to use for debug mode (default: 0)")
    ap.add_argument("--debug-frame", type=int, default=0,
                    help="Frame index to use for debug mode (default: 0)")
    ap.add_argument("--wandb-project", type=str, default="ec-diffuser-debug",
                    help="Wandb project name for debug visualization")

    args = ap.parse_args()

    device = torch.device(args.device if ("cuda" in args.device and torch.cuda.is_available()) else "cpu")

    # Scan voxel cache
    print(f"[voxel] Scanning voxel cache: {args.voxel_cache_dir}")
    voxel_map = scan_voxel_cache(args.voxel_cache_dir)

    demos = sorted(voxel_map.keys(), key=lambda s: int(s.split("_")[-1]))
    if args.max_demos is not None:
        demos = demos[:args.max_demos]

    if len(demos) == 0:
        raise RuntimeError(f"No demos found in voxel cache: {args.voxel_cache_dir}")

    print(f"[voxel] Found {len(demos)} demos")
    if demos:
        sample_demo = demos[0]
        sample_frames = sorted(voxel_map[sample_demo].keys())
        print(f"[voxel] Sample: {sample_demo} has {len(sample_frames)} frames (t={sample_frames[0]}..{sample_frames[-1]})")

    # Load DLP model
    cfg = get_config(args.dlp_cfg)
    model = build_dlp_from_cfg(cfg, device)
    _ = load_checkpoint(args.dlp_ckpt, model, None, None, map_location=device)
    expected_c = int(cfg.get("ch", 3))
    print(f"[dlp] Loaded model, expected channels: {expected_c}")

    # Debug mode: process one frame and exit
    if args.debug:
        demo_name = f"demo_{args.debug_demo}"
        if demo_name not in voxel_map:
            available = list(voxel_map.keys())[:5]
            raise RuntimeError(f"Demo '{demo_name}' not found. Available: {available}...")

        if args.debug_frame not in voxel_map[demo_name]:
            available = sorted(voxel_map[demo_name].keys())[:10]
            raise RuntimeError(f"Frame {args.debug_frame} not found in {demo_name}. Available: {available}...")

        voxel_path = voxel_map[demo_name][args.debug_frame]
        print(f"[debug] Running debug mode on: {voxel_path}")

        run_debug_mode(model, voxel_path, device, wandb_project=args.wandb_project)
        return  # Exit after debug

    # Action converter (only for absolute mode)
    action_converter = None
    if args.action_mode == "absolute":
        from robomimic_converter import RobomimicAbsoluteActionConverter
        action_converter = RobomimicAbsoluteActionConverter(args.h5)
        print(f"[actions] Using ABSOLUTE action mode")
    else:
        print(f"[actions] Using RELATIVE action mode")

    # Process demos
    ep_obs, ep_act, ep_gripper, path_lengths = [], [], [], []
    init_states = []
    demo_indices = []
    total_written = 0

    with h5py.File(args.h5, "r") as h5:
        for demo in demos:
            act_key = f"data/{demo}/actions"
            if act_key not in h5:
                print(f"[warn] {demo}: missing actions in H5, skipping")
                continue

            demo_idx = int(demo.split("_")[-1])

            # Get actions
            if args.action_mode == "absolute":
                actions = action_converter.convert_idx(demo_idx).astype(np.float32)
            else:
                actions = np.asarray(h5[act_key], dtype=np.float32)

            T = actions.shape[0]

            # Check voxel coverage
            avail = voxel_map.get(demo, {})
            if not avail:
                print(f"[warn] {demo}: no voxel frames found, skipping")
                continue

            missing = [tt for tt in range(T) if tt not in avail]
            if missing:
                print(f"[warn] {demo}: missing {len(missing)}/{T} voxel frames, skipping")
                continue

            # Get initial state
            states_keys = [f"data/{demo}/states/states", f"data/{demo}/states"]
            states_key = None
            for k in states_keys:
                if k in h5:
                    states_key = k
                    break
            if states_key is None:
                raise RuntimeError(f"Missing states for {demo}")

            init_state = np.asarray(h5[states_key][0], dtype=np.float64)
            init_states.append(init_state)
            demo_indices.append(demo_idx)

            # Extract gripper state
            gripper_state_full = extract_gripper_state(h5, demo)

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

                # Load voxels for this batch
                vox_list = []
                valid_ts = []
                for tt in ts:
                    voxel_path = voxel_map[demo][tt]
                    vox_t = load_voxel(voxel_path, device)
                    vox_list.append(vox_t)
                    valid_ts.append(tt)

                vox = torch.stack(vox_list, dim=0)  # [B,C,D,H,W]
                if vox.shape[1] != expected_c:
                    raise RuntimeError(f"Channel mismatch: got C={vox.shape[1]} expected C={expected_c}")

                # Run DLP encoder
                with torch.no_grad():
                    out = model(vox, deterministic=True, warmup=False, with_loss=False)
                    toks = pack_tokens_k24(out)

                toks_np = toks.detach().cpu().numpy().astype(np.float32)
                obs_steps.append(toks_np)
                act_steps.append(actions[valid_ts].astype(np.float32))
                gripper_steps.append(gripper_state_full[valid_ts].astype(np.float32))

                total_written += toks_np.shape[0]

            if len(obs_steps) == 0:
                raise RuntimeError(f"No frames written for {demo}")

            obs_ep = np.concatenate(obs_steps, axis=0)
            act_ep = np.concatenate(act_steps, axis=0)
            gripper_ep = np.concatenate(gripper_steps, axis=0)
            L = obs_ep.shape[0]

            ep_obs.append(obs_ep)
            ep_act.append(act_ep)
            ep_gripper.append(gripper_ep)
            path_lengths.append(L)

            print(f"[demo] {demo}: {L} frames (total={total_written})")

            if args.max_frames is not None and total_written >= args.max_frames:
                break

    # Pack into EC-Diffuser format
    E = len(ep_obs)
    Tmax = int(max(path_lengths))
    K = int(ep_obs[0].shape[1])
    Dtok = int(ep_obs[0].shape[2])
    A = int(ep_act[0].shape[1])
    G = int(ep_gripper[0].shape[1])

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

        g = ep_obs[e][L - 1]
        goals[e, :L] = g[None, :, :]

        terminals[e, L - 1, 0] = 1.0

        if L < Tmax:
            observations[e, L:]  = observations[e, L - 1:L]
            goals[e, L:]         = goals[e, L - 1:L]
            actions[e, L:]       = 0.0
            gripper_state[e, L:] = gripper_state[e, L - 1:L]
            timeouts[e, L:, 0]   = 1.0

    path_lengths = np.asarray(path_lengths, dtype=np.int32)
    info_goals_reached = np.ones((E,), dtype=np.float32)
    info_goal_success_frac = np.ones((E,), dtype=np.float32)

    paths_dict = {
        "meta": {
            "K": int(K),
            "Dtok": int(Dtok),
            "A": int(A),
            "G": int(G),
            "Tmax": int(Tmax),
            "repr": "DLP_tokens_k24",
            "action_mode": args.action_mode,
            "demo_indices": np.asarray(demo_indices, dtype=np.int32),
            "gripper_state_format": "pos(3)_rot6d(6)_open(1)",
            "source": "voxel_cache",
            "voxel_cache_dir": args.voxel_cache_dir,
        },
        "observations": observations,
        "actions": actions,
        "gripper_state": gripper_state,
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
    print(f"E={E}, Tmax={Tmax}, K={K}, Dtok={Dtok}, A={A}, G={G}, action_mode={args.action_mode}")


if __name__ == "__main__":
    main()
