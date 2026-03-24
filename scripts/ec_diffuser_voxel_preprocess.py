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
    print("n enc kp: ", cfg["n_kp_enc"])
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


def pack_tokens_k24(out: dict) -> tuple:
    """
    Pack DLP output into K tokens (foreground particles) and separate bg_features.

    Token format per particle: [z(3), z_scale(3), z_depth(1), obj_on(1), z_features(F)]
    Background features are returned separately.

    Returns:
        toks: (B, K, Dtok) - foreground particle tokens
        bg_features: (B, bg_dim) - background features
    """
    z       = out["z"][:, 0]          # (B, K, 3)
    z_scale = out["z_scale"][:, 0]    # (B, K, 3)
    z_depth = out["z_depth"][:, 0]    # (B, K, 1) or (B, K, 3)
    obj_on  = out["obj_on"][:, 0]     # (B, K, 1) or (B, K)
    z_feat  = out["z_features"][:, 0] # (B, K, F)
    z_bg    = out["z_bg_features"][:, 0]  # (B, bg_dim)

    if obj_on.dim() == 2:
        obj_on = obj_on.unsqueeze(-1)

    # Only z_depth's first dim if it has 3 dims (some models output 3, we only need 1)
    if z_depth.shape[-1] == 3:
        z_depth = z_depth[..., :1]

    # Pack foreground tokens (do NOT include bg_features - they're separate)
    toks = torch.cat([z, z_scale, z_depth, obj_on, z_feat], dim=-1)

    return toks, z_bg


# ----------------------------
# Debug visualization
# ----------------------------
def _load_voxel_any(path: str) -> torch.Tensor:
    """
    Load a voxel grid from .pt — handles both compressed (sparse) and
    uncompressed (dense) formats.  Returns dense float32 [C, D, H, W].
    """
    data = torch.load(path, weights_only=False)
    if isinstance(data, dict) and data.get("compressed"):
        shape = data["shape"]
        coords = data["coords"].long()
        values = data["values"].float()
        vox = torch.zeros(shape, dtype=torch.float32)
        vox[:, coords[:, 0], coords[:, 1], coords[:, 2]] = values.T
        return vox
    if isinstance(data, torch.Tensor):
        return data.float()
    raise RuntimeError(f"Unexpected voxel format in {path}: {type(data)}")


def load_voxels_from_nested_cache(cache_dir: str, num_samples: int = 5):
    """
    Load voxels from nested cache structure: cache_dir/demo_X/frameY_voxels.pt
    Returns list of (vox_tensor, vox_path) tuples
    """
    import glob
    from tqdm import tqdm

    if not os.path.isdir(cache_dir):
        return []

    voxels = []

    # Find all demo directories
    demo_dirs = sorted(
        glob.glob(os.path.join(cache_dir, "demo_*")),
        key=lambda x: int(os.path.basename(x).split("_")[1])
    )

    # Collect all voxel paths first
    all_vox_paths = []
    for demo_dir in demo_dirs:
        vox_files = sorted(glob.glob(os.path.join(demo_dir, "frame*_voxels.pt")))
        all_vox_paths.extend(vox_files)
        if len(all_vox_paths) >= num_samples:
            all_vox_paths = all_vox_paths[:num_samples]
            break

    # Load with progress bar
    for vox_path in tqdm(all_vox_paths, desc="Loading voxels"):
        vox = _load_voxel_any(vox_path)
        voxels.append((vox, vox_path))

    return voxels


def run_debug_mode(model, voxel_cache_dir: str, device: torch.device, wandb_project: str = "ec-diffuser-debug", num_samples: int = 3, task_name: str = None):
    """
    Run samples through encode->decode and visualize in wandb.
    Uses cached kmeans (matching the production preprocessing path).
    Shows GT, reconstruction, and difference for each sample.
    """
    import wandb
    from eval.eval_vox import log_rgb_voxels
    from tqdm import tqdm

    # Load voxels from nested cache structure
    voxels = load_voxels_from_nested_cache(voxel_cache_dir, num_samples=num_samples)
    if not voxels:
        raise RuntimeError(f"No voxels found in {voxel_cache_dir}")

    print(f"[debug] Loaded {len(voxels)} voxels")

    # Resolve kmeans cache dir (sibling of voxel dir)
    kmeans_cache_dir = os.path.join(os.path.dirname(voxel_cache_dir), "kmeans_cache")

    # Initialize wandb (resumes existing run if WANDB_RUN_ID/WANDB_RESUME are set)
    prefix = f"{task_name}/" if task_name else ""
    wandb.init(project=wandb_project, name=f"debug-dlp-all-tasks")

    for i, (vox, vox_path) in enumerate(tqdm(voxels, desc=f"Processing {task_name or 'samples'}")):
        print(f"\n[debug] Sample {i}: {vox.shape} from {os.path.basename(vox_path)}")

        # Log GT voxel
        log_rgb_voxels(
            name=f"{prefix}samples/input_{i}",
            rgb_vol=vox,
            alpha_vol=None,
            KPx=None,
            step=i,
            mode="splat",
            topk=60000,
            alpha_thresh=0.05,
            pad=2.0,
            show_axes=True,
        )

        # Load cached kmeans for this frame (matches production path)
        # vox_path: .../voxel_cache/voxel/demo_X/frameY_voxels.pt
        # km_path:  .../voxel_cache/kmeans_cache/demo_X/frameY_kmeans.pt
        demo_name = os.path.basename(os.path.dirname(vox_path))
        frame_name = os.path.basename(vox_path).replace("_voxels.pt", "_kmeans.pt")
        km_path = os.path.join(kmeans_cache_dir, demo_name, frame_name)

        meta = None
        if os.path.exists(km_path):
            km = torch.load(km_path, map_location="cpu", weights_only=False)
            meta = {
                "kmeans_kp": km["kp"].unsqueeze(0).to(device),
                "kmeans_cov": km["cov"].unsqueeze(0).to(device),
            }
            print(f"[debug] Using cached kmeans from {km_path}")
        else:
            print(f"[debug] WARNING: no cached kmeans at {km_path}, recomputing")

        # Run through model (with cached kmeans, matching production path)
        vox_input = vox.unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(vox_input, deterministic=True, warmup=False, with_loss=True, meta=meta)

        # Get reconstruction
        vox_rec = out.get("rec", out.get("reconstruction", None))
        if vox_rec is None:
            vox_rec = model.decode(out)
        vox_rec = vox_rec[0].cpu()

        # Log reconstruction
        log_rgb_voxels(
            name=f"{prefix}samples/rec_{i}",
            rgb_vol=vox_rec,
            alpha_vol=None,
            KPx=None,
            step=i,
            mode="splat",
            topk=60000,
            alpha_thresh=0.05,
            pad=2.0,
            show_axes=True,
        )

        # Log difference
        diff = (vox - vox_rec).abs()
        diff_scaled = diff / (diff.max() + 1e-8)
        log_rgb_voxels(
            name=f"{prefix}samples/diff_{i}",
            rgb_vol=diff_scaled,
            alpha_vol=None,
            KPx=None,
            step=i,
            mode="splat",
            topk=60000,
            alpha_thresh=0.01,
            pad=2.0,
            show_axes=True,
        )

        # Metrics
        mse = float(((vox - vox_rec) ** 2).mean())
        print(f"[debug] Sample {i} MSE: {mse:.6f}")
        wandb.log({f"{prefix}metrics/mse_{i}": mse}, step=i)

    wandb.finish()
    print(f"\n[debug] Done! Check wandb project '{wandb_project}'")


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
    return _load_voxel_any(voxel_path).to(device)


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
                    help="Debug mode: visualize GT vs reconstructed voxels in wandb")
    ap.add_argument("--debug-samples", type=int, default=3,
                    help="Number of samples to visualize in debug mode (default: 3)")
    ap.add_argument("--wandb-project", type=str, default="ec-diffuser-debug",
                    help="Wandb project name for debug visualization")
    ap.add_argument("--task-name", type=str, default=None,
                    help="Task name prefix for wandb log keys (e.g. 'coffee_d2')")

    # Distributed processing
    ap.add_argument("--rank", type=int, default=0,
                    help="Process rank for distributed processing (0-indexed)")
    ap.add_argument("--world-size", type=int, default=1,
                    help="Total number of processes for distributed processing")

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

    # Distributed processing: split demos across ranks
    print(f"[DEBUG] rank={args.rank}, world_size={args.world_size}, total_demos={len(demos)}")
    if args.world_size > 1:
        all_demos = demos
        demos = [d for i, d in enumerate(all_demos) if i % args.world_size == args.rank]
        print(f"[dist] Rank {args.rank}/{args.world_size}: processing {len(demos)}/{len(all_demos)} demos")
        print(f"[dist] Demos for this rank: {demos[:5]}..." if len(demos) > 5 else f"[dist] Demos: {demos}")
    else:
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

    # Debug mode: process samples and exit
    if args.debug:
        print(f"[debug] Running debug mode on voxel cache: {args.voxel_cache_dir}")
        run_debug_mode(model, args.voxel_cache_dir, device, wandb_project=args.wandb_project, num_samples=args.debug_samples, task_name=args.task_name)
        return  # Exit after debug

    # Action converter (only for absolute mode)
    action_converter = None
    if args.action_mode == "absolute":
        from robomimic_converter import RobomimicAbsoluteActionConverter
        action_converter = RobomimicAbsoluteActionConverter(args.h5)
        print(f"[actions] Using ABSOLUTE action mode")
    else:
        print(f"[actions] Using RELATIVE action mode")

    # ---- Pass 1: Precompute kmeans prior for all frames ----
    kmeans_cache_dir = os.path.join(os.path.dirname(args.voxel_cache_dir), "kmeans_cache")
    os.makedirs(kmeans_cache_dir, exist_ok=True)
    kmeans_cache = {}  # {demo: {frame_idx: {"kp": tensor, "cov": tensor}}}

    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor
    import threading

    # Build work list: (demo, frame_idx, voxel_path, km_path), skip cached
    # Pre-scan kmeans dirs once per demo (one listdir instead of N stat calls)
    km_work = []
    km_cached = 0
    for demo in demos:
        kmeans_cache[demo] = {}
        demo_km_dir = os.path.join(kmeans_cache_dir, demo)
        try:
            cached_files = set(os.listdir(demo_km_dir))
        except FileNotFoundError:
            cached_files = set()
        for tt in sorted(voxel_map[demo].keys()):
            km_fname = f"frame{tt}_kmeans.pt"
            if km_fname in cached_files:
                km_cached += 1
            else:
                km_path = os.path.join(demo_km_dir, km_fname)
                km_work.append((demo, tt, voxel_map[demo][tt], km_path))

    print(f"[kmeans] {len(km_work)} frames to compute, {km_cached} already cached -> {kmeans_cache_dir}")

    if km_work:
        # Prefetch queue: background threads load+decompress voxels while GPU computes
        PREFETCH = 16
        prefetch_queue = []
        prefetch_lock = threading.Lock()

        def _load_one(item):
            demo, tt, vox_path, km_path = item
            vox = _load_voxel_any(vox_path)  # decompress on CPU
            return (demo, tt, vox, km_path)

        pbar = tqdm(total=len(km_work), desc="KMeans prior", unit="frame")
        km_computed = 0

        with ThreadPoolExecutor(max_workers=4) as io_pool:
            # Submit all I/O work upfront as futures
            futures = [io_pool.submit(_load_one, item) for item in km_work]

            for fut in futures:
                demo, tt, vox, km_path = fut.result()

                # Run kmeans on single frame (no batch overhead)
                with torch.no_grad():
                    vox_gpu = vox.unsqueeze(0).to(device)
                    kp, cov = model.prior_module.encode_prior(vox_gpu)

                # Save to disk
                km = {"kp": kp[0].cpu(), "cov": cov[0].cpu()}
                os.makedirs(os.path.dirname(km_path), exist_ok=True)
                torch.save(km, km_path)
                kmeans_cache[demo][tt] = km
                km_computed += 1
                pbar.update(1)

        pbar.close()
        print(f"[kmeans] Done: {km_computed} computed, {km_cached} from cache")

    # Load any cached kmeans that we skipped above (parallel I/O)
    km_load_items = []
    for demo in demos:
        demo_km_dir = os.path.join(kmeans_cache_dir, demo)
        try:
            cached_files = set(os.listdir(demo_km_dir))
        except FileNotFoundError:
            cached_files = set()
        for tt in sorted(voxel_map[demo].keys()):
            if tt not in kmeans_cache[demo]:
                km_fname = f"frame{tt}_kmeans.pt"
                if km_fname in cached_files:
                    km_load_items.append((demo, tt, os.path.join(demo_km_dir, km_fname)))

    if km_load_items:
        def _load_km(item):
            demo, tt, path = item
            return demo, tt, torch.load(path, map_location="cpu", weights_only=False)

        print(f"[kmeans] Loading {len(km_load_items)} cached kmeans files...")
        with ThreadPoolExecutor(max_workers=8) as io_pool:
            for demo, tt, km in io_pool.map(_load_km, km_load_items):
                kmeans_cache[demo][tt] = km

    # Process demos
    ep_obs, ep_act, ep_gripper, ep_bg_features, path_lengths = [], [], [], [], []
    init_states = []
    demo_indices = []
    total_written = 0

    with h5py.File(args.h5, "r") as h5:
        pbar = tqdm(demos, desc=f"Rank {args.rank}", unit="demo")
        for demo in pbar:
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
            bg_steps = []

            # Build batch schedule for this demo
            batches = []
            t0 = 0
            while t0 < T:
                if args.max_frames is not None and total_written + sum(len(b) for b in batches) >= args.max_frames:
                    break
                Bcap = min(args.batch, T - t0)
                if args.max_frames is not None:
                    Bcap = min(Bcap, args.max_frames - total_written - sum(len(b) for b in batches))
                if Bcap <= 0:
                    break
                batches.append(list(range(t0, t0 + Bcap)))
                t0 += Bcap

            # Prefetch voxels on CPU threads while GPU computes
            def _prefetch_batch(ts):
                voxels = []
                for tt in ts:
                    vox = _load_voxel_any(voxel_map[demo][tt])
                    voxels.append(vox)
                return torch.stack(voxels, dim=0)

            with ThreadPoolExecutor(max_workers=2) as io_pool:
                pending = None
                if batches:
                    pending = io_pool.submit(_prefetch_batch, batches[0])

                for bi, ts in enumerate(batches):
                    vox = pending.result().to(device)  # [B,C,D,H,W]

                    # Start loading next batch while GPU runs
                    if bi + 1 < len(batches):
                        pending = io_pool.submit(_prefetch_batch, batches[bi + 1])

                    if vox.shape[1] != expected_c:
                        raise RuntimeError(f"Channel mismatch: got C={vox.shape[1]} expected C={expected_c}")

                    # Run DLP encoder (uses cached kmeans if available)
                    with torch.no_grad():
                        meta = None
                        if kmeans_cache is not None:
                            km_list_kp, km_list_cov = [], []
                            for tt in ts:
                                km = kmeans_cache[demo][tt]
                                km_list_kp.append(km["kp"])
                                km_list_cov.append(km["cov"])
                            meta = {
                                "kmeans_kp": torch.stack(km_list_kp).to(device),
                                "kmeans_cov": torch.stack(km_list_cov).to(device),
                            }

                        out = model(vox, deterministic=True, warmup=False, with_loss=False, meta=meta)
                        toks, bg_feats = pack_tokens_k24(out)

                    toks_np = toks.detach().cpu().numpy().astype(np.float32)
                    bg_np = bg_feats.detach().cpu().numpy().astype(np.float32)
                    obs_steps.append(toks_np)
                    bg_steps.append(bg_np)
                    act_steps.append(actions[ts].astype(np.float32))
                    gripper_steps.append(gripper_state_full[ts].astype(np.float32))

                    total_written += toks_np.shape[0]

            if len(obs_steps) == 0:
                raise RuntimeError(f"No frames written for {demo}")

            obs_ep = np.concatenate(obs_steps, axis=0)
            act_ep = np.concatenate(act_steps, axis=0)
            gripper_ep = np.concatenate(gripper_steps, axis=0)
            bg_ep = np.concatenate(bg_steps, axis=0)
            L = obs_ep.shape[0]

            ep_obs.append(obs_ep)
            ep_act.append(act_ep)
            ep_gripper.append(gripper_ep)
            ep_bg_features.append(bg_ep)
            path_lengths.append(L)

            pbar.set_postfix(frames=L, total=total_written)

            if args.max_frames is not None and total_written >= args.max_frames:
                break

    # Pack into EC-Diffuser format
    E = len(ep_obs)
    Tmax = int(max(path_lengths))
    K = int(ep_obs[0].shape[1])
    Dtok = int(ep_obs[0].shape[2])
    A = int(ep_act[0].shape[1])
    G = int(ep_gripper[0].shape[1])
    BG = int(ep_bg_features[0].shape[1])  # background feature dimension

    observations  = np.zeros((E, Tmax, K, Dtok), dtype=np.float32)
    actions       = np.zeros((E, Tmax, A),       dtype=np.float32)
    gripper_state = np.zeros((E, Tmax, G),       dtype=np.float32)
    bg_features   = np.zeros((E, Tmax, BG),      dtype=np.float32)
    rewards       = np.zeros((E, Tmax, 1),       dtype=np.float32)
    terminals     = np.zeros((E, Tmax, 1),       dtype=np.float32)
    timeouts      = np.zeros((E, Tmax, 1),       dtype=np.float32)
    goals         = np.zeros((E, Tmax, K, Dtok), dtype=np.float32)

    for e in range(E):
        L = int(path_lengths[e])
        observations[e, :L]  = ep_obs[e]
        actions[e, :L]       = ep_act[e]
        gripper_state[e, :L] = ep_gripper[e]
        bg_features[e, :L]   = ep_bg_features[e]

        g = ep_obs[e][L - 1]
        goals[e, :L] = g[None, :, :]

        terminals[e, L - 1, 0] = 1.0

        if L < Tmax:
            observations[e, L:]  = observations[e, L - 1:L]
            goals[e, L:]         = goals[e, L - 1:L]
            actions[e, L:]       = 0.0
            gripper_state[e, L:] = gripper_state[e, L - 1:L]
            bg_features[e, L:]   = bg_features[e, L - 1:L]
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
            "BG": int(BG),
            "Tmax": int(Tmax),
            "repr": "DLP_tokens_k24",
            "action_mode": args.action_mode,
            "demo_indices": np.asarray(demo_indices, dtype=np.int32),
            "gripper_state_format": "pos(3)_rot6d(6)_open(1)",
            "bg_features_format": f"z_bg_features({BG})",
            "source": "voxel_cache",
            "voxel_cache_dir": args.voxel_cache_dir,
        },
        "observations": observations,
        "actions": actions,
        "gripper_state": gripper_state,
        "bg_features": bg_features,
        "rewards": rewards,
        "terminals": terminals,
        "timeouts": timeouts,
        "goals": goals,
        "path_lengths": path_lengths,
        "info_goals_reached": info_goals_reached,
        "info_goal_success_frac": info_goal_success_frac,
        "init_states": np.asarray(init_states, dtype=np.float64),
    }

    # For distributed mode, add rank suffix to output filename
    out_pkl = args.out_pkl
    if args.world_size > 1:
        base, ext = os.path.splitext(args.out_pkl)
        out_pkl = f"{base}_rank{args.rank}{ext}"

    os.makedirs(os.path.dirname(out_pkl) or ".", exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(paths_dict, f)

    print(f"\nWrote: {out_pkl}")
    print(f"E={E}, Tmax={Tmax}, K={K}, Dtok={Dtok}, A={A}, G={G}, BG={BG}, action_mode={args.action_mode}")
    if args.world_size > 1:
        print(f"[dist] Rank {args.rank} complete. Merge all rank files with merge_preprocess_shards.py")


if __name__ == "__main__":
    main()
