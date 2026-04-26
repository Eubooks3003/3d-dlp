#!/usr/bin/env python3
"""
Preprocess pre-voxelized RLBench demos for EC-Diffuser (language-conditioned).

Input layout (produced by preprocess_rlbench_voxels.py + rlbench_ply.py):
  {root}/{split}/{task}/all_variations/episodes/episode{N}/
      low_dim_obs.pkl              # pickled rlbench.Demo (list[Observation])
      variation_descriptions.pkl   # list[str], paraphrases of the instruction
      variation_number.pkl         # int
      voxel_cache/
          000000_voxels.pt
          000000_meta.pt
          000000_extras.pt
          ...

Output: a single EC-Diffuser-compatible pickle per task with the same schema
as ec_diffuser_voxel_preprocess.py, plus two new fields:
    language:         list[list[str]]  length E (4 paraphrases per episode)
    variation_number: list[int]        length E

Action space: 10-D = [pos(3), rot6d(6), gripper_open(1)] taken from the NEXT
frame's gripper pose (absolute end-effector control, matches PerAct / RVT /
GNFactor convention). Gripper_state uses the CURRENT frame's gripper pose
in the same 10-D format.

Usage:
  python ec_diffuser_voxel_preprocess_rlbench.py \
      --root /home/ellina/Desktop/data/rlbench \
      --split train_data \
      --task close_jar \
      --dlp-cfg /path/to/rlbench_dlp_config.json \
      --dlp-ckpt /path/to/rlbench_dlp_ckpt.pt \
      --out-pkl /home/ellina/Desktop/data/preprocessed/rlbench_close_jar/close_jar.pkl
"""
import os
import re
import sys
import argparse
import pickle
import glob
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from tqdm import tqdm

# numpy pickle compat shim (same as mimicgen script)
import numpy.core as _core
sys.modules['numpy._core'] = _core
sys.modules['numpy._core.multiarray'] = _core.multiarray

from voxel_models import DLP
from utils.util_func import get_config
from utils.log_utils import load_checkpoint


# ----------------------------
# RLBench Demo: stub-based unpickling (so rlbench need NOT be installed)
# ----------------------------
class _RLBenchStub:
    """Generic stub for any rlbench.* class; pickle populates __dict__ directly."""
    pass


class _StubDemo(list):
    """rlbench.demo.Demo is a list subclass; match that so unpickling works."""
    pass


class _RLBenchUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "rlbench.demo" and name == "Demo":
            return _StubDemo
        if module.startswith("rlbench") or module.startswith("pyrep"):
            return type(name, (_RLBenchStub,), {})
        return super().find_class(module, name)


def load_demo(pkl_path: str):
    """Unpickle low_dim_obs.pkl to a list of Observation stubs.

    rlbench.demo.Demo is a list subclass, but the observations live on the
    ._observations attribute, NOT in the list contents. Check that first.
    """
    with open(pkl_path, "rb") as f:
        demo = _RLBenchUnpickler(f).load()
    if hasattr(demo, "_observations") and len(demo._observations) > 0:
        return list(demo._observations)
    if hasattr(demo, "observations") and len(demo.observations) > 0:
        return list(demo.observations)
    if isinstance(demo, list) and len(demo) > 0:
        return list(demo)
    raise RuntimeError(
        f"Unrecognized / empty low_dim_obs at {pkl_path}. "
        f"type={type(demo).__name__} "
        f"attrs={sorted(vars(demo).keys()) if hasattr(demo, '__dict__') else 'N/A'}"
    )


# ----------------------------
# Geometry: quaternion -> 6D rotation
# ----------------------------
def quat_to_rot6d_xyzw(quat: np.ndarray) -> np.ndarray:
    """Convert (x,y,z,w) quaternion to 6D rotation (first two columns of R, flattened)."""
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    r00 = 1 - 2 * (y * y + z * z)
    r10 = 2 * (x * y + w * z)
    r20 = 2 * (x * z - w * y)
    r01 = 2 * (x * y - w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r21 = 2 * (y * z + w * x)
    return np.stack([r00, r10, r20, r01, r11, r21], axis=-1).astype(np.float32)


# ----------------------------
# RLBench per-frame gripper + action extraction
# ----------------------------
def extract_gripper_and_actions(demo: list) -> tuple:
    """
    From a list of rlbench Observation stubs, build:
      gripper_state: (T, 10) = [pos(3), rot6d(6), gripper_open(1)] at frame t
      actions:       (T, 10) = [pos(3), rot6d(6), gripper_open(1)] at frame t+1
                     (last frame repeats itself; terminals mask handles end-of-episode)

    RLBench Observation.gripper_pose is a length-7 array: [x,y,z, qx,qy,qz,qw].
    RLBench Observation.gripper_open is a float (1.0 open / 0.0 closed).
    """
    T = len(demo)
    pos = np.zeros((T, 3), dtype=np.float32)
    quat = np.zeros((T, 4), dtype=np.float32)
    g_open = np.zeros((T, 1), dtype=np.float32)

    for t, obs in enumerate(demo):
        gp = np.asarray(obs.gripper_pose, dtype=np.float32)
        if gp.shape[0] != 7:
            raise RuntimeError(
                f"Expected gripper_pose length 7, got {gp.shape} at frame {t}"
            )
        pos[t] = gp[:3]
        quat[t] = gp[3:7]
        g_open[t, 0] = float(obs.gripper_open)

    rot6d = quat_to_rot6d_xyzw(quat)
    gripper_state = np.concatenate([pos, rot6d, g_open], axis=-1).astype(np.float32)

    # Actions: next-frame pose (absolute EEF control). Last frame replicates.
    next_idx = np.concatenate([np.arange(1, T), np.array([T - 1])])
    actions = gripper_state[next_idx].copy()

    return gripper_state, actions


# ----------------------------
# Keypose / keyframe discovery
# Direct port of 3D Diffuser Actor's keypoint_discovery
# (utils_with_rlbench.py:766-808). Uses per-frame joint_velocities
# (available on the unpickled rlbench Observation stubs) plus gripper-open
# transitions to identify decision-point frames.
# ----------------------------
def _is_stopped_kp(obs_list, i, stopped_buffer, delta):
    next_is_not_final = (i == len(obs_list) - 2)
    gripper_state_no_change = i < (len(obs_list) - 2) and (
        obs_list[i].gripper_open == obs_list[i + 1].gripper_open
        and obs_list[i].gripper_open == obs_list[max(0, i - 1)].gripper_open
        and obs_list[max(0, i - 2)].gripper_open == obs_list[max(0, i - 1)].gripper_open
    )
    jv = np.asarray(obs_list[i].joint_velocities, dtype=np.float32)
    small_delta = np.allclose(jv, 0, atol=delta)
    return (
        stopped_buffer <= 0
        and small_delta
        and (not next_is_not_final)
        and gripper_state_no_change
    )


def keypoint_discovery(obs_list, stopping_delta: float = 0.1):
    """Return list of per-frame keypose indices for one episode."""
    episode_keypoints = []
    prev_gripper_open = obs_list[0].gripper_open
    stopped_buffer = 0
    for i in range(len(obs_list)):
        stopped = _is_stopped_kp(obs_list, i, stopped_buffer, stopping_delta)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        last = (i == len(obs_list) - 1)
        if i != 0 and (obs_list[i].gripper_open != prev_gripper_open or last or stopped):
            episode_keypoints.append(i)
        prev_gripper_open = obs_list[i].gripper_open
    if (
        len(episode_keypoints) > 1
        and (episode_keypoints[-1] - 1) == episode_keypoints[-2]
    ):
        episode_keypoints.pop(-2)
    return episode_keypoints


# ----------------------------
# Voxel cache loading (RLBench naming: 000000_voxels.pt)
# ----------------------------
def _load_voxel_any(path: str) -> torch.Tensor:
    """Load a voxel grid .pt, supporting sparse-compressed or dense formats."""
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


_FRAME_PAT = re.compile(r"(\d+)_voxels\.pt$")


def scan_episode_voxels(voxel_dir: str) -> dict:
    """Return {frame_idx: voxel_path} for one episode's voxel_cache dir."""
    out = {}
    if not os.path.isdir(voxel_dir):
        return out
    for fn in os.listdir(voxel_dir):
        m = _FRAME_PAT.match(fn)
        if m:
            out[int(m.group(1))] = os.path.join(voxel_dir, fn)
    return out


def discover_episodes(root: str, split: str, task: str) -> list:
    """Return sorted list of (episode_idx, episode_dir) for one task."""
    ep_root = os.path.join(root, split, task, "all_variations", "episodes")
    if not os.path.isdir(ep_root):
        raise RuntimeError(f"Episodes dir not found: {ep_root}")
    eps = []
    for name in os.listdir(ep_root):
        m = re.match(r"episode(\d+)$", name)
        if m:
            eps.append((int(m.group(1)), os.path.join(ep_root, name)))
    eps.sort(key=lambda x: x[0])
    return eps


# ----------------------------
# DLP model building + token packing (identical to mimicgen preprocessor)
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
    """Pack DLP output: fg tokens [z(3), z_scale(3), z_depth(1), obj_on(1), feat(F)], bg feats separate."""
    z = out["z"][:, 0]
    z_scale = out["z_scale"][:, 0]
    z_depth = out["z_depth"][:, 0]
    obj_on = out["obj_on"][:, 0]
    z_feat = out["z_features"][:, 0]
    z_bg = out["z_bg_features"][:, 0]

    if obj_on.dim() == 2:
        obj_on = obj_on.unsqueeze(-1)
    if z_depth.shape[-1] == 3:
        z_depth = z_depth[..., :1]

    toks = torch.cat([z, z_scale, z_depth, obj_on, z_feat], dim=-1)
    return toks, z_bg


# ----------------------------
# Debug: encode/decode a few frames per task, log to wandb
# ----------------------------
def run_debug_mode(model, root, split, tasks, device, expected_c,
                   wandb_project="rlbench-dlp-debug", num_samples=3):
    """Sample a few frames from each task's first episode, run encode->decode,
    and log GT + reconstruction + keypoints + diff to a single wandb run.

    Mirrors the mimicgen debug mode but walks the RLBench layout:
      <root>/<split>/<task>/all_variations/episodes/episode0/voxel_cache/NNNNNN_voxels.pt
      <root>/<split>/<task>/all_variations/episodes/episode0/kmeans_cache/NNNNNN_kmeans.pt
    """
    import wandb
    from eval.eval_vox import log_rgb_voxels

    wandb.init(project=wandb_project, name=f"debug-rlbench-{split}")
    global_step = 0

    for task_name in tasks:
        try:
            eps = discover_episodes(root, split, task_name)
        except RuntimeError as e:
            print(f"[debug] {task_name}: {e} — skipping")
            continue
        if not eps:
            print(f"[debug] {task_name}: no episodes found — skipping")
            continue
        ep_idx, ep_dir = eps[0]

        vox_dir = os.path.join(ep_dir, "voxel_cache")
        km_dir = os.path.join(ep_dir, "kmeans_cache")
        vmap = scan_episode_voxels(vox_dir)
        if not vmap:
            print(f"[debug] {task_name}: no voxels in {vox_dir} — skipping")
            continue

        frames = sorted(vmap.keys())
        if num_samples >= len(frames):
            picked = frames
        else:
            picked = [frames[i]
                      for i in np.linspace(0, len(frames) - 1, num_samples).astype(int)]

        prefix = f"{task_name}/"
        print(f"\n[debug] {task_name}: ep{ep_idx}, frames {picked}")

        for i, tt in enumerate(picked):
            vox = _load_voxel_any(vmap[tt])  # [C, D, H, W], typically RGBO (C=4)
            # The DLP is trained on RGB-only (ch=3); slice off occupancy.
            if vox.shape[0] > expected_c:
                vox = vox[:expected_c].contiguous()

            km_path = os.path.join(km_dir, f"{tt:06d}_kmeans.pt")
            if os.path.exists(km_path):
                km = torch.load(km_path, map_location="cpu", weights_only=False)
                meta = {
                    "kmeans_kp": km["kp"].unsqueeze(0).to(device),
                    "kmeans_cov": km["cov"].unsqueeze(0).to(device),
                }
                print(f"[debug] {task_name} f{tt}: using cached kmeans")
            else:
                meta = None
                print(f"[debug] {task_name} f{tt}: no cached kmeans at {km_path}, recomputing")

            vox_input = vox.unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(vox_input, deterministic=True, warmup=False,
                            with_loss=True, meta=meta)

            vox_rec = out.get("rec", out.get("reconstruction", None))
            if vox_rec is None and hasattr(model, "decode"):
                vox_rec = model.decode(out)
            vox_rec = vox_rec[0].cpu()

            # Keypoint positions: prefer mu_tot, fall back to z_base + mu_offset
            mu_tot = out.get("mu_tot", None)
            if mu_tot is None and "z_base" in out and "mu_offset" in out:
                mu_tot = out["z_base"] + out["mu_offset"]
            kp_pos = mu_tot[0].cpu() if mu_tot is not None else None

            obj_on = out.get("obj_on", None)
            obj_on_vals = obj_on[0].cpu() if obj_on is not None else None

            # GT voxel
            log_rgb_voxels(
                name=f"{prefix}samples/input_{i}",
                rgb_vol=vox, alpha_vol=None, KPx=None,
                step=global_step, mode="splat", topk=60000,
                alpha_thresh=0.05, pad=2.0, show_axes=True,
            )
            # GT + keypoints overlay
            log_rgb_voxels(
                name=f"{prefix}samples/input_kp_{i}",
                rgb_vol=vox, alpha_vol=None, KPx=kp_pos, obj_on=obj_on_vals,
                step=global_step, mode="splat", topk=60000,
                alpha_thresh=0.05, pad=2.0, show_axes=True,
            )
            # Reconstruction + keypoints
            log_rgb_voxels(
                name=f"{prefix}samples/rec_{i}",
                rgb_vol=vox_rec, alpha_vol=None, KPx=kp_pos, obj_on=obj_on_vals,
                step=global_step, mode="splat", topk=60000,
                alpha_thresh=0.05, pad=2.0, show_axes=True,
            )
            # |GT - Rec|
            diff = (vox - vox_rec).abs()
            diff_scaled = diff / (diff.max() + 1e-8)
            log_rgb_voxels(
                name=f"{prefix}samples/diff_{i}",
                rgb_vol=diff_scaled, alpha_vol=None, KPx=None,
                step=global_step, mode="splat", topk=60000,
                alpha_thresh=0.01, pad=2.0, show_axes=True,
            )

            mse = float(((vox - vox_rec) ** 2).mean())
            print(f"[debug] {task_name} f{tt} MSE: {mse:.6f}")
            wandb.log({f"{prefix}metrics/mse": mse}, step=global_step)
            global_step += 1

    wandb.finish()
    print(f"\n[debug] Done! {len(tasks)} task(s) logged to wandb project '{wandb_project}'")


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Preprocess pre-voxelized RLBench demos for EC-Diffuser (language-conditioned)"
    )
    # Data layout
    ap.add_argument("--root", required=True,
                    help="RLBench root (e.g. /home/ellina/Desktop/data/rlbench)")
    ap.add_argument("--split", default="train_data",
                    help="Split folder name under root (e.g. train_data, test_data)")
    ap.add_argument("--task", default=None,
                    help="Task name (e.g. close_jar). Required unless --debug.")
    ap.add_argument("--max-demos", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--stopping-delta", type=float, default=0.1,
                    help="Joint-velocity threshold for keypose stopped-detection "
                         "(matches 3D Diffuser Actor's default).")

    # DLP model
    ap.add_argument("--dlp-cfg", required=True, help="Path to DLP config JSON")
    ap.add_argument("--dlp-ckpt", required=True, help="Path to DLP checkpoint")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8)

    # Output
    ap.add_argument("--out-pkl", default=None,
                    help="Output pickle path. Required unless --debug.")

    # Distributed
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world-size", type=int, default=1)

    # Debug: encode/decode a few frames per task, log to wandb (no pickle written)
    ap.add_argument("--debug", action="store_true",
                    help="Debug mode: encode/decode a few frames per task and log to wandb.")
    ap.add_argument("--debug-samples", type=int, default=3,
                    help="Frames per task to visualize in debug mode.")
    ap.add_argument("--debug-tasks", nargs="+", default=None,
                    help="Task names for debug mode (defaults to [--task] if set).")
    ap.add_argument("--wandb-project", type=str, default="rlbench-dlp-debug",
                    help="Wandb project name for debug visualization.")

    args = ap.parse_args()

    device = torch.device(args.device if ("cuda" in args.device and torch.cuda.is_available()) else "cpu")

    # Load DLP
    cfg = get_config(args.dlp_cfg)
    model = build_dlp_from_cfg(cfg, device)
    _ = load_checkpoint(args.dlp_ckpt, model, None, None, map_location=device)
    expected_c = int(cfg.get("ch", 3))
    print(f"[dlp] Loaded model, expected channels: {expected_c}")

    # Debug mode: visualize a handful of frames per task, then exit.
    if args.debug:
        tasks = args.debug_tasks or ([args.task] if args.task else None)
        if not tasks:
            ap.error("--debug requires --debug-tasks (or --task)")
        run_debug_mode(
            model, args.root, args.split, tasks, device, expected_c,
            wandb_project=args.wandb_project, num_samples=args.debug_samples,
        )
        return

    if not args.task:
        ap.error("--task is required (unless --debug)")
    if not args.out_pkl:
        ap.error("--out-pkl is required (unless --debug)")

    # Discover episodes
    episodes = discover_episodes(args.root, args.split, args.task)
    if args.max_demos is not None:
        episodes = episodes[: args.max_demos]
    if len(episodes) == 0:
        raise RuntimeError(f"No episodes found for {args.task} in {args.split}")

    # Distributed shard
    if args.world_size > 1:
        all_eps = episodes
        episodes = [e for i, e in enumerate(all_eps) if i % args.world_size == args.rank]
        print(f"[dist] Rank {args.rank}/{args.world_size}: {len(episodes)}/{len(all_eps)} eps")
    else:
        print(f"[scan] {len(episodes)} episodes for task={args.task} split={args.split}")

    # kmeans cache lives per-episode alongside voxel_cache
    def kmeans_dir_for(ep_dir: str) -> str:
        return os.path.join(ep_dir, "kmeans_cache")

    # ---- Pass 1: precompute kmeans prior for every (episode, frame) not cached ----
    km_work = []
    km_cached = 0
    voxel_maps = {}  # ep_idx -> {frame_idx: path}
    for ep_idx, ep_dir in episodes:
        vox_dir = os.path.join(ep_dir, "voxel_cache")
        vmap = scan_episode_voxels(vox_dir)
        if not vmap:
            print(f"[warn] no voxels in {vox_dir}, skipping")
            continue
        voxel_maps[ep_idx] = vmap
        km_dir = kmeans_dir_for(ep_dir)
        try:
            cached = set(os.listdir(km_dir))
        except FileNotFoundError:
            cached = set()
        for tt in sorted(vmap.keys()):
            km_fname = f"{tt:06d}_kmeans.pt"
            if km_fname in cached:
                km_cached += 1
            else:
                km_work.append((ep_idx, ep_dir, tt, vmap[tt], os.path.join(km_dir, km_fname)))

    print(f"[kmeans] {len(km_work)} frames to compute, {km_cached} already cached")

    kmeans_cache = {ep_idx: {} for ep_idx, _ in episodes}

    if km_work:
        def _load_one(item):
            ep_idx, ep_dir, tt, vox_path, km_path = item
            vox = _load_voxel_any(vox_path)
            return ep_idx, tt, vox, km_path

        pbar = tqdm(total=len(km_work), desc="KMeans prior", unit="frame")
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(_load_one, it) for it in km_work]
            for fut in futures:
                ep_idx, tt, vox, km_path = fut.result()
                with torch.no_grad():
                    vox_gpu = vox.unsqueeze(0).to(device)
                    kp, cov = model.prior_module.encode_prior(vox_gpu)
                km = {"kp": kp[0].cpu(), "cov": cov[0].cpu()}
                os.makedirs(os.path.dirname(km_path), exist_ok=True)
                torch.save(km, km_path)
                kmeans_cache[ep_idx][tt] = km
                pbar.update(1)
        pbar.close()

    # Load any cached kmeans not in memory
    km_load_items = []
    for ep_idx, ep_dir in episodes:
        if ep_idx not in voxel_maps:
            continue
        km_dir = kmeans_dir_for(ep_dir)
        try:
            cached = set(os.listdir(km_dir))
        except FileNotFoundError:
            cached = set()
        for tt in sorted(voxel_maps[ep_idx].keys()):
            if tt not in kmeans_cache[ep_idx]:
                fn = f"{tt:06d}_kmeans.pt"
                if fn in cached:
                    km_load_items.append((ep_idx, tt, os.path.join(km_dir, fn)))

    if km_load_items:
        def _load_km(item):
            ep_idx, tt, path = item
            return ep_idx, tt, torch.load(path, map_location="cpu", weights_only=False)

        print(f"[kmeans] Loading {len(km_load_items)} cached kmeans files...")
        with ThreadPoolExecutor(max_workers=8) as pool:
            for ep_idx, tt, km in pool.map(_load_km, km_load_items):
                kmeans_cache[ep_idx][tt] = km

    # ---- Pass 2: encode each episode, build arrays ----
    ep_obs, ep_act, ep_gripper, ep_bg = [], [], [], []
    path_lengths = []
    demo_indices = []
    ep_language = []
    ep_variation = []
    ep_keyposes = []
    total_written = 0

    pbar = tqdm(episodes, desc=f"Rank {args.rank}", unit="ep")
    for ep_idx, ep_dir in pbar:
        if ep_idx not in voxel_maps:
            continue
        vmap = voxel_maps[ep_idx]

        # Load Demo -> actions + gripper_state
        demo = load_demo(os.path.join(ep_dir, "low_dim_obs.pkl"))
        gripper_full, actions_full = extract_gripper_and_actions(demo)
        T_demo = gripper_full.shape[0]

        # Keypose discovery (3DDA-style). Uses joint_velocities + gripper_open
        # transitions on the full demo; later clipped to actual stored length L.
        kps_full = keypoint_discovery(demo, args.stopping_delta)

        # Language + variation
        with open(os.path.join(ep_dir, "variation_descriptions.pkl"), "rb") as f:
            lang = pickle.load(f)
        if not isinstance(lang, list) or not all(isinstance(s, str) for s in lang):
            raise RuntimeError(f"Unexpected language format in {ep_dir}: {type(lang)}")
        with open(os.path.join(ep_dir, "variation_number.pkl"), "rb") as f:
            varnum = int(pickle.load(f))

        # Determine frame range: intersect voxel frames with demo length
        vframes = sorted(vmap.keys())
        if vframes[0] != 0:
            print(f"[warn] ep{ep_idx}: voxel frames start at {vframes[0]}, not 0")
        T = min(T_demo, vframes[-1] + 1, len(vframes))
        missing = [t for t in range(T) if t not in vmap]
        if missing:
            print(f"[warn] ep{ep_idx}: missing {len(missing)}/{T} voxel frames, skipping")
            continue

        # Batch schedule
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

        def _prefetch_batch(ts):
            voxels = [_load_voxel_any(vmap[tt]) for tt in ts]
            return torch.stack(voxels, dim=0)

        obs_steps, bg_steps = [], []
        with ThreadPoolExecutor(max_workers=2) as io_pool:
            pending = io_pool.submit(_prefetch_batch, batches[0]) if batches else None
            for bi, ts in enumerate(batches):
                vox = pending.result().to(device)
                if bi + 1 < len(batches):
                    pending = io_pool.submit(_prefetch_batch, batches[bi + 1])

                # DLP trained on RGB-only; if voxels are RGBO, drop occupancy.
                if vox.shape[1] > expected_c:
                    vox = vox[:, :expected_c].contiguous()
                elif vox.shape[1] != expected_c:
                    raise RuntimeError(f"Channel mismatch: got C={vox.shape[1]} expected C={expected_c}")

                with torch.no_grad():
                    km_list_kp, km_list_cov = [], []
                    for tt in ts:
                        km = kmeans_cache[ep_idx][tt]
                        km_list_kp.append(km["kp"])
                        km_list_cov.append(km["cov"])
                    meta = {
                        "kmeans_kp": torch.stack(km_list_kp).to(device),
                        "kmeans_cov": torch.stack(km_list_cov).to(device),
                    }
                    out = model(vox, deterministic=True, warmup=False, with_loss=False, meta=meta)
                    toks, bg_feats = pack_tokens_k24(out)

                obs_steps.append(toks.detach().cpu().numpy().astype(np.float32))
                bg_steps.append(bg_feats.detach().cpu().numpy().astype(np.float32))
                total_written += toks.shape[0]

        if not obs_steps:
            continue

        obs_ep = np.concatenate(obs_steps, axis=0)
        bg_ep = np.concatenate(bg_steps, axis=0)
        L = obs_ep.shape[0]
        act_ep = actions_full[:L].astype(np.float32)
        grip_ep = gripper_full[:L].astype(np.float32)

        ep_obs.append(obs_ep)
        ep_act.append(act_ep)
        ep_gripper.append(grip_ep)
        ep_bg.append(bg_ep)
        path_lengths.append(L)
        demo_indices.append(ep_idx)
        ep_language.append(lang)
        ep_variation.append(varnum)

        # Clip keyposes to the stored frame range [0, L). Always include L-1
        # as the final keypose so trajectories terminate at the last stored frame.
        kps = [int(k) for k in kps_full if k < L]
        if len(kps) == 0 or kps[-1] != L - 1:
            kps.append(L - 1)
        ep_keyposes.append(kps)

        pbar.set_postfix(frames=L, total=total_written)

        if args.max_frames is not None and total_written >= args.max_frames:
            break

    # ---- Pack into EC-Diffuser format ----
    if not ep_obs:
        raise RuntimeError("No episodes produced; nothing to write.")

    E = len(ep_obs)
    Tmax = int(max(path_lengths))
    K = int(ep_obs[0].shape[1])
    Dtok = int(ep_obs[0].shape[2])
    A = int(ep_act[0].shape[1])
    G = int(ep_gripper[0].shape[1])
    BG = int(ep_bg[0].shape[1])

    observations = np.zeros((E, Tmax, K, Dtok), dtype=np.float32)
    actions = np.zeros((E, Tmax, A), dtype=np.float32)
    gripper_state = np.zeros((E, Tmax, G), dtype=np.float32)
    bg_features = np.zeros((E, Tmax, BG), dtype=np.float32)
    rewards = np.zeros((E, Tmax, 1), dtype=np.float32)
    terminals = np.zeros((E, Tmax, 1), dtype=np.float32)
    timeouts = np.zeros((E, Tmax, 1), dtype=np.float32)
    goals = np.zeros((E, Tmax, K, Dtok), dtype=np.float32)

    for e in range(E):
        L = int(path_lengths[e])
        observations[e, :L] = ep_obs[e]
        actions[e, :L] = ep_act[e]
        gripper_state[e, :L] = ep_gripper[e]
        bg_features[e, :L] = ep_bg[e]
        goals[e, :L] = ep_obs[e][L - 1][None, :, :]
        terminals[e, L - 1, 0] = 1.0
        if L < Tmax:
            observations[e, L:] = observations[e, L - 1:L]
            goals[e, L:] = goals[e, L - 1:L]
            gripper_state[e, L:] = gripper_state[e, L - 1:L]
            bg_features[e, L:] = bg_features[e, L - 1:L]
            timeouts[e, L:, 0] = 1.0

    path_lengths_arr = np.asarray(path_lengths, dtype=np.int32)

    # Pack keypose indices into (E, max_kp) with -1 sentinel for padding.
    n_keyposes_arr = np.asarray([len(k) for k in ep_keyposes], dtype=np.int32)
    max_kp = int(n_keyposes_arr.max()) if len(n_keyposes_arr) else 0
    keypose_indices_arr = np.full((len(ep_keyposes), max(max_kp, 1)), -1, dtype=np.int32)
    for ei, kps in enumerate(ep_keyposes):
        keypose_indices_arr[ei, :len(kps)] = np.asarray(kps, dtype=np.int32)
    if len(n_keyposes_arr):
        kp_counts = n_keyposes_arr
        print(f"[keyposes] task={args.task} stopping_delta={args.stopping_delta} "
              f"min={int(kp_counts.min())} median={int(np.median(kp_counts))} "
              f"max={int(kp_counts.max())} mean={float(kp_counts.mean()):.1f}")

    paths_dict = {
        "meta": {
            "K": K, "Dtok": Dtok, "A": A, "G": G, "BG": BG, "Tmax": Tmax,
            "repr": "DLP_tokens_k24",
            "action_mode": "rlbench_absolute_eef_rot6d",
            "gripper_state_format": "pos(3)_rot6d(6)_open(1)",
            "action_format": "next_frame_pos(3)_rot6d(6)_open(1)",
            "bg_features_format": f"z_bg_features({BG})",
            "source": "rlbench_voxel_cache",
            "task": args.task,
            "split": args.split,
            "root": args.root,
            "demo_indices": np.asarray(demo_indices, dtype=np.int32),
            "stopping_delta": float(args.stopping_delta),
            "keypose_format": "(E, max_kp) int32, -1 = padding; n_keyposes gives true count",
        },
        "observations": observations,
        "actions": actions,
        "gripper_state": gripper_state,
        "bg_features": bg_features,
        "rewards": rewards,
        "terminals": terminals,
        "timeouts": timeouts,
        "goals": goals,
        "path_lengths": path_lengths_arr,
        "info_goals_reached": np.ones((E,), dtype=np.float32),
        "info_goal_success_frac": np.ones((E,), dtype=np.float32),
        # RLBench-specific new fields:
        "language": ep_language,                                  # list[list[str]] len E
        "variation_number": np.asarray(ep_variation, dtype=np.int32),  # (E,)
        "keypose_indices": keypose_indices_arr,                   # (E, max_kp) int32, -1 = pad
        "n_keyposes": n_keyposes_arr,                             # (E,) int32
    }

    out_pkl = args.out_pkl
    if args.world_size > 1:
        base, ext = os.path.splitext(args.out_pkl)
        out_pkl = f"{base}_rank{args.rank}{ext}"
    os.makedirs(os.path.dirname(out_pkl) or ".", exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(paths_dict, f)

    print(f"\nWrote: {out_pkl}")
    print(f"E={E}, Tmax={Tmax}, K={K}, Dtok={Dtok}, A={A}, G={G}, BG={BG}")
    print(f"Language: {len(ep_language)} episodes, avg {np.mean([len(l) for l in ep_language]):.1f} paraphrases/ep")


if __name__ == "__main__":
    main()
