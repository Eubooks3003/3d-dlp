#!/usr/bin/env python3
# debug_dlp_vox.py
import os, sys, glob, time, json, argparse
from typing import Optional, Dict, Any, List

import numpy as np
# Fix for loading checkpoints saved with NumPy 2.0+ on older NumPy
# Must register in sys.modules before torch.load unpickles
if 'numpy._core' not in sys.modules:
    import numpy.core
    sys.modules['numpy._core'] = numpy.core
    # Also alias submodules that might be referenced
    for submod in ['multiarray', 'umath', '_multiarray_umath', 'numeric']:
        full_old = f'numpy.core.{submod}'
        full_new = f'numpy._core.{submod}'
        if full_old in sys.modules and full_new not in sys.modules:
            sys.modules[full_new] = sys.modules[full_old]
        elif hasattr(numpy.core, submod) and full_new not in sys.modules:
            sys.modules[full_new] = getattr(numpy.core, submod)

import torch
from torch.utils.data import DataLoader

# --- your codebase imports ---
from utils.util_func import get_config, log_line, prepare_logdir
from datasets.point_cloud_datasets.get_dataset import get_point_cloud_dataset, pc_collate
from datasets.voxelize_ds_wrapper import VoxelizedDataset

from voxel_models import DLP  # voxel-capable DLP (same as training)
from eval.eval_vox import (log_vox_overlay_plotly, log_vox_isoseries, log_cov_ellipsoids_over_voxels,
        extract_volumes_for_vis, print_vol_stats, log_rgb_voxels)

import wandb



@torch.no_grad()
def rgb_distribution_for_plotted_voxels(
    name: str,
    rgb_vol,                      # [3,D,H,W] or [1,3,D,H,W] (torch/np; [0,1] or [-1,1])
    alpha_vol=None,               # [D,H,W] or [1,D,H,W] or None
    *,
    # ---- keep these in sync with log_rgb_voxels ----
    mode="splat",
    topk=60000,
    alpha_thresh=0.25,            # tightened in your plotting code
    rgb_mag_thresh=0.10,
    use_edge_mask=True,
    edge_percentile=80,
    # ---- histogram controls ----
    print_hist=True,
    hist_bins=20,
):
    """
    Computes stats on EXACTLY the voxels selected by your log_rgb_voxels() 'splat' path.
    Returns a dict with stats and selection indices.
    """
    def _to_np(x):
        if x is None: return None
        if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
        return x

    def _as_CDHW(x):
        x = _to_np(x)
        if x is None: return None
        if x.ndim == 5:  # [B,3,D,H,W] or [B,1, D,H,W]
            x = x[0]
        assert x.ndim == 4 and (x.shape[0] in (1,3)), f"rgb shape must be [C,D,H,W], got {x.shape}"
        return x

    def _as_DHW(x):
        x = _to_np(x)
        if x is None: return None
        if x.ndim == 4:  # [1,D,H,W]
            x = x[0]
        assert x.ndim == 3, f"alpha shape must be [D,H,W], got {x.shape}"
        return x

    RGB = _as_CDHW(rgb_vol)            # [3,D,H,W]
    if RGB.shape[0] != 3:
        raise ValueError("rgb_vol must be 3-channel [3,D,H,W] (or [1,3,D,H,W]).")
    D,H,W = RGB.shape[1:]
    ALP = None if alpha_vol is None else _as_DHW(alpha_vol)  # [D,H,W]
    if ALP is not None and ALP.shape != (D,H,W):
        raise ValueError(f"alpha_vol shape {ALP.shape} != rgb spatial {(D,H,W)}")

    # map [-1,1] to [0,1] exactly like the plotter
    rgb_min = min(RGB[0].min(), RGB[1].min(), RGB[2].min())
    if rgb_min < 0.0:
        RGB = (RGB + 1.0) * 0.5
    RGB = np.clip(RGB, 0.0, 1.0)

    # ---- selection mask (identical logic) ----
    # color magnitude
    mag = np.sqrt(RGB[0]**2 + RGB[1]**2 + RGB[2]**2)

    if ALP is None:
        weights = None
        mask = (mag >= rgb_mag_thresh)
    else:
        weights = np.asarray(ALP, dtype=np.float32).clip(0, 1)
        mask = (weights >= alpha_thresh) & (mag >= rgb_mag_thresh)

    # optional: edge emphasis (Sobel on alpha)
    edge = None
    if use_edge_mask and weights is not None:
        try:
            import scipy.ndimage as ndi
            gx = ndi.sobel(weights, axis=2); gy = ndi.sobel(weights, axis=1); gz = ndi.sobel(weights, axis=0)
            edge = np.sqrt(gx*gx + gy*gy + gz*gz)
            if (mask.any() and np.isfinite(edge[mask]).any()) or np.isfinite(edge).any():
                base = edge[mask] if mask.any() else edge
                thr = np.percentile(base, edge_percentile)
                edge_mask = edge >= thr
                mask = mask & edge_mask
        except Exception:
            # fall back silently if scipy not available
            edge = None

    idx = np.argwhere(mask)  # [M,3] (z,y,x)
    M = idx.shape[0]
    if M == 0:
        print(f"\n=== {name}: no voxels selected under current thresholds ===")
        return dict(name=name, count=0, idx=None, stats=None)

    # top-K after filtering (score = alpha*edge or alpha or mag)
    if M > topk:
        if weights is not None:
            score = weights[idx[:,0], idx[:,1], idx[:,2]]
            if use_edge_mask and edge is not None:
                score = score * edge[idx[:,0], idx[:,1], idx[:,2]]
        else:
            score = mag[idx[:,0], idx[:,1], idx[:,2]]
        sel = np.argpartition(score, -topk)[-topk:]
        idx = idx[sel]
        M = idx.shape[0]

    z_i, y_i, x_i = idx[:,0], idx[:,1], idx[:,2]
    r = RGB[0, z_i, y_i, x_i]
    g = RGB[1, z_i, y_i, x_i]
    b = RGB[2, z_i, y_i, x_i]

    # ---- stats ----
    def _stats(v):
        v = v.astype(np.float64).ravel()
        q = np.quantile(v, [0.01, 0.25, 0.5, 0.75, 0.99])
        return dict(
            min=float(v.min()),
            p01=float(q[0]), p25=float(q[1]), med=float(q[2]), p75=float(q[3]), p99=float(q[4]),
            max=float(v.max()),
            mean=float(v.mean()),
            std=float(v.std(ddof=0)),
        )

    sR, sG, sB = _stats(r), _stats(g), _stats(b)

    # histograms (clamped to p1..p99 like before)
    hist = None
    if print_hist:
        def _hist(v, bins):
            q1, q99 = np.quantile(v, [0.01, 0.99])
            if not np.isfinite(q1) or not np.isfinite(q99) or q99 <= q1:
                q1, q99 = float(v.min()), float(v.max())
            vv = np.clip(v, q1, q99)
            h, edges = np.histogram(vv, bins=bins, range=(q1, q99), density=True)
            # compress display to ~8 buckets for readability
            step = max(1, bins // 8)
            out = []
            for i in range(0, bins, step):
                lo = edges[i]; hi = edges[min(i+step, bins)]
                prob = h[i:i+step].sum() * (edges[1]-edges[0])  # integrate density
                out.append((float(lo), float(hi), float(prob)))
            return out
        hist = dict(
            R=_hist(r, hist_bins),
            G=_hist(g, hist_bins),
            B=_hist(b, hist_bins),
        )

    # channel correlations
    def _corr(a, b):
        a = a.astype(np.float64); b = b.astype(np.float64)
        a = (a - a.mean()) / (a.std() + 1e-12)
        b = (b - b.mean()) / (b.std() + 1e-12)
        return float((a*b).mean())
    corr = dict(RG=_corr(r,g), RB=_corr(r,b), GB=_corr(g,b))

    # ---- print ----
    print(f"\n=== {name}: stats on plotted voxels ===")
    print(f"selected = {M} voxels")
    for ch, s in zip("RGB", [sR,sG,sB]):
        print(f"[{ch}] min={s['min']:.5f}  p1={s['p01']:.5f}  p25={s['p25']:.5f}  "
              f"med={s['med']:.5f}  p75={s['p75']:.5f}  p99={s['p99']:.5f}  "
              f"max={s['max']:.5f}  mean={s['mean']:.5f}  std={s['std']:.5f}")
        if print_hist:
            h = hist[ch]
            h_str = " | ".join([f"[{a:.3f},{b:.3f}): {p:.3f}" for (a,b,p) in h])
            print(f"  hist (clamped to p1..p99): {h_str}")
    print(f"corr(R,G)={corr['RG']:.4f}  corr(R,B)={corr['RB']:.4f}  corr(G,B)={corr['GB']:.4f}")

    return dict(
        name=name,
        count=M,
        idx=idx,                      # (z,y,x) indices of selected voxels
        stats=dict(R=sR,G=sG,B=sB),
        corr=corr,
        hist=hist,
    )

def kp_norm_to_voxel_idx(kps, D, H, W, order=("x", "y", "z")):
    """
    Convert keypoints from normalized [-1, 1] space to voxel index coordinates.

    Args:
        kps: [K, 3] keypoints in normalized space
        D, H, W: voxel grid dimensions
        order: order of components in kps, default is (x, y, z)

    Returns:
        [K, 3] keypoints in voxel index space (x_idx, y_idx, z_idx) for Plotly
    """
    if isinstance(kps, torch.Tensor):
        kps = kps.detach().cpu().numpy()
    kps = np.asarray(kps)

    if kps.ndim == 1:
        kps = kps[None, :]

    # Map from [-1, 1] to [0, N-1]
    def to_idx(v, n):
        return np.clip(0.5 * (v + 1.0) * (n - 1), 0, n - 1)

    size = {"x": W, "y": H, "z": D}

    x = to_idx(kps[:, order.index("x")], size["x"])
    y = to_idx(kps[:, order.index("y")], size["y"])
    z = to_idx(kps[:, order.index("z")], size["z"])

    return np.stack([x, y, z], axis=-1)  # [K, 3] in (x, y, z) order for Plotly


def filter_topk_kps_3d(
    z_base_var,      # [B, *G*, K, C]  (C = 3 or 6)
    mu_tot,          # [B, *G*, K, 3]  (must align to same K as z_base_var)
    topk: int,
    obj_on=None,     # [B, *G*, K, 1] or [B, *G*, K] (optional)
    use_posterior_in_score: bool = False,
    eps: float = 1e-6,
):
    B = z_base_var.shape[0]
    C = z_base_var.shape[-1]
    assert C in (3, 6)

    # 1) Squeeze any extra group/time dims so both are [B, K, ·]
    zvar = z_base_var.view(B, -1, C)          # [B, K, C]
    mu   = mu_tot.view(B, -1, 3)              # [B, K, 3]
    K = zvar.size(1)

    # 2) Uncertainty score (prior variance sum; optionally add posterior var)
    prior_var = zvar[..., :3]                 # [B, K, 3]  (≥ 0)
    if use_posterior_in_score and C == 6:
        post_var = zvar[..., 3:6].exp().clamp_min(eps)
        unc = prior_var.sum(-1) + post_var.sum(-1)    # [B, K]
    else:
        unc = prior_var.sum(-1)                        # [B, K]

    # Optional gating by obj_on
    if obj_on is not None:
        g = obj_on
        if g.dim() == 4 and g.size(-1) == 1:  # [B,*G*,K,1]
            g = g.view(B, -1)                 # [B, K]
        else:
            g = g.view(B, -1)
        unc = unc * g

    # 3) Top-k (smallest uncertainty)
    k = min(topk, K)
    _, idx = torch.topk(unc, k=k, dim=1, largest=False)          # [B, k]

    # 4) Gather (no advanced indexing outer-product)
    idx_exp = idx.unsqueeze(-1).expand(B, k, 3)                  # [B, k, 3]
    topk_kp = torch.gather(mu, 1, idx_exp)                       # [B, k, 3]

    bb_scores = -unc                                             # [B, K]
    print("TOPK kp shape:", topk_kp.shape)
    return {
        "indices":   idx,        # [B, k] in the current K space
        "topk_kp":   topk_kp,    # [B, k, 3]
        "unc":       unc,        # [B, K]
        "bb_scores": bb_scores,  # [B, K]
    }

# ------------------------------- helpers -------------------------------

def find_latest_checkpoint(run_save_dir: str) -> Optional[str]:
    """Find the latest checkpoint in a directory (non-recursive)."""
    candidates = []
    for ext in ("*.pth", "*.pt", "*.ckpt", "*.bin"):
        candidates += glob.glob(os.path.join(run_save_dir, ext))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)

def load_model_from_config(cfg: Dict[str, Any], device: torch.device) -> DLP:
    # Use the same field mapping as your training call (including the image_size hack via voxel_grid_whd[0])
    voxel_grid_whd = cfg["voxel_grid_whd"]
    model = DLP(
        # Basics
        cdim=cfg['ch'],
        image_size=voxel_grid_whd[0],
        normalize_rgb=cfg['normalize_rgb'],

        # Keypoint & patch config
        n_kp_per_patch=cfg['n_kp_per_patch'],
        patch_size=cfg['patch_size'],
        anchor_s=cfg['anchor_s'],
        n_kp_enc=cfg['n_kp_enc'],
        n_kp_prior=cfg['n_kp_prior'],

        # Network config
        pad_mode=cfg['pad_mode'],
        dropout=cfg['dropout'],

        # Feature representation
        features_dist=cfg.get('features_dist', 'gauss'),
        learned_feature_dim=cfg['learned_feature_dim'],
        learned_bg_feature_dim=cfg.get('learned_bg_feature_dim', cfg['learned_feature_dim']),
        n_fg_categories=cfg.get('n_fg_categories', 8),
        n_fg_classes=cfg.get('n_fg_classes', 4),
        n_bg_categories=cfg.get('n_bg_categories', 4),
        n_bg_classes=cfg.get('n_bg_classes', 4),

        # Priors
        scale_std=cfg['scale_std'],
        offset_std=cfg['offset_std'],
        obj_on_alpha=cfg['obj_on_alpha'],
        obj_on_beta=cfg['obj_on_beta'],

        # Object decoder arch
        obj_res_from_fc=cfg['obj_res_from_fc'],
        obj_ch_mult_prior=cfg.get('obj_ch_mult_prior', cfg['obj_ch_mult']),
        obj_ch_mult=cfg['obj_ch_mult'],
        obj_base_ch=cfg['obj_base_ch'],
        obj_final_cnn_ch=cfg['obj_final_cnn_ch'],

        # Background decoder arch
        bg_res_from_fc=cfg['bg_res_from_fc'],
        bg_ch_mult=cfg['bg_ch_mult'],
        bg_base_ch=cfg['bg_base_ch'],
        bg_final_cnn_ch=cfg['bg_final_cnn_ch'],

        # General arch options
        use_resblock=cfg['use_resblock'],
        num_res_blocks=cfg['num_res_blocks'],
        cnn_mid_blocks=cfg.get('cnn_mid_blocks', False),
        mlp_hidden_dim=cfg.get('mlp_hidden_dim', 256),

        # PINT
        pint_enc_layers=cfg['pint_enc_layers'],
        pint_enc_heads=cfg['pint_enc_heads'],

        # Dynamics
        timestep_horizon=1,

        # RGBD
        separate_depth_features=cfg.get('separate_depth_features', False),
        depth_feature_dim=cfg.get('depth_feature_dim', 0),
        split_loss=cfg.get('split_loss', False),
        depth_loss_ratio=cfg.get('depth_loss_ratio', 0.5),
    ).to(device)
    model.eval()
    return model

def summarize_kp(model_output: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Compact stats for keypoints to print/log."""
    def _t(t):
        return None if t is None else t.detach().float()
    out = {}
    kp_p = _t(model_output.get("kp_p"))
    if kp_p is not None:
        out["kp/mean_abs_pos"] = kp_p.abs().mean().item()
        out["kp/max_abs_pos"]  = kp_p.abs().max().item()
    mu_scale = _t(model_output.get("mu_scale"))
    if mu_scale is not None:
        out["kp/scale_sigmoid_mean"] = torch.sigmoid(mu_scale).mean().item()
    obj_on = _t(model_output.get("obj_on"))
    if obj_on is not None:
        out["kp/obj_on_mean"] = obj_on.mean().item()
    z_depth = _t(model_output.get("z_depth"))
    if z_depth is not None:
        out["kp/depth_mean"] = z_depth.mean().item()
    return out

def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def make_iso_sweep() -> List[float]:
    """
    Mirror your training sweep, plus a fine sweep if desired.
    Training used: [0.05, 0.1, 0.2, 0.3, 0.4]
    We'll include the exact same by default so it's truly 'same eval'.
    """
    return [0.05, 0.10, 0.20, 0.30, 0.40]

# ------------------------------- main -------------------------------

class VoxelDirDataset(torch.utils.data.Dataset):
    """Simple dataset that loads voxels from a directory of .pt files.

    Supports both flat directories and nested demo_*/frame*_voxels.pt structure.
    """

    def __init__(self, voxel_dir: str, max_files: int = None):
        self.voxel_dir = voxel_dir
        self.files = []

        # Try nested structure first: demo_*/frame*_voxels.pt
        nested_pattern = os.path.join(voxel_dir, "demo_*", "frame*_voxels.pt")
        self.files = sorted(glob.glob(nested_pattern))

        # If no nested files, try flat directory patterns
        if not self.files:
            patterns = [
                os.path.join(voxel_dir, "*_voxels.pt"),
                os.path.join(voxel_dir, "frame_*_voxels.pt"),
                os.path.join(voxel_dir, "*voxels*.pt"),
            ]
            for pattern in patterns:
                self.files = sorted(glob.glob(pattern))
                if self.files:
                    break

        if max_files is not None:
            self.files = self.files[:max_files]

        print(f"[VoxelDirDataset] Found {len(self.files)} voxel files in {voxel_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        vox = torch.load(self.files[idx], map_location="cpu")
        return {"voxels": vox, "path": self.files[idx]}


def main():
    ap = argparse.ArgumentParser("Debug a trained Voxel DLP checkpoint with the same voxel eval as training.")
    ap.add_argument("--config", "-c", type=str, required=True, help="Path to config JSON (hparams.json from training).")
    ap.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint file (.pth/.pt/.ckpt) or directory containing best.pt.")
    ap.add_argument("--run-dir", type=str, default="", help="(Optional) Directory to search for latest checkpoint if --ckpt not given.")
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Dataset split.")
    ap.add_argument("--batch-size", type=int, default=1, help="Batch size for debug pass.")
    ap.add_argument("--max-batches", type=int, default=4, help="How many batches to visualize.")
    ap.add_argument("--wandb", action="store_true", help="If set, logs visualizations to W&B.")
    ap.add_argument("--project", type=str, default="dlp-debug", help="W&B project name.")
    ap.add_argument("--name", type=str, default="", help="Optional W&B run name.")
    ap.add_argument("--voxel-dir", type=str, default=None,
                    help="Load voxels from this directory instead of dataset (e.g., saved mimicgen voxels)")
    ap.add_argument("--save-html", action="store_true", help="Save plotly figures as HTML files instead of logging to W&B.")
    ap.add_argument("--output-dir", type=str, default="./debug_output", help="Directory to save HTML outputs.")
    ap.add_argument("--task", type=str, default=None, help="Focus on a single task (e.g., 'coffee'). Shows --n-samples frames from that task only.")
    ap.add_argument("--n-samples", type=int, default=3, help="Number of samples per task to visualize (default: 3).")
    args = ap.parse_args()

    # Auto-discover checkpoint if directory given or from config directory
    if args.ckpt is None:
        # Try to find checkpoint in same directory as config
        config_dir = os.path.dirname(args.config)
        best_pt = os.path.join(config_dir, "best.pt")
        if os.path.isfile(best_pt):
            args.ckpt = best_pt
            print(f"[info] Auto-discovered checkpoint: {args.ckpt}")
        else:
            print(f"[error] No --ckpt provided and no best.pt found in {config_dir}")
            sys.exit(1)
    elif os.path.isdir(args.ckpt):
        # If ckpt is a directory, look for best.pt inside
        best_pt = os.path.join(args.ckpt, "best.pt")
        if os.path.isfile(best_pt):
            args.ckpt = best_pt
            print(f"[info] Found checkpoint in directory: {args.ckpt}")
        else:
            # Fallback: find latest checkpoint
            ckpt_path = find_latest_checkpoint(args.ckpt)
            if ckpt_path:
                args.ckpt = ckpt_path
                print(f"[info] Found latest checkpoint: {args.ckpt}")
            else:
                print(f"[error] No checkpoint found in {args.ckpt}")
                sys.exit(1)

    # ----- config + device -----
    cfg = get_config(args.config)
    voxel_mode     = cfg["voxel_mode"]
    voxel_grid_whd = cfg["voxel_grid_whd"]
    ch             = cfg["ch"]

    device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device if ('cuda' in device and torch.cuda.is_available()) else 'cpu')
    print(f"[info] Using device: {device}")

    # ----- dataset → voxelized wrapper (mirrors training's input batch['voxels']) -----
    if args.voxel_dir is not None:
        # Load directly from saved voxel files
        print(f"[info] Loading voxels from directory: {args.voxel_dir}")
        base_ds = VoxelDirDataset(args.voxel_dir, max_files=args.max_batches * args.batch_size)
        loader = DataLoader(base_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    else:
        # Load from point cloud dataset (original behavior)
        ds_name = cfg['ds']
        root    = cfg['root']
        task = cfg.get("task", None)
        tasks = cfg.get("tasks", None)
        max_demos = cfg.get("max_demos", None)
        cache_suffix = cfg.get("cache_suffix", "")
        proportion = cfg.get("proportion", 1.0)
        base_ds = get_point_cloud_dataset(
            ds_name, root, mode=args.split, max_points=4096, include_rgb=(ch == 6),
            task=task, tasks=tasks, max_demos=max_demos,
            cache_suffix=cache_suffix, proportion=proportion,
        )

        loader = DataLoader(base_ds, batch_size=args.batch_size,
                            shuffle=True, num_workers=4)

    # ----- model -----
    model = load_model_from_config(cfg, device)

    # ----- checkpoint -----
    ckpt_path = args.ckpt
    if not os.path.isfile(ckpt_path):
        print(f"[!] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"[info] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # unwrap common training bundles
    state = None
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model", "ema", "ema_state_dict", "net", "weights"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                print(f"[info] Using nested state_dict from '{key}'")
                break
    if state is None:
        state = ckpt
        print("[info] Using checkpoint as bare state_dict")

    # strip possible prefixes
    def _strip_prefix(d, prefix="module."):
        if not any(k.startswith(prefix) for k in d.keys()):
            return d
        return { (k[len(prefix):] if k.startswith(prefix) else k): v for k, v in d.items() }

    state = _strip_prefix(state, "module.")
    state = _strip_prefix(state, "model.")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] loaded={len(state)}  missing={len(missing)}  unexpected={len(unexpected)}")
    if missing:
        print(f"[warn] Missing (first 10): {missing[:10]}")
    if unexpected:
        print(f"[warn] Unexpected (first 10): {unexpected[:10]}")

    # ----- Output setup -----
    if args.save_html:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[info] Will save HTML figures to: {args.output_dir}")

    # ----- W&B -----
    if args.wandb:
        ckpt_basename = os.path.splitext(os.path.basename(ckpt_path))[0]
        runname = args.name or f"debug-vox-{ckpt_basename}-{int(time.time())}"
        wandb.init(project=args.project, name=runname,
                   config={"config_path": args.config, "ckpt": ckpt_path, "split": args.split})
        print("[info] W&B logging enabled.")

    def save_or_log_figure(fig, name, step):
        """Save figure as HTML or log to W&B."""
        if args.save_html:
            # Sanitize name for filename
            safe_name = name.replace("/", "_").replace("\\", "_")
            html_path = os.path.join(args.output_dir, f"{safe_name}_step{step}.html")
            fig.write_html(html_path)
            print(f"  Saved: {html_path}")
        if args.wandb:
            wandb.log({name: fig}, step=step)

    # ----- loop -----
    step = 0
    max_batches = args.max_batches
    from collections import defaultdict
    task_sample_count = defaultdict(int)  # track per-task vis count across batches

    def _vis_sample(model_output, b_idx, prefix, vis_step):
        """Visualize one sample: GT, rec, fg, bg with color-coded KPs."""
        gt_vol = model_output['x'][b_idx]
        rec_vol = model_output['rec'][b_idx]
        z_base = model_output["z_base"][b_idx]
        mu_tot = z_base + model_output["mu_offset"][b_idx]
        obj_on_vals = model_output["obj_on"][b_idx].reshape(-1)
        fg_only = model_output['dec_objects'][b_idx]
        bg_only = (model_output['bg_mask'] * model_output['bg'][:, :3])[b_idx]

        # GT with z_base only (no offset) — shows prior anchor positions
        fig = log_rgb_voxels(
            name=f"{prefix}/gt_base", rgb_vol=gt_vol, alpha_vol=None,
            KPx=z_base, obj_on=obj_on_vals, step=None,
            mode="splat", topk=60000, alpha_thresh=0.05, pad=2.0, show_axes=True,
        )
        save_or_log_figure(fig, f"{prefix}/gt_base", vis_step)

        # GT with mu_tot (z_base + offset) — shows final KP positions
        fig = log_rgb_voxels(
            name=f"{prefix}/gt_kp", rgb_vol=gt_vol, alpha_vol=None,
            KPx=mu_tot, obj_on=obj_on_vals, step=None,
            mode="splat", topk=60000, alpha_thresh=0.05, pad=2.0, show_axes=True,
        )
        save_or_log_figure(fig, f"{prefix}/gt_kp", vis_step)

        fig = log_rgb_voxels(
            name=f"{prefix}/rec_kp", rgb_vol=rec_vol, alpha_vol=None,
            KPx=mu_tot, obj_on=obj_on_vals, step=None,
            mode="splat", topk=60000, alpha_thresh=0.05, pad=2.0, show_axes=True,
        )
        save_or_log_figure(fig, f"{prefix}/rec_kp", vis_step)

        fig = log_rgb_voxels(
            name=f"{prefix}/fg_kp", rgb_vol=fg_only, alpha_vol=None,
            KPx=mu_tot, obj_on=obj_on_vals, step=None,
            mode="splat", topk=60000, alpha_thresh=0.05, pad=2.0, show_axes=True,
        )
        save_or_log_figure(fig, f"{prefix}/fg_kp", vis_step)

        fig = log_rgb_voxels(
            name=f"{prefix}/bg", rgb_vol=bg_only, alpha_vol=None,
            KPx=None, step=None,
            mode="splat", topk=60000, alpha_thresh=0.05, pad=2.0, show_axes=True,
        )
        save_or_log_figure(fig, f"{prefix}/bg", vis_step)

    samples_per_task = args.n_samples
    focus_task = args.task  # None = all tasks

    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            vox = batch["voxels"]
            model_output = model(vox, warmup=False, with_loss=True)

            # Print stats for first batch
            if step == 0:
                obj_on = model_output.get('obj_on', None)
                if obj_on is not None:
                    print(f"[obj_on] L1={obj_on.abs().mean():.6f}  mean={obj_on.mean():.6f}")
                if 'loss' in model_output:
                    print(f"[loss] total = {model_output['loss'].item():.4f}")
                print_vol_stats("GT", model_output['x'][0])
                print_vol_stats("REC", model_output['rec'][0])

            batch_tasks = batch.get("task", None)
            is_multitask = batch_tasks is not None and len(set(batch_tasks)) > 1

            if is_multitask:
                for b_idx, t in enumerate(batch_tasks):
                    # Skip tasks we don't care about
                    if focus_task is not None and t != focus_task:
                        continue
                    if task_sample_count[t] >= samples_per_task:
                        continue
                    j = task_sample_count[t]
                    _vis_sample(model_output, b_idx, t, j)
                    task_sample_count[t] += 1

                # Stop condition
                if focus_task is not None:
                    if task_sample_count[focus_task] >= samples_per_task:
                        print(f"[info] Collected {samples_per_task} samples for '{focus_task}', stopping.")
                        break
                else:
                    all_done = all(c >= samples_per_task for c in task_sample_count.values())
                    if all_done:
                        print(f"[info] Collected {samples_per_task} samples for all "
                              f"{len(task_sample_count)} tasks, stopping.")
                    break
            else:
                _vis_sample(model_output, 0, "debug", step)
                step += 1
                if step >= max_batches:
                    break

    print("[done] voxel debug pass complete.")
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
