#!/usr/bin/env python3
"""
Evaluate a checkpoint on validation set for RGB voxel models.
Reports Masked PSNR and Occupancy IoU with mean +/- std.

These metrics match the ones used in train_dlp_voxel.py and eval/eval_vox.py:
  - Occupancy IoU: IoU computed from RGB magnitude (L2 norm > threshold)
  - Masked Color PSNR: PSNR computed only on occupied voxels (masked by GT occupancy)

Usage:
  python scripts/eval_checkpoint_rgb.py /path/to/log_dir
  python scripts/eval_checkpoint_rgb.py /path/to/log_dir --occ_thresh 0.1
  python scripts/eval_checkpoint_rgb.py /path/to/log_dir --checkpoint saves/epoch_100.pt
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.point_cloud_datasets.get_dataset import get_point_cloud_dataset
from utils.log_utils import load_checkpoint


# ============================================================================
# Models (copied from train_baselines.py for standalone use)
# ============================================================================

import torch.nn as nn

class VoxelEncoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=32, latent_dim=256, is_vae=False):
        super().__init__()
        self.is_vae = is_vae
        self.conv1 = nn.Conv3d(in_ch, base_ch, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(base_ch)
        self.conv2 = nn.Conv3d(base_ch, base_ch*2, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(base_ch*2)
        self.conv3 = nn.Conv3d(base_ch*2, base_ch*4, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(base_ch*4)
        self.flatten_dim = base_ch*4 * 8 * 8 * 8
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        if is_vae:
            self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        if self.is_vae:
            logvar = self.fc_logvar(x)
            return mu, logvar
        return mu


class VoxelDecoder(nn.Module):
    def __init__(self, out_ch=3, base_ch=32, latent_dim=256, use_sigmoid=True):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.start_D = self.start_H = self.start_W = 8
        self.start_ch = base_ch * 4
        self.fc = nn.Linear(latent_dim, self.start_ch * self.start_D * self.start_H * self.start_W)
        self.deconv1 = nn.ConvTranspose3d(self.start_ch, base_ch*2, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(base_ch*2)
        self.deconv2 = nn.ConvTranspose3d(base_ch*2, base_ch, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(base_ch)
        self.deconv3 = nn.ConvTranspose3d(base_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), self.start_ch, self.start_D, self.start_H, self.start_W)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = self.deconv3(x)
        if self.use_sigmoid:
            x = torch.sigmoid(x)
        return x


class VoxelAutoencoder(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=32, latent_dim=256, use_sigmoid=True):
        super().__init__()
        self.encoder = VoxelEncoder(in_ch=in_ch, base_ch=base_ch, latent_dim=latent_dim, is_vae=False)
        self.decoder = VoxelDecoder(out_ch=out_ch, base_ch=base_ch, latent_dim=latent_dim, use_sigmoid=use_sigmoid)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z


class VoxelVAE(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=32, latent_dim=256, use_sigmoid=True):
        super().__init__()
        self.encoder = VoxelEncoder(in_ch=in_ch, base_ch=base_ch, latent_dim=latent_dim, is_vae=True)
        self.decoder = VoxelDecoder(out_ch=out_ch, base_ch=base_ch, latent_dim=latent_dim, use_sigmoid=use_sigmoid)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, deterministic=False):
        mu, logvar = self.encoder(x)
        if deterministic:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)
        out = self.decoder(z)
        return out, mu, logvar


# ============================================================================
# RGB Voxel Metrics (matching eval/eval_vox.py)
# ============================================================================

def compute_occupancy_iou(gt_rgb, pred_rgb, occ_thresh=0.1):
    """
    Compute IoU between GT and predicted occupancy masks.
    Occupancy is determined by RGB magnitude (L2 norm) > threshold.

    This matches the implementation in eval/eval_vox.py.

    Args:
        gt_rgb: [C, D, H, W] GT RGB voxels
        pred_rgb: [C, D, H, W] predicted RGB voxels
        occ_thresh: threshold for determining occupancy from RGB magnitude

    Returns:
        float: IoU value
    """
    # Compute RGB magnitude (L2 norm across color channels)
    gt_mag = torch.sqrt((gt_rgb ** 2).sum(dim=0))  # [D, H, W]
    pred_mag = torch.sqrt((pred_rgb ** 2).sum(dim=0))

    # Handle normalized RGB in [-1, 1] range - convert to [0, 1] first
    if gt_rgb.min() < 0:
        gt_rgb_norm = (gt_rgb + 1) * 0.5
        gt_mag = torch.sqrt((gt_rgb_norm ** 2).sum(dim=0))
    if pred_rgb.min() < 0:
        pred_rgb_norm = (pred_rgb + 1) * 0.5
        pred_mag = torch.sqrt((pred_rgb_norm ** 2).sum(dim=0))

    # Create binary occupancy masks
    gt_occ = (gt_mag > occ_thresh).float()
    pred_occ = (pred_mag > occ_thresh).float()

    # Compute IoU
    intersection = (gt_occ * pred_occ).sum()
    union = ((gt_occ + pred_occ) > 0).float().sum()

    if union < 1e-8:
        return 1.0  # Both empty = perfect match

    iou = float(intersection / union)
    return iou


def compute_masked_color_psnr(gt_rgb, pred_rgb, occ_thresh=0.1, max_val=1.0):
    """
    Compute PSNR only on occupied voxels (masked by GT occupancy).

    This matches the implementation in eval/eval_vox.py.

    Args:
        gt_rgb: [C, D, H, W] GT RGB voxels
        pred_rgb: [C, D, H, W] predicted RGB voxels
        occ_thresh: threshold for determining occupancy from RGB magnitude
        max_val: maximum value for PSNR calculation (1.0 for normalized, 255 for uint8)

    Returns:
        float: PSNR value (dB)
    """
    # Ensure same device
    pred_rgb = pred_rgb.to(gt_rgb.device)

    # Handle normalized RGB in [-1, 1] range - convert to [0, 1]
    if gt_rgb.min() < 0:
        gt_rgb = (gt_rgb + 1) * 0.5
        pred_rgb = (pred_rgb + 1) * 0.5

    # Compute GT occupancy mask from RGB magnitude
    gt_mag = torch.sqrt((gt_rgb ** 2).sum(dim=0))  # [D, H, W]
    gt_occ = gt_mag > occ_thresh  # [D, H, W]

    # Expand mask to match RGB channels: [C, D, H, W]
    mask = gt_occ.unsqueeze(0).expand_as(gt_rgb)

    # Count occupied voxels (per-channel)
    n_occupied = mask.sum()

    if n_occupied < 1:
        return float('inf')  # No occupied voxels

    # Compute MSE only on occupied voxels
    diff_sq = (gt_rgb - pred_rgb) ** 2
    mse = (diff_sq * mask).sum() / n_occupied

    if mse < 1e-10:
        return float('inf')  # Perfect reconstruction

    psnr = 10 * torch.log10(torch.tensor(max_val ** 2) / mse)
    return float(psnr)


# ============================================================================
# Main evaluation
# ============================================================================

def load_dlp_model(config, device):
    """Load DLP model from config.

    Uses config values directly (matching train_dlp_voxel.py) to avoid
    shape mismatches from incorrect defaults.
    """
    from voxel_models import DLP

    # Required params from config (no defaults - these must exist)
    ch = config['ch']
    voxel_grid_whd = config['voxel_grid_whd']
    image_size = voxel_grid_whd[0]

    # Feature dimensions
    learned_feature_dim = config['learned_feature_dim']
    learned_bg_feature_dim = config.get('learned_bg_feature_dim', learned_feature_dim)

    # Channel multipliers (might be lists in config)
    obj_ch_mult = tuple(config['obj_ch_mult'])
    obj_ch_mult_prior = tuple(config.get('obj_ch_mult_prior', obj_ch_mult))
    bg_ch_mult = tuple(config['bg_ch_mult'])

    model = DLP(
        cdim=ch,
        image_size=image_size,
        normalize_rgb=config['normalize_rgb'],

        # Keypoint and patch configuration
        n_kp_per_patch=config['n_kp_per_patch'],
        patch_size=config['patch_size'],
        anchor_s=config['anchor_s'],
        n_kp_enc=config['n_kp_enc'],
        n_kp_prior=config['n_kp_prior'],

        # Network configuration
        pad_mode=config['pad_mode'],
        dropout=config['dropout'],

        # Feature representation
        features_dist=config.get('features_dist', 'gauss'),
        learned_feature_dim=learned_feature_dim,
        learned_bg_feature_dim=learned_bg_feature_dim,
        n_fg_categories=config.get('n_fg_categories', 8),
        n_fg_classes=config.get('n_fg_classes', 4),
        n_bg_categories=config.get('n_bg_categories', 4),
        n_bg_classes=config.get('n_bg_classes', 4),

        # Prior distributions parameters
        scale_std=config['scale_std'],
        offset_std=config['offset_std'],
        obj_on_alpha=config['obj_on_alpha'],
        obj_on_beta=config['obj_on_beta'],

        # Object decoder architecture
        obj_res_from_fc=config['obj_res_from_fc'],
        obj_ch_mult_prior=obj_ch_mult_prior,
        obj_ch_mult=obj_ch_mult,
        obj_base_ch=config['obj_base_ch'],
        obj_final_cnn_ch=config['obj_final_cnn_ch'],

        # Background decoder architecture
        bg_res_from_fc=config['bg_res_from_fc'],
        bg_ch_mult=bg_ch_mult,
        bg_base_ch=config['bg_base_ch'],
        bg_final_cnn_ch=config['bg_final_cnn_ch'],

        # Network architecture options
        use_resblock=config['use_resblock'],
        num_res_blocks=config['num_res_blocks'],
        cnn_mid_blocks=config.get('cnn_mid_blocks', False),
        mlp_hidden_dim=config.get('mlp_hidden_dim', 256),

        # Particle interaction transformer (PINT) configuration
        pint_enc_layers=config['pint_enc_layers'],
        pint_enc_heads=config['pint_enc_heads'],

        # Dynamics configuration
        timestep_horizon=1,

        # RGBD Stuff
        separate_depth_features=config['separate_depth_features'],
        depth_feature_dim=config['depth_feature_dim'],
        split_loss=config['split_loss'],
        depth_loss_ratio=config['depth_loss_ratio'],
    ).to(device)

    return model


def infer_mode_from_path(log_dir):
    """Infer model mode from directory name."""
    dirname = os.path.basename(log_dir.rstrip("/")).lower()

    if "occ_vae" in dirname:
        return "occ_vae"
    elif "occ_ae" in dirname:
        return "occ_ae"
    elif "rgb_vae" in dirname:
        return "rgb_vae"
    elif "rgb_ae" in dirname:
        return "rgb_ae"
    elif "_occ" in dirname and "vae" not in dirname:
        return "dlp_occ"
    elif "_rgb" in dirname:
        return "dlp_rgb"
    return None


def load_baseline_model(config, device, mode_override=None):
    """Load baseline VoxelAE or VoxelVAE model."""
    mode = mode_override or config.get("mode", "rgb_ae")
    is_vae = mode.endswith("vae")
    in_ch = 3  # RGB

    base_ch = config.get("base_ch", 32)
    latent_dim = config.get("latent_dim", 256)

    if is_vae:
        model = VoxelVAE(in_ch=in_ch, out_ch=in_ch, base_ch=base_ch, latent_dim=latent_dim)
    else:
        model = VoxelAutoencoder(in_ch=in_ch, out_ch=in_ch, base_ch=base_ch, latent_dim=latent_dim)

    return model.to(device), is_vae


def evaluate(log_dir, checkpoint_name="saves/best.pt", occ_thresh=0.1,
             batch_size=None, device="cuda:0"):
    """
    Evaluate checkpoint on validation set for RGB voxels.

    Returns dict with mean and std for Masked PSNR and Occupancy IoU.
    """
    # Load config
    hparams_path = os.path.join(log_dir, "hparams.json")
    if not os.path.exists(hparams_path):
        raise FileNotFoundError(f"Config not found: {hparams_path}")

    with open(hparams_path) as f:
        config = json.load(f)

    print(f"Loaded config from {hparams_path}")
    print(f"  mode: {config.get('mode', 'unknown')}")
    print(f"  ds: {config.get('ds', 'unknown')}")

    # Load checkpoint
    ckpt_path = os.path.join(log_dir, checkpoint_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Determine model type - try config first, then infer from path
    mode = config.get("mode", "")
    model_type = None

    # First check if mode explicitly says it's a baseline
    if mode in ["rgb_ae", "rgb_vae"]:
        model, is_vae = load_baseline_model(config, device, mode_override=mode)
        model_type = "baseline"
        print(f"  Model type: Baseline ({mode})")
    else:
        # Try to infer from directory name
        inferred = infer_mode_from_path(log_dir)
        if inferred and inferred in ["rgb_ae", "rgb_vae"]:
            mode = inferred
            print(f"  Inferred baseline mode from path: {mode}")
            model, is_vae = load_baseline_model(config, device, mode_override=mode)
            model_type = "baseline"
        elif inferred and inferred == "dlp_rgb":
            mode = inferred
            print(f"  Inferred DLP RGB mode from path: {mode}")
            model = load_dlp_model(config, device)
            is_vae = True
            model_type = "dlp"
            print(f"  Model type: DLP (ch={config.get('ch', 3)}, n_kp_enc={config.get('n_kp_enc', '?')})")
        elif "n_kp_enc" in config and "patch_size" in config and config.get('ch', 1) == 3:
            # Looks like DLP RGB based on config keys
            model = load_dlp_model(config, device)
            is_vae = True
            model_type = "dlp"
            print(f"  Model type: DLP RGB (ch={config.get('ch', 3)}, n_kp_enc={config.get('n_kp_enc', '?')})")
        else:
            raise ValueError(f"Unknown mode: {mode}. Could not determine RGB model type. "
                            f"This script is for RGB voxel models only.")

    # Load weights
    load_checkpoint(ckpt_path, model, None, None, map_location=device)
    model.eval()
    print(f"Loaded checkpoint from {ckpt_path}")

    # Load validation dataset
    ds = config.get("ds", "shapes")
    root = config.get("root", "./data")
    voxel_grid_whd = config.get("voxel_grid_whd", [64, 64, 64])
    voxel_mode = config.get("voxel_mode", "avg_rgb")  # RGB mode

    if batch_size is None:
        batch_size = config.get("batch_size", 4)

    # Get cache dir for voxels if specified
    voxel_root = config.get("voxel_root", None)

    # MimicGen specific
    task = config.get("task", None)
    max_demos = config.get("max_demos", None)
    cache_suffix = config.get("cache_suffix", "")

    try:
        val_ds = get_point_cloud_dataset(
            ds, root, mode="val", max_points=4096, include_rgb=True,
            voxelize=True, voxel_grid_whd=tuple(voxel_grid_whd), voxel_mode=voxel_mode,
            cache_dir=voxel_root, task=task, max_demos=max_demos, cache_suffix=cache_suffix,
        )
    except Exception as e:
        print(f"[Warning] Could not load val dataset, trying test: {e}")
        try:
            val_ds = get_point_cloud_dataset(
                ds, root, mode="test", max_points=4096, include_rgb=True,
                voxelize=True, voxel_grid_whd=tuple(voxel_grid_whd), voxel_mode=voxel_mode,
                cache_dir=voxel_root, task=task, max_demos=max_demos, cache_suffix=cache_suffix,
            )
        except Exception as e2:
            print(f"[Warning] Could not load test dataset either, trying train: {e2}")
            val_ds = get_point_cloud_dataset(
                ds, root, mode="train", max_points=4096, include_rgb=True,
                voxelize=True, voxel_grid_whd=tuple(voxel_grid_whd), voxel_mode=voxel_mode,
                cache_dir=voxel_root, task=task, max_demos=max_demos, cache_suffix=cache_suffix,
            )

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    print(f"Loaded validation set: {len(val_ds)} samples")
    print(f"  Occupancy threshold: {occ_thresh}")

    # Evaluate
    all_ious = []
    all_masked_psnrs = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            vox = batch["voxels"].to(device).float()
            x = vox[:, :3, ...]  # RGB channels

            # Forward pass - different for DLP vs baseline
            if model_type == "dlp":
                model_output = model(x, deterministic=True, warmup=False, with_loss=False)
                rec = model_output['rec']
            elif is_vae:
                rec, mu, logvar = model(x, deterministic=True)
            else:
                rec, z = model(x)

            # Per-sample metrics
            B = x.shape[0]
            for b in range(B):
                # Occupancy IoU (from RGB magnitude)
                iou = compute_occupancy_iou(x[b], rec[b], occ_thresh=occ_thresh)
                all_ious.append(iou)

                # Masked Color PSNR (only on occupied voxels)
                masked_psnr = compute_masked_color_psnr(x[b], rec[b], occ_thresh=occ_thresh)
                if np.isfinite(masked_psnr):
                    all_masked_psnrs.append(masked_psnr)

    # Compute statistics
    results = {
        "n_samples": len(all_ious),
        "model_type": model_type,
        "occ_thresh": occ_thresh,
        # Main metrics: Occupancy IoU
        "occ_iou_mean": float(np.mean(all_ious)),
        "occ_iou_std": float(np.std(all_ious)),
        "occ_iou_min": float(np.min(all_ious)),
        "occ_iou_max": float(np.max(all_ious)),
    }

    # Main metrics: Masked PSNR
    if all_masked_psnrs:
        results["masked_psnr_mean"] = float(np.mean(all_masked_psnrs))
        results["masked_psnr_std"] = float(np.std(all_masked_psnrs))
        results["masked_psnr_min"] = float(np.min(all_masked_psnrs))
        results["masked_psnr_max"] = float(np.max(all_masked_psnrs))
        results["n_valid_psnr"] = len(all_masked_psnrs)

    return results, all_ious, all_masked_psnrs


def main():
    parser = argparse.ArgumentParser(description="Evaluate RGB voxel checkpoint on validation set")
    parser.add_argument("log_dirs", type=str, nargs="+", help="Path(s) to log directory")
    parser.add_argument("--checkpoint", type=str, default="saves/best.pt",
                        help="Checkpoint path relative to log_dir (default: saves/best.pt)")
    parser.add_argument("--occ_thresh", type=float, default=0.1,
                        help="Threshold for occupancy from RGB magnitude (default: 0.1)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (default: from config)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device (default: cuda:0)")

    args = parser.parse_args()

    all_results = []

    for log_dir in args.log_dirs:
        print(f"\n{'='*60}")
        print(f"Evaluating: {log_dir}")
        print("=" * 60)

        try:
            results, all_ious, all_masked_psnrs = evaluate(
                log_dir=log_dir,
                checkpoint_name=args.checkpoint,
                occ_thresh=args.occ_thresh,
                batch_size=args.batch_size,
                device=args.device,
            )

            results["log_dir"] = log_dir
            results["name"] = os.path.basename(log_dir.rstrip("/"))
            all_results.append(results)

            print(f"\n  Samples: {results['n_samples']}")
            print(f"  Occ IoU:     {results['occ_iou_mean']:.4f} +/- {results['occ_iou_std']:.4f}")
            if "masked_psnr_mean" in results:
                print(f"  Masked PSNR: {results['masked_psnr_mean']:.2f} +/- {results['masked_psnr_std']:.2f} dB")

            # Save individual results
            out_path = os.path.join(log_dir, "eval_rgb_results.json")
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Saved results to {out_path}")

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            all_results.append({"log_dir": log_dir, "error": str(e)})

    # Print summary table
    if len(args.log_dirs) > 1:
        print("\n" + "=" * 90)
        print("SUMMARY (RGB Voxel Evaluation)")
        print("=" * 90)
        print(f"{'Name':<45} {'Occ IoU':<22} {'Masked PSNR (dB)':<20}")
        print("-" * 90)
        for r in all_results:
            name = r.get("name", os.path.basename(r["log_dir"]))[:43]
            if "error" in r:
                print(f"{name:<45} ERROR: {r['error'][:40]}")
            else:
                iou_str = f"{r['occ_iou_mean']:.4f} +/- {r['occ_iou_std']:.4f}"
                if "masked_psnr_mean" in r:
                    psnr_str = f"{r['masked_psnr_mean']:.2f} +/- {r['masked_psnr_std']:.2f}"
                else:
                    psnr_str = "N/A"
                print(f"{name:<45} {iou_str:<22} {psnr_str:<20}")
        print("=" * 90)


if __name__ == "__main__":
    main()
