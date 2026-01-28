#!/usr/bin/env python3
"""
Visualization script for baseline AE/VAE models trained via train_baselines.py.
Generates the same sample visualizations as visualize_voxel_paper.py for fair comparison.

Saves Plotly JSON files that can be loaded and rendered with custom styling.

Usage:
    python visualize_baseline_paper.py --run-dir /path/to/baseline_run --max-samples 50
"""
import os
import sys
import glob
import json
import argparse
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import plotly.graph_objects as go
import plotly.io as pio

# Fix for loading checkpoints saved with NumPy 2.0+ on older NumPy
if 'numpy._core' not in sys.modules:
    import numpy.core
    sys.modules['numpy._core'] = numpy.core
    for submod in ['multiarray', 'umath', '_multiarray_umath', 'numeric']:
        full_old = f'numpy.core.{submod}'
        full_new = f'numpy._core.{submod}'
        if full_old in sys.modules and full_new not in sys.modules:
            sys.modules[full_new] = sys.modules[full_old]
        elif hasattr(numpy.core, submod) and full_new not in sys.modules:
            sys.modules[full_new] = getattr(numpy.core, submod)

from utils.util_func import get_config
from datasets.point_cloud_datasets.get_dataset import get_point_cloud_dataset


# ============================================================================
# Models (copied from train_baselines.py to avoid import issues)
# ============================================================================

class VoxelEncoder(nn.Module):
    """Encoder for voxel grids. Works for any number of input channels."""
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
    """Decoder for voxel grids. Works for any number of output channels."""
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
    """Voxel Autoencoder (deterministic)."""
    def __init__(self, in_ch=3, out_ch=3, base_ch=32, latent_dim=256, use_sigmoid=True):
        super().__init__()
        self.encoder = VoxelEncoder(in_ch=in_ch, base_ch=base_ch, latent_dim=latent_dim, is_vae=False)
        self.decoder = VoxelDecoder(out_ch=out_ch, base_ch=base_ch, latent_dim=latent_dim, use_sigmoid=use_sigmoid)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z


class VoxelVAE(nn.Module):
    """Voxel Variational Autoencoder."""
    def __init__(self, in_ch=3, out_ch=3, base_ch=32, latent_dim=256, use_sigmoid=True):
        super().__init__()
        self.encoder = VoxelEncoder(in_ch=in_ch, base_ch=base_ch, latent_dim=latent_dim, is_vae=True)
        self.decoder = VoxelDecoder(out_ch=out_ch, base_ch=base_ch, latent_dim=latent_dim, use_sigmoid=use_sigmoid)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(z)
        return out, mu, logvar


# ============================================================================
# Visualization helpers (reused from visualize_voxel_paper.py)
# ============================================================================

def save_plotly_json(fig: go.Figure, filepath: str):
    """Save a plotly figure as JSON that can be loaded later."""
    fig_json = pio.to_json(fig)
    with open(filepath, 'w') as f:
        f.write(fig_json)
    print(f"  Saved: {filepath}")


def lock_scene(fig: go.Figure, D: int, H: int, W: int, camera: Optional[Dict] = None):
    """Force a shared scene box for consistent zoom/scale across figures."""
    if camera is None:
        camera = dict(
            eye=dict(x=1.5, y=1.5, z=1.2),
            up=dict(x=0, y=0, z=1),
        )
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, W-1], autorange=False),
            yaxis=dict(range=[0, H-1], autorange=False),
            zaxis=dict(range=[0, D-1], autorange=False),
            aspectmode="cube",
            camera=camera,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )


def create_rgb_voxel_figure(
    rgb_vol,
    alpha_thresh: float = 0.05,
    topk: int = 60000,
    marker_size: int = 2
) -> go.Figure:
    """Create a plotly figure with RGB voxels."""
    if isinstance(rgb_vol, torch.Tensor):
        rgb_vol = rgb_vol.detach().cpu().numpy()

    if rgb_vol.ndim == 5:
        rgb_vol = rgb_vol[0]

    # Normalize first, then compute magnitude
    if rgb_vol.min() < 0:
        rgb_vol = (rgb_vol + 1) * 0.5
    rgb_vol = np.clip(rgb_vol, 0, 1)

    mag = np.sqrt((rgb_vol ** 2).sum(axis=0))

    mask = mag > alpha_thresh
    idx = np.argwhere(mask)

    if len(idx) == 0:
        return go.Figure()

    if len(idx) > topk:
        scores = mag[mask]
        sel = np.argpartition(scores, -topk)[-topk:]
        idx = idx[sel]

    z_i, y_i, x_i = idx[:, 0], idx[:, 1], idx[:, 2]
    r = (rgb_vol[0, z_i, y_i, x_i] * 255).astype(np.uint8)
    g = (rgb_vol[1, z_i, y_i, x_i] * 255).astype(np.uint8)
    b = (rgb_vol[2, z_i, y_i, x_i] * 255).astype(np.uint8)

    colors = [f"rgba({r[i]},{g[i]},{b[i]},0.8)" for i in range(len(r))]

    fig = go.Figure(data=[
        go.Scatter3d(
            x=x_i.astype(float).tolist(),
            y=y_i.astype(float).tolist(),
            z=z_i.astype(float).tolist(),
            mode='markers',
            marker=dict(size=marker_size, color=colors),
            name='RGB voxels'
        )
    ])

    return fig


def create_occupancy_voxel_figure(
    occ_vol,
    alpha_thresh: float = 0.1,
    topk: int = 60000,
    marker_size: int = 2,
    color: str = "rgba(100, 100, 255, 0.8)"
) -> go.Figure:
    """Create a plotly figure with occupancy voxels (single channel)."""
    if isinstance(occ_vol, torch.Tensor):
        occ_vol = occ_vol.detach().cpu().numpy()

    if occ_vol.ndim == 5:
        occ_vol = occ_vol[0]

    # Handle [C, D, H, W] -> use first channel or mean
    if occ_vol.ndim == 4:
        occ_vol = occ_vol[0]  # [D, H, W]

    occ_vol = np.clip(occ_vol, 0, 1)

    mask = occ_vol > alpha_thresh
    idx = np.argwhere(mask)

    if len(idx) == 0:
        return go.Figure()

    if len(idx) > topk:
        scores = occ_vol[mask]
        sel = np.argpartition(scores, -topk)[-topk:]
        idx = idx[sel]

    z_i, y_i, x_i = idx[:, 0], idx[:, 1], idx[:, 2]

    fig = go.Figure(data=[
        go.Scatter3d(
            x=x_i.astype(float).tolist(),
            y=y_i.astype(float).tolist(),
            z=z_i.astype(float).tolist(),
            mode='markers',
            marker=dict(size=marker_size, color=color),
            name='Occupancy voxels'
        )
    ])

    return fig


def print_vol_stats(name: str, vol):
    """Print volume statistics for debugging."""
    if isinstance(vol, torch.Tensor):
        vol = vol.detach().cpu().numpy()
    print(f"  [{name}] shape={vol.shape}, min={vol.min():.4f}, max={vol.max():.4f}, "
          f"mean={vol.mean():.4f}, std={vol.std():.4f}")


# ============================================================================
# Dataset loading
# ============================================================================

class VoxelDirDataset(torch.utils.data.Dataset):
    """Simple dataset that loads voxels from a directory of .pt files."""

    def __init__(self, voxel_dir: str, max_files: int = None):
        self.voxel_dir = voxel_dir
        self.files = []

        nested_pattern = os.path.join(voxel_dir, "demo_*", "frame*_voxels.pt")
        self.files = sorted(glob.glob(nested_pattern))

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

        print(f"[VoxelDirDataset] Found {len(self.files)} voxel files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        vox = torch.load(self.files[idx], map_location="cpu")
        return {"voxels": vox, "path": self.files[idx]}


def find_latest_checkpoint(run_save_dir: str) -> Optional[str]:
    """Find the latest checkpoint in a directory."""
    candidates = []
    for ext in ("*.pth", "*.pt", "*.ckpt", "*.bin"):
        candidates += glob.glob(os.path.join(run_save_dir, ext))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate paper visualization figures for baseline AE/VAE models (saves Plotly JSON)"
    )
    parser.add_argument("--run-dir", "-r", type=str, default=None,
                        help="Path to run folder containing hparams.json and saves/best.pt")
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Path to config JSON (optional if --run-dir provided)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to checkpoint file (optional if --run-dir provided)")
    parser.add_argument("--voxel-dir", type=str, default=None,
                        help="Load voxels from this directory instead of dataset")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=4,
                        help="Maximum number of samples to visualize")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save Plotly JSON files (default: run_dir/paper_figures)")
    parser.add_argument("--alpha-thresh", type=float, default=0.1,
                        help="Alpha threshold for voxel visibility (default 0.1, same as train_baselines)")
    parser.add_argument("--start-sample", type=int, default=0,
                        help="Index of first sample to process (for matching with DLP visualization)")
    args = parser.parse_args()

    # Handle --run-dir: auto-discover config and checkpoint
    if args.run_dir is not None:
        run_dir = args.run_dir

        # Find config (hparams.json)
        if args.config is None:
            hparams_path = os.path.join(run_dir, "hparams.json")
            if os.path.isfile(hparams_path):
                args.config = hparams_path
                print(f"[info] Found config: {args.config}")
            else:
                print(f"[error] No hparams.json found in {run_dir}")
                sys.exit(1)

        # Find checkpoint (saves/best.pt)
        if args.ckpt is None:
            best_pt = os.path.join(run_dir, "saves", "best.pt")
            if os.path.isfile(best_pt):
                args.ckpt = best_pt
                print(f"[info] Found checkpoint: {args.ckpt}")
            else:
                for ckpt_path in [
                    os.path.join(run_dir, "best.pt"),
                    os.path.join(run_dir, "saves", "checkpoint.pt"),
                ]:
                    if os.path.isfile(ckpt_path):
                        args.ckpt = ckpt_path
                        print(f"[info] Found checkpoint: {args.ckpt}")
                        break
                else:
                    ckpt_path = find_latest_checkpoint(os.path.join(run_dir, "saves"))
                    if ckpt_path:
                        args.ckpt = ckpt_path
                        print(f"[info] Found checkpoint: {args.ckpt}")
                    else:
                        print(f"[error] No checkpoint found in {run_dir}/saves/")
                        sys.exit(1)

        if args.output_dir is None:
            args.output_dir = os.path.join(run_dir, "paper_figures")

    # Validate required args
    if args.config is None:
        print("[error] Must provide either --run-dir or --config")
        sys.exit(1)

    if args.output_dir is None:
        args.output_dir = "./paper_figures"

    # Auto-discover checkpoint from config dir if still not found
    if args.ckpt is None:
        config_dir = os.path.dirname(args.config)
        best_pt = os.path.join(config_dir, "best.pt")
        if os.path.isfile(best_pt):
            args.ckpt = best_pt
            print(f"[info] Auto-discovered checkpoint: {args.ckpt}")
        else:
            print(f"[error] No --ckpt provided and no best.pt found in {config_dir}")
            sys.exit(1)
    elif os.path.isdir(args.ckpt):
        best_pt = os.path.join(args.ckpt, "best.pt")
        if os.path.isfile(best_pt):
            args.ckpt = best_pt
        else:
            ckpt_path = find_latest_checkpoint(args.ckpt)
            if ckpt_path:
                args.ckpt = ckpt_path
            else:
                print(f"[error] No checkpoint found in {args.ckpt}")
                sys.exit(1)

    # Load config
    cfg = get_config(args.config)
    voxel_grid_whd = cfg["voxel_grid_whd"]
    W, H, D = voxel_grid_whd

    # Detect mode from hparams
    mode = cfg.get("mode", "rgb_ae")
    is_rgb = mode.startswith("rgb")
    is_vae = mode.endswith("vae")
    in_ch = 3 if is_rgb else 1
    occ_thresh = cfg.get("occ_thresh", 1e-6)

    print(f"[info] Mode: {mode} (is_rgb={is_rgb}, is_vae={is_vae})")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[info] Using device: {device}")
    print(f"[info] Voxel grid: W={W}, H={H}, D={D}")

    # Load dataset - SAME as visualize_voxel_paper.py for consistent sampling
    if args.voxel_dir is not None:
        dataset = VoxelDirDataset(args.voxel_dir, max_files=args.start_sample + args.max_samples)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    else:
        ds_name = cfg['ds']
        root = cfg['root']
        voxel_mode = cfg.get("voxel_mode", "avg_rgb") if is_rgb else "occupancy"

        base_ds = get_point_cloud_dataset(
            ds_name, root, mode=args.split, max_points=4096,
            include_rgb=is_rgb,
            voxelize=True,
            voxel_grid_whd=tuple(voxel_grid_whd),
            voxel_mode=voxel_mode,
        )
        dataloader = DataLoader(base_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Build model
    base_ch = cfg.get("base_ch", 32)
    latent_dim = cfg.get("latent_dim", 256)

    if is_vae:
        model = VoxelVAE(in_ch=in_ch, out_ch=in_ch, base_ch=base_ch, latent_dim=latent_dim).to(device)
    else:
        model = VoxelAutoencoder(in_ch=in_ch, out_ch=in_ch, base_ch=base_ch, latent_dim=latent_dim).to(device)

    print(f"[info] Model: {'VoxelVAE' if is_vae else 'VoxelAutoencoder'}(in_ch={in_ch}, latent_dim={latent_dim})")

    # Load checkpoint
    print(f"[info] Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)

    state = None
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model", "ema", "ema_state_dict", "net", "weights"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
    if state is None:
        state = ckpt

    def _strip_prefix(d, prefix):
        if not any(k.startswith(prefix) for k in d.keys()):
            return d
        return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in d.items()}

    state = _strip_prefix(state, "module.")
    state = _strip_prefix(state, "model.")
    model.load_state_dict(state, strict=False)
    model.eval()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process samples
    sample_idx = 0
    processed_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Skip samples before start_sample
            if sample_idx < args.start_sample:
                sample_idx += batch["voxels"].shape[0]
                continue

            vox = batch["voxels"].to(device).float()

            # Prepare input based on mode
            if is_rgb:
                x = vox[:, :3, ...]
            else:
                # Occupancy mode
                if vox.shape[1] >= 3:
                    mag = vox[:, :3, ...].abs().mean(dim=1, keepdim=True)
                    x = (mag > occ_thresh).float()
                else:
                    x = vox[:, :1, ...]

            # Forward pass
            if is_vae:
                rec_x, mu, logvar = model(x)
            else:
                rec_x, z = model(x)

            # Process each sample in batch
            for b in range(x.shape[0]):
                if processed_count >= args.max_samples:
                    break

                gt_vol = x[b]
                rec_vol = rec_x[b]

                # Create subfolder for this sample
                sample_output_dir = os.path.join(args.output_dir, f"sample_{sample_idx:02d}")
                os.makedirs(sample_output_dir, exist_ok=True)

                print(f"\n=== Sample {sample_idx} ===")
                print_vol_stats("GT", gt_vol)
                print_vol_stats("REC", rec_vol)

                if is_rgb:
                    # RGB mode: create RGB voxel figures
                    fig_gt = create_rgb_voxel_figure(gt_vol, alpha_thresh=args.alpha_thresh)
                    lock_scene(fig_gt, D, H, W)
                    save_plotly_json(fig_gt, os.path.join(sample_output_dir, "gt_rgb.plotly.json"))

                    fig_rec = create_rgb_voxel_figure(rec_vol, alpha_thresh=args.alpha_thresh)
                    lock_scene(fig_rec, D, H, W)
                    save_plotly_json(fig_rec, os.path.join(sample_output_dir, "rec_rgb.plotly.json"))

                else:
                    # Occupancy mode: create occupancy figures
                    fig_gt = create_occupancy_voxel_figure(gt_vol, alpha_thresh=args.alpha_thresh)
                    lock_scene(fig_gt, D, H, W)
                    save_plotly_json(fig_gt, os.path.join(sample_output_dir, "gt_occ.plotly.json"))

                    fig_rec = create_occupancy_voxel_figure(rec_vol, alpha_thresh=args.alpha_thresh)
                    lock_scene(fig_rec, D, H, W)
                    save_plotly_json(fig_rec, os.path.join(sample_output_dir, "rec_occ.plotly.json"))

                sample_idx += 1
                processed_count += 1

            if processed_count >= args.max_samples:
                break

    print(f"\n[done] Generated visualizations for {processed_count} samples in {args.output_dir}")
    print(f"\nTo render figures, use a script like:")
    print("""
import json
import plotly.graph_objects as go

with open("sample_00/gt_rgb.plotly.json") as f:
    fig_dict = json.load(f)

fig = go.Figure(fig_dict)
# Customize styling as needed
fig.show()
""")


if __name__ == "__main__":
    main()
