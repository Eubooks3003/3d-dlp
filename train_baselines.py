#!/usr/bin/env python3
"""
Unified baseline training for voxel autoencoders.

4 modes:
  - rgb_ae:   RGB voxel autoencoder (3 channels, reconstruction loss only)
  - rgb_vae:  RGB voxel VAE (3 channels, reconstruction + KL divergence)
  - occ_ae:   Occupancy voxel autoencoder (1 channel, reconstruction loss only)
  - occ_vae:  Occupancy voxel VAE (1 channel, reconstruction + KL divergence)

Usage:
  python train_baselines.py --mode rgb_ae -d shapes
  python train_baselines.py --mode rgb_vae -d shapes --kl_weight 0.001
  python train_baselines.py --mode occ_ae -d shapes
  python train_baselines.py --mode occ_vae -d shapes --kl_weight 0.001
"""

import os
import argparse
from collections import defaultdict

import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import wandb

from datasets.point_cloud_datasets.get_dataset import get_point_cloud_dataset, pc_collate
from datasets.voxelize_ds_wrapper import VoxelizedDataset
from utils.util_func import (
    get_config, prepare_logdir, save_config, log_line,
    LinearWithWarmupScheduler, save_code_backup,
)
from utils.log_utils import save_checkpoint
from eval.eval_vox import log_rgb_voxels

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# ============================================================================
# Models
# ============================================================================

class VoxelEncoder(nn.Module):
    """Encoder for voxel grids. Works for any number of input channels."""
    def __init__(self, in_ch=3, base_ch=32, latent_dim=256, is_vae=False):
        super().__init__()
        self.is_vae = is_vae

        # [B,in_ch,64,64,64] -> [B,32,32,32,32]
        self.conv1 = nn.Conv3d(in_ch, base_ch, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(base_ch)

        # [B,32,32,32,32] -> [B,64,16,16,16]
        self.conv2 = nn.Conv3d(base_ch, base_ch*2, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(base_ch*2)

        # [B,64,16,16,16] -> [B,128,8,8,8]
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
# Losses
# ============================================================================

def recon_loss(x, rec_x, loss_type="mse", occ=None, fg_weight=1.0, bg_weight=1.0):
    """Reconstruction loss with optional occupancy weighting."""
    assert x.shape == rec_x.shape, f"shape mismatch: x {x.shape}, rec_x {rec_x.shape}"

    if loss_type == "mse":
        err = (rec_x - x) ** 2
    elif loss_type == "l1":
        err = (rec_x - x).abs()
    elif loss_type == "bce":
        # Binary cross-entropy for occupancy
        err = F.binary_cross_entropy(rec_x, x, reduction='none')
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    if occ is not None:
        fg = (occ > 0.5).float()
        bg = 1.0 - fg
        w = fg_weight * fg + bg_weight * bg
        w = w.expand_as(err)
        err = err * w
        loss = err.sum() / w.sum().clamp_min(1.0)
    else:
        loss = err.mean()

    # PSNR (for logging)
    with torch.no_grad():
        mse_per_batch = ((rec_x - x) ** 2).view(x.shape[0], -1).mean(dim=1)
        psnr_per_batch = -10.0 * torch.log10(mse_per_batch + 1e-8)
        psnr = psnr_per_batch.mean()

    return loss, psnr


def kl_divergence(mu, logvar):
    """KL divergence loss for VAE: KL(q(z|x) || p(z)) where p(z) = N(0,I)."""
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl.mean()


# ============================================================================
# Training helpers
# ============================================================================

def _to_float_safe(x, default=None):
    if x is None:
        return default
    try:
        if isinstance(x, (float, int)):
            return float(x)
        if hasattr(x, "detach"):
            return float(x.detach().mean().item())
        return float(x)
    except Exception:
        return default


def wandb_log_lossdict(loss_dict: dict, step: int):
    flat = {}
    for k, v in loss_dict.items():
        val = _to_float_safe(v)
        if val is not None:
            flat[k] = val
    if flat:
        wandb.log(flat, step=step)
    return flat


class EpochAverager:
    def __init__(self):
        self.store = defaultdict(list)

    def add(self, logged_loss_floats: dict):
        for k, v in logged_loss_floats.items():
            if v is not None:
                self.store[k].append(float(v))

    def means(self):
        return {k: float(np.mean(v)) for k, v in self.store.items() if len(v) > 0}


def build_epoch_log(epoch, means, mode):
    parts = [f"epoch {epoch:04d}", f"[{mode}]", f"loss {means.get('loss', 0.0):.4f}"]
    if 'loss_rec' in means:
        parts.append(f"rec {means['loss_rec']:.4f}")
    if 'loss_kl' in means:
        parts.append(f"kl {means['loss_kl']:.4f}")
    if 'psnr' in means:
        parts.append(f"psnr {means['psnr']:.2f} dB")
    return " | ".join(parts)


# ============================================================================
# Main training
# ============================================================================

def train_baseline(config_path, mode, kl_weight=0.001):
    """
    Train a baseline voxel autoencoder.

    Args:
        config_path: Path to config JSON
        mode: One of 'rgb_ae', 'rgb_vae', 'occ_ae', 'occ_vae'
        kl_weight: Weight for KL divergence (VAE modes only)
    """
    # Parse mode
    is_rgb = mode.startswith("rgb")
    is_vae = mode.endswith("vae")
    in_ch = 3 if is_rgb else 1

    # Config
    try:
        config = get_config(config_path)
    except FileNotFoundError:
        raise SystemExit("config file not found")

    hparams = config
    hparams["mode"] = mode
    hparams["kl_weight"] = kl_weight if is_vae else 0.0

    ds = config["ds"]
    root = config["root"]
    voxel_grid_whd = config["voxel_grid_whd"]

    # For occupancy mode, we need to change voxel_mode
    if is_rgb:
        voxel_mode = config.get("voxel_mode", "avg_rgb")
    else:
        voxel_mode = "occupancy"
        hparams["voxel_mode"] = voxel_mode

    device_str = config.get("device", "cuda:0")
    if "cuda" in device_str:
        device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # Optimization params
    batch_size = config.get("batch_size", 4)
    lr = config.get("lr", 1e-3)
    num_epochs = config.get("num_epochs", 100)
    start_epoch = int(config.get("start_epoch", 0))
    weight_decay = config.get("weight_decay", 0.0)
    adam_betas = tuple(config.get("adam_betas", (0.9, 0.999)))
    adam_eps = config.get("adam_eps", 1e-8)
    use_scheduler = config.get("use_scheduler", False)
    scheduler_gamma = config.get("scheduler_gamma", 0.99)
    warmup_epoch = config.get("warmup_epoch", 0)
    eval_epoch_freq = config.get("eval_epoch_freq", 10)

    # Loss params
    occ_thresh = config.get("occ_thresh", 1e-6)
    fg_weight = config.get("fg_weight", 5.0)
    bg_weight = config.get("bg_weight", 0.1)

    # Use BCE for occupancy, MSE for RGB
    loss_type = "bce" if not is_rgb else "mse"

    # W&B / checkpoint
    run_prefix = config.get("run_prefix", "")
    monitor = config.get("monitor", "loss")
    monitor_mode = config.get("monitor_mode", "min")
    assert monitor_mode in ("min", "max")
    best_val = float("inf") if monitor_mode == "min" else -float("inf")

    run_name = f"{ds}_{mode}{run_prefix}"
    log_dir = prepare_logdir(runname=run_name, src_dir="./logs")
    fig_dir = os.path.join(log_dir, "figures")
    save_dir = os.path.join(log_dir, "saves")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    ckpt_best = os.path.join(save_dir, "best.pt")

    save_config(log_dir, hparams)
    backup_info = save_code_backup(".", backup_dir=os.path.join(save_dir, "code_backup"))
    log_line(log_dir, backup_info)
    print(backup_info)

    # Dataset
    base_ds = get_point_cloud_dataset(
        ds,
        root,
        mode="train",
        max_points=4096,
        include_rgb=is_rgb,
    )

    dataloader = DataLoader(base_ds, batch_size=batch_size, shuffle=True, num_workers=4)

    # Model
    base_ch = config.get("base_ch", 32)
    latent_dim = config.get("latent_dim", 256)

    if is_vae:
        model = VoxelVAE(in_ch=in_ch, out_ch=in_ch, base_ch=base_ch, latent_dim=latent_dim).to(device)
        model_info = f"VoxelVAE(in_ch={in_ch}, latent_dim={latent_dim}, grid={voxel_grid_whd})"
    else:
        model = VoxelAutoencoder(in_ch=in_ch, out_ch=in_ch, base_ch=base_ch, latent_dim=latent_dim).to(device)
        model_info = f"VoxelAutoencoder(in_ch={in_ch}, latent_dim={latent_dim}, grid={voxel_grid_whd})"

    log_line(log_dir, model_info)
    print(model_info)
    print(f"Mode: {mode} | is_rgb={is_rgb} | is_vae={is_vae} | in_ch={in_ch}")

    # Optimizer / scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=adam_betas, eps=adam_eps, weight_decay=weight_decay)
    if use_scheduler:
        scheduler = LinearWithWarmupScheduler(
            optimizer, gamma=scheduler_gamma, verbose=False,
            steps=(max(warmup_epoch, 1), max(warmup_epoch, 1) + 1),
            factors=(1.0, 1.0, 1.0 * scheduler_gamma),
        )
    else:
        scheduler = None

    # W&B
    wandb.init(
        name=run_name,
        config=hparams,
        resume="never",
    )

    iteration = 0

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_avg = EpochAverager()
        pbar = tqdm(dataloader)

        for batch_idx, batch in enumerate(pbar):
            vox = batch["voxels"].to(device).float()

            if is_rgb:
                x = vox[:, :3, ...]  # RGB channels
            else:
                # For occupancy: compute occupancy from RGB magnitude or use first channel
                if vox.shape[1] >= 3:
                    # Compute occupancy from RGB
                    mag = vox[:, :3, ...].abs().mean(dim=1, keepdim=True)
                    x = (mag > occ_thresh).float()
                else:
                    x = vox[:, :1, ...]

            optimizer.zero_grad()

            if is_vae:
                rec_x, mu, logvar = model(x)
                loss_rec, psnr = recon_loss(x, rec_x, loss_type=loss_type)
                loss_kl = kl_divergence(mu, logvar)
                loss = loss_rec + kl_weight * loss_kl

                loss_dict = {
                    "loss": loss,
                    "loss_rec": loss_rec,
                    "loss_kl": loss_kl,
                    "psnr": psnr,
                }
            else:
                rec_x, z = model(x)
                loss_rec, psnr = recon_loss(x, rec_x, loss_type=loss_type)
                loss = loss_rec

                loss_dict = {
                    "loss": loss,
                    "loss_rec": loss_rec,
                    "psnr": psnr,
                }

            loss.backward()
            optimizer.step()

            # Log
            logged = wandb_log_lossdict(loss_dict, iteration)
            epoch_avg.add(logged)

            pbar.set_description_str(f"epoch #{epoch} [{mode}]")
            pbar.set_postfix(
                loss=logged.get("loss", 0.0),
                rec=logged.get("loss_rec", 0.0),
                psnr=logged.get("psnr", 0.0),
            )

            iteration += 1

        pbar.close()

        # End of epoch logging
        means = epoch_avg.means()
        log_str = build_epoch_log(epoch, means, mode)
        print(log_str)
        log_line(log_dir, log_str)
        wandb.log({f"epoch/{k}": v for k, v in means.items()}, step=iteration)
        wandb.log({"epoch_idx": epoch}, step=iteration)

        # Monitor for best checkpoint
        monitor_map = {"loss": "loss", "rec": "loss_rec", "psnr": "psnr"}
        mon_key = monitor_map.get(monitor, "loss")
        monitored = means.get(mon_key, means.get("loss", None))

        if monitored is not None and not np.isfinite(monitored):
            print(f"[warn] monitored metric {mon_key} is non-finite ({monitored}); skipping model selection this epoch.")
            monitored = None

        if monitored is not None:
            improved = (monitored < best_val) if monitor_mode == "min" else (monitored > best_val)
            if improved:
                best_val = monitored
                save_checkpoint(
                    ckpt_best,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_val,
                    extra={"monitored": monitored, "best_update": True, "mode": mode},
                )
                print(f"[ckpt] New best ({mon_key}={monitored:.6f}) at epoch {epoch:04d} -> saved best.pt")

        # Visualization
        if epoch % eval_epoch_freq == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                batch_vis = next(iter(dataloader))
                vox_vis = batch_vis["voxels"].to(device).float()

                if is_rgb:
                    x_vis = vox_vis[:, :3, ...]
                else:
                    if vox_vis.shape[1] >= 3:
                        mag = vox_vis[:, :3, ...].abs().mean(dim=1, keepdim=True)
                        x_vis = (mag > occ_thresh).float()
                    else:
                        x_vis = vox_vis[:, :1, ...]

                if is_vae:
                    rec_vis, _, _ = model(x_vis)
                else:
                    rec_vis, _ = model(x_vis)

                b0 = 0
                gt_vol = x_vis[b0].detach().cpu()
                rec_vol = rec_vis[b0].detach().cpu()

                # For visualization, expand occupancy to 3 channels
                if not is_rgb:
                    gt_vol = gt_vol.repeat(3, 1, 1, 1)
                    rec_vol = rec_vol.repeat(3, 1, 1, 1)

                log_rgb_voxels(
                    name=f"{mode}/gt",
                    rgb_vol=gt_vol,
                    alpha_vol=None,
                    KPx=None,
                    step=iteration,
                    mode="splat",
                    topk=60000,
                    alpha_thresh=0.05,
                    pad=2.0,
                    show_axes=True,
                )
                log_rgb_voxels(
                    name=f"{mode}/rec",
                    rgb_vol=rec_vol,
                    alpha_vol=None,
                    KPx=None,
                    step=iteration,
                    mode="splat",
                    topk=60000,
                    alpha_thresh=0.05,
                    pad=2.0,
                    show_axes=True,
                )

        if scheduler is not None:
            scheduler.step()

    wandb.finish()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Baseline Training for Voxel AE/VAE")
    parser.add_argument(
        "-d", "--dataset", type=str, default="shapes",
        help="Dataset name or path to config JSON"
    )
    parser.add_argument(
        "--mode", type=str, default="rgb_ae",
        choices=["rgb_ae", "rgb_vae", "occ_ae", "occ_vae"],
        help="Training mode: rgb_ae, rgb_vae, occ_ae, occ_vae"
    )
    parser.add_argument(
        "--kl_weight", type=float, default=0.001,
        help="KL divergence weight for VAE modes (default: 0.001)"
    )

    args = parser.parse_args()

    ds = args.dataset
    if ds.endswith("json"):
        conf_path = ds
    else:
        conf_path = os.path.join("./configs", f"{ds}.json")

    train_baseline(conf_path, mode=args.mode, kl_weight=args.kl_weight)
