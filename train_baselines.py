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
  python train_baselines.py --mode rgb_vae -d shapes --kl_weight 1e-5
  python train_baselines.py --mode occ_ae -d shapes
  python train_baselines.py --mode occ_vae -d shapes --kl_weight 1e-5
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
from eval.eval_vox import (
    log_rgb_voxels, plot_loss_curves,
    compute_occupancy_iou, compute_masked_color_psnr,
)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# ============================================================================
# Models
# ============================================================================

class ResBlock3d(nn.Module):
    """Residual block for 3D convolutions."""
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv3d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(ch)
        self.conv2 = nn.Conv3d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(ch)

    def forward(self, x):
        residual = x
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = self.bn2(self.conv2(x))
        return F.leaky_relu(x + residual, 0.2)


class VoxelEncoder(nn.Module):
    """Encoder for voxel grids. Works for any number of input channels."""
    def __init__(self, in_ch=3, base_ch=32, latent_dim=512, is_vae=False):
        super().__init__()
        self.is_vae = is_vae

        # [B,in_ch,64,64,64] -> [B,base_ch,32,32,32]
        self.down1 = nn.Sequential(
            nn.Conv3d(in_ch, base_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(base_ch),
            nn.LeakyReLU(0.2),
            ResBlock3d(base_ch),
        )
        # [B,base_ch,32] -> [B,base_ch*2,16]
        self.down2 = nn.Sequential(
            nn.Conv3d(base_ch, base_ch * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(base_ch * 2),
            nn.LeakyReLU(0.2),
            ResBlock3d(base_ch * 2),
        )
        # [B,base_ch*2,16] -> [B,base_ch*4,8]
        self.down3 = nn.Sequential(
            nn.Conv3d(base_ch * 2, base_ch * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(base_ch * 4),
            nn.LeakyReLU(0.2),
            ResBlock3d(base_ch * 4),
        )
        # [B,base_ch*4,8] -> [B,base_ch*8,4]
        self.down4 = nn.Sequential(
            nn.Conv3d(base_ch * 4, base_ch * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(base_ch * 8),
            nn.LeakyReLU(0.2),
            ResBlock3d(base_ch * 8),
        )

        self.flatten_dim = base_ch * 8 * 4 * 4 * 4
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)

        if is_vae:
            self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        if self.is_vae:
            logvar = self.fc_logvar(x)
            return mu, logvar
        return mu


class VoxelDecoder(nn.Module):
    """Decoder for voxel grids. Works for any number of output channels."""
    def __init__(self, out_ch=3, base_ch=32, latent_dim=512, use_sigmoid=True):
        super().__init__()
        self.use_sigmoid = use_sigmoid

        self.start_ch = base_ch * 8
        self.fc = nn.Linear(latent_dim, self.start_ch * 4 * 4 * 4)

        # [B,base_ch*8,4] -> [B,base_ch*4,8]
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(self.start_ch, base_ch * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(base_ch * 4),
            nn.LeakyReLU(0.2),
            ResBlock3d(base_ch * 4),
        )
        # [B,base_ch*4,8] -> [B,base_ch*2,16]
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(base_ch * 4, base_ch * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(base_ch * 2),
            nn.LeakyReLU(0.2),
            ResBlock3d(base_ch * 2),
        )
        # [B,base_ch*2,16] -> [B,base_ch,32]
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(base_ch * 2, base_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(base_ch),
            nn.LeakyReLU(0.2),
            ResBlock3d(base_ch),
        )
        # [B,base_ch,32] -> [B,out_ch,64]
        self.up4 = nn.ConvTranspose3d(base_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        x = self.fc(z).view(z.size(0), self.start_ch, 4, 4, 4)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        if self.use_sigmoid:
            x = torch.sigmoid(x)
        return x


class VoxelAutoencoder(nn.Module):
    """Voxel Autoencoder (deterministic)."""
    def __init__(self, in_ch=3, out_ch=3, base_ch=32, latent_dim=512, use_sigmoid=True):
        super().__init__()
        self.encoder = VoxelEncoder(in_ch=in_ch, base_ch=base_ch, latent_dim=latent_dim, is_vae=False)
        self.decoder = VoxelDecoder(out_ch=out_ch, base_ch=base_ch, latent_dim=latent_dim, use_sigmoid=use_sigmoid)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z


class VoxelVAE(nn.Module):
    """Voxel Variational Autoencoder."""
    def __init__(self, in_ch=3, out_ch=3, base_ch=32, latent_dim=512, use_sigmoid=True):
        super().__init__()
        self.encoder = VoxelEncoder(in_ch=in_ch, base_ch=base_ch, latent_dim=latent_dim, is_vae=True)
        self.decoder = VoxelDecoder(out_ch=out_ch, base_ch=base_ch, latent_dim=latent_dim, use_sigmoid=use_sigmoid)

    def reparameterize(self, mu, logvar):
        logvar = logvar.clamp(-20.0, 10.0)  # prevent exp() overflow
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

def train_baseline(config_path, mode, kl_weight=1e-5, kl_warmup_epochs=10):
    """
    Train a baseline voxel autoencoder.

    Args:
        config_path: Path to config JSON
        mode: One of 'rgb_ae', 'rgb_vae', 'occ_ae', 'occ_vae'
        kl_weight: Weight for KL divergence (VAE modes only)
        kl_warmup_epochs: Epochs to linearly ramp KL weight from 0 to kl_weight (VAE only)
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

    # Force correct voxel_mode based on training mode (config may have DLP-specific value)
    if is_rgb:
        voxel_mode = "avg_rgb"
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
    lambda_color = config.get("baseline_lambda_color", 5.0)  # chroma loss weight (RGB only, per-voxel scale)

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

    # Dataset - wrap with VoxelizedDataset to get voxels
    base_ds = get_point_cloud_dataset(
        ds,
        root,
        mode="train",
        max_points=4096,
        include_rgb=is_rgb,
        voxelize=True,
        voxel_grid_whd=tuple(voxel_grid_whd),
        voxel_mode=voxel_mode,
    )

    dataloader = DataLoader(base_ds, batch_size=batch_size, shuffle=True, num_workers=4)

    # Validation dataset - wrap with VoxelizedDataset to get voxels
    try:
        val_ds = get_point_cloud_dataset(
            ds,
            root,
            mode="val",
            max_points=4096,
            include_rgb=is_rgb,
            voxelize=True,
            voxel_grid_whd=tuple(voxel_grid_whd),
            voxel_mode=voxel_mode,
        )
        val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4) if len(val_ds) > 0 else None
        val_ds_len = len(val_ds) if val_ds else 0
    except Exception as e:
        print(f"[Warning] Could not load validation dataset: {e}")
        val_ds = None
        val_dataloader = None
        val_ds_len = 0
    print(f"Train dataset: {len(base_ds)} samples, Val dataset: {val_ds_len} samples")

    # Model
    base_ch = config.get("base_ch", 32)
    latent_dim = config.get("latent_dim", 512)

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

    # Loss tracking for curves
    losses = []
    losses_rec = []
    losses_kl = []
    val_losses = []
    val_losses_rec = []
    val_losses_kl = []

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
            else:
                rec_x, z = model(x)

            # For RGB: occupancy-weighted MSE + foreground chroma loss (matches DLP)
            if is_rgb:
                occ_mask = (x.abs().mean(dim=1, keepdim=True) > occ_thresh).float()
                loss_rec, psnr = recon_loss(x, rec_x, loss_type=loss_type,
                                            occ=occ_mask, fg_weight=fg_weight, bg_weight=bg_weight)
                # Chroma loss: penalize wrong hue/saturation on foreground voxels
                L_gt = x.mean(dim=1, keepdim=True)        # luminance [B,1,D,H,W]
                L_rec = rec_x.mean(dim=1, keepdim=True)
                C_gt = x - L_gt                            # chrominance [B,3,D,H,W]
                C_rec = rec_x - L_rec
                fg = occ_mask.expand_as(C_gt)
                chroma_err = (C_rec - C_gt) ** 2 * fg
                # Normalize by foreground count (same per-voxel scale as recon_loss)
                loss_chroma = chroma_err.sum() / fg.sum().clamp_min(1.0)
                loss_rec = loss_rec + lambda_color * loss_chroma
            else:
                loss_rec, psnr = recon_loss(x, rec_x, loss_type=loss_type)
                loss_chroma = None

            if is_vae:
                # Clamp logvar before computing KL to prevent exp() overflow
                logvar_clamped = logvar.clamp(-20.0, 10.0)
                loss_kl = kl_divergence(mu, logvar_clamped)
                # Linear KL warmup to prevent posterior collapse
                kl_frac = min(1.0, epoch / max(kl_warmup_epochs, 1))
                if kl_frac > 0:
                    loss = loss_rec + (kl_weight * kl_frac) * loss_kl
                else:
                    loss = loss_rec  # epoch 0: pure reconstruction, avoids 0*NaN

                loss_dict = {
                    "loss": loss,
                    "loss_rec": loss_rec,
                    "loss_kl": loss_kl.detach(),
                    "psnr": psnr,
                    "kl_frac": kl_frac,
                }
            else:
                loss = loss_rec

                loss_dict = {
                    "loss": loss,
                    "loss_rec": loss_rec,
                    "psnr": psnr,
                }

            if loss_chroma is not None:
                loss_dict["loss_chroma"] = loss_chroma.detach()

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

        # Store epoch losses for loss curves
        losses.append(means.get('loss', 0.0))
        losses_rec.append(means.get('loss_rec', 0.0))
        losses_kl.append(means.get('loss_kl', 0.0))

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

        # ------- VALIDATION -------
        if (epoch % eval_epoch_freq == 0 or epoch == num_epochs - 1) and val_dataloader is not None:
            print(f"\n[Validation] Running validation at epoch {epoch}...")
            model.eval()

            val_epoch_losses = []
            val_epoch_losses_rec = []
            val_epoch_losses_kl = []
            val_psnrs = []

            # Main comparison metrics (same as train_dlp_voxel.py)
            occ_ious = []
            masked_psnrs = []

            with torch.no_grad():
                for val_batch in val_dataloader:
                    vox = val_batch["voxels"].to(device).float()

                    if is_rgb:
                        x_val = vox[:, :3, ...]
                    else:
                        if vox.shape[1] >= 3:
                            mag = vox[:, :3, ...].abs().mean(dim=1, keepdim=True)
                            x_val = (mag > occ_thresh).float()
                        else:
                            x_val = vox[:, :1, ...]

                    if is_vae:
                        rec_val, mu_val, logvar_val = model(x_val)
                    else:
                        rec_val, z_val = model(x_val)

                    # Use same occupancy-weighted + chroma loss as training for RGB
                    if is_rgb:
                        occ_mask_val = (x_val.abs().mean(dim=1, keepdim=True) > occ_thresh).float()
                        loss_rec_val, psnr_val = recon_loss(x_val, rec_val, loss_type=loss_type,
                                                            occ=occ_mask_val, fg_weight=fg_weight, bg_weight=bg_weight)
                        L_gt_v = x_val.mean(dim=1, keepdim=True)
                        L_rec_v = rec_val.mean(dim=1, keepdim=True)
                        C_gt_v = x_val - L_gt_v
                        C_rec_v = rec_val - L_rec_v
                        fg_v = occ_mask_val.expand_as(C_gt_v)
                        chroma_err_v = (C_rec_v - C_gt_v) ** 2 * fg_v
                        loss_rec_val = loss_rec_val + lambda_color * chroma_err_v.sum() / fg_v.sum().clamp_min(1.0)
                    else:
                        loss_rec_val, psnr_val = recon_loss(x_val, rec_val, loss_type=loss_type)

                    if is_vae:
                        logvar_val_clamped = logvar_val.clamp(-20.0, 10.0)
                        loss_kl_val = kl_divergence(mu_val, logvar_val_clamped)
                        loss_val = loss_rec_val + kl_weight * loss_kl_val
                        val_epoch_losses_kl.append(float(loss_kl_val.item()))
                    else:
                        loss_val = loss_rec_val

                    val_epoch_losses.append(float(loss_val.item()))
                    val_epoch_losses_rec.append(float(loss_rec_val.item()))
                    val_psnrs.append(float(psnr_val.item()))

                    # Compute main comparison metrics: Occ IoU + Masked Color PSNR
                    # (same metrics as train_dlp_voxel.py for fair comparison)
                    if is_rgb:
                        B = x_val.shape[0]
                        for b in range(B):
                            iou = compute_occupancy_iou(x_val[b], rec_val[b], occ_thresh=occ_thresh)
                            masked_psnr = compute_masked_color_psnr(x_val[b], rec_val[b], occ_thresh=occ_thresh)
                            occ_ious.append(iou)
                            if np.isfinite(masked_psnr):
                                masked_psnrs.append(masked_psnr)

            # Compute mean validation metrics
            val_loss_mean = float(np.mean(val_epoch_losses))
            val_loss_rec_mean = float(np.mean(val_epoch_losses_rec))
            val_psnr_mean = float(np.mean(val_psnrs))

            val_losses.append(val_loss_mean)
            val_losses_rec.append(val_loss_rec_mean)

            # Compute main comparison metrics
            occ_iou_mean = float(np.mean(occ_ious)) if occ_ious else None
            masked_psnr_mean = float(np.mean(masked_psnrs)) if masked_psnrs else None

            # Log validation results - main metrics first (Occ IoU + Masked PSNR)
            val_log_str = f"[Val] epoch {epoch:04d}"
            if occ_iou_mean is not None:
                val_log_str += f" | Occ IoU: {occ_iou_mean:.4f}"
            if masked_psnr_mean is not None:
                val_log_str += f" | Masked PSNR: {masked_psnr_mean:.2f} dB"
            val_log_str += f" | loss: {val_loss_mean:.4f} | rec: {val_loss_rec_mean:.4f} | psnr: {val_psnr_mean:.2f} dB"
            if is_vae and val_epoch_losses_kl:
                val_loss_kl_mean = float(np.mean(val_epoch_losses_kl))
                val_losses_kl.append(val_loss_kl_mean)
                val_log_str += f" | kl: {val_loss_kl_mean:.4f}"
            print(val_log_str)
            log_line(log_dir, val_log_str)

            # Log to wandb
            val_wandb_dict = {
                "val/loss": val_loss_mean,
                "val/loss_rec": val_loss_rec_mean,
                "val/psnr": val_psnr_mean,
            }
            # Main comparison metrics
            if occ_iou_mean is not None:
                val_wandb_dict["val/occ_iou"] = occ_iou_mean
            if masked_psnr_mean is not None:
                val_wandb_dict["val/masked_color_psnr"] = masked_psnr_mean
            wandb.log(val_wandb_dict, step=iteration)
            if is_vae and val_epoch_losses_kl:
                wandb.log({"val/loss_kl": val_loss_kl_mean}, step=iteration)

            # Plot and save loss curves
            loss_curve_path = os.path.join(fig_dir, f"loss_curves_epoch{epoch:04d}.png")
            plot_loss_curves(
                train_losses=losses,
                val_losses=val_losses,
                train_losses_rec=losses_rec,
                val_losses_rec=val_losses_rec,
                train_losses_kl=losses_kl if is_vae else None,
                val_losses_kl=val_losses_kl if is_vae else None,
                save_path=loss_curve_path,
                title=f"Training Progress - {mode} - Epoch {epoch}",
            )
            wandb.log({"loss_curves": wandb.Image(loss_curve_path)}, step=iteration)

            model.train()

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
        "--kl_weight", type=float, default=1e-5,
        help="KL divergence weight for VAE modes (default: 1e-5)"
    )
    parser.add_argument(
        "--kl_warmup_epochs", type=int, default=10,
        help="Epochs to linearly ramp KL weight from 0 (default: 10)"
    )

    args = parser.parse_args()

    ds = args.dataset
    if ds.endswith("json"):
        conf_path = ds
    else:
        conf_path = os.path.join("./configs", f"{ds}.json")

    train_baseline(conf_path, mode=args.mode, kl_weight=args.kl_weight, kl_warmup_epochs=args.kl_warmup_epochs)
