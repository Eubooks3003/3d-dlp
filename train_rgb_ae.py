# train_voxel_ae.py

"""
Single-GPU training of a simple voxel RGB autoencoder
"""

import os
import argparse
from collections import defaultdict

import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")

import torch
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

# our AE + loss
from rgb_autoencoder import VoxelRGBAutoencoder, voxel_rgb_recon_loss

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# ---------- small helpers (reused from your training style) ----------

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


def build_epoch_log(epoch, means):
    parts = [f"epoch {epoch:04d}", f"loss {means.get('loss', 0.0):.4f}"]
    if 'loss_rec' in means:
        parts.append(f"rec {means['loss_rec']:.4f}")
    if 'psnr' in means:
        parts.append(f"psnr {means['psnr']:.2f} dB")
    return " | ".join(parts)


# ---------- main training ----------

def train_voxel_ae(config_path="./configs/shapes.json"):
    # ---- config ----
    try:
        config = get_config(config_path)
    except FileNotFoundError:
        raise SystemExit("config file not found")

    hparams = config

    ds = config["ds"]
    root = config["root"]
    ch = config.get("ch", 3)               # expected voxel channels (RGB=3)
    voxel_grid_whd = config["voxel_grid_whd"]
    voxel_mode = config.get("voxel_mode", "avg_rgb")  # we assume this yields RGB in channels 0:3

    device_str = config.get("device", "cuda:0")
    if "cuda" in device_str:
        device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # optimization
    batch_size   = config.get("batch_size", 4)
    lr           = config.get("lr", 1e-3)
    num_epochs   = config.get("num_epochs", 100)
    start_epoch  = int(config.get("start_epoch", 0))
    weight_decay = config.get("weight_decay", 0.0)
    adam_betas   = tuple(config.get("adam_betas", (0.9, 0.999)))
    adam_eps     = config.get("adam_eps", 1e-8)
    use_scheduler= config.get("use_scheduler", False)
    scheduler_gamma = config.get("scheduler_gamma", 0.99)
    warmup_epoch    = config.get("warmup_epoch", 0)

    eval_epoch_freq = config.get("eval_epoch_freq", 10)

    # W&B / checkpoint
    run_prefix = config.get("run_prefix", "")
    monitor = config.get("monitor", "loss")
    mode    = config.get("monitor_mode", "min")
    assert mode in ("min", "max")
    best_val = float("inf") if mode == "min" else -float("inf")

    run_name = f"{ds}_voxelAE{run_prefix}"
    log_dir  = prepare_logdir(runname=run_name, src_dir="./logs")
    fig_dir  = os.path.join(log_dir, "figures")
    save_dir = os.path.join(log_dir, "saves")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    ckpt_best = os.path.join(save_dir, "best.pt")

    save_config(log_dir, hparams)
    backup_info = save_code_backup(".", backup_dir=os.path.join(save_dir, "code_backup"))
    log_line(log_dir, backup_info)
    print(backup_info)

    # ---- dataset + dataloader ----
    # base dataset is point-cloud; VoxelizedDataset wraps it into voxel grids
    base_ds = get_point_cloud_dataset(
        ds,
        root,
        mode="train",
        max_points=4096,
        include_rgb=True,  # we want RGB info
    )

    # vox_ds = VoxelizedDataset(
    #     base_ds=base_ds,
    #     grid_whd=voxel_grid_whd,
    #     mode=voxel_mode,           # set to "avg_rgb" in config for RGB voxels
    #     bounds_mode="global",
    #     keep_points=False,
    #     device=torch.device("cpu"),
    #     cache_dir=config.get("voxel_cache_dir", None),
    #     force_rebuild=config.get("voxel_force_rebuild", False),
    # )

    dataloader = DataLoader(base_ds, batch_size=batch_size, shuffle=True, num_workers=4)

    # ---- model ----
    ae_in_ch = ch
    model = VoxelRGBAutoencoder(in_ch=ae_in_ch, out_ch=ae_in_ch, base_ch=32, latent_dim=256).to(device)

    model_info = f"VoxelRGBAutoencoder(in_ch={ae_in_ch}, latent_dim=256, grid={voxel_grid_whd})"
    log_line(log_dir, model_info)
    print(model_info)

    # ---- optimizer / scheduler ----
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=adam_betas, eps=adam_eps, weight_decay=weight_decay)
    if use_scheduler:
        scheduler = LinearWithWarmupScheduler(
            optimizer, gamma=scheduler_gamma, verbose=False,
            steps=(max(warmup_epoch, 1), max(warmup_epoch, 1) + 1),
            factors=(1.0, 1.0, 1.0 * scheduler_gamma),
        )
    else:
        scheduler = None

    # ---- wandb ----
    wandb.init(
        name=run_name,
        config=config,
        resume="never",
    )

    epoch_avg = EpochAverager()
    iteration = 0

    # ---------- training loop ----------
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_avg = EpochAverager()
        pbar = tqdm(dataloader)

        for batch in pbar:
            # batch["voxels"]: [B,C,D,H,W]
            vox = batch["voxels"].to(device).float()
            # take first C=ch channels as RGB (configure ch=3 for RGB-only)
            x = vox[:, :ae_in_ch, ...]   # [B,C,D,H,W]

            optimizer.zero_grad()
            rec_x = model(x)            # [B,C,D,H,W]

            # if you also have occupancy: occ = batch.get("occ", None)
            occ = None
            loss, loss_dict = voxel_rgb_recon_loss(
                x, rec_x,
                loss_type="mse",
                occ=occ,
                fg_weight=1.0,
                bg_weight=1.0,
            )

            loss.backward()
            optimizer.step()

            # log per-iter
            logged = wandb_log_lossdict(loss_dict, iteration)
            epoch_avg.add(logged)

            pbar.set_description_str(f"epoch #{epoch}")
            pbar.set_postfix(
                loss=logged.get("loss", 0.0),
                rec=logged.get("loss_rec", 0.0),
                psnr=logged.get("psnr", 0.0),
            )

            iteration += 1

        pbar.close()

        # ---- end of epoch logging ----
        means = epoch_avg.means()
        log_str = build_epoch_log(epoch, means)
        print(log_str)
        log_line(log_dir, log_str)
        wandb.log({f"epoch/{k}": v for k, v in means.items()}, step=iteration)
        wandb.log({"epoch_idx": epoch}, step=iteration)

        # ---- monitor metric for "best" checkpoint ----
        monitor_map = {
            "loss": "loss",
            "rec": "loss_rec",
            "psnr": "psnr",
        }
        mon_key = monitor_map.get(monitor, "loss")
        monitored = means.get(mon_key, means.get("loss", None))

        if monitored is not None and not np.isfinite(monitored):
            print(f"[warn] monitored metric {mon_key} is non-finite ({monitored}); skipping model selection this epoch.")
            monitored = None

        if monitored is not None:
            improved = (monitored < best_val) if mode == "min" else (monitored > best_val)
            if improved:
                best_val = monitored
                save_checkpoint(
                    ckpt_best,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_val,
                    extra={"monitored": monitored, "best_update": True},
                )
                print(f"[ckpt] New best ({mon_key}={monitored:.6f}) at epoch {epoch:04d} -> saved best.pt")

        # ---- optional eval / visualization ----
        if epoch % eval_epoch_freq == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                batch = next(iter(dataloader))
                vox = batch["voxels"].to(device).float()
                x = vox[:, :ae_in_ch, ...]
                rec_x = model(x)

                b0 = 0
                gt_vol  = x[b0].detach().cpu()      # [C,D,H,W]
                rec_vol = rec_x[b0].detach().cpu()  # [C,D,H,W]

                log_rgb_voxels(
                    name="ae/gt_rgb",
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
                    name="ae/rec_rgb",
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
    parser = argparse.ArgumentParser(description="Voxel RGB AE Single-GPU Training")
    parser.add_argument(
        "-d", "--dataset", type=str, default="shapes",
        help="dataset to train the model on"
    )
    args = parser.parse_args()
    ds = args.dataset
    if ds.endswith("json"):
        conf_path = ds
    else:
        conf_path = os.path.join("./configs", f"{ds}.json")
    train_voxel_ae(conf_path)
