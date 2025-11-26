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

    # ---- NEW: masking & weights ----
    occ_thresh      = config.get("occ_thresh", 1e-6)    # threshold on |RGB| to treat voxel as occupied
    fg_weight       = config.get("fg_weight", 5.0)      # fg weight inside occ region
    bg_weight       = config.get("bg_weight", 0.1)      # bg weight outside occ region
    lambda_masked   = config.get("lambda_masked", 1.0)  # weight on masked recon term

    # ---- NEW: chroma loss & debugging flags ----
    lambda_chroma   = config.get("lambda_chroma", 1.0)  # weight on chroma-only loss
    overfit_one     = config.get("overfit_one", False)  # if True, train on a single fixed batch
    debug_stats     = config.get("debug_stats", True)   # log per-channel stats
    debug_every     = config.get("debug_every", 500)    # how often to log stats (iters)

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
    base_ds = get_point_cloud_dataset(
        ds,
        root,
        mode="train",
        max_points=4096,
        include_rgb=True,  # we want RGB info
    )

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

    # ---- optional: fixed batch for overfitting experiment ----
    fixed_batch = None
    if overfit_one:
        fixed_batch = next(iter(dataloader))
        print("[AE] overfit_one=True -> will train on a single fixed batch")

    iteration = 0

    # ---------- training loop ----------
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_avg = EpochAverager()
        pbar = tqdm(dataloader)

        for batch_idx, batch in enumerate(pbar):
            if overfit_one and fixed_batch is not None:
                batch = fixed_batch  # ignore dataloader batch, always reuse the same one

            vox = batch["voxels"].to(device).float()
            # take first C=ch channels as RGB (configure ch=3 for RGB-only)
            x = vox[:, :ae_in_ch, ...]   # [B,C,D,H,W]

            # ---- occupancy mask from GT ----
            mag = x.abs().mean(dim=1, keepdim=True)         # [B,1,D,H,W]
            occ = (mag > occ_thresh).float()                # [B,1,D,H,W]

            optimizer.zero_grad()
            rec_x = model(x)            # [B,C,D,H,W]

            # ---- 1) plain (unmasked) recon loss ----
            loss_plain, loss_dict_plain = voxel_rgb_recon_loss(
                x, rec_x,
                loss_type="mse",
                occ=None,          # no mask here
                fg_weight=1.0,
                bg_weight=1.0,
            )

            # ---- 2) GT-derived occupancy-masked loss (same as before) ----
            with torch.no_grad():
                mag_dbg = x.abs().mean(dim=1, keepdim=True)      # [B,1,D,H,W]
                occ_dbg = (mag_dbg > occ_thresh).float()         # [B,1,D,H,W]

            loss_masked, loss_dict_masked = voxel_rgb_recon_loss(
                x, rec_x,
                loss_type="mse",
                occ=occ_dbg,              # use GT occupancy to reweight foreground
                fg_weight=fg_weight,
                bg_weight=bg_weight,
            )

            # ---- 3) chroma loss *only on foreground* ----
            # luminance
            L_gt = x.mean(dim=1, keepdim=True)         # [B,1,D,H,W]
            L_pr = rec_x.mean(dim=1, keepdim=True)     # [B,1,D,H,W]
            # chroma
            C_gt = x - L_gt                            # [B,C,D,H,W]
            C_pr = rec_x - L_pr                        # [B,C,D,H,W]

            # foreground mask
            fg_mask = occ_dbg                          # [B,1,D,H,W]
            fg_chroma = fg_mask.expand_as(C_gt)        # [B,C,D,H,W]

            chroma_sq = (C_pr - C_gt) ** 2 * fg_chroma
            # normalize by number of fg voxels so scale is stable
            fg_norm = fg_chroma.sum().clamp_min(1.0)
            loss_chroma = chroma_sq.sum() / fg_norm

            # # ---- 4) optional BG-zero penalty: discourage noisy background ----
            # bg_mask = 1.0 - occ_dbg                    # [B,1,D,H,W]
            # bg_rgb  = rec_x * bg_mask                  # [B,C,D,H,W]
            # bg_norm = bg_mask.expand_as(rec_x).sum().clamp_min(1.0)
            # loss_bg_zero = (bg_rgb ** 2).sum() / bg_norm

            # ---- 5) combine losses ----
            loss = (
                loss_plain
                + lambda_masked * loss_masked
                + lambda_chroma * loss_chroma

            )

            # merged logging dict
            loss_dict = {
                "loss": loss,
                "loss_plain": loss_plain,
                "loss_masked": loss_masked,
                "loss_chroma": loss_chroma,
                "loss_rec": loss_plain,                      # keep 'loss_rec' ≈ original semantics
                "psnr": loss_dict_plain.get("psnr", None),
                "psnr_masked": loss_dict_masked.get("psnr", None),
            }

            loss.backward()
            optimizer.step()

            # ---- per-channel debug stats ----
            if debug_stats and (iteration < 10 or iteration % debug_every == 0):
                with torch.no_grad():
                    B_cur = x.shape[0]
                    # flatten spatial dims
                    x_flat = x.view(B_cur, ae_in_ch, -1)
                    rec_flat = rec_x.view(B_cur, ae_in_ch, -1)

                    x_mean = x_flat.mean(dim=-1)      # [B,C]
                    x_std  = x_flat.std(dim=-1)       # [B,C]
                    r_mean = rec_flat.mean(dim=-1)    # [B,C]
                    r_std  = rec_flat.std(dim=-1)     # [B,C]

                    # just log batch-mean over samples
                    x_mean_b = x_mean.mean(dim=0)     # [C]
                    x_std_b  = x_std.mean(dim=0)
                    r_mean_b = r_mean.mean(dim=0)
                    r_std_b  = r_std.mean(dim=0)

                    debug_log = {}
                    for c in range(ae_in_ch):
                        debug_log[f"x/ch_mean_c{c}"] = float(x_mean_b[c].item())
                        debug_log[f"x/ch_std_c{c}"]  = float(x_std_b[c].item())
                        debug_log[f"rec/ch_mean_c{c}"] = float(r_mean_b[c].item())
                        debug_log[f"rec/ch_std_c{c}"]  = float(r_std_b[c].item())

                    debug_log["occ/mean"] = float(occ_dbg.mean().item())

                    wandb.log(debug_log, step=iteration)

                    # also print once in a while
                    print(f"[debug] iter {iteration} channel stats:")
                    print("  x_mean:", [round(v, 4) for v in x_mean_b.tolist()])
                    print("  x_std :", [round(v, 4) for v in x_std_b.tolist()])
                    print("  rec_mean:", [round(v, 4) for v in r_mean_b.tolist()])
                    print("  rec_std :", [round(v, 4) for v in r_std_b.tolist()])

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

            # if overfitting one batch, we don't need to iterate whole loader
            if overfit_one:
                # just do a few steps per epoch to see the trend
                if batch_idx >= 9:   # e.g., 10 iters/epoch
                    break

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
                batch_vis = next(iter(dataloader))
                vox_vis = batch_vis["voxels"].to(device).float()
                x_vis = vox_vis[:, :ae_in_ch, ...]
                rec_vis = model(x_vis)

                b0 = 0
                gt_vol  = x_vis[b0].detach().cpu()      # [C,D,H,W]
                rec_vol = rec_vis[b0].detach().cpu()    # [C,D,H,W]

                mag_vis = x_vis.abs().mean(dim=1, keepdim=True)   # [B,1,D,H,W]
                occ_vis = (mag_vis > occ_thresh).float()
                occ_vol = occ_vis[b0, 0].detach().cpu()           # [D,H,W]
                occ_rgb = occ_vol.unsqueeze(0).repeat(3, 1, 1, 1) # [3,D,H,W]

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
                log_rgb_voxels(
                    name="ae/occ_mask",
                    rgb_vol=occ_rgb,
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
