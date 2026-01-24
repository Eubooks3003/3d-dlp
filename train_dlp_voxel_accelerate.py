"""
Multi-GPU/Multi-Node training of DLP Voxel using HuggingFace Accelerate.
Works on both x86 and ARM (aarch64) machines.

Usage:
    # Single GPU (default)
    python train_dlp_voxel_accelerate.py -d configs/your_config.json

    # Multi-GPU with accelerate
    accelerate launch --num_processes=4 train_dlp_voxel_accelerate.py -d configs/your_config.json

    # For ARM/aarch64 machines, use gloo backend:
    accelerate launch --num_processes=4 --dynamo_backend=no train_dlp_voxel_accelerate.py -d configs/your_config.json

    # Or set environment variable for gloo:
    ACCELERATE_TORCH_DEVICE=cpu accelerate launch train_dlp_voxel_accelerate.py -d configs/your_config.json
"""
# imports
import os
import platform

# For ARM compatibility: prefer gloo backend if NCCL is problematic
if platform.machine() in ('aarch64', 'arm64'):
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")

import numpy as np
from tqdm import tqdm
import matplotlib
import argparse
# torch
import torch
from utils.loss_functions import calc_reconstruction_loss, LossLPIPS
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.optim as optim
# modules
from voxel_models import DLP
# datasets
from datasets.point_cloud_datasets.get_dataset import get_point_cloud_dataset, pc_collate
from datasets.voxelize_ds_wrapper import VoxelizedDataset
# util functions
from utils.util_func import (plot_keypoints_on_image_batch, prepare_logdir, save_config, log_line,
                             plot_bb_on_image_batch_from_z_scale_nms, plot_bb_on_image_batch_from_masks_nms,
                             create_segmentation_map, get_config, LinearWithWarmupScheduler, format_epoch_summary,
                             plot_training_metrics, save_metrics_data, save_code_backup, depth_to_rgb)
from utils.rgbd_utils import get_depth_range, normalize_rgbd
from utils.log_utils import (save_checkpoint, load_checkpoint, log_block_grads, log_param_updates, plot_grad_flow,
                            topk_indices_from_output, wandb_log_iter_losses)
from eval.eval_model import evaluate_validation_elbo
from eval.eval_gen_metrics import eval_dlp_im_metric
from eval.eval_vox import (log_vox_overlay_plotly, log_vox_isoseries, log_cov_ellipsoids_over_voxels,
                           extract_volumes_for_vis, print_vol_stats, log_voxel_rec_distributions, filter_topk_kps_3d,
                           log_rgb_voxels, evaluate_validation_voxel, plot_loss_curves)
import wandb

# Accelerate imports
from accelerate import Accelerator, DistributedDataParallelKwargs

matplotlib.use("Agg")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


from collections import defaultdict
import time
def cuda_sync():
    """Synchronize CUDA for accurate timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


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

def wandb_log_lossdict(loss_dict: dict, step: int, accelerator=None):
    """Logs all scalar-like entries in loss_dict to W&B under their existing keys."""
    flat = {}
    for k, v in loss_dict.items():
        val = _to_float_safe(v)
        if val is not None:
            flat[k] = val
    if flat and (accelerator is None or accelerator.is_main_process):
        wandb.log(flat, step=step)
    return flat  # return what was logged (as floats) for local use

class EpochAverager:
    """Collect per-iteration floats and compute epoch means for the keys we've actually seen."""
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
    if 'loss_rec' in means: parts.append(f"rec {means['loss_rec']:.4f}")
    if 'kl' in means:       parts.append(f"KL {means['kl']:.4f}")

    bracket = []
    for k,label in [
        ('loss_kl_kp','kp'), ('loss_kl_feat','feat'),
        ('loss_kl_scale','scale'), ('loss_kl_obj_on','obj'),
        ('loss_kl_depth','depth'), ('loss_kl_context','ctx')
    ]:
        if k in means: bracket.append(f"{label} {means[k]:.3f}")
    if bracket: parts.append("[" + ", ".join(bracket) + "]")

    if 'obj_on_l1' in means: parts.append(f"on_L1 {means['obj_on_l1']:.3f}")
    return " | ".join(parts)


def train_dlp_voxel_accelerate(config_path='./configs/shapes.json'):
    # Initialize accelerator
    # find_unused_parameters=False is more efficient but may cause issues if model has unused params
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    # load config
    try:
        config = get_config(config_path)
    except FileNotFoundError:
        raise SystemExit("config file not found")

    modality = config.get('modality', 'image')
    if modality != "point_cloud":
        raise NotImplementedError("This is the training code for point-cloud DLP only. For image DLP, use train_dlp.py")

    hparams = config  # to save a copy of the hyper-parameters

    # data and general
    ds = config['ds']
    ch = config['ch']  # image channels
    image_size = config['image_size']
    root = config['root']  # dataset root

    run_prefix = config['run_prefix']
    load_model = config['load_model']
    pretrained_path = config['pretrained_path']  # path of pretrained model to load, if None, train from scratch

    # model
    pad_mode = config['pad_mode']
    n_kp_per_patch = config['n_kp_per_patch']  # kp per patch in prior, best to leave at 1
    n_kp_prior = config['n_kp_prior']  # number of prior kp to filter for the kl
    n_kp_enc = config['n_kp_enc']  # total posterior kp
    patch_size = config['patch_size']  # prior patch size
    anchor_s = config['anchor_s']  # posterior patch/glimpse ratio of image size

    features_dist = config.get('features_dist', 'gauss')
    learned_feature_dim = config['learned_feature_dim']
    learned_bg_feature_dim = config.get('learned_bg_feature_dim', learned_feature_dim)
    n_fg_categories = config.get('n_fg_categories', 8)
    n_fg_classes = config.get('n_fg_classes', 4)
    n_bg_categories = config.get('n_bg_categories', 4)
    n_bg_classes = config.get('n_bg_classes', 4)

    dropout = config['dropout']
    use_resblock = config['use_resblock']

    pint_enc_layers = config['pint_enc_layers']
    pint_enc_heads = config['pint_enc_heads']

    normalize_rgb = config['normalize_rgb']
    obj_res_from_fc = config["obj_res_from_fc"]
    obj_ch_mult = config["obj_ch_mult"]
    obj_ch_mult_prior = config.get("obj_ch_mult_prior", obj_ch_mult)
    obj_base_ch = config["obj_base_ch"]
    obj_final_cnn_ch = config["obj_final_cnn_ch"]
    bg_res_from_fc = config["bg_res_from_fc"]
    bg_ch_mult = config["bg_ch_mult"]
    bg_base_ch = config["bg_base_ch"]
    bg_final_cnn_ch = config["bg_final_cnn_ch"]
    num_res_blocks = config["num_res_blocks"]
    cnn_mid_blocks = config.get('cnn_mid_blocks', False)
    mlp_hidden_dim = config.get('mlp_hidden_dim', 256)

    # optimization
    batch_size = config['batch_size']
    lr = config['lr']
    num_epochs = config['num_epochs']
    start_epoch = config.get('start_epoch', 0)
    weight_decay = config['weight_decay']
    adam_betas = config['adam_betas']
    adam_eps = config['adam_eps']
    use_scheduler = config['use_scheduler']
    scheduler_gamma = config['scheduler_gamma']
    warmup_epoch = config['warmup_epoch']
    recon_loss_type = config['recon_loss_type']
    beta_kl = config['beta_kl']
    beta_rec = config['beta_rec']
    beta_obj = config.get('beta_obj', 0.0)
    lambda_color = config['lambda_color']
    kl_balance = config['kl_balance']

    # priors
    scale_std = config['scale_std']
    offset_std = config['offset_std']
    obj_on_alpha = config['obj_on_alpha']
    obj_on_beta = config['obj_on_beta']

    # evaluation
    eval_epoch_freq = config['eval_epoch_freq']
    eval_im_metrics = config['eval_im_metrics']

    # visualization
    topk = min(config['topk'], config['n_kp_enc'])
    iou_thresh = config['iou_thresh']

    # RGBD Stuff
    separate_depth_features = config["separate_depth_features"]
    depth_feature_dim = config["depth_feature_dim"]
    split_loss = config["split_loss"]
    depth_loss_ratio = config["depth_loss_ratio"]

    # Point Cloud Stuff
    decoder_point_mode = config["decoder_point_mode"]

    # Voxel Stuff
    voxel_mode = config["voxel_mode"]
    voxel_grid_whd = config["voxel_grid_whd"]
    voxel_root = config.get("voxel_root", None)

    # MimicGen specific
    task = config.get("task", None)  # task name for mimicgen_voxel
    max_demos = config.get("max_demos", None)  # limit number of demos
    cache_suffix = config.get("cache_suffix", "")  # e.g., "_debug" for voxel_cache_debug

    # Prior mode configuration
    prior_mode = config.get("prior_mode", "kmeans")  # "kmeans" | "ssm_raw" | "ssm_enc"
    raw_heatmap_mode = config.get("raw_heatmap_mode", "luma")  # "luma" | "rgb_norm" | "channel0" | "learned_1x1"

    # Dataset
    dataset = get_point_cloud_dataset(
        ds, root, mode="train",
        voxelize=True, voxel_grid_whd=voxel_grid_whd,
        voxel_mode="occupancy",
        cache_dir=voxel_root,
        task=task,
        max_demos=max_demos,
        cache_suffix=cache_suffix,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=8, pin_memory=True,
                            persistent_workers=True, prefetch_factor=4)

    # Validation dataset
    val_dataset = get_point_cloud_dataset(
        ds, root, mode="val",
        voxelize=True, voxel_grid_whd=voxel_grid_whd,
        voxel_mode="occupancy",
        cache_dir=voxel_root,
        task=task,
        max_demos=max_demos,
        cache_suffix=cache_suffix,
    )

    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )


    # Model
    model = DLP(
        cdim=ch,
        image_size=voxel_grid_whd[0],
        normalize_rgb=normalize_rgb,
        n_kp_per_patch=n_kp_per_patch,
        patch_size=patch_size,
        anchor_s=anchor_s,
        n_kp_enc=n_kp_enc,
        n_kp_prior=n_kp_prior,
        pad_mode=pad_mode,
        dropout=dropout,
        features_dist=features_dist,
        learned_feature_dim=learned_feature_dim,
        learned_bg_feature_dim=learned_bg_feature_dim,
        n_fg_categories=n_fg_categories,
        n_fg_classes=n_fg_classes,
        n_bg_categories=n_bg_categories,
        n_bg_classes=n_bg_classes,
        scale_std=scale_std,
        offset_std=offset_std,
        obj_on_alpha=obj_on_alpha,
        obj_on_beta=obj_on_beta,
        obj_res_from_fc=obj_res_from_fc,
        obj_ch_mult_prior=obj_ch_mult_prior,
        obj_ch_mult=obj_ch_mult,
        obj_base_ch=obj_base_ch,
        obj_final_cnn_ch=obj_final_cnn_ch,
        bg_res_from_fc=bg_res_from_fc,
        bg_ch_mult=bg_ch_mult,
        bg_base_ch=bg_base_ch,
        bg_final_cnn_ch=bg_final_cnn_ch,
        use_resblock=use_resblock,
        num_res_blocks=num_res_blocks,
        cnn_mid_blocks=cnn_mid_blocks,
        mlp_hidden_dim=mlp_hidden_dim,
        pint_enc_layers=pint_enc_layers,
        pint_enc_heads=pint_enc_heads,
        timestep_horizon=1,
        separate_depth_features=separate_depth_features,
        depth_feature_dim=depth_feature_dim,
        split_loss=split_loss,
        depth_loss_ratio=depth_loss_ratio,
        # Prior mode configuration
        prior_mode=prior_mode,
        raw_heatmap_mode=raw_heatmap_mode,
    )


    model_info = model.info()
    if accelerator.is_main_process:
        print(model_info)

    # Prepare saving location
    run_name = run_prefix
    log_dir = prepare_logdir(runname=run_name, src_dir='./logs', accelerator=accelerator)
    fig_dir = os.path.join(log_dir, 'figures')
    save_dir = os.path.join(log_dir, 'saves')

    # Checkpoint config: ONLY SAVE BEST
    monitor = config.get("monitor", "loss")
    mode = config.get("monitor_mode", "min")
    assert mode in ("min", "max")
    best_val = float("inf") if mode == "min" else -float("inf")

    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        save_config(log_dir, hparams)
        log_line(log_dir, model_info)
        backup_info = save_code_backup('.', backup_dir=os.path.join(log_dir, 'saves', 'code_backup'))
        log_line(log_dir, backup_info)
        print(backup_info)

    ckpt_best = os.path.join(save_dir, "best.pt")

    # Get the range of the keypoints
    kp_range = model.kp_range

    # Prepare loss functions
    if recon_loss_type == "vgg":
        recon_loss_func = LossLPIPS(normalized_rgb=normalize_rgb).to(accelerator.device)
    else:
        recon_loss_func = calc_reconstruction_loss

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=adam_betas, eps=adam_eps, weight_decay=weight_decay)

    # Prepare model, optimizer, dataloaders with accelerator
    model, optimizer, dataloader, val_dataloader = accelerator.prepare(model, optimizer, dataloader, val_dataloader)

    # Scheduler (after accelerator.prepare for optimizer)
    if use_scheduler:
        scheduler = LinearWithWarmupScheduler(optimizer, gamma=scheduler_gamma, verbose=False,
                                              steps=(max(warmup_epoch, 1), max(warmup_epoch, 1) + 1),
                                              factors=(1.0, 1.0, 1.0 * scheduler_gamma))
    else:
        scheduler = None

    # Optional resume / preload
    start_epoch = int(config.get('start_epoch', 0))
    if load_model and pretrained_path is not None:
        try:
            unwrapped_model = accelerator.unwrap_model(model)
            resume_info = load_checkpoint(pretrained_path, unwrapped_model, optimizer, scheduler,
                                         map_location=accelerator.device)
            if start_epoch == 0 and resume_info.get("epoch", 0) > 0:
                start_epoch = int(resume_info["epoch"] + 1)
            best_val = resume_info.get("best_metric", best_val)
            accelerator.print(f"[ckpt] Resumed training from epoch {start_epoch}, best={best_val:.6f}")
        except Exception as e:
            accelerator.print(f"[ckpt] Resume skipped: {e}")

    # Initialize validation statistics
    valid_loss = best_valid_loss = 1e8
    valid_losses = []
    best_valid_epoch = 0

    # Training/validation loss tracking for loss curves
    train_losses = []
    train_losses_rec = []
    train_losses_kl = []
    val_losses = []
    val_losses_rec = []
    val_losses_kl = []

    # Image metrics
    if eval_im_metrics:
        val_lpipss = []
        best_val_lpips_epoch = 0
        val_lpips = best_val_lpips = 1e8
    else:
        best_val_lpips_epoch = None
        val_lpips = best_val_lpips = None

    # Iteration counter
    iteration = 0

    # Initialize wandb only on main process
    if accelerator.is_main_process:
        wandb.init(
            name=run_name,
            config=config,
            resume="never",
        )

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_avg = EpochAverager()

        pbar = tqdm(iterable=dataloader, disable=not accelerator.is_local_main_process)
        for it, batch in enumerate(pbar):
            t0 = time.time()

            # Get batch (already on CPU from dataloader)
            vox_cpu = batch["voxels"]
            cuda_sync()
            t1 = time.time()

            # Host to device transfer + channels_last_3d for better perf
            vox = vox_cpu.to(accelerator.device, non_blocking=True)
            vox = vox.contiguous(memory_format=torch.channels_last_3d)
            cuda_sync()
            t2 = time.time()

            warmup = (epoch < warmup_epoch)

            # Forward pass
            model_output = model(vox, warmup=warmup, with_loss=True,
                                 beta_kl=beta_kl,
                                 beta_rec=beta_rec, kl_balance=kl_balance,
                                 recon_loss_type=recon_loss_type,
                                 recon_loss_func=recon_loss_func,
                                 beta_obj=beta_obj, lambda_color=lambda_color)
            loss = model_output['loss_dict']['loss']
            cuda_sync()
            t3 = time.time()

            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            cuda_sync()
            t4 = time.time()

            # Optimizer step
            optimizer.step()
            cuda_sync()
            t5 = time.time()

            # Print timing for first 10 iterations of first epoch
            if accelerator.is_main_process and epoch == start_epoch and it < 10:
                print(f"iter {it:03d} | "
                      f"get_batch {t1-t0:.3f}s | "
                      f"H2D {t2-t1:.3f}s | "
                      f"fwd {t3-t2:.3f}s | "
                      f"bwd {t4-t3:.3f}s | "
                      f"step {t5-t4:.3f}s | "
                      f"total {t5-t0:.3f}s")

            # Collect losses for epoch averaging (no per-iter wandb logging)
            all_losses = model_output['loss_dict']
            logged = {k: _to_float_safe(v) for k, v in all_losses.items()}
            epoch_avg.add(logged)

            # Progress bar
            def pick(key, default=0.0):
                return logged.get(key, default)

            if epoch < warmup_epoch:
                pbar.set_description_str(f'epoch #{epoch} (warmup)')
            else:
                pbar.set_description_str(f'epoch #{epoch}')

            pbar.set_postfix(
                loss=pick('loss'),
                rec=pick('loss_rec'),
                KL=pick('kl'),
            )

            iteration += 1

        pbar.close()

        # End of epoch
        means = epoch_avg.means()

        # Pretty print (robust to missing keys)
        log_str = build_epoch_log(epoch, means)
        accelerator.print(log_str)

        # Store epoch losses for loss curves
        train_losses.append(means.get('loss', 0.0))
        train_losses_rec.append(means.get('loss_rec', 0.0))
        train_losses_kl.append(means.get('kl', 0.0))

        if accelerator.is_main_process:
            log_line(log_dir, log_str)
            wandb.log({**{f"epoch/{k}": v for k, v in means.items()},
                "epoch_idx": epoch}, step=iteration)

        # Scheduler step
        if use_scheduler and scheduler is not None:
            scheduler.step()

        # Decide monitored metric
        monitor_map = {
            "loss": "loss",
            "rec": "loss_rec",
            "kl": "kl",
            "vox/psnr_c0": "vox/psnr_c0",
        }
        mon_key = monitor_map.get(monitor, "loss")
        monitored = means.get(mon_key, means.get("loss", None))

        # Guard against NaN/Inf
        if monitored is not None and (not np.isfinite(monitored)):
            accelerator.print(f"[warn] monitored metric {mon_key} is non-finite ({monitored}); skipping model selection this epoch.")
            monitored = None

        # Save "best" if improved (only main process)
        if monitored is not None and accelerator.is_main_process:
            improved = (monitored < best_val) if mode == "min" else (monitored > best_val)
            if improved:
                best_val = monitored
                unwrapped_model = accelerator.unwrap_model(model)
                save_checkpoint(ckpt_best, unwrapped_model, optimizer, scheduler, epoch, best_val,
                                extra={"monitored": monitored, "best_update": True})
                print(f"[ckpt] New best ({mon_key}={monitored:.6f}) at epoch {epoch:04d} -> saved best.pt")

        # Synchronize before eval
        accelerator.wait_for_everyone()

        # ------- VALIDATION -------
        if epoch % eval_epoch_freq == 0 or epoch == num_epochs - 1:
            accelerator.print(f"\n[Validation] Running validation at epoch {epoch}...")

            # Run validation (use unwrapped model for eval)
            unwrapped_model = accelerator.unwrap_model(model)
            val_results = evaluate_validation_voxel(
                model=unwrapped_model,
                val_dataloader=val_dataloader,
                device=accelerator.device,
                config=config,
                epoch=epoch,
                recon_loss_func=recon_loss_func,
                beta_kl=beta_kl,
                beta_rec=beta_rec,
                beta_obj=beta_obj,
                kl_balance=kl_balance,
                lambda_color=lambda_color,
                recon_loss_type=recon_loss_type,
                fig_dir=fig_dir,
                save_visuals=accelerator.is_main_process,
                topk=topk,
                iteration=iteration,
                accelerator=accelerator,
            )

            # Store validation losses
            val_losses.append(val_results.get('val_loss', 0.0))
            val_losses_rec.append(val_results.get('val_loss_rec', 0.0))
            val_losses_kl.append(val_results.get('val_kl', 0.0))

            # Log validation metrics - main metrics: Occ IoU + Masked Color PSNR
            val_log_str = f"[Val] epoch {epoch:04d}"
            if 'occ_iou' in val_results:
                val_log_str += f" | Occ IoU: {val_results['occ_iou']:.4f}"
            if 'masked_color_psnr' in val_results:
                val_log_str += f" | Masked PSNR: {val_results['masked_color_psnr']:.2f} dB"
            val_log_str += f" | val_loss: {val_results.get('val_loss', 0.0):.4f}"
            if 'val_loss_rec' in val_results:
                val_log_str += f" | val_rec: {val_results['val_loss_rec']:.4f}"
            if 'val_kl' in val_results:
                val_log_str += f" | val_kl: {val_results['val_kl']:.4f}"
            accelerator.print(val_log_str)

            if accelerator.is_main_process:
                log_line(log_dir, val_log_str)
                wandb.log({f"val/{k}": v for k, v in val_results.items()}, step=iteration)

                # Plot and save loss curves
                loss_curve_path = os.path.join(fig_dir, f"loss_curves_epoch{epoch:04d}.png")
                plot_loss_curves(
                    train_losses=train_losses,
                    val_losses=val_losses,
                    train_losses_rec=train_losses_rec,
                    val_losses_rec=val_losses_rec,
                    train_losses_kl=train_losses_kl,
                    val_losses_kl=val_losses_kl,
                    save_path=loss_curve_path,
                    title=f"Training Progress - Epoch {epoch}",
                )
                wandb.log({"loss_curves": wandb.Image(loss_curve_path)}, step=iteration)

            model.train()  # Back to training mode
            torch.cuda.empty_cache()  # Clear memory fragmentation after validation

        # TRAIN VISUALS (voxel version) - only main process
        if (epoch % eval_epoch_freq == 0 or epoch == num_epochs - 1) and accelerator.is_main_process:
            b0 = 0

            gt_vol = model_output['x'][b0]
            rec_vol = model_output['rec'][b0]

            print("gt vol: ", gt_vol.shape)

            with torch.no_grad():
                out = filter_topk_kps_3d(
                    z_base_var=model_output["z_base_var"],
                    mu_tot=model_output["z_base"] + model_output["mu_offset"],
                    topk=config['topk'],
                    obj_on=model_output.get("obj_on", None),
                    use_posterior_in_score=False
                )
                indices = out["indices"]
                topk_kp = out["topk_kp"]
                bb_scores = out["bb_scores"]

            topk_kp_b0 = topk_kp[b0]
            cov_b0 = model_output["cov_kp"][b0]
            kp_xyz = model_output["kp_p"]
            z_base_cov_b0 = model_output["z_base_cov"][b0]

            print("z base: ", model_output["z_base"].shape)
            z_base_b0 = model_output["z_base"][b0]
            mu_tot_b0 = z_base_b0 + model_output["mu_offset"][b0]

            print("mu tot: ", mu_tot_b0.shape)
            print("GT VOL: ", gt_vol.shape)

            # GT LOGGING
            log_rgb_voxels(
                name="gt/rgb_splat_kp",
                rgb_vol=gt_vol,
                alpha_vol=None,
                KPx=mu_tot_b0,
                step=iteration,
                mode="splat",
                topk=60000,
                alpha_thresh=0.05,
                pad=2.0,
                show_axes=True,
            )

            log_rgb_voxels(
                name="gt/rgb_splat_no_offset",
                rgb_vol=gt_vol,
                alpha_vol=None,
                KPx=z_base_b0,
                step=iteration,
                mode="splat",
                topk=60000,
                alpha_thresh=0.05,
                pad=2.0,
                show_axes=True,
            )

            log_rgb_voxels(
                name="gt/rgb_splat",
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

            # REC LOGGING
            log_rgb_voxels(
                name="rec/rgb_splat_kp",
                rgb_vol=rec_vol,
                alpha_vol=None,
                KPx=mu_tot_b0,
                step=iteration,
                mode="splat",
                topk=60000,
                alpha_thresh=0.05,
                pad=2.0,
                show_axes=True,
            )

            log_rgb_voxels(
                name="rec/rgb_splat",
                rgb_vol=rec_vol,
                alpha_vol=None,
                step=iteration,
                mode="splat",
                topk=60000,
                alpha_thresh=0.05,
                pad=2.0,
                show_axes=True,
            )

            log_rgb_voxels(
                name="rec/rgb_splat_no_offset",
                rgb_vol=rec_vol,
                alpha_vol=None,
                KPx=z_base_b0,
                step=iteration,
                mode="splat",
                topk=60000,
                alpha_thresh=0.05,
                pad=2.0,
                show_axes=True,
            )

            # Log foreground and background separately
            fg_vol = model_output.get('rgb_obj')
            bg_vol = model_output.get('bg_rgb')

            if fg_vol is not None:
                fg_vol_b0 = fg_vol[b0]
                log_rgb_voxels(
                    name="rec/fg_splat",
                    rgb_vol=fg_vol_b0,
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
                    name="rec/fg_splat_kp",
                    rgb_vol=fg_vol_b0,
                    alpha_vol=None,
                    KPx=mu_tot_b0,
                    step=iteration,
                    mode="splat",
                    topk=60000,
                    alpha_thresh=0.05,
                    pad=2.0,
                    show_axes=True,
                )

            if bg_vol is not None:
                bg_vol_b0 = bg_vol[b0]
                log_rgb_voxels(
                    name="rec/bg_splat",
                    rgb_vol=bg_vol_b0,
                    alpha_vol=None,
                    KPx=None,
                    step=iteration,
                    mode="splat",
                    topk=60000,
                    alpha_thresh=0.05,
                    pad=2.0,
                    show_axes=True,
                )

        accelerator.wait_for_everyone()

    # Finish wandb
    if accelerator.is_main_process:
        wandb.finish()

    # Return unwrapped model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    return unwrapped_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DLP Voxel Multi-GPU Training with Accelerate")
    parser.add_argument("-d", "--dataset", type=str, default='shapes',
                        help="dataset or config path (e.g., configs/your_config.json)")
    args = parser.parse_args()
    ds = args.dataset

    if ds.endswith('json'):
        conf_path = ds
    else:
        conf_path = os.path.join('./configs', f'{ds}.json')

    train_dlp_voxel_accelerate(conf_path)
