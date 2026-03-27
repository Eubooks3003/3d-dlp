"""
Single-GPU training of DLPv2
"""
import argparse
from collections import defaultdict
import os

import matplotlib
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from voxel_models import DLP
from datasets.point_cloud_datasets.get_dataset import get_point_cloud_dataset
from utils.loss_functions import calc_reconstruction_loss, LossLPIPS
from utils.util_func import (prepare_logdir, save_config, log_line, get_config,
                             LinearWithWarmupScheduler, save_code_backup)
from utils.log_utils import save_checkpoint, load_checkpoint
from eval.eval_vox import log_rgb_voxels, evaluate_validation_voxel, plot_loss_curves

matplotlib.use("Agg")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


@torch.no_grad()
def precompute_kmeans_cache(dataset, prior_module, cache_dir=None):
    """
    Iterate over dataset on CPU, run the DLPPrior's kmeans for each sample,
    and save the results to disk.

    Path resolution (in priority order):
      1. dataset.get_kmeans_path(idx) — per-task/demo/frame paths (mimicgen)
      2. cache_dir/{idx:06d}_kmeans.pt  — flat-index fallback (other datasets)

    Skips samples whose cache file already exists.
    Returns the number of newly computed samples.
    """
    use_method = hasattr(dataset, "get_kmeans_path")
    if not use_method and cache_dir is None:
        return 0
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device_orig = next(prior_module.parameters()).device
    prior_module = prior_module.cpu().eval()
    n_new = 0
    for idx in tqdm(range(len(dataset)), desc="Precomputing kmeans cache"):
        if use_method:
            out_path = dataset.get_kmeans_path(idx)
        else:
            out_path = os.path.join(cache_dir, f"{idx:06d}_kmeans.pt")

        if out_path is None or os.path.exists(out_path):
            continue

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sample = dataset[idx]
        x = sample["voxels"].unsqueeze(0).float()  # [1, C, D, H, W]
        kp, cov = prior_module.encode_prior(x)      # [1, K, 3], [1, K, 3, 3]
        torch.save({"kp": kp[0].cpu(), "cov": cov[0].cpu()}, out_path)
        n_new += 1
    prior_module.to(device_orig)
    return n_new


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


class EpochAverager:
    """Collect per-iteration floats and compute epoch means for the keys we’ve actually seen."""
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


def train_dlp_pc(config_path='./configs/shapes.json'):
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
    root = config['root']  # dataset root

    run_prefix = config['run_prefix']
    load_model = config['load_model']
    pretrained_path = config['pretrained_path']  # path of pretrained model to load, if None, train from scratch

    device = config['device']
    if 'cuda' in device:
        device = torch.device(f'{device}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
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
    n_fg_categories = config.get('n_fg_categories', 8)  # Number of foreground feature categories (if categorical)
    n_fg_classes = config.get('n_fg_classes', 4)  # Number of foreground feature classes per category
    n_bg_categories = config.get('n_bg_categories', 4)  # Number of background feature categories
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
    kl_balance = config['kl_balance']  # balance between visual features and the other particle attributes

    # priors
    scale_std = config['scale_std']
    offset_std = config['offset_std']
    obj_on_alpha = config['obj_on_alpha']  # transparency beta distribution "a"
    obj_on_beta = config['obj_on_beta']  # transparency beta distribution "b"

    # evaluation
    eval_epoch_freq = 1

    # visualization
    topk = min(config['topk'], config['n_kp_enc'])  # top-k particles to plot

    #RGBD Stuff
    separate_depth_features = config["separate_depth_features"]  # use separate depth feature encoding
    depth_feature_dim = config["depth_feature_dim"]  # depth feature dimension if separate encoding
    split_loss = config["split_loss"]  # split loss into components for logging
    depth_loss_ratio = config["depth_loss_ratio"]  # weight of depth loss if split_loss is True

    # Voxel Stuff
    voxel_grid_whd = config["voxel_grid_whd"]
    voxel_root = config.get("voxel_root", None)

    # MimicGen specific
    task = config.get("task", None)  # task name for mimicgen_voxel
    tasks = config.get("tasks", None)  # task list for mimicgen_multitask_voxel
    task_cache_suffixes = config.get("task_cache_suffixes", None)
    max_demos = config.get("max_demos", None)  # limit number of demos
    cache_suffix = config.get("cache_suffix", "")  # e.g., "_debug" for voxel_cache_debug
    proportion = config.get("proportion", 1.0)

    # -------------------------------------------------
    # KMEANS CACHING
    # -------------------------------------------------
    precompute_kmeans = config.get("precompute_kmeans", False)
    # Flat-index fallback for non-mimicgen datasets (VoxelizedDataset, RealWorldVoxelDataset)
    kmeans_cache_base = config.get("kmeans_cache_dir", None)
    if kmeans_cache_base is None and voxel_root is not None and not precompute_kmeans:
        kmeans_cache_base = os.path.join(voxel_root, "kmeans_cache")
    kmeans_train_dir = os.path.join(kmeans_cache_base, "train") if kmeans_cache_base else None
    kmeans_val_dir = os.path.join(kmeans_cache_base, "val") if kmeans_cache_base else None

    dataset = get_point_cloud_dataset(
        ds, root, mode="train",
        voxelize=True, voxel_grid_whd=voxel_grid_whd,
        voxel_mode="occupancy",
        cache_dir=voxel_root,
        task=task,
        tasks=tasks,
        task_cache_suffixes=task_cache_suffixes,
        max_demos=max_demos,
        cache_suffix=cache_suffix,
        proportion=proportion,
        precompute_kmeans=precompute_kmeans,
        kmeans_cache_dir=kmeans_train_dir,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Validation dataset
    val_dataset = get_point_cloud_dataset(
        ds, root, mode="val",
        voxelize=True, voxel_grid_whd=voxel_grid_whd,
        voxel_mode="occupancy",
        cache_dir=voxel_root,
        task=task,
        tasks=tasks,
        task_cache_suffixes=task_cache_suffixes,
        max_demos=max_demos,
        cache_suffix=cache_suffix,
        proportion=proportion,
        precompute_kmeans=precompute_kmeans,
        kmeans_cache_dir=kmeans_val_dir,
    )

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # model

    model = DLP(
        cdim=ch,  # Number of input image channels
        image_size=voxel_grid_whd[0],  # TODO: this is a jank way of dealing with sizes
        normalize_rgb=normalize_rgb,  # If True, normalize RGB to [-1, 1], else keep [0, 1]

        # Keypoint and patch configuration
        n_kp_per_patch=n_kp_per_patch,  # Number of proposal/prior keypoints to extract per patch
        patch_size=patch_size,  # Size of patches for keypoint proposal network
        anchor_s=anchor_s,  # Glimpse size ratio relative to image size
        n_kp_enc=n_kp_enc,  # Number of posterior keypoints to learn
        n_kp_prior=n_kp_prior,  # Number of keypoints to filter from prior proposals

        # Network configuration
        pad_mode=pad_mode,  # Padding mode for CNNs ('zeros' or 'replicate')
        dropout=dropout,  # Dropout rate for transformers

        # Feature representation
        features_dist=features_dist,  # Distribution type for features ('gauss' or 'categorical')
        learned_feature_dim=learned_feature_dim,  # Dimension of learned visual features
        learned_bg_feature_dim=learned_bg_feature_dim,
        # Background feature dimension (if None, equals learned_feature_dim)
        n_fg_categories=n_fg_categories,  # Number of foreground feature categories (if categorical)
        n_fg_classes=n_fg_classes,  # Number of foreground feature classes per category
        n_bg_categories=n_bg_categories,  # Number of background feature categories
        n_bg_classes=n_bg_classes,  # Number of background feature classes per category

        # Prior distributions parameters
        scale_std=scale_std,  # Prior standard deviation for scale
        offset_std=offset_std,  # Prior standard deviation for offset
        obj_on_alpha=obj_on_alpha,  # Alpha parameter for transparency Beta distribution
        obj_on_beta=obj_on_beta,  # Beta parameter for transparency Beta distribution

        # Object decoder architecture
        obj_res_from_fc=obj_res_from_fc,  # Initial resolution for object encoder-decoder
        obj_ch_mult_prior=obj_ch_mult_prior,  # Channel multipliers for prior patch encoder (kp proposals)
        obj_ch_mult=obj_ch_mult,  # Channel multipliers for object encoder-decoder
        obj_base_ch=obj_base_ch,  # Base channels for object encoder-decoder
        obj_final_cnn_ch=obj_final_cnn_ch,  # Final CNN channels for object encoder-decoder

        # Background decoder architecture
        bg_res_from_fc=bg_res_from_fc,  # Initial resolution for background encoder-decoder
        bg_ch_mult=bg_ch_mult,  # Channel multipliers for background encoder-decoder
        bg_base_ch=bg_base_ch,  # Base channels for background encoder-decoder
        bg_final_cnn_ch=bg_final_cnn_ch,  # Final CNN channels for background encoder-decoder

        # Network architecture options
        use_resblock=use_resblock,  # Use residual blocks in encoders-decoders
        num_res_blocks=num_res_blocks,  # Number of residual blocks per resolution
        cnn_mid_blocks=cnn_mid_blocks,  # Use middle blocks in CNN
        mlp_hidden_dim=mlp_hidden_dim,  # Hidden dimension for MLPs

        # Particle interaction transformer (PINT) configuration
        pint_enc_layers=pint_enc_layers,  # Number of PINT encoder layers
        pint_enc_heads=pint_enc_heads,  # Number of PINT encoder attention heads

        # Dynamics configuration
        timestep_horizon=1,
        
        #RGBD Stuff
        separate_depth_features=separate_depth_features,
        depth_feature_dim=depth_feature_dim,
        split_loss=split_loss,
        depth_loss_ratio=depth_loss_ratio,

        ).to(device)

    # -------------------------------------------------
    # OFFLINE KMEANS CACHE
    # -------------------------------------------------
    if precompute_kmeans or kmeans_cache_base:
        print(f"[kmeans] Precomputing kmeans cache ...")
        n_new = precompute_kmeans_cache(dataset, model.prior_module, cache_dir=kmeans_train_dir)
        n_new += precompute_kmeans_cache(val_dataset, model.prior_module, cache_dir=kmeans_val_dir)
        print(f"[kmeans] Done ({n_new} newly computed)")

    model_info = model.info()
    # print(model_info)
    # prepare saving location
        
    run_name = run_prefix
    # run_name = f'{ds}_gdlp' + run_prefix
    log_dir = prepare_logdir(runname=run_name, src_dir='./logs')
    fig_dir = os.path.join(log_dir, 'figures')
    save_dir = os.path.join(log_dir, 'saves')

    # ---- Checkpoint config: ONLY SAVE BEST ----
    monitor = config.get("monitor", "loss")          # which metric to track for "best"
    mode = config.get("monitor_mode", "min")         # "min" or "max"
    assert mode in ("min", "max")
    best_val = float("inf") if mode == "min" else -float("inf")

    os.makedirs(save_dir, exist_ok=True)
    ckpt_best = os.path.join(save_dir, "best.pt")
    ckpt_last = os.path.join(save_dir, "last.pt")

    # ---- Optional resume / preload ----
    start_epoch = int(config.get('start_epoch', 0))
    if load_model and pretrained_path is not None:
        try:
            resume_info = load_checkpoint(pretrained_path, model, None, None, map_location=device)
            # if this looks like a full ckpt and user wants true resume, do it after optimizer is created
            print(f"[ckpt] Loaded weights from {pretrained_path} (full={resume_info['is_full_ckpt']})")
        except Exception as e:
            print(f"[ckpt] Failed to load {pretrained_path}: {e}")


    save_config(log_dir, hparams)
    log_line(log_dir, model_info)
    # save a backup of the code for this run
    backup_info = save_code_backup('.', backup_dir=os.path.join(log_dir, 'saves', 'code_backup'))
    log_line(log_dir, backup_info)
    print(backup_info)

    # prepare loss functions
    if recon_loss_type == "vgg":
        recon_loss_func = LossLPIPS(normalized_rgb=normalize_rgb).to(device)
    else:
        recon_loss_func = calc_reconstruction_loss

    # optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=adam_betas, eps=adam_eps, weight_decay=weight_decay)
    if use_scheduler:
        scheduler = LinearWithWarmupScheduler(optimizer, gamma=scheduler_gamma, verbose=False,
                                              steps=(max(warmup_epoch, 1), max(warmup_epoch, 1) + 1),
                                              factors=(1.0, 1.0, 1.0 * scheduler_gamma))
    else:
        scheduler = None

    # If pretrained_path was a full checkpoint and we want to resume optimizer/scheduler
    if load_model and pretrained_path is not None:
        try:
            resume_info = load_checkpoint(pretrained_path, model, optimizer, scheduler, map_location=device)
            # carry over epoch/best if start_epoch isn't forced by config
            if start_epoch == 0 and resume_info.get("epoch", 0) > 0:
                start_epoch = int(resume_info["epoch"] + 1)
            best_val = resume_info.get("best_metric", best_val)
            print(f"[ckpt] Resumed training from epoch {start_epoch}, best={best_val:.6f}")
        except Exception as e:
            print(f"[ckpt] Resume skipped: {e}")


    # log statistics (for loss curves)
    losses = []
    losses_rec = []
    losses_kl = []
    val_losses = []
    val_losses_rec = []
    val_losses_kl = []

    # iteration counter
    iteration = 0

    wandb.init(
        name=run_name,
        config=config,
        resume="never", 
    )
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_avg = EpochAverager()

        pbar = tqdm(iterable=dataloader)
        for batch in pbar:
            vox = batch["voxels"].to(device)
            meta = batch.get("meta", None)

            warmup = (epoch < warmup_epoch)
            # forward pass
            model_output = model(vox, warmup=warmup, with_loss=True,
                                 beta_kl=beta_kl,
                                 beta_rec=beta_rec, kl_balance=kl_balance,
                                 recon_loss_type=recon_loss_type,
                                 recon_loss_func=recon_loss_func,
                                 beta_obj=beta_obj,
                                 meta=meta)

            # Compute & backprop
            all_losses = model_output['loss_dict']
            loss = all_losses['loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Collect losses for epoch averaging (no per-iter wandb logging)
            logged = {k: _to_float_safe(v) for k, v in all_losses.items()}
            epoch_avg.add(logged)

            # Progress bar
            def pick(key, default=0.0):
                return logged.get(key, default)

            if epoch < warmup_epoch:
                pbar.set_description_str(f'epoch #{epoch} (warmup)')
            else:
                pbar.set_description_str(f'epoch #{epoch}')

            # Average offset magnitude per particle: mean over batch of per-kp L2 norms
            mu_offset = model_output.get("mu_offset", None)
            avg_offset = mu_offset.norm(dim=-1).mean().item() if mu_offset is not None else 0.0

            pbar.set_postfix(
                loss=pick('loss'),
                rec=pick('loss_rec'),
                KL=pick('kl'),
                obj_l1=pick('obj_on_l1'),
                off=f'{avg_offset:.3f}',
                fg_e=f'{pick("fg_mse"):.3f}',        # MSE on fg voxels only (should decrease)
                bg_m=f'{pick("bg_mag"):.4f}',        # bg decoder output magnitude (-1=missing, 0=dead)
                r_kl=f'{pick("rec_kl_ratio"):.0f}', # rec/KL ratio, >100 means KL is negligible
                psnr=f'{pick("psnr"):.1f}',
            )

            iteration += 1
           

        pbar.close()

        # End of epoch - compute mean losses
        means = epoch_avg.means()

        # pretty print (robust to missing keys)
        log_str = build_epoch_log(epoch, means)
        print(log_str)
        log_line(log_dir, log_str)

        # Store epoch losses for loss curves
        losses.append(means.get('loss', 0.0))
        losses_rec.append(means.get('loss_rec', 0.0))
        losses_kl.append(means.get('kl', 0.0))

        # Collect all epoch-end logs into a single dict to avoid multiple wandb.log at same step
        epoch_log_dict = {**{f"epoch/{k}": v for k, v in means.items()}, "epoch_idx": epoch}

        # Decide monitored metric (robust to missing keys)
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
            print(f"[warn] monitored metric {mon_key} is non-finite ({monitored}); skipping model selection this epoch.")
            monitored = None


        # Always save last.pt for crash recovery / resume
        save_checkpoint(ckpt_last, model, optimizer, scheduler, epoch, best_val,
                        extra={"monitored": monitored})

        # Save best.pt if improved
        if monitored is not None:
            improved = (monitored < best_val) if mode == "min" else (monitored > best_val)
            if improved:
                best_val = monitored
                save_checkpoint(ckpt_best, model, optimizer, scheduler, epoch, best_val,
                                extra={"monitored": monitored, "best_update": True})
                print(f"[ckpt] New best ({mon_key}={monitored:.6f}) at epoch {epoch:04d} -> saved best.pt")



        # ------- VALIDATION -------
        # if epoch % eval_epoch_freq == 0 or epoch == num_epochs - 1:
        #     print(f"\n[Validation] Running validation at epoch {epoch}...")

        #     # Run validation
        #     val_results = evaluate_validation_voxel(
        #         model=model,
        #         val_dataloader=val_dataloader,
        #         device=device,
        #         config=config,
        #         epoch=epoch,
        #         recon_loss_func=recon_loss_func,
        #         beta_kl=beta_kl,
        #         beta_rec=beta_rec,
        #         beta_obj=beta_obj,
        #         kl_balance=kl_balance,
        #         lambda_color=lambda_color,
        #         recon_loss_type=recon_loss_type,
        #         fig_dir=fig_dir,
        #         save_visuals=True,
        #         topk=topk,
        #         iteration=iteration,
        #     )

        #     # Store validation losses
        #     val_losses.append(val_results.get('val_loss', 0.0))
        #     val_losses_rec.append(val_results.get('val_loss_rec', 0.0))
        #     val_losses_kl.append(val_results.get('val_kl', 0.0))

        #     # Log validation metrics - main metrics: Occ IoU + Masked Color PSNR (with std)
        #     val_log_str = f"[Val] epoch {epoch:04d}"
        #     if 'occ_iou' in val_results:
        #         iou_str = f"{val_results['occ_iou']:.4f}"
        #         if 'occ_iou_std' in val_results:
        #             iou_str += f" (±{val_results['occ_iou_std']:.4f})"
        #         val_log_str += f" | IoU: {iou_str}"
        #     if 'masked_color_psnr' in val_results:
        #         psnr_str = f"{val_results['masked_color_psnr']:.2f}"
        #         if 'masked_color_psnr_std' in val_results:
        #             psnr_str += f" (±{val_results['masked_color_psnr_std']:.2f})"
        #         val_log_str += f" | PSNR: {psnr_str} dB"
        #     val_log_str += f" | loss: {val_results.get('val_loss', 0.0):.4f}"
        #     if 'val_loss_rec' in val_results:
        #         val_log_str += f" | rec: {val_results['val_loss_rec']:.4f}"
        #     if 'val_kl' in val_results:
        #         val_log_str += f" | kl: {val_results['val_kl']:.4f}"
        #     print(val_log_str)
        #     log_line(log_dir, val_log_str)

        #     # Add validation metrics to epoch log dict (don't log separately)
        #     epoch_log_dict.update({f"val/{k}": v for k, v in val_results.items()})

        #     # Plot and save loss curves
        #     loss_curve_path = os.path.join(fig_dir, f"loss_curves_epoch{epoch:04d}.png")
        #     plot_loss_curves(
        #         train_losses=losses,
        #         val_losses=val_losses,
        #         train_losses_rec=losses_rec,
        #         val_losses_rec=val_losses_rec,
        #         train_losses_kl=losses_kl,
        #         val_losses_kl=val_losses_kl,
        #         save_path=loss_curve_path,
        #         title=f"Training Progress - Epoch {epoch}",
        #     )

        #     # Add loss curves image to epoch log dict
        #     epoch_log_dict["loss_curves"] = wandb.Image(loss_curve_path)

        #     model.train()  # Back to training mode

        # Log all epoch metrics in a single call (avoids issues with multiple logs at same step)
        wandb.log(epoch_log_dict, step=iteration)

        # ------- TRAIN VISUALS -------
        if epoch % eval_epoch_freq == 0 or epoch == num_epochs - 1:
            import plotly.graph_objects as go

            def _add_colored_kp_crosses(fig, kp_positions, obj_on_vals, D, H, W, half=2.0):
                """Add KP crosses color-coded by obj_on: red=off, green=on."""
                if kp_positions is None:
                    return
                if torch.is_tensor(kp_positions):
                    kp_np = kp_positions.detach().cpu().tolist()
                    kp_np = np.asarray(kp_np, dtype=np.float32)
                else:
                    kp_np = np.asarray(kp_positions, dtype=np.float32)
                kp_np = kp_np.reshape(-1, 3)

                if torch.is_tensor(obj_on_vals):
                    on_np = obj_on_vals.detach().cpu().tolist()
                    on_np = np.asarray(on_np, dtype=np.float32).reshape(-1)
                else:
                    on_np = np.asarray(obj_on_vals, dtype=np.float32).reshape(-1)

                # Convert from normalized [-1,1] to voxel indices
                # DLP z_pos is (x, y, z) order — see spatial_transform() in util_func.py
                # PyTorch affine_grid: x->W, y->H, z->D
                x_vox = (kp_np[:, 0] + 1) / 2 * (W - 1)
                y_vox = (kp_np[:, 1] + 1) / 2 * (H - 1)
                z_vox = (kp_np[:, 2] + 1) / 2 * (D - 1)

                for i in range(len(kp_np)):
                    x, y, z = float(x_vox[i]), float(y_vox[i]), float(z_vox[i])
                    on = float(np.clip(on_np[i], 0, 1))
                    if on < 0.5:
                        r, g, b = 255, int(255 * on * 2), 0
                    else:
                        r, g, b = int(255 * (1 - (on - 0.5) * 2)), 255, 0
                    color = f"rgb({r},{g},{b})"

                    xs = [x - half, x + half, None, x, x, None, x, x, None]
                    ys = [y, y, None, y - half, y + half, None, y, y, None]
                    zs = [z, z, None, z, z, None, z - half, z + half, None]
                    fig.add_trace(go.Scatter3d(
                        x=xs, y=ys, z=zs, mode="lines",
                        line=dict(width=6, color=color),
                        name=f"KP{i} on={on:.2f}",
                        showlegend=False,
                    ))

            def _vis_one_sample(b_idx, prefix, model_output, iteration):
                """Visualize one sample: GT, rec, fg, bg with color-coded KPs."""
                gt_vol = model_output['x'][b_idx]
                rec_vol = model_output['rec'][b_idx]
                mu_tot = model_output["z_base"][b_idx] + model_output["mu_offset"][b_idx]
                obj_on_vals = model_output["obj_on"][b_idx].reshape(-1)
                fg_only = model_output['dec_objects'][b_idx]
                bg_only = (model_output['bg_mask'] * model_output['bg'][:, :3])[b_idx]

                D, H, W = gt_vol.shape[1], gt_vol.shape[2], gt_vol.shape[3]

                # GT with color-coded KPs
                fig = log_rgb_voxels(
                    name=f"{prefix}/gt_kp", rgb_vol=gt_vol,
                    alpha_vol=None, KPx=None, step=None,
                    mode="splat", topk=60000, alpha_thresh=0.05, pad=2.0, show_axes=True,
                )
                _add_colored_kp_crosses(fig, mu_tot, obj_on_vals, D, H, W)
                wandb.log({f"{prefix}/gt_kp": fig}, step=iteration)

                # Rec with color-coded KPs
                fig = log_rgb_voxels(
                    name=f"{prefix}/rec_kp", rgb_vol=rec_vol,
                    alpha_vol=None, KPx=None, step=None,
                    mode="splat", topk=60000, alpha_thresh=0.05, pad=2.0, show_axes=True,
                )
                _add_colored_kp_crosses(fig, mu_tot, obj_on_vals, D, H, W)
                wandb.log({f"{prefix}/rec_kp": fig}, step=iteration)

                # FG with color-coded KPs
                fig = log_rgb_voxels(
                    name=f"{prefix}/fg_kp", rgb_vol=fg_only,
                    alpha_vol=None, KPx=None, step=None,
                    mode="splat", topk=60000, alpha_thresh=0.05, pad=2.0, show_axes=True,
                )
                _add_colored_kp_crosses(fig, mu_tot, obj_on_vals, D, H, W)
                wandb.log({f"{prefix}/fg_kp": fig}, step=iteration)

                # BG only (no KPs)
                log_rgb_voxels(
                    name=f"{prefix}/bg", rgb_vol=bg_only,
                    alpha_vol=None, KPx=None, step=iteration,
                    mode="splat", topk=60000, alpha_thresh=0.05, pad=2.0, show_axes=True,
                )

            # Check if multi-task: look for task labels in batch
            batch_tasks = batch.get("task", None)
            is_multitask = batch_tasks is not None and len(set(batch_tasks)) > 1

            if is_multitask:
                # Group batch indices by task, render up to 3 per task
                # Use wandb step = iteration * 1000 + sample_idx so the slider
                # scrubs through samples within each task panel
                from collections import defaultdict
                task_indices = defaultdict(list)
                for b_idx, t in enumerate(batch_tasks):
                    task_indices[t].append(b_idx)

                for task_name, indices in sorted(task_indices.items()):
                    for j, b_idx in enumerate(indices[:3]):
                        sample_step = iteration * 1000 + j
                        _vis_one_sample(b_idx, f"train/{task_name}", model_output, sample_step)
            else:
                # Single task: render first sample
                _vis_one_sample(0, "train", model_output, iteration)


    wandb.finish()


    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DLP Single-GPU Training")
    parser.add_argument("-d", "--dataset", type=str, default='shapes',
                        help="dataset of to train the model on: ['traffic', 'clevrer', 'obj3d128', 'phyre']")
    args = parser.parse_args()
    ds = args.dataset
    # TODO: Create a separate folder for PC configs since the params are pretty different
    if ds.endswith('json'):
        conf_path = ds
    else:
        conf_path = os.path.join('./configs', f'{ds}.json')

    train_dlp_pc(conf_path)
