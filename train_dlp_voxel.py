"""
Single-GPU training of DLPv2
"""
# imports
import numpy as np
import os
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
from models import DLP
from voxel_models import VoxelDLP
# datasets
from datasets.get_dataset import get_image_dataset
from datasets.point_cloud_datasets.get_dataset import get_point_cloud_dataset, pc_collate
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
import eval.eval_vox as eval_vox
from eval.eval_pc import clean_pts, log_pc_plotly, log_pc_overlay_plotly, select_topk_keypoints
import wandb

matplotlib.use("Agg")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


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
    image_size = config['image_size']
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
    kl_balance = config['kl_balance']  # balance between visual features and the other particle attributes

    # priors
    scale_std = config['scale_std']
    offset_std = config['offset_std']
    obj_on_alpha = config['obj_on_alpha']  # transparency beta distribution "a"
    obj_on_beta = config['obj_on_beta']  # transparency beta distribution "b"

    # evaluation
    eval_epoch_freq = config['eval_epoch_freq']
    eval_im_metrics = config['eval_im_metrics']

    # visualization
    topk = min(config['topk'], config['n_kp_enc'])  # top-k particles to plot
    iou_thresh = config['iou_thresh']  # threshold for NMS for plotting bounding boxes

    #RGBD Stuff
    separate_depth_features = config["separate_depth_features"]  # use separate depth feature encoding
    depth_feature_dim = config["depth_feature_dim"]  # depth feature dimension if separate encoding
    split_loss = config["split_loss"]  # split loss into components for logging
    depth_loss_ratio = config["depth_loss_ratio"]  # weight of depth loss if split_loss is True

    # Point Cloud Stuff

    decoder_point_mode = config["decoder_point_mode"]
    points_per_object = config["points_per_object"]
    points_per_scale = config["points_per_scale"]
    points_bg = config["points_bg"]

    # Voxel Stuff
    voxel_mode = config["voxel_mode"]
    voxel_grid_whd = config["voxel_grid_whd"]

    dataset = get_point_cloud_dataset(ds, root, mode='train', max_points=4096, include_rgb=(ch == 6))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=pc_collate)


    # model

    model = VoxelDLP(
        cdim=ch,  # Number of input image channels
        image_size=image_size,  # Input image size (assumed square)
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
        
        # PC Stuff
        decoder_point_mode=decoder_point_mode,
        points_per_object=points_per_object,
        points_per_scale=points_per_scale,
        points_bg=points_bg,

        # Voxel Stuff
        voxel_mode=voxel_mode,
        voxel_grid_whd=voxel_grid_whd,
        ).to(device)
        
    model_info = model.info()
    # print(model_info)
    # prepare saving location
    run_name = f'{ds}_gdlp' + run_prefix
    log_dir = prepare_logdir(runname=run_name, src_dir='./logs')
    fig_dir = os.path.join(log_dir, 'figures')
    save_dir = os.path.join(log_dir, 'saves')

    # ---- Checkpoint config ----
    save_every = int(config.get("save_every", 1))   # save epoch snapshot every N epochs
    monitor = config.get("monitor", "loss")          # which metric to track for "best"
    mode = config.get("monitor_mode", "min")         # "min" or "max"
    assert mode in ("min", "max")
    best_val = float("inf") if mode == "min" else -float("inf")

    # Paths
    os.makedirs(save_dir, exist_ok=True)
    ckpt_last = os.path.join(save_dir, "last.pt")
    ckpt_best = os.path.join(save_dir, "best.pt")

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

    # get the range of the keypoints, it is [-1, 1] by default
    kp_range = model.kp_range
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


    # log statistics
    losses = []
    losses_rec = []
    losses_kl = []
    losses_kl_kp = []
    losses_kl_feat = []
    losses_kl_scale = []
    losses_kl_depth = []
    losses_kl_obj_on = []



    # initialize validation statistics
    valid_loss = best_valid_loss = 1e8
    valid_losses = []
    best_valid_epoch = 0

    # save PSNR values of the reconstruction
    psnrs = []

    # image metrics
    if eval_im_metrics:
        val_lpipss = []
        best_val_lpips_epoch = 0
        val_lpips = best_val_lpips = 1e8
    else:
        best_val_lpips_epoch = None
        val_lpips = best_val_lpips = None

    # iteration counter
    iteration = 0
    
    run_name = f"{voxel_mode}-{voxel_grid_whd[0]}-{decoder_point_mode}"
    wandb.init(
        name=run_name,
        config=config,
        reinit=True,
    )
    for epoch in range(start_epoch, num_epochs):
        model.train()
        batch_losses = []
        batch_losses_rec = []
        batch_losses_kl = []
        batch_losses_kl_kp = []
        batch_losses_kl_feat = []
        batch_losses_kl_scale = []
        batch_losses_kl_obj_on = []

        # recon components (DCD & optional weighted CD)
        losses_rec_dcd = []
        losses_rec_cd_w = []

        obj_on_l1_list = []
        obj_on_mean_list = []
        mu_scale_mean_list = []

        

        pbar = tqdm(iterable=dataloader)
        for batch in pbar:
            pts  = batch["points"].to(device)   # [B, N, 3]
            mask = batch["mask"].to(device) 

            warmup = (epoch < warmup_epoch)
            # forward pass
            model_output = model(pts, mask, warmup=warmup, with_loss=True,
                                 beta_kl=beta_kl,
                                 beta_rec=beta_rec, kl_balance=kl_balance,
                                 recon_loss_type=recon_loss_type,
                                 recon_loss_func=recon_loss_func,
                                 beta_obj=beta_obj)
            
            # --- pick tensors inside model_output to watch gradient flow through ---
            watch_tensors = {}
            for name in ["kp_p", "z", "z_scale", "mu_offset", "mu_scale",
                        "z_features", "z_obj_on", "z_depth"]:
                t = model_output.get(name, None)
                if t is not None and t.requires_grad:
                    t.retain_grad()                # allow reading .grad on non-leaf tensors
                    watch_tensors[name] = t

            # calculate loss
            all_losses = model_output['loss_dict']
            loss = all_losses['loss']

            optimizer.zero_grad()
            loss.backward()

            # --- log per-tensor grad magnitudes (mean and max) ---
            for k, v in watch_tensors.items():
                gm = (v.grad.abs().mean().item() if v.grad is not None else 0.0)
                gM = (v.grad.abs().max().item()  if v.grad is not None else 0.0)
                wandb.log({f"grad/{k}_mean": gm, f"grad/{k}_max": gM}, step=iteration)
            log_block_grads(model, iteration)
            plot_grad_flow(list(model.named_parameters()), iteration)

            optimizer.step()

            log_param_updates(model, iteration)
            wandb_log_iter_losses(all_losses, iteration)

            iteration += 1

            # keypoint spread / motion
            kp = model_output.get("kp_p")                          # [B,K,3] in [-1,1]
            obj_on = model_output.get("obj_on")                    # [B,K,1] or [B,K]
            if kp is not None:
                spread = kp.std(dim=1).mean().item()              # avg spatial spread
                wandb.log({"kp/spread": spread}, step=iteration)
                if hasattr(model, "_kp_prev"):
                    drift = (kp - model._kp_prev).pow(2).sum(-1).sqrt().mean().item()
                    wandb.log({"kp/drift_per_step": drift}, step=iteration)
                model._kp_prev = kp.detach().clone()

            # how many are "on"
            if obj_on is not None:
                on_frac = obj_on.sigmoid().mean().item() if obj_on.dim()==3 else obj_on.mean().item()
                wandb.log({"kp/on_frac": on_frac}, step=iteration)


            # --- helpers ---
            def _to_float(x, default=0.0):
                if x is None:
                    return float(default)
                if isinstance(x, (float, int)):
                    return float(x)
                try:
                    return float(x.detach().mean().item())
                except Exception:
                    return float(default)

            # --- unpack logged losses from calc_static_elbo_pc ---
            loss            = all_losses['loss']
            loss_rec        = all_losses['loss_rec']                 # combined recon (DCD + weighted CD * weight)
            rec_dcd         = all_losses.get('rec_dcd', None)        # pure DCD term
            rec_cd_weighted = all_losses.get('rec_cd_weighted', None)# optional weighted CD term

            loss_kl         = all_losses['kl']
            loss_kl_kp      = all_losses.get('loss_kl_kp', None)
            loss_kl_scale   = all_losses.get('loss_kl_scale', None)
            loss_kl_feat    = all_losses.get('loss_kl_feat', None)
            loss_kl_obj_on  = all_losses.get('loss_kl_obj_on', None)
            obj_on_l1       = all_losses.get('obj_on_l1', None)

            # --- a few encoder-side sanity stats (guarded) ---
            obj_on          = model_output.get('obj_on', None)         # [B,K,1] or None
            obj_on_mean     = _to_float(obj_on)                        # fraction of active objs (roughly)
            mu_scale        = model_output.get('mu_scale', None)       # [B,K,3] or [B,K,1]
            mu_scale_mean   = _to_float(torch.sigmoid(mu_scale) if mu_scale is not None else None)

            # point count sanity
            valid_points    = _to_float(mask.float().sum(dim=1) if mask is not None else None)
            valid_points    = int(valid_points) if not isinstance(valid_points, float) else valid_points

            # --- collect per-batch scalars ---
            batch_losses.append(_to_float(loss))
            batch_losses_rec.append(_to_float(loss_rec))
            batch_losses_kl.append(_to_float(loss_kl))
            batch_losses_kl_kp.append(_to_float(loss_kl_kp))
            batch_losses_kl_feat.append(_to_float(loss_kl_feat))
            batch_losses_kl_scale.append(_to_float(loss_kl_scale))
            batch_losses_kl_obj_on.append(_to_float(loss_kl_obj_on))

            # recon components
            if rec_dcd is not None:
                losses_rec_dcd.append(_to_float(rec_dcd))
            if rec_cd_weighted is not None:
                losses_rec_cd_w.append(_to_float(rec_cd_weighted))

            # obj_on stats
            obj_on_l1_list.append(_to_float(obj_on_l1))
            obj_on_mean_list.append(obj_on_mean)
            mu_scale_mean_list.append(mu_scale_mean)



            # optional: track obj_on stats
            obj_on_l1_list.append(_to_float(obj_on_l1))
            obj_on_mean_list.append(obj_on_mean)
            mu_scale_mean_list.append(mu_scale_mean)

            # --- tqdm/postfix (compact) ---
            if epoch < warmup_epoch:
                pbar.set_description_str(f'epoch #{epoch} (warmup)')
            else:
                pbar.set_description_str(f'epoch #{epoch}')

            pbar.set_postfix(
                loss=_to_float(loss),
                rec=_to_float(loss_rec),
                KL=_to_float(loss_kl),
                kp=_to_float(loss_kl_kp),
                feat=_to_float(loss_kl_feat),
                scale=_to_float(loss_kl_scale),
                obj=_to_float(loss_kl_obj_on),
            )


            break  # for debug
        pbar.close()
        # at end of epoch
        losses.append(float(np.mean(batch_losses)))
        losses_rec.append(float(np.mean(batch_losses_rec)))
        losses_kl.append(float(np.mean(batch_losses_kl)))
        losses_kl_kp.append(float(np.mean(batch_losses_kl_kp)))
        losses_kl_feat.append(float(np.mean(batch_losses_kl_feat)))
        losses_kl_scale.append(float(np.mean(batch_losses_kl_scale)))
        losses_kl_obj_on.append(float(np.mean(batch_losses_kl_obj_on)))

        mean_dcd   = float(np.mean(losses_rec_dcd)) if len(losses_rec_dcd) else None
        mean_cdw   = float(np.mean(losses_rec_cd_w)) if len(losses_rec_cd_w) else None
        mean_on_l1 = float(np.mean(obj_on_l1_list)) if len(obj_on_l1_list) else None
        mean_on_p  = float(np.mean(obj_on_mean_list)) if len(obj_on_mean_list) else None
        mean_s     = float(np.mean(mu_scale_mean_list)) if len(mu_scale_mean_list) else None

        log_str = (
            f"epoch {epoch:04d} | "
            f"loss {losses[-1]:.4f} | rec {losses_rec[-1]:.4f}"
            f"{'' if mean_dcd is None else f' (dcd {mean_dcd:.4f}'}"
            f"{'' if mean_cdw is None else f', cdw {mean_cdw:.4f})' if mean_dcd is not None else f' (cdw {mean_cdw:.4f})'} | "
            f"KL {losses_kl[-1]:.4f} [kp {losses_kl_kp[-1]:.3f}, feat {losses_kl_feat[-1]:.3f}, "
            f"scale {losses_kl_scale[-1]:.3f}, obj {losses_kl_obj_on[-1]:.3f}] | "
            f"on_L1 {0.0 if mean_on_l1 is None else mean_on_l1:.3f} | "
            f"on̄ {0.0 if mean_on_p is None else mean_on_p:.3f} | "
            f"s̄ {0.0 if mean_s is None else mean_s:.3f}"
        )

        print(log_str)
        log_line(log_dir, log_str)

        # ---- Decide monitored metric ----
        # You already have epoch means: losses[-1], losses_rec[-1], etc.
        if monitor == "loss":
            monitored = losses[-1]
        elif monitor == "rec":
            monitored = losses_rec[-1]
        elif monitor == "kl":
            monitored = losses_kl[-1]
        elif monitor == "vox/psnr_c0":
            # if you computed metrics dict above (optional)
            monitored = metrics.get("vox/psnr_c0", None) if "metrics" in locals() else None
        else:
            # fallback to total loss
            monitored = losses[-1]

        # ---- Save "last" every epoch ----
        save_checkpoint(ckpt_last, model, optimizer, scheduler, epoch, best_val,
                        extra={"monitored": monitored, "monitor": monitor, "mode": mode})

        # ---- Save "best" if improved ----
        if monitored is not None:
            improved = (monitored < best_val) if mode == "min" else (monitored > best_val)
            if improved:
                best_val = monitored
                save_checkpoint(ckpt_best, model, optimizer, scheduler, epoch, best_val,
                                extra={"monitored": monitored, "best_update": True})
                print(f"[ckpt] New best ({monitor}={monitored:.6f}) at epoch {epoch:04d} -> saved best.pt")

        # ---- Periodic epoch snapshot ----
        if save_every and (epoch % save_every == 0 or epoch == num_epochs - 1):
            snap_path = os.path.join(save_dir, f"epoch_{epoch:04d}.pt")
            save_checkpoint(snap_path, model, optimizer, scheduler, epoch, best_val,
                            extra={"monitored": monitored, "snapshot": True})

        if epoch % eval_epoch_freq == 0 or epoch == num_epochs - 1:
            b0 = 0

            # --- GT (mask-filtered) ---
            # gt_clean = clean_pts(pts[b0], mask[b0] if mask is not None else None)  # [Ng,3]
            gt_clean = model_output.get('pts_norm', None)

            # --- REC ---
            rec_pts  = model_output.get('points_scene', None)
            rec_pts  = clean_pts(rec_pts[b0]) if rec_pts is not None else None
            rec_cols = model_output.get('rec_colors', None)
            rec_cols = rec_cols[b0] if rec_cols is not None else None
            ids      = model_output.get('assign_ids', None)
            ids      = ids[b0] if ids is not None else None

            print("REc points: ", rec_pts.shape)

            # --- KPs ---
            kp_xyz = model_output.get('kp_p', None)  # [B,K,3] in [-1,1]
            if kp_xyz is not None:
                with torch.no_grad():
                    idx, kp_topk, scores_topk = select_topk_keypoints(model_output, topk, prefer_logvar=True)
            else:
                kp_topk = None

            # --- Plotly logs (GT / REC / overlays) ---
            log_pc_overlay_plotly("gt/gt_with_kp", gt_clean, None, kps=kp_xyz,
                                color_mode="source", step=iteration, point_size_gt=2, point_size_rec=2)
            log_pc_overlay_plotly("rec/rec_with_kp", None, rec_pts, kps=kp_xyz,
                                color_mode="source", step=iteration, point_size_gt=2, point_size_rec=2)

            log_pc_overlay_plotly("gt/gt", gt_clean, None, kps=None,
                                color_mode="source", step=iteration, point_size_gt=2, point_size_rec=2)
            log_pc_overlay_plotly("rec/rec", None, rec_pts, kps=None,
                                color_mode="source", step=iteration, point_size_gt=2, point_size_rec=2)

            log_pc_overlay_plotly("viz/overlay_source_with_kp", gt_clean, rec_pts, kps=kp_xyz,
                                color_mode="source", step=iteration, point_size_gt=2, point_size_rec=2)
            log_pc_overlay_plotly("viz/overlay_source", gt_clean, rec_pts, kps=None,
                                color_mode="source", step=iteration, point_size_gt=2, point_size_rec=2)

            if kp_topk is not None:
                log_pc_overlay_plotly("gt/gt_topk", gt_clean, None, kps=kp_topk,
                                    color_mode="source", step=iteration, point_size_gt=2, point_size_rec=2)
                log_pc_overlay_plotly("rec/rec_topk", None, rec_pts, kps=kp_topk,
                                    color_mode="source", step=iteration, point_size_gt=2, point_size_rec=2)
                log_pc_overlay_plotly("viz/overlay_source_topk", gt_clean, rec_pts, kps=kp_topk,
                                    color_mode="source", step=iteration, point_size_gt=2, point_size_rec=2)

            # --- scalar metrics to W&B (only if computed) ---
            metrics = {
                "rec/dcd":        mean_dcd,
                "rec/cd_weighted": mean_cdw,
                "obj/on_L1":      mean_on_l1,
                "obj/on_prob":    mean_on_p,
                "obj/scale_mean": mean_s,
            }
            metrics = {k: v for k, v in metrics.items() if v is not None}
            if metrics:
                wandb.log(metrics, step=epoch)




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
