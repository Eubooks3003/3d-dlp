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
from voxel_models import DLP
# datasets
from datasets.get_dataset import get_image_dataset
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
                           log_rgb_voxels)
import wandb

matplotlib.use("Agg")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


from collections import defaultdict

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
    """Logs all scalar-like entries in loss_dict to W&B under their existing keys."""
    flat = {}
    for k, v in loss_dict.items():
        val = _to_float_safe(v)
        if val is not None:
            flat[k] = val
    if flat:
        wandb.log(flat, step=step)
    return flat  # return what was logged (as floats) for local use

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

    # Voxel Stuff
    voxel_mode = config["voxel_mode"]
    voxel_grid_whd = config["voxel_grid_whd"]

    dataset = get_point_cloud_dataset(ds, root, mode='train', max_points=4096, include_rgb=(ch == 6))
    # vox_ds = VoxelizedDataset(
    #     base_ds=dataset,
    #     grid_whd=voxel_grid_whd,          # (W,H,D) in your class name, but tensors come out [C,D,H,W]
    #     mode="occupancy",                 # "occupancy" | "density" | "moments" | "avg_rgb"
    #     bounds_mode="global",           # "per_item" | "global" | ((pmin),(pmax))
    #     keep_points=False,              # True to also return original points
    #     device=torch.device("cpu"),     # keep CPU if you use DataLoader workers
    #     cache_dir="/home/ellina/Desktop/Code/voxel_ds_moments",  # speeds up future runs
    #     force_rebuild=False
    # )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

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
    
    run_name = f"{recon_loss_type}-beta_rec-{beta_rec}-beta_kl-{beta_kl}-kl_balance-{kl_balance}"
    wandb.init(
        name=run_name,
        config=config,
        resume="never", 
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

        epoch_avg = EpochAverager()

        

        pbar = tqdm(iterable=dataloader)
        for batch in pbar:
            vox  = batch["voxels"].to(device)  
            print("VOX SHAPE:", vox.shape)
            # mask = batch["mask"].to(device) 

            warmup = (epoch < warmup_epoch)
            # forward pass
            model_output = model(vox, warmup=warmup, with_loss=True,
                                            beta_kl=beta_kl,
                                            beta_rec=beta_rec, kl_balance=kl_balance,
                                            recon_loss_type=recon_loss_type,
                                            recon_loss_func=recon_loss_func,
                                            beta_obj=beta_obj)
            
            with torch.no_grad():
                # --- decoder-side diagnostics if present ---
                d = model_output.get("diag", None)
                if d:
                    # Means across batch for concise scalars
                    wandb.log({
                        "spawn/r_s0": float(d["radii"][:,0].mean().item()) if d["radii"].size(1) >= 1 else None,
                        "spawn/r_s1": float(d["radii"][:,1].mean().item()) if d["radii"].size(1) >= 2 else None,
                        "spawn/u_std_s0": float(d["u_std"][:,0].mean().item()) if d["u_std"].size(1) >= 1 else None,
                        "spawn/u_std_s1": float(d["u_std"][:,1].mean().item()) if d["u_std"].size(1) >= 2 else None,
                        "spawn/scale_mean": float(d["scale_mean"].mean().item()),
                        "spawn/d2kp_mean": float(d["d2kp_mean"].mean().item()),
                        "spawn/d2kp_std":  float(d["d2kp_std"].mean().item()),
                    }, step=iteration)
                # --- object-cloud spread in world space ---
            pts_obj = model_output.get("points_obj", None)     # [B,K,M,3]
            kp      = model_output.get("kp_p", None)           # [B,K,3]
            if (pts_obj is not None) and (kp is not None):
                # per-object mean radial distance and std
                d = (pts_obj - kp.unsqueeze(2)).norm(dim=-1)   # [B,K,M]
                world_rad_mean = d.mean(dim=(1,2)).mean().item()
                world_rad_std  = d.std(dim=(1,2)).mean().item()
                wandb.log({
                    "obj/rad_mean": world_rad_mean,
                    "obj/rad_std":  world_rad_std
                }, step=iteration)

                # min-NN distance inside each object (sampled to keep it cheap)
                # subsample points to avoid O(M^2)
                B,K,M,_ = pts_obj.shape
                m_sub = min(256, M)
                idx = torch.randperm(M, device=pts_obj.device)[:m_sub]
                Psub = pts_obj[:,:,idx]                        # [B,K,m_sub,3]
                # compute KNN=2 to skip self
                D = torch.cdist(Psub.reshape(B*K, m_sub, 3), Psub.reshape(B*K, m_sub, 3), p=2)
                D[torch.arange(B*K, device=D.device).unsqueeze(-1), torch.arange(m_sub, device=D.device)] = 1e9
                min_nn = D.min(dim=-1).values.mean().item()
                wandb.log({"obj/min_nn_dist_mean": min_nn}, step=iteration)
            # --- pick tensors inside model_output to watch gradient flow through ---
            watch_tensors = {}
            for name in ["kp_p", "z", "z_scale", "mu_offset", "mu_scale",
                        "z_features", "z_obj_on", "z_depth"]:
                t = model_output.get(name, None)
                if t is not None and t.requires_grad:
                    t.retain_grad()                # allow reading .grad on non-leaf tensors
                    watch_tensors[name] = t

            # ---- compute & backprop ----
            all_losses = model_output['loss_dict']          # whatever your calc_* returned
            loss = all_losses['loss']                       # must exist
            optimizer.zero_grad()
            loss.backward()

            # grads (unchanged)
            for k, v in watch_tensors.items():
                gm = (v.grad.abs().mean().item() if v.grad is not None else 0.0)
                gM = (v.grad.abs().max().item()  if v.grad is not None else 0.0)
                wandb.log({f"grad/{k}_mean": gm, f"grad/{k}_max": gM}, step=iteration)
            log_block_grads(model, iteration)
            plot_grad_flow(list(model.named_parameters()), iteration)

            optimizer.step()

            # ---- log exactly what your loss dict contains ----
            logged = wandb_log_lossdict(all_losses, iteration)   # returns dict of floats
            epoch_avg.add(logged)

            # ---- compact tqdm postfix using available keys ----
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
                kp=pick('loss_kl_kp'),
                feat=pick('loss_kl_feat'),
                scale=pick('loss_kl_scale'),
                obj=pick('loss_kl_obj_on'),
            )

            iteration += 1


            break  # for debug
        pbar.close()
        # at end of epoch
        # end of epoch
        means = epoch_avg.means()   # {'loss': ..., 'loss_rec': ..., 'kl': ..., ...} only for keys that appeared

        # pretty print (robust to missing keys)
        log_str = build_epoch_log(epoch, means)
        print(log_str)
        log_line(log_dir, log_str)

        wandb.log({**{f"epoch/{k}": v for k, v in means.items()},
            "epoch_idx": epoch}, step=iteration)

        # choose monitored metric robustly
        monitor_map = {
            "loss": "loss",
            "rec": "loss_rec",
            "kl": "kl",
            "vox/psnr_c0": "vox/psnr_c0",  # if you logged it in loss_dict or elsewhere
        }
        mon_key = monitor_map.get(monitor, "loss")
        monitored = means.get(mon_key, means.get("loss", None))

        # ---- Decide monitored metric (robust to missing keys) ----
        means = epoch_avg.means()  # e.g., {'loss': ..., 'loss_rec': ..., 'kl': ..., ...}

        def pick(mkey, default=None):
            return means.get(mkey, default)

        monitor_map = {
            "loss": "loss",
            "rec": "loss_rec",
            "kl": "kl",
            "vox/psnr_c0": "vox/psnr_c0",  # log this into means if you want to monitor it
        }
        mon_key   = monitor_map.get(monitor, "loss")
        monitored = pick(mon_key, pick("loss", None))  # fallback to total loss if chosen key missing

        # Optional: guard against NaN/Inf
        if monitored is not None and (not np.isfinite(monitored)):
            print(f"[warn] monitored metric {mon_key} is non-finite ({monitored}); skipping model selection this epoch.")
            monitored = None

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
                print(f"[ckpt] New best ({mon_key}={monitored:.6f}) at epoch {epoch:04d} -> saved best.pt")

        # ---- Periodic epoch snapshot ----
        if save_every and (epoch % save_every == 0 or epoch == num_epochs - 1):
            snap_path = os.path.join(save_dir, f"epoch_{epoch:04d}.pt")
            save_checkpoint(snap_path, model, optimizer, scheduler, epoch, best_val,
                            extra={"monitored": monitored, "snapshot": True})

        # ------- EVAL (voxel version) -------
        if epoch % eval_epoch_freq == 0 or epoch == num_epochs - 1:
            b0 = 0

            # gt_vol, rec_vol = extract_volumes_for_vis(model_output, occ_channel=0)

            gt_vol = model_output['x'][b0]
            rec_vol = model_output['rec'][b0]

            print("gt vol: ", gt_vol.shape)

            with torch.no_grad():
                # z_base_var: [B,K,6], mu_tot: [B,K,3], obj_on: [B,K,1]
                out = filter_topk_kps_3d(
                    z_base_var=model_output["z_base_var"],
                    mu_tot=model_output["z_base"] + model_output["mu_offset"],
                    topk=config['topk'],
                    obj_on=model_output.get("obj_on", None),
                    use_posterior_in_score=False  # set True to include posterior uncertainty
                )
                indices  = out["indices"]
                topk_kp  = out["topk_kp"]
                bb_scores= out["bb_scores"]

            b0 = 0  # first in batch
            topk_kp_b0 = topk_kp[b0]  # [k, 3]
            cov_b0 = model_output["cov_kp"][b0]  # [K, 6]
            kp_xyz = model_output["kp_p"]  # [B, K, 3]

            z_base_cov_b0 = model_output["z_base_cov"][b0]  # [K, 6]
            
            print("z base: ", model_output["z_base"].shape)
            z_base_b0 = model_output["z_base"][b0]  # [K, 3]
            mu_tot_b0 = z_base_b0 + model_output["mu_offset"][b0]  # [K, 3]
            kp_order = ("x","y","z")  # your kp_xyz is in (x,y,z) order

            print("mu tot: ", mu_tot_b0.shape)
            print("GT VOL: ", gt_vol.shape)
            log_rgb_voxels(
                name="gt/rgb_splat",
                rgb_vol=gt_vol,
                alpha_vol=None,          # None if you don’t have GT α
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
                alpha_vol=None,          # None if you don’t have GT α
                step=iteration,
                mode="splat",
                topk=60000,
                alpha_thresh=0.05,
                pad=2.0,
                show_axes=True,
            )


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
