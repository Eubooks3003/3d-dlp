#!/usr/bin/env python3
# debug_dlp_vox.py
import os, sys, glob, time, json, argparse
from typing import Optional, Dict, Any, List

import numpy as np
import torch
from torch.utils.data import DataLoader

# --- your codebase imports ---
from utils.util_func import get_config, log_line, prepare_logdir
from datasets.point_cloud_datasets.get_dataset import get_point_cloud_dataset, pc_collate
from datasets.voxelize_ds_wrapper import VoxelizedDataset

from voxel_models import DLP  # voxel-capable DLP (same as training)
from eval.eval_vox import (log_vox_overlay_plotly, log_vox_isoseries, topk_kps,
        extract_volumes_for_vis, print_vol_stats)

import wandb

# ------------------------------- helpers -------------------------------

def find_latest_checkpoint(run_save_dir: str) -> Optional[str]:
    """Find the latest checkpoint in a directory (non-recursive)."""
    candidates = []
    for ext in ("*.pth", "*.pt", "*.ckpt", "*.bin"):
        candidates += glob.glob(os.path.join(run_save_dir, ext))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)

def load_model_from_config(cfg: Dict[str, Any], device: torch.device) -> DLP:
    # Use the same field mapping as your training call (including the image_size hack via voxel_grid_whd[0])
    voxel_grid_whd = cfg["voxel_grid_whd"]
    model = DLP(
        # Basics
        cdim=cfg['ch'],
        image_size=voxel_grid_whd[0],
        normalize_rgb=cfg['normalize_rgb'],

        # Keypoint & patch config
        n_kp_per_patch=cfg['n_kp_per_patch'],
        patch_size=cfg['patch_size'],
        anchor_s=cfg['anchor_s'],
        n_kp_enc=cfg['n_kp_enc'],
        n_kp_prior=cfg['n_kp_prior'],

        # Network config
        pad_mode=cfg['pad_mode'],
        dropout=cfg['dropout'],

        # Feature representation
        features_dist=cfg.get('features_dist', 'gauss'),
        learned_feature_dim=cfg['learned_feature_dim'],
        learned_bg_feature_dim=cfg.get('learned_bg_feature_dim', cfg['learned_feature_dim']),
        n_fg_categories=cfg.get('n_fg_categories', 8),
        n_fg_classes=cfg.get('n_fg_classes', 4),
        n_bg_categories=cfg.get('n_bg_categories', 4),
        n_bg_classes=cfg.get('n_bg_classes', 4),

        # Priors
        scale_std=cfg['scale_std'],
        offset_std=cfg['offset_std'],
        obj_on_alpha=cfg['obj_on_alpha'],
        obj_on_beta=cfg['obj_on_beta'],

        # Object decoder arch
        obj_res_from_fc=cfg['obj_res_from_fc'],
        obj_ch_mult_prior=cfg.get('obj_ch_mult_prior', cfg['obj_ch_mult']),
        obj_ch_mult=cfg['obj_ch_mult'],
        obj_base_ch=cfg['obj_base_ch'],
        obj_final_cnn_ch=cfg['obj_final_cnn_ch'],

        # Background decoder arch
        bg_res_from_fc=cfg['bg_res_from_fc'],
        bg_ch_mult=cfg['bg_ch_mult'],
        bg_base_ch=cfg['bg_base_ch'],
        bg_final_cnn_ch=cfg['bg_final_cnn_ch'],

        # General arch options
        use_resblock=cfg['use_resblock'],
        num_res_blocks=cfg['num_res_blocks'],
        cnn_mid_blocks=cfg.get('cnn_mid_blocks', False),
        mlp_hidden_dim=cfg.get('mlp_hidden_dim', 256),

        # PINT
        pint_enc_layers=cfg['pint_enc_layers'],
        pint_enc_heads=cfg['pint_enc_heads'],

        # Dynamics
        timestep_horizon=1,

        # RGBD
        separate_depth_features=cfg.get('separate_depth_features', False),
        depth_feature_dim=cfg.get('depth_feature_dim', 0),
        split_loss=cfg.get('split_loss', False),
        depth_loss_ratio=cfg.get('depth_loss_ratio', 0.5),
    ).to(device)
    model.eval()
    return model

def summarize_kp(model_output: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Compact stats for keypoints to print/log."""
    def _t(t):
        return None if t is None else t.detach().float()
    out = {}
    kp_p = _t(model_output.get("kp_p"))
    if kp_p is not None:
        out["kp/mean_abs_pos"] = kp_p.abs().mean().item()
        out["kp/max_abs_pos"]  = kp_p.abs().max().item()
    mu_scale = _t(model_output.get("mu_scale"))
    if mu_scale is not None:
        out["kp/scale_sigmoid_mean"] = torch.sigmoid(mu_scale).mean().item()
    obj_on = _t(model_output.get("obj_on"))
    if obj_on is not None:
        out["kp/obj_on_mean"] = obj_on.mean().item()
    z_depth = _t(model_output.get("z_depth"))
    if z_depth is not None:
        out["kp/depth_mean"] = z_depth.mean().item()
    return out

def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def make_iso_sweep() -> List[float]:
    """
    Mirror your training sweep, plus a fine sweep if desired.
    Training used: [0.05, 0.1, 0.2, 0.3, 0.4]
    We'll include the exact same by default so it's truly 'same eval'.
    """
    return [0.05, 0.10, 0.20, 0.30, 0.40]

# ------------------------------- main -------------------------------

def main():
    ap = argparse.ArgumentParser("Debug a trained Voxel DLP checkpoint with the same voxel eval as training.")
    ap.add_argument("--config", "-c", type=str, required=True, help="Path to config JSON (same used for training).")
    ap.add_argument("--run-dir", type=str, required=True, help="Directory containing checkpoints (e.g., ./logs/.../saves).")
    ap.add_argument("--ckpt", type=str, default="", help="Explicit checkpoint path (overrides latest in run-dir).")
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Dataset split.")
    ap.add_argument("--batch-size", type=int, default=1, help="Batch size for debug pass.")
    ap.add_argument("--max-batches", type=int, default=4, help="How many batches to visualize.")
    ap.add_argument("--wandb", action="store_true", help="If set, logs visualizations to W&B.")
    ap.add_argument("--project", type=str, default="dlp-debug", help="W&B project name.")
    ap.add_argument("--name", type=str, default="", help="Optional W&B run name.")
    args = ap.parse_args()

    # ----- config + device -----
    cfg = get_config(args.config)
    voxel_mode     = cfg["voxel_mode"]
    voxel_grid_whd = cfg["voxel_grid_whd"]
    ch             = cfg["ch"]

    device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device if ('cuda' in device and torch.cuda.is_available()) else 'cpu')
    print(f"[info] Using device: {device}")

    # ----- dataset → voxelized wrapper (mirrors training’s input batch['voxels']) -----
    ds_name = cfg['ds']
    root    = cfg['root']

    base_ds = get_point_cloud_dataset(
        ds_name, root, mode=args.split, max_points=4096, include_rgb=(ch == 6)
    )
    loader = DataLoader(base_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # ----- model -----
    model = load_model_from_config(cfg, device)

    # ----- checkpoint -----
    ckpt_path = args.ckpt or find_latest_checkpoint(os.path.join(args.run_dir))
    if ckpt_path is None:
        print(f"[!] No checkpoint found in {args.run_dir}")
        sys.exit(1)

    print(f"[info] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # unwrap common training bundles
    state = None
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model", "ema", "ema_state_dict", "net", "weights"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                print(f"[info] Using nested state_dict from '{key}'")
                break
    if state is None:
        state = ckpt
        print("[info] Using checkpoint as bare state_dict")

    # strip possible prefixes
    def _strip_prefix(d, prefix="module."):
        if not any(k.startswith(prefix) for k in d.keys()):
            return d
        return { (k[len(prefix):] if k.startswith(prefix) else k): v for k, v in d.items() }

    state = _strip_prefix(state, "module.")
    state = _strip_prefix(state, "model.")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] loaded={len(state)}  missing={len(missing)}  unexpected={len(unexpected)}")
    if missing:
        print(f"[warn] Missing (first 10): {missing[:10]}")
    if unexpected:
        print(f"[warn] Unexpected (first 10): {unexpected[:10]}")

    # ----- W&B -----
    if args.wandb:
        runname = args.name or f"debug-vox-{os.path.basename(args.run_dir)}-{int(time.time())}"
        wandb.init(project=args.project, name=runname,
                   config={"config_path": args.config, "ckpt": ckpt_path, "split": args.split})
        print("[info] W&B logging enabled.")

    # ----- loop -----
    step = 0
    max_batches = args.max_batches
    iso_main = 0.67
    iso_sweep = make_iso_sweep()

    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            # training fed: vox = batch["voxels"]
            vox  = batch["voxels"]  # [B, C, D, H, W] or [B, D, H, W] depending on your wrapper
            # forward (no loss)
            model_output = model(vox, warmup=False, with_loss=False)

            gt_vol, rec_vol = extract_volumes_for_vis(model_output, occ_channel=0)
            print_vol_stats("GT", gt_vol)
            print_vol_stats("REC", rec_vol)

            # keypoints in normalized scene coords, shape [B,K,3] (order z,y,x)
            kp_xyz = model_output.get('kp_p', None)


            # Top-K by variance (mirrors training call)
            idx, kp_topk, score_topk, scores_all = topk_kps(model_output, cfg['topk'], use_mu="kp_p", gate_with_obj_on=True, eps=1e-12)
            # take first in batch for voxel plots (consistent with training eval snippet)
            print("KP TOPK: ", kp_topk.shape)
            print("KP TOPK SCORES: ", score_topk)
            b0 = 0
            kp_b0    = None if kp_xyz is None else kp_xyz[b0]
            kpt_b0   = None if kp_topk is None else kp_topk[b0]
            score_b0 = None if score_topk is None else score_topk[b0]
            print("KP TOPK: ", kpt_b0.shape)
            print("KP TOPK SCORES: ", score_b0)

            # ------ Voxel overlays (same as training) ------
            log_vox_overlay_plotly("vox/overlay_main", gt_vol, rec_vol, kps=kp_b0,
                                   iso_levels=[iso_main], step=step)
            log_vox_overlay_plotly("gt/gt", gt_vol, None, kps=None,
                                   iso_levels=[iso_main], step=step)
            log_vox_overlay_plotly("rec/rec", None, rec_vol, kps=None,
                                   iso_levels=[iso_main], step=step)
            log_vox_overlay_plotly("gt/gt_topk", gt_vol, None, kps=kpt_b0,
                                   iso_levels=[iso_main], step=step)
            log_vox_overlay_plotly("rec/rec_topk", None, rec_vol, kps=kpt_b0,
                                   iso_levels=[iso_main], step=step)

            # --- compact KP stats ---
            kp_stats = summarize_kp(model_output)
            if args.wandb and kp_stats:
                wandb.log(kp_stats, step=step)
            else:
                print(f"[kp] step={step} :: {json.dumps({k: round(v,5) for k,v in kp_stats.items()})}")

            step += 1
            if step >= max_batches:
                break

    print("[done] voxel debug pass complete.")
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
