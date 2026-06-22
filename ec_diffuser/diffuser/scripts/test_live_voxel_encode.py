#!/usr/bin/env python
"""
Live voxelization + DLP encoding smoke test for the rlbench branch.

Launches RLBench inside the wrapper, resets to a task variation, runs the
end-to-end live pipeline once:

  RGBD from each enabled camera
    -> backproject to world
    -> fuse + voxelize (RGBO)
    -> DLP prior (kmeans kp, cov)
    -> DLP full forward with precomputed prior in meta
    -> particle tokens (K, Dtok) + bg features

Reports shapes/ranges for each stage, and (optional) logs the voxel grid +
decoded reconstruction + keypoints to wandb.

Usage:
    cd /home/ellina/Desktop/EC-Diffuser
    LPWM_DEV=/home/ellina/Desktop/lpwm-dev \
    xvfb-run -a /home/ellina/miniconda3/envs/ecdiffuser/bin/python \
        diffuser/scripts/test_live_voxel_encode.py \
        --task close_jar \
        --dlp-cfg  /home/ellina/Desktop/data/preprocessed_voxel_tokens/rlbench_close_jar/dlp_config.json \
        --dlp-ckpt /home/ellina/Desktop/data/preprocessed_voxel_tokens/rlbench_close_jar/dlp_ckpt.pt \
        --device cuda:0

Add --wandb-project <name> to also log voxels + decoded recon + kp to wandb.
Add --steps 5 to take a few env steps and re-encode after each.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch


# -----------------------------------------------------------------------------#
# Path setup: make EC-Diffuser, lpwm-dev (voxel_models), and any sibling deps
# importable. Mirrors train.py's resolution.
# -----------------------------------------------------------------------------#
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIFFUSER_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))             # EC-Diffuser/diffuser
EC_DIFFUSER_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))    # EC-Diffuser
for _p in (EC_DIFFUSER_ROOT, DIFFUSER_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_LPWM_CANDIDATES = []
_env_lpwm = os.environ.get("LPWM_DEV")
if _env_lpwm:
    _LPWM_CANDIDATES.append(_env_lpwm)
_LPWM_CANDIDATES.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "lpwm-dev")))
for _p in _LPWM_CANDIDATES:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)
        print(f"[setup] lpwm-dev on path: {_p}", flush=True)
        break


# -----------------------------------------------------------------------------#
# DLP loading (mirrors precompute_kmeans_rlbench.build_prior, but full model)
# -----------------------------------------------------------------------------#
def load_dlp(cfg_path: str, ckpt_path: str, device: torch.device):
    from voxel_models import DLP
    from utils.util_func import get_config
    from utils.log_utils import load_checkpoint

    cfg = get_config(cfg_path)
    model = DLP(
        cdim=cfg["ch"],
        image_size=cfg["voxel_grid_whd"][0],
        normalize_rgb=cfg["normalize_rgb"],
        n_kp_per_patch=cfg["n_kp_per_patch"],
        patch_size=cfg["patch_size"],
        anchor_s=cfg["anchor_s"],
        n_kp_enc=cfg["n_kp_enc"],
        n_kp_prior=cfg["n_kp_prior"],
        pad_mode=cfg["pad_mode"],
        dropout=cfg["dropout"],
        features_dist=cfg.get("features_dist", "gauss"),
        learned_feature_dim=cfg["learned_feature_dim"],
        learned_bg_feature_dim=cfg.get("learned_bg_feature_dim", cfg["learned_feature_dim"]),
        n_fg_categories=cfg.get("n_fg_categories", 8),
        n_fg_classes=cfg.get("n_fg_classes", 4),
        n_bg_categories=cfg.get("n_bg_categories", 4),
        n_bg_classes=cfg.get("n_bg_classes", 4),
        scale_std=cfg["scale_std"],
        offset_std=cfg["offset_std"],
        obj_on_alpha=cfg["obj_on_alpha"],
        obj_on_beta=cfg["obj_on_beta"],
        obj_res_from_fc=cfg["obj_res_from_fc"],
        obj_ch_mult_prior=cfg.get("obj_ch_mult_prior", cfg["obj_ch_mult"]),
        obj_ch_mult=cfg["obj_ch_mult"],
        obj_base_ch=cfg["obj_base_ch"],
        obj_final_cnn_ch=cfg["obj_final_cnn_ch"],
        bg_res_from_fc=cfg["bg_res_from_fc"],
        bg_ch_mult=cfg["bg_ch_mult"],
        bg_base_ch=cfg["bg_base_ch"],
        bg_final_cnn_ch=cfg["bg_final_cnn_ch"],
        use_resblock=cfg["use_resblock"],
        num_res_blocks=cfg["num_res_blocks"],
        cnn_mid_blocks=cfg.get("cnn_mid_blocks", False),
        mlp_hidden_dim=cfg.get("mlp_hidden_dim", 256),
        pint_enc_layers=cfg["pint_enc_layers"],
        pint_enc_heads=cfg["pint_enc_heads"],
        timestep_horizon=1,
        separate_depth_features=cfg.get("separate_depth_features", False),
        depth_feature_dim=cfg.get("depth_feature_dim", 0),
        split_loss=cfg.get("split_loss", False),
        depth_loss_ratio=cfg.get("depth_loss_ratio", 1.0),
    ).to(device)
    model.eval()
    if ckpt_path:
        _ = load_checkpoint(ckpt_path, model, None, None, map_location=device)
    return model, cfg


# -----------------------------------------------------------------------------#
# Reporting helpers
# -----------------------------------------------------------------------------#
def _stat(name, t):
    if isinstance(t, torch.Tensor):
        a = t.detach().cpu().numpy()
    else:
        a = np.asarray(t)
    print(f"  {name:30s} shape={tuple(a.shape)}  dtype={a.dtype}  "
          f"min={a.min():.4f}  max={a.max():.4f}  mean={a.mean():.4f}")


def _report_stage(label, **tensors):
    print(f"\n[{label}]")
    for k, v in tensors.items():
        _stat(k, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="close_jar",
                    help="snake_case RLBench task name")
    ap.add_argument("--variation", type=int, default=0)
    ap.add_argument("--cams", nargs="+",
                    default=["front", "overhead", "left_shoulder", "right_shoulder"],
                    help="cameras to enable for live voxelization")
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--grid-whd", type=int, nargs=3, default=None,
                    help="voxel grid (W H D); default = DLP cfg's voxel_grid_whd")
    ap.add_argument("--max-fuse-points", type=int, default=200000)
    ap.add_argument("--no-normalize", action="store_true",
                    help="disable per-frame [-1,1] normalization")
    ap.add_argument("--workspace-bounds", type=float, nargs=6, default=None,
                    metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
                    help="world-frame crop bounds applied before normalization "
                         "(default: PerAct/RLBench standard, see wrapper). "
                         "Pass --workspace-bounds 0 0 0 0 0 0 to disable.")
    ap.add_argument("--offline-cache", default=None,
                    help="path to one offline _voxels.pt to also decode + log "
                         "(e.g. .../episode0/voxel_cache/000000_voxels.pt). "
                         "Useful as a ground-truth A/B vs the live voxel.")
    ap.add_argument("--dlp-cfg", required=True)
    ap.add_argument("--dlp-ckpt", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=0,
                    help="take N no-op env steps after reset, re-encoding each time")
    ap.add_argument("--wandb-project", default=None,
                    help="if set, log voxels + decoded recon + kp to wandb")
    ap.add_argument("--wandb-run", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)

    # 1) DLP
    print(f"[load] DLP from cfg={args.dlp_cfg}")
    print(f"             ckpt={args.dlp_ckpt}")
    t0 = time.monotonic()
    dlp_model, dlp_cfg = load_dlp(args.dlp_cfg, args.dlp_ckpt, device)
    print(f"[load] DLP loaded in {time.monotonic() - t0:.1f}s")
    print(f"[load] cfg keys of interest: voxel_grid_whd={dlp_cfg.get('voxel_grid_whd')} "
          f"ch={dlp_cfg.get('ch')} n_kp_enc={dlp_cfg.get('n_kp_enc')} "
          f"learned_feature_dim={dlp_cfg.get('learned_feature_dim')} "
          f"learned_bg_feature_dim={dlp_cfg.get('learned_bg_feature_dim')}")

    # 2) Wrapper
    from diffuser.envs.rlbench_dlp_wrapper import (
        RLBenchDLPEnv, DEFAULT_WORKSPACE_BOUNDS,
    )
    if args.workspace_bounds is None:
        wb = DEFAULT_WORKSPACE_BOUNDS
    elif all(v == 0.0 for v in args.workspace_bounds):
        wb = None  # explicit "no crop" sentinel
    else:
        wb = tuple(args.workspace_bounds)
    print(f"[wrapper] workspace_bounds={wb}")

    env = RLBenchDLPEnv(
        task_name=args.task,
        dlp_model=dlp_model,
        dlp_cfg=dlp_cfg,
        cams=args.cams,
        image_size=args.image_size,
        headless=True,
        episode_length=400,
        grid_whd=tuple(args.grid_whd) if args.grid_whd else None,
        max_fuse_points=args.max_fuse_points,
        normalize_to_unit_cube=not args.no_normalize,
        workspace_bounds=wb,
        device=args.device,
    )

    # 3) wandb (optional)
    use_wandb = args.wandb_project is not None
    log_rgb_voxels = None
    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run or f"live_voxel_{args.task}",
            config={
                "task": args.task, "variation": args.variation,
                "cams": args.cams, "image_size": args.image_size,
                "grid_whd": tuple(env.grid_whd),
                "dlp_cfg": args.dlp_cfg, "dlp_ckpt": args.dlp_ckpt,
            },
        )
        try:
            from eval.eval_vox import log_rgb_voxels as _log_rgb_voxels
            log_rgb_voxels = _log_rgb_voxels
        except Exception as e:
            print(f"[wandb] log_rgb_voxels import failed ({e}); skipping voxel viz", flush=True)

    # 4) reset + encode
    try:
        print(f"\n[reset] task={args.task} variation={args.variation}")
        t0 = time.monotonic()
        obs_dict = env.reset(variation=args.variation)
        print(f"[reset] reset+encode in {time.monotonic() - t0:.2f}s")

        print(f"\n[language] {obs_dict['language']!r}")
        print(f"[variation] {obs_dict['variation_number']}")

        _report_stage(
            "stage 1: live encoding outputs",
            tokens=obs_dict["obs"],                  # (K, Dtok)
            bg_features=obs_dict["bg_features"],     # (bg_dim,)
            gripper_state=obs_dict["gripper_state"], # (10,)
            voxel=env._last_voxel,                   # (4, D, H, W)
            kp=env._last_kp,                         # (K, 3)
            cov=env._last_cov,                       # (K, 3, 3)
        )

        # Sanity: token feature_dim should be 8 + learned_feature_dim
        F = int(dlp_cfg.get("learned_feature_dim", 0))
        expected_dtok = 8 + F
        actual_dtok = obs_dict["obs"].shape[-1]
        print(f"\n[check] token Dtok={actual_dtok}, expected {expected_dtok} "
              f"(=8 + learned_feature_dim={F}): "
              f"{'OK' if actual_dtok == expected_dtok else 'MISMATCH'}")

        # Sanity: bg_features dim should equal learned_bg_feature_dim
        bg_dim_expected = int(dlp_cfg.get("learned_bg_feature_dim",
                                          dlp_cfg.get("learned_feature_dim", 0)))
        bg_dim_actual = obs_dict["bg_features"].shape[-1]
        print(f"[check] bg_features dim={bg_dim_actual}, expected {bg_dim_expected}: "
              f"{'OK' if bg_dim_actual == bg_dim_expected else 'MISMATCH'}")

        # Sanity: K
        K_expected = int(dlp_cfg.get("n_kp_enc", 0))
        K_actual = obs_dict["obs"].shape[0]
        print(f"[check] K (#particles)={K_actual}, expected {K_expected}: "
              f"{'OK' if K_actual == K_expected else 'MISMATCH'}")

        # 5) decoded reconstruction (gt vs full vs bg vs fg) and wandb logging
        # model.forward already runs decode_all internally and returns 'rec_rgb',
        # 'bg_rgb', 'dec_objects' (fg-with-alpha) etc. — same keys train_dlp_voxel
        # uses for val visuals.
        with torch.no_grad():
            vox_b = env._last_voxel.unsqueeze(0).to(device)
            rgb_only = vox_b[:, :int(dlp_cfg.get("ch", 3))].contiguous()
            kp_b = env._last_kp.unsqueeze(0).to(device)
            cov_b = env._last_cov.unsqueeze(0).to(device)
            meta = {"kmeans_kp": kp_b, "kmeans_cov": cov_b}
            full = dlp_model(rgb_only, deterministic=True, warmup=False,
                             with_loss=False, meta=meta)

        # Pull recon volumes — same indexing pattern as
        # train_dlp_voxel.py:_vis_one_sample (line 670+):
        #   gt   = model_output['x'][b]                              # [C,D,H,W]
        #   rec  = model_output['rec'][b]                            # [C,D,H,W]
        #   fg   = model_output['dec_objects'][b]                    # [C,D,H,W]  (already composited)
        #   bg   = (model_output['bg_mask'] * model_output['bg'][:,:3])[b]   # [C,D,H,W]
        b0 = 0
        gt_vol  = full["x"][b0]
        rec_vol = full["rec"][b0] if full.get("rec") is not None else full["rec_rgb"][b0]
        fg_vol  = full["dec_objects"][b0] if full.get("dec_objects") is not None else None
        bg_vol  = None
        if full.get("bg_mask") is not None and full.get("bg") is not None:
            bg_vol = (full["bg_mask"] * full["bg"][:, :3])[b0]
        # mu_tot[b0] is [T, K, 3] (T=1)
        mu_tot_b0 = full["mu_tot"][b0]
        if mu_tot_b0.dim() == 3:
            mu_tot_b0 = mu_tot_b0[0]

        print("\n[decoded recon shapes]")
        _stat("gt", gt_vol)
        _stat("rec_full", rec_vol)
        if fg_vol is not None: _stat("rec_fg", fg_vol)
        if bg_vol is not None: _stat("rec_bg", bg_vol)
        _stat("mu_tot (kp)", mu_tot_b0)

        if use_wandb and log_rgb_voxels is not None:
            base = f"live/{args.task}/var{args.variation}/reset"
            log_rgb_voxels(name=f"{base}/gt",       rgb_vol=gt_vol.cpu(),  KPx=mu_tot_b0.cpu(), step=0)
            log_rgb_voxels(name=f"{base}/rec_full", rgb_vol=rec_vol.cpu(), KPx=mu_tot_b0.cpu(), step=0)
            if fg_vol is not None:
                log_rgb_voxels(name=f"{base}/rec_fg", rgb_vol=fg_vol.cpu(), KPx=mu_tot_b0.cpu(), step=0)
            if bg_vol is not None:
                log_rgb_voxels(name=f"{base}/rec_bg", rgb_vol=bg_vol.cpu(), KPx=None, step=0)
            print(f"[wandb] logged gt/rec_full/rec_fg/rec_bg under '{base}'")

        # ---------- Optional: offline-cache A/B comparison ----------
        if args.offline_cache:
            print(f"\n[offline] loading {args.offline_cache}")
            import sys as _sys
            import numpy.core as _np_core
            _sys.modules['numpy._core'] = _np_core
            _sys.modules['numpy._core.multiarray'] = _np_core.multiarray
            cached = torch.load(args.offline_cache, weights_only=False)
            if isinstance(cached, dict) and cached.get("compressed"):
                shape = cached["shape"]; coords = cached["coords"].long()
                values = cached["values"].float()
                vox_off = torch.zeros(shape, dtype=torch.float32)
                vox_off[:, coords[:, 0], coords[:, 1], coords[:, 2]] = values.T
            else:
                vox_off = cached.float()
            print(f"[offline] vox shape={tuple(vox_off.shape)}  "
                  f"non_empty={int((vox_off[:3].abs().sum(0) > 0).sum())}")

            # Voxel-level diff vs the live wrapper output
            live_vox = env._last_voxel
            if live_vox.shape == vox_off.shape:
                rgb_diff = (live_vox[:3] - vox_off[:3]).abs()
                print(f"[diff live vs offline] "
                      f"rgb max={rgb_diff.max().item():.5f} "
                      f"mean={rgb_diff.mean().item():.7f} "
                      f"non-empty: live={int((live_vox[:3].abs().sum(0)>0).sum())} "
                      f"offline={int((vox_off[:3].abs().sum(0)>0).sum())}")

            # DLP-encode the offline voxel and log everything under offline/...
            with torch.no_grad():
                vob = vox_off.unsqueeze(0).to(device)
                rgb_off = vob[:, :int(dlp_cfg.get("ch", 3))].contiguous()
                kp_o, cov_o = dlp_model.prior_module.encode_prior(rgb_off)
                full_off = dlp_model(rgb_off, deterministic=True, warmup=False,
                                     with_loss=False,
                                     meta={"kmeans_kp": kp_o, "kmeans_cov": cov_o})
            gt_o   = full_off["x"][0]
            rec_o  = full_off["rec"][0] if full_off.get("rec") is not None else full_off["rec_rgb"][0]
            fg_o   = full_off["dec_objects"][0] if full_off.get("dec_objects") is not None else None
            bg_o = None
            if full_off.get("bg_mask") is not None and full_off.get("bg") is not None:
                bg_o = (full_off["bg_mask"] * full_off["bg"][:, :3])[0]
            mu_o = full_off["mu_tot"][0]
            if mu_o.dim() == 3: mu_o = mu_o[0]

            if use_wandb and log_rgb_voxels is not None:
                base_o = f"offline/{args.task}/frame0"
                log_rgb_voxels(name=f"{base_o}/gt",       rgb_vol=gt_o.cpu(),  KPx=mu_o.cpu(), step=0)
                log_rgb_voxels(name=f"{base_o}/rec_full", rgb_vol=rec_o.cpu(), KPx=mu_o.cpu(), step=0)
                if fg_o is not None:
                    log_rgb_voxels(name=f"{base_o}/rec_fg", rgb_vol=fg_o.cpu(), KPx=mu_o.cpu(), step=0)
                if bg_o is not None:
                    log_rgb_voxels(name=f"{base_o}/rec_bg", rgb_vol=bg_o.cpu(), KPx=None,      step=0)
                print(f"[wandb] logged offline gt/rec_full/rec_fg/rec_bg under '{base_o}'")

        # 7) optional follow-up steps with a no-op action (current pose, gripper open)
        for s in range(args.steps):
            print(f"\n[step {s+1}/{args.steps}] no-op (current pose)")
            cur_pose = env._last_gripper_pose
            # Build a no-op 10D action: keep current pos+rot6d, gripper open
            quat = cur_pose[3:7]
            from diffuser.envs.rlbench_dlp_wrapper import quat_xyzw_to_rot6d
            action = np.concatenate([
                cur_pose[:3], quat_xyzw_to_rot6d(quat), np.array([1.0])
            ]).astype(np.float32)
            t0 = time.monotonic()
            obs_dict, reward, done, info = env.step(action)
            dt = time.monotonic() - t0
            if obs_dict is None:
                print(f"  step failed: {info}")
                break
            print(f"  step+encode: {dt:.2f}s  reward={reward}  done={done}")
            if use_wandb and log_rgb_voxels is not None:
                # For each follow-up step also re-decode and log gt/full/bg/fg.
                with torch.no_grad():
                    vox_b = env._last_voxel.unsqueeze(0).to(device)
                    rgb_only = vox_b[:, :int(dlp_cfg.get("ch", 3))].contiguous()
                    kp_b = env._last_kp.unsqueeze(0).to(device)
                    cov_b = env._last_cov.unsqueeze(0).to(device)
                    full = dlp_model(rgb_only, deterministic=True, warmup=False,
                                     with_loss=False,
                                     meta={"kmeans_kp": kp_b, "kmeans_cov": cov_b})
                gt_vol = full["x"][0]
                rec_vol = full["rec"][0] if full.get("rec") is not None else full["rec_rgb"][0]
                fg_vol = full["dec_objects"][0] if full.get("dec_objects") is not None else None
                bg_vol = None
                if full.get("bg_mask") is not None and full.get("bg") is not None:
                    bg_vol = (full["bg_mask"] * full["bg"][:, :3])[0]
                mu_tot_b0 = full["mu_tot"][0]
                if mu_tot_b0.dim() == 3:
                    mu_tot_b0 = mu_tot_b0[0]
                base = f"live/{args.task}/var{args.variation}/step{s+1}"
                log_rgb_voxels(name=f"{base}/gt", rgb_vol=gt_vol.cpu(),
                               KPx=mu_tot_b0.cpu(), step=s + 1)
                log_rgb_voxels(name=f"{base}/rec_full", rgb_vol=rec_vol.cpu(),
                               KPx=mu_tot_b0.cpu(), step=s + 1)
                if bg_vol is not None:
                    log_rgb_voxels(name=f"{base}/rec_bg", rgb_vol=bg_vol.cpu(),
                                   KPx=None, step=s + 1)
                if fg_vol is not None:
                    log_rgb_voxels(name=f"{base}/rec_fg", rgb_vol=fg_vol.cpu(),
                                   KPx=mu_tot_b0.cpu(), step=s + 1)
            if done:
                print("  episode terminated; stopping")
                break

        print("\n[done]")

    finally:
        env.shutdown()
        if use_wandb:
            import wandb
            wandb.finish()


if __name__ == "__main__":
    main()
