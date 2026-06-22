#!/usr/bin/env python
"""
Diagnostic rollout: runs 1 episode and logs decoded DLP observations to wandb
at each replan step — both the current observation and the model's imagined future.

Usage:
    cd /home/ellina/Desktop/EC-Diffuser
    python diffuser/scripts/debug_rollout.py \
        --config mimicgen_coffee_dlp \
        --mode 16C_dlp \
        --ckpt_path /path/to/checkpoint.pt \
        --max_steps 300 \
        --seed 42
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import argparse
import json
import numpy as np
import torch
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIFFUSER_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
EC_DIFFUSER_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
LPWM_DEV = os.path.abspath(os.path.join(EC_DIFFUSER_ROOT, "..", "lpwm-dev"))

for p in [EC_DIFFUSER_ROOT, DIFFUSER_ROOT, LPWM_DEV]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from dlp_utils import log_rgb_voxels


def build_dlp_from_cfg(cfg, device, DLPClass):
    model = DLPClass(
        cdim=cfg["ch"], image_size=cfg["voxel_grid_whd"][0],
        normalize_rgb=cfg["normalize_rgb"], n_kp_per_patch=cfg["n_kp_per_patch"],
        patch_size=cfg["patch_size"], anchor_s=cfg["anchor_s"],
        n_kp_enc=cfg["n_kp_enc"], n_kp_prior=cfg["n_kp_prior"],
        pad_mode=cfg["pad_mode"], dropout=cfg["dropout"],
        features_dist=cfg.get("features_dist", "gauss"),
        learned_feature_dim=cfg["learned_feature_dim"],
        learned_bg_feature_dim=cfg.get("learned_bg_feature_dim", cfg["learned_feature_dim"]),
        n_fg_categories=cfg.get("n_fg_categories", 8),
        n_fg_classes=cfg.get("n_fg_classes", 4),
        n_bg_categories=cfg.get("n_bg_categories", 4),
        n_bg_classes=cfg.get("n_bg_classes", 4),
        scale_std=cfg["scale_std"], offset_std=cfg["offset_std"],
        obj_on_alpha=cfg["obj_on_alpha"], obj_on_beta=cfg["obj_on_beta"],
        obj_res_from_fc=cfg["obj_res_from_fc"],
        obj_ch_mult_prior=cfg.get("obj_ch_mult_prior", cfg["obj_ch_mult"]),
        obj_ch_mult=cfg["obj_ch_mult"], obj_base_ch=cfg["obj_base_ch"],
        obj_final_cnn_ch=cfg["obj_final_cnn_ch"],
        bg_res_from_fc=cfg["bg_res_from_fc"], bg_ch_mult=cfg["bg_ch_mult"],
        bg_base_ch=cfg["bg_base_ch"], bg_final_cnn_ch=cfg["bg_final_cnn_ch"],
        use_resblock=cfg["use_resblock"], num_res_blocks=cfg["num_res_blocks"],
        cnn_mid_blocks=cfg.get("cnn_mid_blocks", False),
        mlp_hidden_dim=cfg.get("mlp_hidden_dim", 256),
        pint_enc_layers=cfg["pint_enc_layers"], pint_enc_heads=cfg["pint_enc_heads"],
        timestep_horizon=1,
        separate_depth_features=cfg.get("separate_depth_features", False),
        depth_feature_dim=cfg.get("depth_feature_dim", 0),
        split_loss=cfg.get("split_loss", False),
        depth_loss_ratio=cfg.get("depth_loss_ratio", 1.0),
    ).to(device)
    model.eval()
    return model


def load_dlp_lpwm(dlp_cfg_path, dlp_ckpt_path, device):
    from utils.util_func import get_config
    from utils.log_utils import load_checkpoint
    from voxel_models import DLP as DLPClass
    dev = torch.device(device)
    cfg = get_config(dlp_cfg_path)
    model = build_dlp_from_cfg(cfg, dev, DLPClass)
    _ = load_checkpoint(dlp_ckpt_path, model, None, None, map_location=dev)
    model.eval()
    return model, cfg


@torch.no_grad()
def decode_and_log(envw, toks_np, bg_features, tag, step):
    """Decode DLP tokens to voxels and log to wandb as 3D plotly with color-coded keypoints."""
    import plotly.graph_objects as go

    dlp_dict = envw.unpack_tokens_to_dlp_format(toks_np)
    z = dlp_dict["z"]
    z_scale = dlp_dict["z_scale"]
    z_depth = dlp_dict["z_depth"]
    obj_on = dlp_dict["obj_on"]
    z_features = dlp_dict["z_features"]

    # Extract ALL keypoint positions and obj_on values
    kp_norm = z[0, 0].cpu().numpy()  # (K, 3) in [-1, 1]
    obj_on_vals = obj_on[0, 0, :, 0].cpu().numpy()  # (K,)

    # Build z_bg tensor from bg_features
    z_bg = None
    if bg_features is not None:
        bg_np = np.asarray(bg_features).flatten()
        z_bg = torch.from_numpy(bg_np).float().to(z.device)
        z_bg = z_bg.unsqueeze(0).unsqueeze(0)  # (1, 1, bg_dim)

    dec = envw.dlp.decode_all(
        z, z_scale, z_features, obj_on, z_depth,
        z_bg, None, warmup=False
    )

    # Get the volume to determine D, H, W for KP coordinate conversion
    vol_key = "rec_rgb" if "rec_rgb" in dec else "dec_objects_trans"
    vol = dec[vol_key][0].cpu().numpy()  # (3, D, H, W)
    _, D, H, W = vol.shape

    # Convert ALL KPs from [-1, 1] to voxel index space
    # DLP z_pos is (x, y, z) order — see spatial_transform() in util_func.py
    # PyTorch affine_grid: x->W, y->H, z->D
    kp_vox = np.empty_like(kp_norm)
    kp_vox[:, 0] = (kp_norm[:, 0] + 1) / 2 * (W - 1)  # x -> W
    kp_vox[:, 1] = (kp_norm[:, 1] + 1) / 2 * (H - 1)  # y -> H
    kp_vox[:, 2] = (kp_norm[:, 2] + 1) / 2 * (D - 1)  # z -> D

    # Log voxels WITHOUT KPs (we'll add them manually with colors)
    fig = log_rgb_voxels(
        name=tag, rgb_vol=vol,
        alpha_vol=None, KPx=None,
        step=None,  # don't log yet
        mode="splat", topk=60000, alpha_thresh=0.05,
        pad=2.0, show_axes=True,
    )

    # Add color-coded KP crosses: green=on, red=off
    half = 2.0
    for i, (kp, on_val) in enumerate(zip(kp_vox, obj_on_vals)):
        x, y, z_coord = float(kp[0]), float(kp[1]), float(kp[2])
        # Color: red (off) -> yellow (0.5) -> green (on)
        on = float(np.clip(on_val, 0, 1))
        if on < 0.5:
            r, g, b = 255, int(255 * on * 2), 0
        else:
            r, g, b = int(255 * (1 - (on - 0.5) * 2)), 255, 0
        color = f"rgb({r},{g},{b})"

        xs = [x - half, x + half, None, x, x, None, x, x, None]
        ys = [y, y, None, y - half, y + half, None, y, y, None]
        zs = [z_coord, z_coord, None, z_coord, z_coord, None,
              z_coord - half, z_coord + half, None]
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(width=6, color=color),
            name=f"KP{i} on={on:.2f}",
            showlegend=False,
        ))

    wandb.log({tag: fig}, step=step)


def main():
    parser = argparse.ArgumentParser(description="Debug rollout with DLP visualization")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, default="16C_dlp")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--random_init", action="store_true", default=True)
    parser.add_argument("--no_random_init", dest="random_init", action="store_false")
    parser.add_argument("--log_every_replan", type=int, default=1,
                        help="Log 3D obs every N replans (1=every replan)")
    parser.add_argument("--save_video", type=str, default=None,
                        help="Path to save rollout video (e.g., rollout.mp4)")
    args = parser.parse_args()

    # Init wandb
    run_name = f"debug_{args.config}_seed{args.seed}"
    wandb.init(project="ec-diffuser-debug", name=run_name, config=vars(args))

    import diffuser.utils as utils
    from diffuser.utils.arrays import set_global_device

    # Load config
    config_module = __import__(f"config.{args.config}", fromlist=[args.config])
    mode_args = config_module.mode_to_args.get(args.mode, {})
    base_args = config_module.base["diffusion"]
    merged_args = {**base_args, **mode_args}

    class Args:
        pass
    cfg = Args()
    for k, v in merged_args.items():
        setattr(cfg, k, v)
    cfg.device = args.device
    set_global_device(cfg.device)

    # Load DLP
    print("Loading DLP model...")
    dlp_model, _ = load_dlp_lpwm(
        getattr(cfg, 'dlp_cfg'), getattr(cfg, 'dlp_ckpt'), cfg.device
    )

    # Load dataset + diffusion model
    print("Loading dataset and model...")
    dataset_path = getattr(cfg, 'override_dataset_path')
    cfg.dataset_path = dataset_path
    cfg.savepath = os.path.dirname(args.ckpt_path).replace("/ckpt", "")

    dataset_config = utils.Config(
        cfg.loader, savepath=None, dataset_path=cfg.dataset_path,
        dataset_name=cfg.dataset, horizon=cfg.horizon,
        obs_only=getattr(cfg, 'obs_only', False),
        action_only=getattr(cfg, 'action_only', False),
        normalizer=cfg.normalizer, particle_normalizer=cfg.particle_normalizer,
        preprocess_fns=cfg.preprocess_fns, use_padding=cfg.use_padding,
        max_path_length=cfg.max_path_length, overfit=False,
        single_view=(getattr(cfg, 'input_type', 'dlp') == "dlp" and not cfg.multiview),
        action_z_scale=getattr(cfg, 'action_z_scale', 1.0),
        use_gripper_obs=getattr(cfg, 'use_gripper_obs', False),
        use_bg_obs=getattr(cfg, 'use_bg_obs', False),
    )
    dataset = dataset_config()

    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim
    gripper_dim = getattr(dataset, 'gripper_dim', 0)
    bg_dim = getattr(dataset, 'bg_dim', 0)

    model_config = utils.Config(
        cfg.model, savepath=None, features_dim=cfg.features_dim,
        action_dim=action_dim, hidden_dim=cfg.hidden_dim,
        projection_dim=cfg.projection_dim, n_head=cfg.n_heads,
        n_layer=cfg.n_layers, dropout=cfg.dropout, block_size=cfg.horizon,
        positional_bias=cfg.positional_bias, max_particles=cfg.max_particles,
        multiview=cfg.multiview, device=cfg.device,
        gripper_dim=gripper_dim, bg_dim=bg_dim,
    )
    diffusion_config = utils.Config(
        cfg.diffusion, savepath=None, horizon=cfg.horizon,
        observation_dim=observation_dim, action_dim=action_dim,
        gripper_dim=gripper_dim, bg_dim=bg_dim,
        n_timesteps=cfg.n_diffusion_steps, loss_type=cfg.loss_type,
        clip_denoised=cfg.clip_denoised, predict_epsilon=cfg.predict_epsilon,
        action_weight=cfg.action_weight, loss_weights=cfg.loss_weights,
        loss_discount=cfg.loss_discount, device=cfg.device,
        obs_only=getattr(cfg, 'obs_only', False),
        action_only=getattr(cfg, 'action_only', False),
    )

    renderer = None
    try:
        render_config = utils.Config(cfg.renderer, savepath=None, env=None,
                                     particle_dim=cfg.features_dim)
        renderer = render_config()
    except Exception:
        pass

    model = model_config()
    diffusion = diffusion_config(model)
    trainer_config = utils.Config(
        utils.Trainer, savepath=None, train_batch_size=cfg.batch_size,
        train_lr=cfg.learning_rate,
        gradient_accumulate_every=cfg.gradient_accumulate_every,
        ema_decay=cfg.ema_decay, sample_freq=cfg.sample_freq,
        save_freq=cfg.save_freq, label_freq=int(cfg.n_train_steps // cfg.n_saves),
        save_parallel=cfg.save_parallel, results_folder=cfg.savepath,
        bucket=cfg.bucket, n_reference=cfg.n_reference,
    )
    trainer = trainer_config(diffusion, dataset, renderer)

    # Load checkpoint
    ckpt_data = torch.load(args.ckpt_path, map_location=cfg.device)
    trainer.step = ckpt_data['step']
    trainer.model.load_state_dict(ckpt_data['model'])
    trainer.ema_model.load_state_dict(ckpt_data['ema'])
    print(f"Loaded checkpoint at step {trainer.step}")

    # Setup env
    from diffuser.envs.mimicgen_dlp_wrapper import DatasetGoalProvider
    from diffuser.eval_utils import extract_mimicgen_task_name, setup_mimicgen_env

    goal_provider = DatasetGoalProvider(dataset_path, shuffle=True)
    calib_h5_path = getattr(cfg, 'calib_h5_path')
    use_absolute_actions = getattr(cfg, 'use_absolute_actions', False)
    task = extract_mimicgen_task_name(calib_h5_path)
    max_steps = args.max_steps
    exe_steps = getattr(cfg, 'exe_steps', 8)
    grid_dhw = getattr(cfg, 'mimicgen_grid_dhw', (128, 128, 128))
    cams = tuple(getattr(cfg, 'mimicgen_cams', ["agentview", "sideview"]))
    pixel_stride = getattr(cfg, 'mimicgen_pixel_stride', 1)

    print(f"Task: {task}, max_steps: {max_steps}, exe_steps: {exe_steps}")

    env = setup_mimicgen_env(cfg, use_absolute_actions=use_absolute_actions)

    from diffuser.envs.mimicgen_dlp_wrapper import MimicGenDLPWrapper
    envw = MimicGenDLPWrapper(
        env=env, dlp_model=dlp_model, device=args.device,
        cams=cams, grid_dhw=grid_dhw, pixel_stride=pixel_stride,
        calib_h5_path=calib_h5_path, goal_provider=goal_provider,
        random_init=args.random_init, normalize_to_unit_cube=False, task=task,
    )

    device = torch.device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ========== RUN 1 EPISODE ==========
    print(f"\nRunning 1 episode (seed={args.seed}, random_init={args.random_init})")
    obs_vec = envw.reset()

    # Log initial observation
    K = envw.last_toks.shape[0]
    Dtok = envw.last_toks.shape[1]
    print(f"Obs: K={K}, Dtok={Dtok}, obs_dim={obs_vec.shape}, a_dim={action_dim}, "
          f"gripper_dim={gripper_dim}, bg_dim={bg_dim}")

    decode_and_log(envw, envw.last_toks, envw.last_bg_features,
                   "obs_current", step=0)
    if envw.last_vox is not None:
        log_rgb_voxels(
            name="gt_voxels", rgb_vol=envw.last_vox,
            alpha_vol=None, KPx=None,
            step=0, mode="splat", topk=60000, alpha_thresh=0.05,
            pad=2.0, show_axes=True,
        )

    # Tracking
    action_log = []
    replan_steps = []
    action_buffer = None
    action_idx = 0
    plan_idx = 0

    video_frames = []

    # Switch correction hack (kitchen only): proportional correction when loitering near switch
    # Only armed after the first flip — the policy handles that on its own.
    is_kitchen = (task is not None and "kitchen" in task)
    SWITCH_POS = np.array([-0.1389, 0.0760, 0.9899])
    LOITER_RADIUS = 0.05   # 5cm
    LOITER_STEPS = 15
    SWITCH_GAIN = 0.3
    loiter_count = 0
    CORRECTION_ARMED_AFTER = 500

    use_gripper_obs = getattr(dataset, 'use_gripper_obs', False)
    use_bg_obs = getattr(dataset, 'use_bg_obs', False)
    obs_start_idx = action_dim + gripper_dim + bg_dim
    H = dataset.horizon

    t = 0
    done = False
    while t < max_steps and not done:
        need_replan = (action_buffer is None) or (action_idx >= exe_steps)

        if need_replan:
            replan_steps.append(t)

            # Build condition (matches training eval exactly)
            cond_parts = []
            if use_gripper_obs and gripper_dim > 0:
                gripper_cond = envw.get_gripper_cond(horizon=H)
                if gripper_cond is not None and 0 in gripper_cond:
                    g_norm = dataset.normalizer.normalize(
                        gripper_cond[0][None], "gripper_state")[0]
                    cond_parts.append(torch.from_numpy(g_norm).float().to(device))
            if use_bg_obs and bg_dim > 0:
                bg_cond = envw.get_bg_cond(horizon=H)
                if bg_cond is not None and 0 in bg_cond:
                    b_norm = dataset.normalizer.normalize(
                        bg_cond[0][None], "bg_features")[0]
                    cond_parts.append(torch.from_numpy(b_norm).float().to(device))
            obs_norm = dataset.normalizer.normalize(obs_vec[None], "observations")[0]
            cond_parts.append(torch.from_numpy(obs_norm).float().to(device))

            cond_0 = torch.cat(cond_parts, dim=-1)[None, :]  # (1, total_dim)
            cond = {0: cond_0}

            # Sample trajectory
            sample = trainer.ema_model(cond, verbose=False)
            traj = sample.trajectories[0]  # (H, transition_dim)
            action_buffer = traj[:, :action_dim].detach().cpu().numpy()
            action_idx = 0

            # === LOG 3D — same name, step=t for slider ===
            should_log = (plan_idx % args.log_every_replan == 0)
            if should_log:
                # 1) Current observation decoded (scrub with slider)
                decode_and_log(envw, envw.last_toks, envw.last_bg_features,
                               "obs_current", step=t)

                # 2) GT voxels from simulator (before DLP encoding)
                if envw.last_vox is not None:
                    gt_vox = envw.last_vox  # (3, D, H, W) or (C, D, H, W)
                    log_rgb_voxels(
                        name="gt_voxels", rgb_vol=gt_vox,
                        alpha_vol=None, KPx=None,
                        step=t, mode="splat", topk=60000, alpha_thresh=0.05,
                        pad=2.0, show_axes=True,
                    )

                # 2) Model's predicted observations
                pred_obs_norm = traj[:, obs_start_idx:].detach().cpu().numpy()
                pred_obs = dataset.normalizer.unnormalize(pred_obs_norm, "observations")

                pred_bg = None
                if use_bg_obs and bg_dim > 0:
                    bg_s = action_dim + gripper_dim
                    bg_e = bg_s + bg_dim
                    pred_bg_norm = traj[:, bg_s:bg_e].detach().cpu().numpy()
                    pred_bg = dataset.normalizer.unnormalize(pred_bg_norm, "bg_features")

                # Log predicted at start, mid, end of horizon (each its own slider panel)
                for h_idx in [0, H // 2, H - 1]:
                    pred_toks = pred_obs[h_idx].reshape(K, Dtok)
                    bg_at_h = pred_bg[h_idx] if pred_bg is not None else envw.last_bg_features
                    decode_and_log(envw, pred_toks, bg_at_h,
                                   f"pred_h{h_idx:02d}", step=t)

            plan_idx += 1

        # Execute action
        a_norm = action_buffer[action_idx]
        a = dataset.normalizer.unnormalize(a_norm[None], "actions")[0]

        # Switch correction (kitchen only): detect loitering and nudge toward switch
        dist_to_switch = -1.0
        if is_kitchen:
            eef_pos = envw.last_gripper_state[:3] if envw.last_gripper_state is not None else None
            if eef_pos is not None:
                dist_to_switch = np.linalg.norm(eef_pos - SWITCH_POS)
                if t > CORRECTION_ARMED_AFTER:
                    if dist_to_switch < LOITER_RADIUS:
                        loiter_count += 1
                    else:
                        loiter_count = 0
                    if loiter_count > LOITER_STEPS:
                        direction = (SWITCH_POS - eef_pos)
                        direction = direction / (np.linalg.norm(direction) + 1e-8)
                        a[:3] = direction * 0.5
                        print(f"  [CORRECTION] t={t} dist={dist_to_switch:.4f} loiter={loiter_count} eef={eef_pos}")
                if t % 50 == 0 or dist_to_switch < LOITER_RADIUS:
                    print(f"  t={t} dist_to_switch={dist_to_switch:.4f} loiter={loiter_count} eef={eef_pos}")

        action_log.append({
            "t": t, "plan_idx": plan_idx - 1, "action_idx": action_idx,
            "x": float(a[0]), "y": float(a[1]), "z": float(a[2]),
            "grip": float(a[6]) if len(a) > 6 else 0.0,
            "loiter": loiter_count, "dist_switch": float(dist_to_switch),
        })

        # Log per-step action scalars (line charts in wandb)
        wandb.log({
            "actions/x": float(a[0]), "actions/y": float(a[1]),
            "actions/z": float(a[2]),
            "actions/gripper": float(a[6]) if len(a) > 6 else 0.0,
            "step": t,
        })

        obs_vec, r, done, info = envw.step(a)
        action_idx += 1
        t += 1

        # Collect video frame
        if args.save_video:
            raw_obs = envw.env.get_observation()
            for cam_key in ["agentview_image", "sideview_image"]:
                if cam_key in raw_obs:
                    frame = np.asarray(raw_obs[cam_key])
                    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
                        frame = np.transpose(frame, (1, 2, 0))
                    if frame.dtype != np.uint8:
                        if frame.max() <= 1.0:
                            frame = (frame * 255).clip(0, 255).astype(np.uint8)
                        else:
                            frame = frame.clip(0, 255).astype(np.uint8)
                    video_frames.append(frame)
                    break

        if info.get("success", False):
            print(f"SUCCESS at t={t}")
            done = True

    success = bool(info.get("success", False))
    print(f"\nEpisode done: t={t}, success={success}, replans={plan_idx}")

    # Save video
    if args.save_video and len(video_frames) > 0:
        import imageio
        os.makedirs(os.path.dirname(args.save_video) or '.', exist_ok=True)
        imageio.mimsave(args.save_video, video_frames, fps=30)
        print(f"Saved video ({len(video_frames)} frames) to: {args.save_video}")

    # Log final observation
    decode_and_log(envw, envw.last_toks, envw.last_bg_features,
                   "obs_current", step=t)

    # ========== ACTION TRAJECTORY PLOT ==========
    ts = [a["t"] for a in action_log]
    xs = [a["x"] for a in action_log]
    ys = [a["y"] for a in action_log]
    zs = [a["z"] for a in action_log]
    gs = [a["grip"] for a in action_log]

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for ax, vals, label in zip(axes, [xs, ys, zs, gs], ["x", "y", "z", "gripper"]):
        ax.plot(ts, vals, linewidth=1)
        ax.set_ylabel(label)
        for rp in replan_steps:
            ax.axvline(rp, color="red", alpha=0.3, linewidth=0.5)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("timestep")
    axes[0].set_title(f"Actions over time (success={success}, replans={plan_idx})")
    plt.tight_layout()
    wandb.log({"actions/trajectory_plot": wandb.Image(fig)})
    plt.close(fig)

    # Log summary
    wandb.log({
        "summary/success": int(success),
        "summary/episode_length": t,
        "summary/n_replans": plan_idx,
        "summary/seed": args.seed,
    })

    # Save action log as artifact
    log_path = os.path.join(wandb.run.dir, "action_log.json")
    with open(log_path, "w") as f:
        json.dump({"actions": action_log, "replan_steps": replan_steps,
                    "success": success, "length": t}, f, indent=2)
    wandb.save(log_path)

    try:
        env.close()
    except Exception:
        pass

    wandb.finish()
    print("Done. Check wandb for 3D visualizations.")


if __name__ == "__main__":
    main()
