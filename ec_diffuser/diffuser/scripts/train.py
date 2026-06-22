import warnings
warnings.filterwarnings("ignore")

import os
import sys

import wandb
import torch

import diffuser.utils as utils
from diffuser.utils.arrays import set_global_device
from diffuser.utils.args import ArgsParser

import logging
logging.basicConfig(level=logging.WARNING, force=True) 


# -----------------------------------------------------------------------------#
#                        make lpwm-dev importable                               #
# -----------------------------------------------------------------------------#

# Resolve lpwm-dev so `from voxel_models import DLP` works. Strategy:
#   1) explicit LPWM_DEV env var (most portable across machines)
#   2) sibling of EC-Diffuser: train.py is at EC-Diffuser/diffuser/scripts/train.py
#      so EC-Diffuser/.. is 3 levels up, then look for ./lpwm-dev there
# If voxel_models is already importable (e.g. installed in the env), nothing
# extra is needed.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_LPWM_CANDIDATES = []
_env_lpwm = os.environ.get("LPWM_DEV")
if _env_lpwm:
    _LPWM_CANDIDATES.append(_env_lpwm)
_LPWM_CANDIDATES.append(os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "..", "lpwm-dev")))
for _p in _LPWM_CANDIDATES:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)
        print(f"[train] added lpwm-dev to sys.path: {_p}", flush=True)
        break
else:
    # No directory matched; rely on voxel_models being installed in the env.
    print("[train] no lpwm-dev dir found via LPWM_DEV or sibling; "
          "expecting voxel_models to be importable from env", flush=True)


# -----------------------------------------------------------------------------#
#                          LPWM DLP loading (cfg + ckpt)                        #
# -----------------------------------------------------------------------------#

def build_dlp_from_cfg(cfg, device, DLPClass):
    # mirrors your preprocessing constructor exactly
    model = DLPClass(
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
    return model


def load_dlp_lpwm(dlp_cfg_path: str, dlp_ckpt_path: str, device: str):
    # from lpwm-dev
    from utils.util_func import get_config
    from utils.log_utils import load_checkpoint
    from voxel_models import DLP as DLPClass

    dev = torch.device(device)
    cfg = get_config(dlp_cfg_path)
    model = build_dlp_from_cfg(cfg, dev, DLPClass)
    _ = load_checkpoint(dlp_ckpt_path, model, None, None, map_location=dev)
    model.eval()
    return model, cfg


# -----------------------------------------------------------------------------#
#                                   setup                                      #
# -----------------------------------------------------------------------------#

args = ArgsParser().parse_args("diffusion")

# -----------------------------------------------------------------------------#
#                          (optional) Accelerate / DDP                          #
# -----------------------------------------------------------------------------#
# When launched with `accelerate launch --num_processes=N scripts/train.py ...`
# (or `torchrun --nproc_per_node=N`), HF Accelerate sets LOCAL_RANK / WORLD_SIZE
# in the environment. We detect that and instantiate an Accelerator so the
# Trainer wraps model/optimizer/dataloader in DDP. Plain `python scripts/train.py
# ...` keeps the single-GPU path untouched (accelerator stays None).
accelerator = None
_is_distributed = (
    "LOCAL_RANK" in os.environ
    or int(os.environ.get("WORLD_SIZE", "1")) > 1
)
if _is_distributed:
    from accelerate import Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulate_every,
    )
    # Each rank owns one GPU (cuda:LOCAL_RANK). Override args.device so the
    # model/diffusion configs (which use args.device for `.to(device)`) build
    # on the right physical GPU per rank.
    args.device = str(accelerator.device)
    set_global_device(accelerator.device)
    print(
        f"[train] Accelerate: num_processes={accelerator.num_processes} "
        f"process_index={accelerator.process_index} "
        f"is_main={accelerator.is_main_process} device={accelerator.device}",
        flush=True,
    )
else:
    set_global_device(args.device)

_is_main_process = accelerator is None or accelerator.is_main_process

eval_backend = getattr(args, "eval_backend", "none")   # "none" | "mimicgen" | "isaac"
eval_freq = int(getattr(args, "eval_freq", 0) or 0)
do_eval = (eval_freq > 0) and (eval_backend != "none")

print(f"[eval cfg] eval_backend={eval_backend} eval_freq={eval_freq} do_eval={do_eval}", flush=True)


# -----------------------------------------------------------------------------#
#                                   dataset                                    #
# -----------------------------------------------------------------------------#

dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, "dataset_config.pkl"),
    dataset_path=args.dataset_path,
    dataset_name=args.dataset,
    horizon=args.horizon,
    obs_only=args.obs_only,
    action_only=args.action_only,
    normalizer=args.normalizer,
    particle_normalizer=args.particle_normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
    overfit=args.overfit,
    single_view=(args.input_type == "dlp" and not args.multiview),
    action_z_scale=getattr(args, 'action_z_scale', 1.0),
    use_gripper_obs=getattr(args, 'use_gripper_obs', False),
    use_bg_obs=getattr(args, 'use_bg_obs', False),
    keypose_mode=getattr(args, 'keypose_mode', False),
)

render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, "render_config.pkl"),
    env=None,
    particle_dim=args.features_dim,
)

dataset = dataset_config()

print("DATASTE TYPE: ", dataset.__class__.__name__)
renderer = render_config()

# Load DLP for renderer reference renders (independent of eval backend)
dlp_cfg_path = getattr(args, "dlp_cfg", None)
dlp_ckpt_path = getattr(args, "dlp_ckpt", None)
if dlp_cfg_path and dlp_ckpt_path:
    print(f"[renderer] loading DLP for reference renders: cfg={dlp_cfg_path} ckpt={dlp_ckpt_path}", flush=True)
    _renderer_dlp, _ = load_dlp_lpwm(dlp_cfg_path, dlp_ckpt_path, args.device)
    renderer.latent_rep_model = _renderer_dlp
else:
    print("[renderer] no DLP cfg/ckpt provided, reference renders will be skipped", flush=True)

print("renderer: ", renderer)

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim
gripper_dim = getattr(dataset, 'gripper_dim', 0)
bg_dim = getattr(dataset, 'bg_dim', 0)

if gripper_dim > 0:
    print(f"[train] Using gripper observations: gripper_dim={gripper_dim}")
if bg_dim > 0:
    print(f"[train] Using background features: bg_dim={bg_dim}")


# -----------------------------------------------------------------------------#
#                              model & trainer                                 #
# -----------------------------------------------------------------------------#

model_config = utils.Config(
    args.model,
    savepath=(args.savepath, "model_config.pkl"),
    features_dim=args.features_dim,
    action_dim=action_dim,
    hidden_dim=args.hidden_dim,
    projection_dim=args.projection_dim,
    n_head=args.n_heads,
    n_layer=args.n_layers,
    dropout=args.dropout,
    block_size=args.horizon,
    positional_bias=args.positional_bias,
    max_particles=args.max_particles,
    multiview=args.multiview,
    device=args.device,
    gripper_dim=gripper_dim,
    bg_dim=bg_dim,
    lang_dim=getattr(args, 'lang_dim', 0),
    act_pos_dim=getattr(args, 'act_pos_dim', 3),
    act_rot_dim=getattr(args, 'act_rot_dim', 6),
    act_grip_dim=getattr(args, 'act_grip_dim', 1),
    prop_pos_dim=getattr(args, 'prop_pos_dim', 3),
    prop_rot_dim=getattr(args, 'prop_rot_dim', 6),
    prop_grip_dim=getattr(args, 'prop_grip_dim', 1),
    split_action_tokens=getattr(args, 'split_action_tokens', None),
    use_cond_tokens=getattr(args, 'keypose_mode', False),
)

diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, "diffusion_config.pkl"),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    gripper_dim=gripper_dim,
    bg_dim=bg_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    clip_denoised=args.clip_denoised,
    predict_epsilon=args.predict_epsilon,
    action_weight=args.action_weight,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    device=args.device,
    obs_only=args.obs_only,
    action_only=args.action_only,
    keypose_mode=getattr(args, 'keypose_mode', False),
)

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, "trainer_config.pkl"),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
)

model = model_config()
diffusion = diffusion_config(model)
# Pass the Accelerator at call time (not via Config) so it doesn't end up in
# the pickled `trainer_config.pkl` -- Accelerator instances aren't picklable.
trainer = trainer_config(diffusion, dataset, renderer, accelerator=accelerator)


# -----------------------------------------------------------------------------#
#                         test forward & backward pass                          #
# -----------------------------------------------------------------------------#

# Skip the sanity forward/backward when running under DDP -- DDP's gradient
# sync hooks fire only when going through the wrapped module's __call__, and
# leaving stray grads from this test would interfere with the first real step.
# Single-GPU mode keeps the existing test.
if accelerator is None:
    print("Testing forward...", end=" ", flush=True)
    batch = utils.batchify(dataset[0])
    loss, _ = diffusion.loss(*batch)
    loss.backward()
    print("✓", flush=True)


# -----------------------------------------------------------------------------#
#                                    wandb                                     #
# -----------------------------------------------------------------------------#

# Only the main process talks to wandb. Other ranks would either duplicate the
# run or trip wandb's lockfile.
if _is_main_process:
    wandb_run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group_name,
        config=args,
        sync_tensorboard=False,
        settings=wandb.Settings(start_method="fork"),
    )
    wandb_run.name = f"{args.dataset}_H{args.horizon}_exe{getattr(args, 'exe_steps', 1)}"
else:
    # Disable wandb on non-main ranks so any stray `wandb.log(...)` is a no-op.
    os.environ["WANDB_MODE"] = "disabled"
    wandb_run = None


# -----------------------------------------------------------------------------#
#                               mimicgen eval wiring                            #
# -----------------------------------------------------------------------------#

dlp_model = None
dlp_cfg = None
make_env_fn = None
calib_h5_path = None
goal_provider = None  # NEW: for dataset-based goal conditioning

if do_eval and eval_backend == "mimicgen":
    calib_h5_path = getattr(args, "calib_h5_path", None)
    dlp_ckpt = getattr(args, "dlp_ckpt", None)
    dlp_cfg_path = getattr(args, "dlp_cfg", None)

    if calib_h5_path is None:
        raise RuntimeError("eval_backend='mimicgen' requires --calib_h5_path")
    if dlp_ckpt is None:
        raise RuntimeError("eval_backend='mimicgen' requires --dlp_ckpt")
    if dlp_cfg_path is None:
        raise RuntimeError("eval_backend='mimicgen' requires --dlp_cfg (LPWM cfg used to build DLP)")

    print(f"[mimicgen eval] loading DLP from cfg={dlp_cfg_path} ckpt={dlp_ckpt}", flush=True)
    dlp_model, dlp_cfg = load_dlp_lpwm(dlp_cfg_path, dlp_ckpt, args.device)

    renderer.latent_rep_model = dlp_model

    # NEW: Set up goal provider from dataset for init_state + goal pairing
    # This ensures eval uses the same (init_state, goal) pairs as training
    from diffuser.envs.mimicgen_dlp_wrapper import DatasetGoalProvider
    print("=" * 60)
    print("[mimicgen eval] Setting up DatasetGoalProvider for goal conditioning")
    print(f"[mimicgen eval]   Loading from: {args.dataset_path}")
    goal_provider = DatasetGoalProvider(args.dataset_path, shuffle=True)
    print("[mimicgen eval] ✓ Goal provider ready - will use paired (init_state, goal_tokens)")
    print("=" * 60, flush=True)

    # NEW: Check if using absolute actions (control_delta=False)
    use_absolute_actions = getattr(args, "use_absolute_actions", True)
    print(f"[mimicgen eval] use_absolute_actions = {use_absolute_actions}", flush=True)

    # Extract task name from HDF5 for task-specific voxel bounds
    from diffuser.eval_utils import extract_mimicgen_task_name
    mimicgen_task = getattr(args, "mimicgen_task", None)  # Allow config override
    if mimicgen_task is None:
        mimicgen_task = extract_mimicgen_task_name(calib_h5_path)
    if mimicgen_task is not None:
        print(f"[mimicgen eval] Using task-specific bounds for task: '{mimicgen_task}'")
    else:
        print(f"[mimicgen eval] WARNING: Could not determine task name, using default bounds")

    def make_env_fn():
        from diffuser.eval_utils import setup_mimicgen_env
        return setup_mimicgen_env(args, use_absolute_actions=use_absolute_actions)


# -----------------------------------------------------------------------------#
#                          rlbench eval wiring                                  #
# -----------------------------------------------------------------------------#

if do_eval and eval_backend == "rlbench":
    dlp_ckpt = getattr(args, "dlp_ckpt", None)
    dlp_cfg_path = getattr(args, "dlp_cfg", None)
    if dlp_ckpt is None or dlp_cfg_path is None:
        raise RuntimeError("eval_backend='rlbench' requires --dlp_ckpt and --dlp_cfg")

    if dlp_model is None:
        # Renderer already loaded a DLP at module level (renderer.latent_rep_model);
        # reuse if possible, otherwise load a fresh one for the wrapper.
        dlp_model = getattr(renderer, "latent_rep_model", None)
        if dlp_model is None:
            print(f"[rlbench eval] loading DLP from cfg={dlp_cfg_path} ckpt={dlp_ckpt}", flush=True)
            dlp_model, dlp_cfg = load_dlp_lpwm(dlp_cfg_path, dlp_ckpt, args.device)
        else:
            from utils.util_func import get_config
            dlp_cfg = get_config(dlp_cfg_path)

    # Cache the wrapper across eval cycles. Headless CoppeliaSim's OpenGL
    # has a known cross-launch texture-loss bug — only the first launch keeps
    # textures, every subsequent launch in the same process renders washed-out.
    _rlbench_env_cache = {"env": None}

    def make_env_fn():
        if _rlbench_env_cache["env"] is not None:
            return _rlbench_env_cache["env"]
        from diffuser.envs.rlbench_dlp_wrapper import RLBenchDLPEnv
        env = RLBenchDLPEnv(
            task_name=args.dataset,
            dlp_model=dlp_model,
            dlp_cfg=dlp_cfg,
            cams=getattr(args, "rlbench_eval_cams",
                         ["front", "overhead", "left_shoulder", "right_shoulder"]),
            image_size=int(getattr(args, "rlbench_eval_image_size", 128)),
            headless=True,
            episode_length=int(getattr(args, "rlbench_eval_max_steps", 200)),
            device=args.device,
        )
        _rlbench_env_cache["env"] = env
        return env

    def make_policy_fn():
        from diffuser.sampling import LanguageConditionedPolicy
        return LanguageConditionedPolicy(
            diffusion_model=trainer.ema_model,
            normalizer=dataset.normalizer,
            preprocess_fns=[],
            clip_model_name=getattr(args, "clip_model_name", "openai/clip-vit-base-patch32"),
            lang_pooled=bool(getattr(args, "lang_pooled", False)),
            max_lang_tokens=int(getattr(args, "max_lang_tokens", 32)),
        )


# -----------------------------------------------------------------------------#
#                                  main loop                                   #
# -----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

# Auto-resume from latest checkpoint if one exists
start_epoch = 0
if trainer.load_latest():
    start_epoch = trainer.step // int(args.n_steps_per_epoch)
    print(f"[resume] Resuming from step {trainer.step}, skipping to epoch {start_epoch}/{n_epochs}", flush=True)

for i in range(start_epoch, n_epochs):
    print(f"Epoch {i} / {n_epochs} | {args.savepath}", flush=True)
    trainer.train(n_train_steps=args.n_steps_per_epoch)

    # Sync ranks before main-only eval. RLBench rollouts use CoppeliaSim
    # (single-process, GPU-agnostic) so they only run on rank 0; other ranks
    # would try to launch a second sim and stall. Wait here so non-main ranks
    # don't race ahead into the next training epoch while rank 0 is still
    # rolling out.
    if accelerator is not None:
        accelerator.wait_for_everyone()

    # eval AFTER each epoch; with eval_freq=1 it runs every epoch.
    # Eval rollouts only run on the main process; non-main ranks fall through
    # and the wait_for_everyone above resyncs them at the next epoch.
    if do_eval and _is_main_process and ((i + 1) % eval_freq == 0):
        # Save checkpoint before eval (synced with eval_freq)
        trainer.save(i)
        print(f"[eval] starting {eval_backend} eval at epoch={i} step={trainer.step}", flush=True)

        if eval_backend == "mimicgen":
            sim_stats = trainer.eval_mimicgen_rollouts(
                make_env_fn=make_env_fn,
                dlp_model=dlp_model,
                calib_h5_path=calib_h5_path,
                n_episodes=getattr(args, "mimicgen_eval_episodes", 5),
                max_steps=getattr(args, "mimicgen_max_steps", 50),
                bounds_xyz=getattr(args, "mimicgen_bounds_xyz", ((-2, 2), (-2, 2), (-0.2, 2.5))),
                grid_dhw=getattr(args, "mimicgen_grid_dhw", (128, 128, 128)),
                cams=getattr(args, "mimicgen_cams", ("agentview", "sideview")),
                pixel_stride=getattr(args, "mimicgen_pixel_stride", 2),
                goal_from_env_fn=getattr(args, "goal_from_env_fn", None),
                goal_provider=goal_provider,  # NEW: dataset-based goal provider
                random_init=getattr(args, "random_init_eval", False),  # NEW: random vs dataset init
                task=mimicgen_task,  # Task name for task-specific voxel bounds
                renderer_3d=renderer,
                exe_steps=getattr(args, "exe_steps", 1),  # ACTION CHUNKING: how many actions to execute per plan
            )

            # avoid double-prefix if eval returns sim/... already
            log_stats = {k if k.startswith("sim/") else f"sim/{k}": v for k, v in sim_stats.items()}
            wandb.log({"step": trainer.step, **log_stats})
            print(f"[mimicgen eval] epoch={i} :: {sim_stats}", flush=True)

        elif eval_backend == "rlbench":
            sim_stats = trainer.eval_rlbench_rollouts(
                make_env_fn=make_env_fn,
                make_policy_fn=make_policy_fn,
                n_episodes=int(getattr(args, "rlbench_eval_episodes", 2)),
                max_steps=int(getattr(args, "rlbench_eval_max_steps", 200)),
                exe_steps=int(getattr(args, "exe_steps", 1)),
                video_fps=int(getattr(args, "rlbench_eval_video_fps", 10)),
                wandb_step=trainer.step,
            )

            log_stats = {k if k.startswith("sim/") else f"sim/{k}": v
                         for k, v in sim_stats.items()}
            wandb.log({"step": trainer.step, **log_stats})
            print(f"[rlbench eval] epoch={i} :: {sim_stats}", flush=True)

        else:
            raise RuntimeError(
                f"eval_backend={eval_backend} not supported in this script without extra wiring."
            )
