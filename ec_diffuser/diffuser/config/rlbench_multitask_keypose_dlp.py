"""Multi-task keypose v2 config across all 10 RLBench-PerAct tasks.

Drives a single keypose-mode policy on all tasks jointly (same setup
as 3DDA / RVT / PerAct headline numbers). Language conditioning disambiguates
which task per episode.

Paths default to lambda's remote layout (/home/ubuntu/tal-lpwm-neurips-2026/...).
For local rollout/dev rewrite the paths to /home/ellina/Desktop/data/... per
project_remote_data_path memory.

DLP ckpts: each task has its own dlp_ckpt.pt and dlp_config.json under its
per-task subdir. They are byte-identical today (same multi-task DLP run, copied
into each task dir; md5 a49c1fa6...), so a single representative ckpt is loaded
into the model at training time. The full per-task list is exposed below
(TASK_DLP_CKPTS / TASK_DLP_CFGS) so future per-task eval can pick the correct
one if the per-task ckpts ever diverge.
"""

from diffuser.utils import watch

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('seed', 'seed'),
]

logbase = 'data'

# Lambda root for keypose-aware preprocessed voxel tokens. Rewrite per-machine.
DATA_ROOT = '/home/ellina/Desktop/data/preprocessed_voxel_tokens_with_keyposes'

# All 10 PerAct keypose tasks. Each pkl has shape (100 episodes, max_path_per_task, ...).
# Global max_path = 927 (stack_blocks); global max_keyposes = 24 (stack_blocks).
TASK_NAMES = [
    'close_jar',
    'meat_off_grill',
    'open_drawer',
    'push_buttons',
    'put_item_in_drawer',
    'reach_and_drag',
    'slide_block_to_color_target',
    'stack_blocks',
    'sweep_to_dustpan_of_size',
    'turn_tap',
]

# Per-task data references. The lists are positionally aligned -- index i
# refers to the same task across pkl / dlp_ckpt / dlp_cfg.
TASK_PKLS      = [f'{DATA_ROOT}/rlbench_{t}/{t}.pkl'             for t in TASK_NAMES]
TASK_DLP_CKPTS = [f'{DATA_ROOT}/rlbench_{t}/dlp_ckpt.pt'         for t in TASK_NAMES]
TASK_DLP_CFGS  = [f'{DATA_ROOT}/rlbench_{t}/dlp_config.json'     for t in TASK_NAMES]

# Representative single-DLP path used at training time. All 10 are byte-identical
# today, so picking any is equivalent. Per-task eval should index TASK_DLP_CKPTS
# by task to remain correct if/when per-task ckpts diverge.
_REP_TASK_IDX = TASK_NAMES.index('meat_off_grill')

# RLBench multitask language-conditioned config (keypose v2).
# IMPORTANT: key must match mode computed in setup.py: "{num_entity}C_{input_type}"
mode_to_args = {
  '16C_dlp': {
    'keypose_mode': True,

    # Multi-task: list of pkls to concatenate. The dataset loader detects the
    # list and concatenates episode-axis after padding to global max_T/max_kp.
    'dataset': 'multitask',
    'override_dataset_path': TASK_PKLS,
    'calib_h5_path': None,

    # Single-DLP path for the model's encoder at training time (representative).
    'dlp_ckpt': TASK_DLP_CKPTS[_REP_TASK_IDX],
    'dlp_ctor': "voxel_models:DLP",
    'dlp_cfg': TASK_DLP_CFGS[_REP_TASK_IDX],

    # Per-task DLP references for downstream per-task eval. Not yet consumed
    # by the train loop; intended for the multi-task eval extension that swaps
    # tasks via RLBenchDLPEnv.set_task and (optionally) per-task DLPs.
    'task_names':     TASK_NAMES,
    'task_dlp_ckpts': TASK_DLP_CKPTS,
    'task_dlp_cfgs':  TASK_DLP_CFGS,

    'features_dim': 12,       # z(3)+scale(3)+depth(1)+obj_on(1)+feat(4)
    'gripper_dim': 10,        # pos(3)+rot6d(6)+open(1)
    'use_gripper_obs': True,  # cond-tokens path: current gripper feeds into cond[0]
    'gripper_state_mask_ratio': 0.0,
    'bg_dim': 2,
    'use_bg_obs': True,
    'max_particles': 40,
    'multiview': False,
    'device': 'cuda:0',
    'max_path_length': 927,   # global max across the 10 tasks (stack_blocks)
    'max_demos': 100,         # per-task; total = 10 * 100 = 1000 episodes
    # In-training eval is gated off (eval_backend='none') because make_env_fn
    # uses task_name=args.dataset='multitask', which isn't a real RLBench task.
    # Re-enable once per-task eval-loop is wired (RLBenchDLPEnv.set_task +
    # iterate over task_names). Run external per-task paper_eval after training
    # for now.
    'eval_freq': 60,
    'eval_backend': 'none',
    'rlbench_eval_episodes': 2,
    'rlbench_eval_max_steps': 20,  # keypose attempts per episode
    'rlbench_eval_cams': ['front', 'overhead', 'left_shoulder', 'right_shoulder'],
    'rlbench_eval_image_size': 128,
    'rlbench_eval_video_fps': 10,
    'n_steps_per_epoch': 500,
    # --- RLBench-specific ---
    'action_dim': 10,
    'lang_dim': 512,
    'lang_pooled': False,
    'max_lang_tokens': 32,
    'clip_model_name': 'openai/clip-vit-base-patch32',
    'lang_device': 'cpu',
    # -------------------------
    "use_absolute_actions": True,
    'horizon': 1,    # predict only the next keypose (3DDA-style); cond is current keypose
    'exe_steps': 1,
    "random_init": True,
    "random_init_eval": True,
  },
}


base = {
    'diffusion': {
        'model': 'models.AdaLNPINTDenoiser',
        'diffusion': 'models.GaussianDiffusion',

        'keypose_mode': True,
        'horizon': 1,
        'features_dim': 7,
        'hidden_dim': 256,
        'projection_dim': 256,
        'n_heads': 8,
        'n_layers': 12,        # multi-task: data is plentiful, capacity helps
        'dropout': 0.0,

        'n_diffusion_steps': 100,
        'action_weight': 1,

        'max_particles': 64,
        'positional_bias': False,
        'multiview': False,

        # dataset
        'loader': 'datasets.LanguageConditionedDataset',
        'normalizer': 'GaussianNormalizer',
        'particle_normalizer': 'ParticleGaussianNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 10,
        'obs_only': False,
        'action_only': False,
        'action_z_scale': 1.0,
        'gripper_state_mask_ratio': 0.0,

        # serialization
        'logbase': logbase,
        'prefix': 'diffusion/rlbench_multitask_keypose/',
        'exp_name': watch(args_to_watch),

        # training
        'n_steps_per_epoch': 200,
        'loss_type': 'l1',
        'n_train_steps': 2e6,
        'batch_size': 128,      # multi-task: 10x data, larger batch keeps tasks balanced per step
        'learning_rate': 8e-5,
        'gradient_accumulate_every': 1,
        'ema_decay': 0.995,
        'save_freq': 10_000,
        'eval_freq': 10**9,
        'sample_freq': 1,
        'n_saves': 2,
        'save_parallel': False,
        'n_reference': 1,
        'bucket': None,
        'device': 'cuda:0',
        'seed': 0,
        'renderer': 'utils.ParticleRenderer3D',
        'predict_epsilon': False,
        'env_config_dir': 'env_config/n_cubes',

        'loss_weights': None,
        'loss_discount': 1,

        'exe_steps': 1,
    },

    'plan': {
        'policy': 'sampling.LanguageConditionedPolicy',
        'max_episode_length': 50,
        'batch_size': 1,
        'preprocess_fns': [],
        'device': 'cuda:0',
        'seed': 0,
        'exe_steps': 1,

        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/rlbench_multitask_keypose/',
        'exp_name': watch(args_to_watch),
        'vis_freq': 10,
        'max_render': 8,

        'diffusion_epoch': 'latest',
        'horizon': 2,
        'n_diffusion_steps': 5,
        'verbose': False,
        'suffix': 'f:step_{diffusion_epoch}',
    },
}
