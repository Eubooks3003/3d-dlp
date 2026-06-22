from diffuser.utils import watch

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('seed', 'seed'),
]

logbase = 'data'

# RLBench 3D-DLP language-conditioned config for task: close_jar.
# Multi-entity (3DDA-style): action and proprio each split into 3 sub-tokens
# (pos / rot / grip) with separate projections and decoder heads.
# IMPORTANT: key must match mode computed in setup.py: "{num_entity}C_{input_type}"
mode_to_args = {
  '16C_dlp': {
    'keypose_mode': True,

    'dataset': 'close_jar',
    'override_dataset_path': '/home/ellina/Desktop/data/preprocessed_voxel_tokens_with_keyposes/rlbench_close_jar/close_jar.pkl',
    'calib_h5_path': None,  # RLBench does not use a robomimic calib HDF5
    'dlp_ckpt': '/home/ellina/Desktop/data/preprocessed_voxel_tokens_with_keyposes/rlbench_close_jar/dlp_ckpt.pt',
    'dlp_ctor': "voxel_models:DLP",
    'dlp_cfg': '/home/ellina/Desktop/data/preprocessed_voxel_tokens_with_keyposes/rlbench_close_jar/dlp_config.json',
    'features_dim': 12,       # z(3)+scale(3)+depth(1)+obj_on(1)+feat(4)
    'gripper_dim': 10,        # pos(3)+rot6d(6)+open(1)
    'use_gripper_obs': True,  # cond-tokens path: current gripper feeds into cond[0]
    'split_action_tokens': True,  # multi-entity: decode pos/rot/grip as 3 separate heads (action AND proprio)
    'gripper_state_mask_ratio': 0.0,
    'bg_dim': 2,              # learned_bg_feature_dim
    'use_bg_obs': True,
    'max_particles': 40,
    'multiview': False,
    'device': 'cuda:0',
    'max_path_length': 400,   # pkl T dim (must match preprocess output)
    'max_demos': 100,
    'eval_freq': 10,
    'eval_backend': 'none',
    # ---- RLBench live eval (logs videos to wandb, not to disk) ----
    'rlbench_eval_episodes': 2,
    'rlbench_eval_max_steps': 200,
    'rlbench_eval_cams': ['front', 'overhead', 'left_shoulder', 'right_shoulder'],
    'rlbench_eval_image_size': 128,
    'rlbench_eval_video_fps': 10,
    'n_steps_per_epoch': 500,
    # --- RLBench-specific ---
    'action_dim': 10,          # [pos(3), rot6d(6), open(1)] absolute EEF control
    'act_pos_dim': 3,
    'act_rot_dim': 6,
    'act_grip_dim': 1,
    'prop_pos_dim': 3,
    'prop_rot_dim': 6,
    'prop_grip_dim': 1,
    'lang_dim': 512,           # CLIP ViT-B/32 hidden size
    'lang_pooled': False,      # use full CLIP token sequence (not just EOS-pooled)
    'max_lang_tokens': 32,     # truncate CLIP 77-token sequence to this
    'clip_model_name': 'openai/clip-vit-base-patch32',
    'lang_device': 'cpu',
    # -------------------------
    "use_absolute_actions": True,
    'horizon': 1,    # predict only the next keypose (3DDA-style); cond is current keypose
    'exe_steps': 1,  # apply 1 keypose, replan
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
        'n_layers': 12,
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
        'prefix': 'diffusion/rlbench_close_jar_keypose_multientity/',
        'exp_name': watch(args_to_watch),

        # training
        'n_steps_per_epoch': 200,
        'loss_type': 'l1',
        'n_train_steps': 2e6,
        'batch_size': 16,
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
        'prefix': 'plans/rlbench_close_jar_keypose_multientity/',
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
