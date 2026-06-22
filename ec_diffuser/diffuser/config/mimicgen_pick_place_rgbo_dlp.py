from diffuser.utils import watch

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('seed', 'seed'),
]

logbase = 'data'

# IMPORTANT: key must match mode computed in setup.py: "{num_entity}C_{input_type}"
mode_to_args = {
  '16C_dlp': {
    'dataset': 'pick_place',
    'override_dataset_path': '/home/ellina/Desktop/data/preprocessed_rgbo/pick_place_d0/pick_place_d0.pkl',
    'calib_h5_path': '/home/ellina/Desktop/data/3D-DLP-mimicgen-data/core/pick_place_d0.hdf5',
    'dlp_ckpt': '/home/ellina/Desktop/data/preprocessed_rgbo/pick_place_d0/dlp_ckpt.pt',
    'dlp_ctor': "voxel_models:DLP",
    'dlp_cfg': '/home/ellina/Desktop/data/preprocessed_rgbo/pick_place_d0/dlp_config.json',
    'features_dim': 16,       # Dtok: z(3)+scale(3)+depth(1)+obj_on(1)+feat(8)
    'gripper_dim': 10,        # G: pos(3)+rot6d(6)+open(1)
    'use_gripper_obs': True,
    'gripper_state_mask_ratio': 0.0,
    'bg_dim': 2,              # BG: learned_bg_feature_dim
    'use_bg_obs': True,
    'max_particles': 40,
    'multiview': False,
    'device': 'cuda:0',
    'max_path_length': 798,   # Tmax from RGBO preprocessed pkl
    'max_demos': 200,
    'eval_freq': 0,
    'eval_backend': 'none',
    'n_steps_per_epoch': 500,
    "mimicgen_cams": ["agentview", "frontview"],  # PickPlace has no sideview
    "mimicgen_camera_width": 256,
    "mimicgen_camera_height": 256,
    "mimicgen_max_steps": 1000,
    "mimicgen_pixel_stride": 1,
    "use_absolute_actions": False,
    'horizon': 16,
    'exe_steps': 8,
    "random_init": True,
    "random_init_eval": True,
  },
}


base = {
    'diffusion': {
        'model': 'models.AdaLNPINTDenoiser',
        'diffusion': 'models.GaussianDiffusion',

        'horizon': 5,
        'features_dim': 7,
        'hidden_dim': 256,
        'projection_dim': 256,
        'n_heads': 8,
        'n_layers': 6,
        'dropout': 0.0,

        'n_diffusion_steps': 5,
        'action_weight': 50,

        'max_particles': 64,
        'positional_bias': False,
        'multiview': False,

        # dataset
        'loader': 'datasets.GoalDataset',
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
        'prefix': 'diffusion/mimicgen_pick_place_rgbo/',
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

        # (safe to include; many configs rely on these existing)
        'loss_weights': None,
        'loss_discount': 1,

        # ACTION CHUNKING for eval: execute N actions per plan before replanning
        'exe_steps': 3,  # With horizon=5, execute 3 actions before replanning

    },

    'plan': {
        # not used while eval_freq is huge
        'policy': 'sampling.GoalConditionedPolicy',
        'max_episode_length': 50,
        'batch_size': 1,
        'preprocess_fns': [],
        'device': 'cuda:0',
        'seed': 0,
        'exe_steps': 3,  # ACTION CHUNKING: execute 3 actions per plan before replanning (was 1)

        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/mimicgen_pick_place_rgbo/',
        'exp_name': watch(args_to_watch),
        'vis_freq': 10,
        'max_render': 8,

        'diffusion_epoch': 'latest',
        'horizon': 5,
        'n_diffusion_steps': 5,
        'verbose': False,
        'suffix': 'f:step_{diffusion_epoch}',

    },
}
