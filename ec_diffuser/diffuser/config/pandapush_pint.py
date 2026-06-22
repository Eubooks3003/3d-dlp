import socket

from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('seed', 'seed'),
]

#------------------------ overrides ------------------------#
# Check `read_config` function from diffuser/utils/setup.py for how the overrides are applied
logbase = 'data'

mode_to_args = {
    '1C_dlp': {'env_config_dir': 'env_config/n_cubes',
            'features_dim': 10,
            'n_diffusion_steps': 5,
            'max_path_length': 30,
            },
    '2C_dlp': {'env_config_dir': 'env_config/n_cubes',
            'features_dim': 10,
            'n_diffusion_steps': 5,
            'max_path_length': 50,
            },
    '3C_dlp': {'env_config_dir': 'env_config/n_cubes',
            'features_dim': 10,
            'n_diffusion_steps': 5,
            'max_path_length': 100,
            },
    '1C_dlp_pusht': {'env_config_dir': 'env_config/push_one_t',
            'features_dim': 10,
            'n_diffusion_steps': 5,
            'max_path_length': 50,
    },
    '2C_dlp_pusht': {'env_config_dir': 'env_config/push_n_t',
            'features_dim': 12,
            'n_diffusion_steps': 5,
            'max_path_length': 100,
    },
    '3C_dlp_pusht': {'env_config_dir': 'env_config/push_n_t',
            'features_dim': 12,
            'n_diffusion_steps': 5,
            'max_path_length': 150,
            'hidden_dim': 512,
            'projection_dim': 512,
            'n_heads': 8,
            'n_layers': 12,
    },
    '3C_dlp_randcolor': {'env_config_dir': 'env_config/generalization_num_cubes',
            'features_dim': 10,
            'n_diffusion_steps': 100,
            'max_path_length': 100,
            'hidden_dim': 512,
            'projection_dim': 512,
            'n_heads': 8,
            'n_layers': 12,
            },
    '1C_dlp_kitchen': {
        'dataset': 'kitchen',
        'features_dim': 10,
        'multiview': False,
        'n_diffusion_steps': 5,
        'model': 'models.AdaLNPINTDenoiser',
        'particle_normalizer': 'ParticleLimitsNormalizer',
        'horizon': 5,
        'max_path_length': 409,
        'device': 'cuda:0',
        'droupout': 0.0,
        'renderer': 'utils.ParticleRenderer',
        'eval_freq': 200,
        'n_train_steps': 9e5,
},
}

base = {
    'diffusion': {
        'env_config_dir':'env_config/n_cubes',
        'model': 'models.AdaLNPINTDenoiser',
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 5,
        'features_dim': 4,
        'hidden_dim': 256,
        'projection_dim': 256,
        'n_heads': 8,
        'n_layers': 6,
        'dropout': 0.0,
        'n_diffusion_steps': 5,
        'action_weight': 10,
        'max_particles': None,
        "positional_bias": False,
        "multiview": True,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'renderer': 'utils.ParticleRenderer',

        ## dataset
        'loader': 'datasets.GoalDataset',
        'normalizer': 'SafeLimitsNormalizer',
        'particle_normalizer': 'ParticleLimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 30,
        'obs_only': False,
        'action_only': False,
        'gripper_state_mask_ratio': 0.0,  # Ratio of samples where gripper state is masked (0.0=never, 1.0=always)

        ## serialization
        'logbase': logbase,
        'prefix': 'diffusion/',
        'exp_name': watch(args_to_watch),

        'n_steps_per_epoch': 1000,
        'loss_type': 'l1',
        'n_train_steps': 5e5,
        'batch_size': 32,
        'learning_rate': 8e-5,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'eval_freq': 20,
        'sample_freq': 0,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 4,
        'bucket': None,
        'device': 'cuda:1',
        'seed': None,
    },

    'plan': {
        'policy': 'sampling.GoalConditionedPolicy',
        'max_episode_length': 50,
        'batch_size': 1,
        'preprocess_fns': [],
        'device': 'cuda:1',
        'seed': None,
        'exe_steps': 1,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/',
        'exp_name': watch(args_to_watch),
        'vis_freq': 10,
        'max_render': 8,

        ## diffusion model
        'diffusion_epoch': 'latest',
        'horizon': 5,
        'n_diffusion_steps': 5,
        'verbose': False,
        'suffix': 'f:step_{diffusion_epoch}',
    },
}