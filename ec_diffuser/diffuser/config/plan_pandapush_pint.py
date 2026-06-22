import socket

from diffuser.utils import watch

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('seed', 'seed'),
]
logbase = 'data'
loadbase = 'ecdiffuser-data/pretrained_models'
#------------------------ overrides ------------------------#
entity_to_steps = {1:30, 2: 50, 3: 100, 4: 150, 5: 200, 6:200}
PushT_entity_to_steps = {1:50, 2: 100, 3: 150, 4: 200, 5: 250, 6:300}
kitchen_entity_to_steps = {1:500}

mode_to_args = {
    'dlp_kitchen': {
                    'dataset': 'kitchen',
                    'n_diffusion_steps': 5,
                    'horizon': 5,
                    'device': 'cuda:1',
                    'diffusion_loadpath': 'diffusion/kitchen_1C_dlp_40kp_0.25anchor_s_H5_T5',
                    'override_dataset_path': 'ecdiffuser-data/kitchen/dlp_kitchen_dataset_40kp_64kpp_4zdim.pkl',
                    },
    'dlp': {'env_config_dir': 'env_config/generalization_num_cubes',
            'n_diffusion_steps': 100,
            'horizon': 5,
            'device': 'cuda:1',
            'diffusion_loadpath': 'diffusion/3C_adalnpintlarge_dlp_randcolor_H5_T100',
            'override_dataset_path': 'ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl',
            },
    'dlp_pusht': {'env_config_dir': 'env_config/push_n_t',
                  'n_diffusion_steps': 5,
                  'horizon': 5,
                  'device': 'cuda:1',
                  'diffusion_loadpath': 'diffusion/PushT_3C_dlp_pintlarge_H5_T5',
                  'vis_freq': 999,
                  'override_dataset_path': 'ecdiffuser-data/push_t/3T/panda_push_replay_buffer_dlp.pkl',
                  },
}

#------------------------ base -----------------------------#
base = {

    'plan': {
        'env_config_dir': 'env_config/n_cubes',
        'policy': 'sampling.GoalConditionedPolicy',
        'exe_steps': 1,
        'batch_size': 1,
        'preprocess_fns': [],
        'device': 'cuda:1',
        'seed': None,
        'multiview': True,

        ## serialization
        'loadbase': loadbase,
        'logbase': logbase,
        'prefix': 'plans/',
        'exp_name': watch(args_to_watch),
        'vis_freq': 10,
        'max_render': 8,

        ## diffusion model
        'horizon': 5,
        'n_diffusion_steps': 5,

        ## loading
        'diffusion_loadpath': '',

        'diffusion_epoch': 'latest',

        'verbose': True,
        'suffix': 'f:step_{diffusion_epoch}',
    },
}