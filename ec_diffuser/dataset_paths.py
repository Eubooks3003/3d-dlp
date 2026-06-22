
import os
##### paths to datasets for training diffusor models

def get_diffuser_dataset_path(dataset_loadbase, num_entity, input_type, rand_color=False, push_t=False, kitchen=False):
    if input_type == 'dlp':
        if kitchen:
            assert num_entity == 1
            return os.path.join(dataset_loadbase, 'kitchen', 'dlp_kitchen_dataset_40kp_64kpp_4zdim.pkl')
        if push_t:
            return os.path.join(dataset_loadbase, 'push_t', f'{num_entity}T', 'panda_push_replay_buffer_dlp.pkl')
        else:
            if rand_color:
                return os.path.join(dataset_loadbase, 'push_cubes', f'{num_entity}C_randcolor', 'panda_push_replay_buffer_dlp.pkl')
            else:
                return os.path.join(dataset_loadbase, 'push_cubes', f'{num_entity}C', 'panda_push_replay_buffer_dlp.pkl')
    raise ValueError(f'input_type {input_type} not supported')
