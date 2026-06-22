import diffuser.utils as utils

class ArgsParser(utils.Parser):
    wandb_entity: str = 'ellina-zhang0827'
    wandb_project: str = 'test_release_ecdiffuser'
    dataset: str = 'panda_push'
    dataset_loadbase: str = 'ecdiffuser-data'
    config: str = 'config.pandapush_pint'
    input_type: str = 'dlp'
    num_entity: int = 1
    push_t_num_color: int = 1
    rand_color: bool = False
    push_t: bool = False
    kitchen: bool = False
    overfit: bool = False
    seed: int = 42
    exp_note: str = 'adalnpint'
    vis_traj_wandb: bool = True
    planning_only: bool = False
    random_init_eval: bool = False  # If True, use random env reset instead of dataset init states during eval