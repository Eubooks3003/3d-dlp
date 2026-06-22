import warnings
warnings.filterwarnings('ignore')
import os
from diffuser.eval_utils import setup_isaac_env, evaluate_policy, wandb_log_eval_stats
from diffuser.utils.arrays import set_global_device
import diffuser.utils as utils
import wandb
from diffuser.utils.args import ArgsParser
#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#
if __name__ == '__main__':

    args = ArgsParser().parse_args('plan')

    set_global_device(args.device)

    ## load diffusion model from disk
    diffusion_experiment = utils.load_diffusion(
        args.loadbase, args.dataset, args.diffusion_loadpath,
        epoch=args.diffusion_epoch, seed=args.seed, is_diffusion=True,
        override_dataset_path=args.override_dataset_path
    )
    diffusion = diffusion_experiment.ema
    dataset = diffusion_experiment.dataset
    renderer = diffusion_experiment.renderer

    logger_config = utils.Config(
        utils.Logger,
        renderer=renderer,
        logpath=args.savepath,
        vis_freq=args.vis_freq,
        max_render=args.max_render,
    )

    ## policies are wrappers around an unconditional diffusion model
    policy_config = utils.Config(
        args.policy,
        diffusion_model=diffusion,
        normalizer=dataset.normalizer,
        preprocess_fns=args.preprocess_fns,
        verbose=False,
        horizon=args.horizon,
    )

    logger = logger_config()
    policy = policy_config()


#-----------------------------------------------------------------------------#
#---------------------------eval policy on environment------------------------#
#-----------------------------------------------------------------------------#
    wandb_run = wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                group=args.wandb_group_name,
                config=args,
                sync_tensorboard=False,
                settings=wandb.Settings(start_method="fork"),
            )
    savepath_folder_name = args.savepath.split('/')[-2]
    wandb_run.name = savepath_folder_name

    if args.kitchen:
        import franka_kitchen.kitchen_env
        from franka_kitchen.kitchen_utils import evaluate_kitchen, get_kitchen_goal_fn
        from dlp_utils import load_pretrained_rep_model
        import gym
        from franka_kitchen.kitchen_env import KitchenWrapper
        from franka_kitchen.kitchen_utils import VideoRecorder
        env = gym.make('kitchen-v0')
        env = KitchenWrapper(env, 'kitchen-v0', visual_input=True)
        video = VideoRecorder(dir_name=args.savepath)
        pretrained_dlp_path = 'ecdiffuser-data/latent_rep_chkpts/dlp_kitchen'
        latent_rep_model = load_pretrained_rep_model(dir_path=pretrained_dlp_path, model_type=args.input_type, pandapush=False).to(args.device)
        video = VideoRecorder(dir_name=args.savepath)
        goal_fn = get_kitchen_goal_fn(dataset)
        evaluate_kitchen(policy, env, latent_rep_model, goal_fn, args, video, 0, args.savepath, num_evals=100, save_img=False)
    else:
        env = setup_isaac_env(args)
        if args.input_type == 'dlp':
            renderer.env = env
        stat_save_path = os.path.join(args.savepath, 'eval_stats.pkl')
        eval_stat_dict = evaluate_policy(policy, env, args, logger, num_eval_episodes=100, exe_steps=args.exe_steps, stat_save_path=stat_save_path)
        wandb_log_eval_stats(env, eval_stat_dict, args)