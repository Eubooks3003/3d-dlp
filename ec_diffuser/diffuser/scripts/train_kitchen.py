import warnings
warnings.filterwarnings('ignore')
import franka_kitchen.kitchen_env
from franka_kitchen.kitchen_utils import evaluate_kitchen, get_kitchen_goal_fn

from diffuser.utils.arrays import set_global_device

import diffuser.utils as utils
import wandb
from dlp_utils import load_pretrained_rep_model
from diffuser.utils.args import ArgsParser
#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

args = ArgsParser().parse_args('diffusion')
## set default device for data and models to be args.device
set_global_device(args.device)

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#
dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
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
    single_view=(args.input_type == 'dlp' and not args.multiview),
)


render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, 'render_config.pkl'),
    env=None,
    num_entity=args.num_entity,
    particle_dim=args.features_dim,
    single_view=(args.input_type == 'dlp' and not args.multiview),
)

dataset = dataset_config()
renderer = render_config()

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#
model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
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
)

diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
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
)

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
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

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

model = model_config()

diffusion = diffusion_config(model)

trainer = trainer_config(diffusion, dataset, renderer)

plan_args = ArgsParser().parse_args('plan', savepath=args.savepath)

logger_config = utils.Config(
    utils.Logger,
    renderer=renderer,
    logpath=plan_args.savepath,
    vis_freq=plan_args.vis_freq,
    max_render=plan_args.max_render,
)

## policies are wrappers around an unconditional diffusion model
policy_config = utils.Config(
    plan_args.policy,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=plan_args.preprocess_fns,
    verbose=False,
    horizon=plan_args.horizon,
)

logger = logger_config()
policy = policy_config()



#-----------------------------------------------------------------------------#
#---------------------------- create environments ----------------------------#
#-----------------------------------------------------------------------------#
import gym
from franka_kitchen.kitchen_env import KitchenWrapper
from franka_kitchen.kitchen_utils import VideoRecorder
env = gym.make('kitchen-v0')
env = KitchenWrapper(env, 'kitchen-v0', visual_input=True)

pretrained_dlp_path = 'ecdiffuser-data/latent_rep_chkpts/dlp_kitchen'
latent_rep_model = load_pretrained_rep_model(dir_path=pretrained_dlp_path, model_type=args.input_type, pandapush=False).to(args.device)
video = VideoRecorder(dir_name=plan_args.savepath)
goal_fn = get_kitchen_goal_fn(dataset)
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#
print('Testing forward...', end=' ', flush=True)
batch = utils.batchify(dataset[0])
loss, _ = diffusion.loss(*batch)
loss.backward()
print('✓')

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#
wandb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            group=args.wandb_group_name,
            config=args,
            sync_tensorboard=False,
            settings=wandb.Settings(start_method="fork"),
        )
savepath_folder_name = args.savepath.split('/')[-1]
wandb_run.name = savepath_folder_name

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    trainer.train(n_train_steps=args.n_steps_per_epoch)

    if i % args.eval_freq == 0:
        evaluate_kitchen(policy, env, latent_rep_model, goal_fn, plan_args, video, i, plan_args.savepath, num_evals=100)