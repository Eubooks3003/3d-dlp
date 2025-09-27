import torch
from dlp_policy import DLPPolicy
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)  # supress pytorch's torch.load future warnings

if __name__ == '__main__':
    path_to_config = './configs/panda_policy.json'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    model = DLPPolicy(path_to_config).to(device)

    # disable gradients
    model.requires_grad_(False)
    # change to eval mode
    model.eval()

    """
    if 'path_to_policy_ckpt' in 'panda_policy.json' is not None, calling
    `model = DLPPolicy(path_to_config).to(device)`
    will automatically load the checkpoint.
    """

    # demo
    ch = 3
    image_size = 128
    batch_size = 2
    n_views = 2  # this is also set in 'panda_policy.json' under `n_views`

    """
    See `dlp_policy.py` for more examples how to use `.sample()` to get the particles and render rgb from them.
    """

    print("----------------------------------")
    """
    Scenario 1: no history, predict a single action.
    We observe a single image each time and produce an action.
    """
    print("Scenario 1: no history, predict a single action")
    deterministic_latent_action_prediction = False  # if True, take only mu without sampling
    # Not sure how deterministic = True will behave, need to test
    n_env_steps = 5
    actions = []
    x_goal = torch.rand(batch_size, n_views, ch, image_size, image_size, device=device)
    for i in range(n_env_steps):
        # observation and goal
        x = torch.rand(batch_size, n_views, ch, image_size, image_size, device=device)
        # model expects: [bs * n_views, timesteps, ch, h, w]
        x = x.view(-1, 1, *x.shape[2:])  # [bs * n_views, 1, ch, h, w]
        x_goal = x_goal.view(-1, 1, *x_goal.shape[2:])  # [bs * n_views, 1, ch, h, w]
        # predict action
        action = model.act(x, x_goal, n_steps=1, deterministic=deterministic_latent_action_prediction)
        # [bs, 1, action_dim]
        print(f'env step {i}: x: {x.shape}, x_goal: {x_goal.shape}, action: {action.shape}')
        actions.append(action)
    actions = torch.cat(actions, dim=1)  # [bs, env_timesteps, action_dim]
    print(f'total actions: {actions.shape}')

    print("----------------------------------")
    """
    Scenario 2: use history, predict a single action.
    We observe a sequence of images each time and produce an action.
    """
    print("Scenario 2: use history, predict a single action.")
    deterministic_latent_action_prediction = False  # if True, take only mu without sampling
    # Not sure how deterministic = True will behave, need to test
    n_env_steps = 10
    actions = []
    obs = []
    x_goal = torch.rand(batch_size, n_views, ch, image_size, image_size, device=device)
    for i in range(n_env_steps):
        # observation and goal
        x = torch.rand(batch_size, n_views, ch, image_size, image_size, device=device)
        # model expects: [bs * n_views, timesteps, ch, h, w]
        x = x.view(-1, 1, *x.shape[2:])  # [bs * n_views, 1, ch, h, w]
        x_goal = x_goal.view(-1, 1, *x_goal.shape[2:])  # [bs * n_views, 1, ch, h, w]
        obs.append(x)
        x_in = torch.cat(obs, dim=1)  # [bs * n_views, timesteps, ch, h, w]
        # predict action
        action = model.act(x_in, x_goal, n_steps=1, deterministic=deterministic_latent_action_prediction)
        # [bs, 1, action_dim]
        print(f'env step {i}: x:_in {x_in.shape}, x_goal: {x_goal.shape}, action: {action.shape}')
        actions.append(action)
    actions = torch.cat(actions, dim=1)  # [bs, env_timesteps, action_dim]
    print(f'total actions: {actions.shape}')

    print("----------------------------------")
    """
    Scenario 3: use history, predict multiple future actions.
    We observe a sequence of images each time and produce multiple future actions.
    """
    print("Scenario 3: use history, predict multiple future actions.")
    deterministic_latent_action_prediction = False  # if True, take only mu without sampling
    # Not sure how deterministic = True will behave, need to test
    n_env_steps = 5
    n_action_samples = 3
    actions = []
    obs = torch.rand(batch_size * n_views, 1, ch, image_size, image_size, device=device)
    x_goal = torch.rand(batch_size, n_views, ch, image_size, image_size, device=device)
    for i in range(n_env_steps):
        # observation and goal
        # we now take `n_action_samples` steps and produce same number of obs
        x = torch.rand(batch_size, n_action_samples, n_views, ch, image_size, image_size, device=device)
        # model expects: [bs * n_views, timesteps, ch, h, w]
        x = x.permute(0, 2, 1, 3, 4, 5)  # [bs, n_views, n_action_samples, ch, h, w]
        x = x.reshape(-1, *x.shape[2:]) # [bs * n_views, n_action_samples, ch, h, w]
        x_goal = x_goal.view(-1, 1, *x_goal.shape[2:])  # [bs * n_views, 1, ch, h, w]
        obs = torch.cat([obs, x], dim=1)
        x_in = obs[:, -model.dlp_model.timestep_horizon:]  # [bs * n_views, timesteps, ch, h, w]
        # can spare some compute by only using the last context window frames of the model
        # predict action
        action = model.act(x_in, x_goal, n_steps=n_action_samples, deterministic=deterministic_latent_action_prediction)
        # [bs, n_action_samples, action_dim]
        print(f'env step {i}: x:_in {x_in.shape}, x_goal: {x_goal.shape}, action: {action.shape}')
        actions.append(action)
    actions = torch.cat(actions, dim=1)  # [bs, env_timesteps, action_dim]
    print(f'total actions: {actions.shape}')

    print("----------------------------------")
    """
    Scenario 4: general example to use the posterior ctx encoder instead of the prior.
    We observe a sequence of images each time and produce multiple future actions.
    """
    print("Scenario 4: general example to use the posterior ctx encoder instead of the prior.")
    deterministic_latent_action_prediction = False  # if True, take only mu without sampling
    # Not sure how deterministic = True will behave, need to test
    use_posterior_ctx = True
    posterior_ctx_img = False  # if True, first decode the imagined trajectory and then re-encode with the posterior
    print(f'use_posterior_ctx: {use_posterior_ctx}, posterior_ctx_img: {posterior_ctx_img}')
    n_env_steps = 5
    n_action_samples = 1
    actions = []
    obs = torch.rand(batch_size * n_views, 1, ch, image_size, image_size, device=device)
    x_goal = torch.rand(batch_size, n_views, ch, image_size, image_size, device=device)
    for i in range(n_env_steps):
        # observation and goal
        # we now take `n_action_samples` steps and produce same number of obs
        x = torch.rand(batch_size, n_action_samples, n_views, ch, image_size, image_size, device=device)
        # model expects: [bs * n_views, timesteps, ch, h, w]
        x = x.permute(0, 2, 1, 3, 4, 5)  # [bs, n_views, n_action_samples, ch, h, w]
        x = x.reshape(-1, *x.shape[2:])  # [bs * n_views, n_action_samples, ch, h, w]
        x_goal = x_goal.view(-1, 1, *x_goal.shape[2:])  # [bs * n_views, 1, ch, h, w]
        obs = torch.cat([obs, x], dim=1)
        x_in = obs[:, -model.dlp_model.timestep_horizon:]  # [bs * n_views, timesteps, ch, h, w]
        # can spare some compute by only using the last context window frames of the model
        # predict action
        action = model.act(x_in, x_goal, n_steps=n_action_samples, deterministic=deterministic_latent_action_prediction,
                           use_posterior_ctx=use_posterior_ctx, posterior_ctx_img=posterior_ctx_img)
        # [bs, n_action_samples, action_dim]
        print(f'env step {i}: x:_in {x_in.shape}, x_goal: {x_goal.shape}, action: {action.shape}')
        actions.append(action)
    actions = torch.cat(actions, dim=1)  # [bs, env_timesteps, action_dim]
    print(f'total actions: {actions.shape}')


    """
    Output
    ----------------------------------
    Scenario 1: no history, predict a single action
    env step 0: x: torch.Size([4, 1, 3, 128, 128]), x_goal: torch.Size([4, 1, 3, 128, 128]), action: torch.Size([2, 1, 3])
    env step 1: x: torch.Size([4, 1, 3, 128, 128]), x_goal: torch.Size([4, 1, 3, 128, 128]), action: torch.Size([2, 1, 3])
    env step 2: x: torch.Size([4, 1, 3, 128, 128]), x_goal: torch.Size([4, 1, 3, 128, 128]), action: torch.Size([2, 1, 3])
    env step 3: x: torch.Size([4, 1, 3, 128, 128]), x_goal: torch.Size([4, 1, 3, 128, 128]), action: torch.Size([2, 1, 3])
    env step 4: x: torch.Size([4, 1, 3, 128, 128]), x_goal: torch.Size([4, 1, 3, 128, 128]), action: torch.Size([2, 1, 3])
    total actions: torch.Size([2, 5, 3])
    ----------------------------------
    Scenario 2: use history, predict a single action.
    env step 0: x:_in torch.Size([4, 1, 3, 128, 128]), x_goal: torch.Size([4, 1, 3, 128, 128]), action: torch.Size([2, 1, 3])
    env step 1: x:_in torch.Size([4, 2, 3, 128, 128]), x_goal: torch.Size([4, 1, 3, 128, 128]), action: torch.Size([2, 1, 3])
    env step 2: x:_in torch.Size([4, 3, 3, 128, 128]), x_goal: torch.Size([4, 1, 3, 128, 128]), action: torch.Size([2, 1, 3])
    env step 3: x:_in torch.Size([4, 4, 3, 128, 128]), x_goal: torch.Size([4, 1, 3, 128, 128]), action: torch.Size([2, 1, 3])
    env step 4: x:_in torch.Size([4, 5, 3, 128, 128]), x_goal: torch.Size([4, 1, 3, 128, 128]), action: torch.Size([2, 1, 3])
    env step 5: x:_in torch.Size([4, 6, 3, 128, 128]), x_goal: torch.Size([4, 1, 3, 128, 128]), action: torch.Size([2, 1, 3])
    env step 6: x:_in torch.Size([4, 7, 3, 128, 128]), x_goal: torch.Size([4, 1, 3, 128, 128]), action: torch.Size([2, 1, 3])
    env step 7: x:_in torch.Size([4, 8, 3, 128, 128]), x_goal: torch.Size([4, 1, 3, 128, 128]), action: torch.Size([2, 1, 3])
    env step 8: x:_in torch.Size([4, 9, 3, 128, 128]), x_goal: torch.Size([4, 1, 3, 128, 128]), action: torch.Size([2, 1, 3])
    env step 9: x:_in torch.Size([4, 10, 3, 128, 128]), x_goal: torch.Size([4, 1, 3, 128, 128]), action: torch.Size([2, 1, 3])
    total actions: torch.Size([2, 10, 3])
    ----------------------------------
    Scenario 3: use history, predict multiple future actions.
    env step 0: x:_in torch.Size([4, 4, 3, 128, 128]), x_goal: torch.Size([4, 1, 3, 128, 128]), action: torch.Size([2, 3, 3])
    env step 1: x:_in torch.Size([4, 7, 3, 128, 128]), x_goal: torch.Size([4, 1, 3, 128, 128]), action: torch.Size([2, 3, 3])
    env step 2: x:_in torch.Size([4, 10, 3, 128, 128]), x_goal: torch.Size([4, 1, 3, 128, 128]), action: torch.Size([2, 3, 3])
    env step 3: x:_in torch.Size([4, 13, 3, 128, 128]), x_goal: torch.Size([4, 1, 3, 128, 128]), action: torch.Size([2, 3, 3])
    env step 4: x:_in torch.Size([4, 16, 3, 128, 128]), x_goal: torch.Size([4, 1, 3, 128, 128]), action: torch.Size([2, 3, 3])
    env step 5: x:_in torch.Size([4, 19, 3, 128, 128]), x_goal: torch.Size([4, 1, 3, 128, 128]), action: torch.Size([2, 3, 3])
    env step 6: x:_in torch.Size([4, 20, 3, 128, 128]), x_goal: torch.Size([4, 1, 3, 128, 128]), action: torch.Size([2, 3, 3])
    env step 7: x:_in torch.Size([4, 20, 3, 128, 128]), x_goal: torch.Size([4, 1, 3, 128, 128]), action: torch.Size([2, 3, 3])
    env step 8: x:_in torch.Size([4, 20, 3, 128, 128]), x_goal: torch.Size([4, 1, 3, 128, 128]), action: torch.Size([2, 3, 3])
    env step 9: x:_in torch.Size([4, 20, 3, 128, 128]), x_goal: torch.Size([4, 1, 3, 128, 128]), action: torch.Size([2, 3, 3])
    total actions: torch.Size([2, 30, 3])
    
    ----------------------------------
    Scenario 4: general example to use the posterior ctx encoder instead of the prior.
    use_posterior_ctx: True, posterior_ctx_img: False
    env step 0: x:_in torch.Size([4, 2, 3, 128, 128]), x_goal: torch.Size([4, 1, 3, 128, 128]), action: torch.Size([2, 1, 3])
    env step 1: x:_in torch.Size([4, 3, 3, 128, 128]), x_goal: torch.Size([4, 1, 3, 128, 128]), action: torch.Size([2, 1, 3])
    env step 2: x:_in torch.Size([4, 4, 3, 128, 128]), x_goal: torch.Size([4, 1, 3, 128, 128]), action: torch.Size([2, 1, 3])
    env step 3: x:_in torch.Size([4, 5, 3, 128, 128]), x_goal: torch.Size([4, 1, 3, 128, 128]), action: torch.Size([2, 1, 3])
    env step 4: x:_in torch.Size([4, 6, 3, 128, 128]), x_goal: torch.Size([4, 1, 3, 128, 128]), action: torch.Size([2, 1, 3])
    total actions: torch.Size([2, 5, 3])
    
    ----------------------------------
    Scenario 4: general example to use the posterior ctx encoder instead of the prior.
    use_posterior_ctx: True, posterior_ctx_img: True
    env step 0: x:_in torch.Size([4, 2, 3, 128, 128]), x_goal: torch.Size([4, 1, 3, 128, 128]), action: torch.Size([2, 1, 3])
    env step 1: x:_in torch.Size([4, 3, 3, 128, 128]), x_goal: torch.Size([4, 1, 3, 128, 128]), action: torch.Size([2, 1, 3])
    env step 2: x:_in torch.Size([4, 4, 3, 128, 128]), x_goal: torch.Size([4, 1, 3, 128, 128]), action: torch.Size([2, 1, 3])
    env step 3: x:_in torch.Size([4, 5, 3, 128, 128]), x_goal: torch.Size([4, 1, 3, 128, 128]), action: torch.Size([2, 1, 3])
    env step 4: x:_in torch.Size([4, 6, 3, 128, 128]), x_goal: torch.Size([4, 1, 3, 128, 128]), action: torch.Size([2, 1, 3])
    total actions: torch.Size([2, 5, 3])
    
    """
