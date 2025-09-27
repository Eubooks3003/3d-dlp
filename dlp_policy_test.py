import torch
from torchvision import transforms
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dlp_policy import DLPPolicy
from PIL import Image
matplotlib.use('tkAgg')

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def animate_tensor(
        tensor,
        x_goal=None,
        interval=120,
        pause_duration=2.0,
        plot=True,
        save_gif=False,
        save_path=None,
):
    """
    Animate a tensor of shape [n_views, timesteps, ch, h, w], with optional goal images.
    The last frame is held for pause_duration seconds.

    Args:
        tensor (torch.Tensor): [n_views, timesteps, ch, h, w]
        x_goal (torch.Tensor or None): [n_views, 1, ch, h, w] or None
        interval (int): ms between frames
        pause_duration (float): pause on last frame (seconds)
        plot (bool): Show animation in window
        save_gif (bool): Save GIF animation to disk
        save_path (str or None): Path (including filename.gif) to save GIF. Required if save_gif is True.
    """
    # Move to CPU and convert to numpy
    tensor = tensor.detach().cpu()
    n_views, timesteps, ch, h, w = tensor.shape
    frames = tensor.permute(1, 0, 3, 4, 2).numpy()

    # Prepare goal images
    if x_goal is not None:
        x_goal = x_goal.detach().cpu()
        assert x_goal.shape == (n_views, 1, ch, h, w), "x_goal shape mismatch"
        goal_imgs = x_goal.squeeze(1).permute(0, 2, 3, 1).numpy()
    else:
        goal_imgs = [None] * n_views

    # Setup figure: 2 rows (animation + goal), n_views columns
    fig, axes = plt.subplots(2, n_views, figsize=(n_views * 3, 6))
    if n_views == 1:
        axes = np.expand_dims(axes, axis=1)

    ims = []

    # Animation frames
    for t in range(timesteps):
        ims_t = []
        for v in range(n_views):
            frame = frames[t, v]
            ax = axes[0][v]
            if frame.shape[2] == 1:
                im = ax.imshow(frame[..., 0], cmap='gray', animated=True)
            else:
                im = ax.imshow(frame, animated=True)
            ax.axis('off')
            if t == 0:
                ax.set_title(f'View {v}')
            ims_t.append(im)
        ims.append(ims_t)

    # Hold on last frame for pause
    pause_frames = int(pause_duration * 1000 // interval)
    last_frame = ims[-1]
    for _ in range(pause_frames):
        ims.append(last_frame)

    # Draw goal images
    for v in range(n_views):
        ax_goal = axes[1][v]
        goal = goal_imgs[v]
        if goal is not None:
            if goal.shape[2] == 1:
                ax_goal.imshow(goal[..., 0], cmap='gray')
            else:
                ax_goal.imshow(goal)
        ax_goal.axis('off')
        ax_goal.set_title("Goal")

    ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=True, repeat_delay=1000)
    plt.tight_layout()

    if plot:
        plt.show()
    if save_gif:
        if save_path is None:
            raise ValueError("save_path must be provided to save GIF.")
        ani.save(save_path, writer='pillow')

    plt.close(fig)


if __name__ == '__main__':
    # dataset = 'panda'
    dataset = 'ogbench'
    # dataset = 'frankakitchen'

    if dataset == 'panda':
        path_to_config = './configs/panda_policy.json'  # panda
    elif dataset == 'ogbench':
        path_to_config = './configs/ogbench_policy.json'  # ogbench
    elif dataset == 'frankakitchen':
        path_to_config = './configs/frankakitchen_policy.json'
    else:
        raise NotImplementedError

    save_anim = False

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    model = DLPPolicy(path_to_config).to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(model.dlp_model.image_size),
        transforms.CenterCrop(model.dlp_model.image_size),
        transforms.ToTensor()
    ])

    interval = 120

    # panda
    if dataset == 'panda':
        ep = 900
        start_frame = 0
        goal_frame = 30
        interval = 60
        task = '3C'
        path_to_episode_v1 = f'C:/Users/tabad/Downloads/panda_ds/{task}/view1/ep{ep:04d}'
        path_to_episode_v2 = f'C:/Users/tabad/Downloads/panda_ds/{task}/view2/ep{ep:04d}'

        path_to_start_frame_v1 = os.path.join(path_to_episode_v1, f'frame{start_frame:04d}.png')
        path_to_start_frame_v2 = os.path.join(path_to_episode_v2, f'frame{start_frame:04d}.png')

        path_to_goal_frame_v1 = os.path.join(path_to_episode_v1, f'frame{goal_frame:04d}.png')
        path_to_goal_frame_v2 = os.path.join(path_to_episode_v2, f'frame{goal_frame:04d}.png')

        path_to_goal_frame_v1 = os.path.join(path_to_episode_v1, f'goal.png')
        path_to_goal_frame_v2 = os.path.join(path_to_episode_v2, f'goal.png')

        start_img_v1 = transform(Image.open(path_to_start_frame_v1))[:3]
        start_img_v2 = transform(Image.open(path_to_start_frame_v2))[:3]

        goal_img_v1 = transform(Image.open(path_to_goal_frame_v1))[:3]
        goal_img_v2 = transform(Image.open(path_to_goal_frame_v2))[:3]

        start_img = torch.stack([start_img_v1, start_img_v2], dim=0)[None, :, None]
        goal_img = torch.stack([goal_img_v1, goal_img_v2], dim=0)[None, :, None]

        n_steps = 80
        x = start_img.view(-1, *start_img.shape[2:]).to(device)
        x_goal = goal_img.view(-1, *goal_img.shape[2:]).to(device)
        print(f'x: {x.shape}, x_goal: {x_goal.shape}')
        with torch.no_grad():
            model_output = model.sample(x, x_goal, n_steps=n_steps, render=True, deterministic=False)
        rec_rgb = model_output['rec']  # [n_views, timesteps, ch, h, w]
        print(f'rec_rgb: {rec_rgb.shape}')
    elif dataset == 'ogbench':
        example_path = './example_data/ogbench/cube_train_ep0015'
        # example_path = './example_data/ogbench/scene_train_ep0118'
        start_frame = 0
        goal_frame = 250
        interval = 60

        path_to_start_frame = os.path.join(example_path, f'frame{start_frame:04d}.png')
        path_to_goal_frame = os.path.join(example_path, f'frame{goal_frame:04d}.png')
        start_img = transform(Image.open(path_to_start_frame))[:3]
        goal_img = transform(Image.open(path_to_goal_frame))[:3]
        n_steps = 400
        x = start_img[None, None].to(device)
        x_goal = goal_img[None, None].to(device)
        print(f'x: {x.shape}, x_goal: {x_goal.shape}')
        with torch.no_grad():
            model_output = model.sample(x, x_goal, n_steps=n_steps, render=True, deterministic=False)
        rec_rgb = model_output['rec']  # [n_views, timesteps, ch, h, w]
        print(f'rec_rgb: {rec_rgb.shape}')
    elif dataset == 'frankakitchen':
        example_path = './example_data/franka_kitchen/ep0540'
        start_frame = 0
        goal_frame = 150
        interval = 60

        path_to_start_frame = os.path.join(example_path, f'frame{start_frame:04d}.png')
        path_to_goal_frame = os.path.join(example_path, f'frame{goal_frame:04d}.png')
        start_img = transform(Image.open(path_to_start_frame))[:3]
        goal_img = transform(Image.open(path_to_goal_frame))[:3]
        n_steps = 200
        x = start_img[None, None].to(device)
        x_goal = goal_img[None, None].to(device)
        print(f'x: {x.shape}, x_goal: {x_goal.shape}')
        with torch.no_grad():
            model_output = model.sample(x, x_goal, n_steps=n_steps, render=True, deterministic=False)
        rec_rgb = model_output['rec']  # [n_views, timesteps, ch, h, w]
        print(f'rec_rgb: {rec_rgb.shape}')
        actions = model_output['actions']
        print(f'actions: {actions.shape}')

    save_path = os.path.join('./checkpoints', dataset, f'anim_s{start_frame}g{goal_frame}n{n_steps}.gif')
    animate_tensor(rec_rgb, x_goal, interval=interval, save_gif=save_anim, save_path=save_path)