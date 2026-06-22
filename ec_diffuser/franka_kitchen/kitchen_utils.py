import os
import time
import pickle
import wandb
from moviepy.editor import ImageSequenceClip
import imageio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
from collections import deque
from typing import Callable, Dict, Optional, Tuple, Union
import gym
import einops
from PIL import Image
from dlp_utils import extract_dlp_features_with_bg, get_recon_from_dlps

class VideoRecorder(object):
    ### Adapted from https://github.com/jayLEE0301/vq_bet_official/blob/main/examples/video.py
    def __init__(self, dir_name, height=256, width=256, fps=30):
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, obs):
        if self.enabled:
            self.frames.append(obs)
            # self.frames.append(cv2.cvtColor(obs, cv2.COLOR_BGR2RGB))

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)

def get_kitchen_goal_fn(
    dataset,
    goal_conditional: Optional[str] = 'future',
    goal_seq_len: Optional[int] = 1,
    visualize_goal: Optional[bool] = False,
) -> Callable[
    [gym.Env, torch.Tensor, torch.Tensor, torch.Tensor],
    Union[
        torch.Tensor,
        Tuple[torch.Tensor, Dict[str, torch.Tensor]],
        Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
        None,
    ],
]:  
    ### Adapted from https://github.com/jayLEE0301/vq_bet_official/blob/main/examples/kitchen_env.py
    all_goal_idx_permutations = np.random.permutation(dataset.fields.observations.shape[0])
    if goal_conditional == "future":
        assert (
            goal_seq_len is not None
        ), "goal_seq_len must be provided if goal_conditional is 'future'"

        def future_goal_fn(env, goal_idx, frame_idx):  # type: ignore
            idx = all_goal_idx_permutations[goal_idx]
            obs, onehot = dataset.fields['observations'][idx], dataset.fields['onehot_goals'][idx]  # seq_len x obs_dim
            path_length = dataset.fields['path_lengths'][idx]
            info = {}
            if frame_idx == 0:
                onehot = einops.reduce(onehot, "T C -> C", "max")
                info["onehot_goal"] = onehot
                init_qpos = env.set_task_goal(onehot)
                goal_obs = env.reset_model_with_state(init_qpos, process_img=False)
                info["goal_image"] = goal_obs
                env.set_task_goal(onehot)
            obs = obs[path_length - goal_seq_len: path_length]
            return obs, info

        goal_fn = future_goal_fn

    elif goal_conditional == "onehot":

        def onehot_goal_fn(env, goal_idx, frame_idx):
            onehot_goals = dataset.fields['onehot_goals'][goal_idx] # seq_len x obs_dim
            env.set_task_goal(onehot_goals)
            return onehot_goals[min(frame_idx, len(onehot_goals) - 1)], {}
        goal_fn = onehot_goal_fn

    else:
        raise ValueError(f"goal_conditional {goal_conditional} not recognized")

    return goal_fn

def evaluate_kitchen(policy, env, latent_encoder, goal_fn, args, video, epoch, save_path, num_evals=100, num_eval_per_goal=1, save_img=False):
    @torch.no_grad()
    def eval_on_env(
        args,
        env,
        num_evals=100,
        num_eval_per_goal=1,
        videorecorder=None,
        epoch=None,
    ):
        avg_reward = 0
        action_list = []
        completion_id_list = []
        avg_max_coverage = []
        avg_final_coverage = []
        for goal_idx in tqdm(range(num_evals)):
            if videorecorder is not None:
                videorecorder.init(enabled=(goal_idx == 0))
            for _ in range(num_eval_per_goal):
                step = 0
                goal, goal_info = goal_fn(env, goal_idx, step)
                obs_stack = deque(maxlen=1)
                obs_stack.append(env.reset())
                done, total_reward = False, 0
                if len(goal.shape) == 2:
                    goal = goal.reshape(1, -1, 10)
                if save_img:
                    save_folder = f"{save_path}/traj_{goal_idx}"
                    os.makedirs(save_folder, exist_ok=True)
                    Image.fromarray(goal_info["goal_image"]).save(f"{save_folder}/goal_image.png")
                while not done:
                    obs_torch = torch.from_numpy(np.stack(obs_stack)).float().to(args.device)
                    obs_torch = torch.nn.functional.interpolate(obs_torch, latent_encoder.image_size, mode='bilinear', align_corners=False)
                    dlp_obs, bg_dlp = extract_dlp_features_with_bg(obs_torch, latent_encoder, args.device)
                    # if step == 0:
                    #     goal_image = get_recon_from_dlps(goal, bg_dlp, latent_encoder, args.device)
                    #     Image.fromarray(goal_image).save(f"{save_path}/goal_image_{epoch}_{goal_idx}.png")
                    conditions = {0: dlp_obs.view(1, -1).cpu().numpy(), args.horizon-1: goal.reshape(1, -1)}
                    action_0, samples = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)                
                    action = samples.actions[0, 0]
                    obs, reward, done, info = env.step(action)
                    if save_img:
                        img = obs_torch.cpu().numpy()
                        Image.fromarray(img).save(f"{save_folder}/img_{step}.png")
                    if videorecorder.enabled:
                        videorecorder.record(info["images"])
                    step += 1
                    total_reward += reward
                    obs_stack.append(obs)
                    if step >= 500:
                        break
                avg_reward += total_reward
                completion_id_list.append(info["all_completions_ids"])
            videorecorder.save("eval_{}_{}.mp4".format(epoch, goal_idx))
        return (
            avg_reward / (num_evals * num_eval_per_goal),
            completion_id_list,
            avg_max_coverage,
            avg_final_coverage,
        )

    policy.diffusion_model.eval()
    avg_reward, completion_id_list, max_coverage, final_coverage = eval_on_env(
        args,
        env,
        videorecorder=video,
        epoch=epoch,
        num_evals=num_evals,
        num_eval_per_goal=num_eval_per_goal,
    )
    with open("{}/completion_idx_{}.json".format(save_path, epoch), "wb") as fp:
        pickle.dump(completion_id_list, fp)
    wandb.log({"eval_on_env": avg_reward})
    policy.diffusion_model.train()