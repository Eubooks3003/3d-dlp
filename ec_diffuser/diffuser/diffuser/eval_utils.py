# import os
# from isaac_panda_push_env import IsaacPandaPush
# from isaac_env_wrappers import IsaacPandaPushGoalSB3Wrapper
# from dlp_utils import check_config, load_pretrained_rep_model
# import yaml
# from pathlib import Path
# import time
# import pickle
# import wandb
# from moviepy.editor import ImageSequenceClip
# import matplotlib.pyplot as plt
# import numpy as np
# from tqdm import tqdm
# import torch
# import cv2 as cv
# from collections import deque
# from dlp_utils import extract_dlp_features_with_bg

# def setup_isaac_env(args):
#     config_dir = args.env_config_dir

#     # load config files
#     config = yaml.safe_load(Path(f'{config_dir}/Config.yaml').read_text())
#     isaac_env_cfg = yaml.safe_load(Path(f'{config_dir}/IsaacPandaPushConfig.yaml').read_text())

#     # overwrite env_config/ based on args
#     if args.planning_only:
#         isaac_env_cfg['env']['numObjects'] = args.num_entity
#         isaac_env_cfg["env"]["episodeLength"] = args.max_episode_length
#         if args.push_t:
#             isaac_env_cfg['env']['numColors'] = args.push_t_num_color
#             isaac_env_cfg['env']['numGoalObjects'] = args.push_t_num_color
#         isaac_env_cfg['env']['numObjects'] = args.num_entity
#     else:
#         if args.push_t:
#             isaac_env_cfg['env']['numColors'] = args.num_entity
#             isaac_env_cfg['env']['numGoalObjects'] = args.num_entity
#     if not args.multiview:
#         config['Model']['numViews'] = 1

#     check_config(config, isaac_env_cfg, None)
#     envs = IsaacPandaPush(
#         cfg=isaac_env_cfg,
#         rl_device=f"cuda:{config['cudaDevice']}",
#         sim_device=f"cuda:{config['cudaDevice']}",
#         graphics_device_id=config['cudaDevice'],
#         headless=True,
#         virtual_screen_capture=False,
#         force_render=False,
#     )

#     latent_rep_model = load_pretrained_rep_model(dir_path=config['Model']['latentRepPath'], model_type=config['Model']['obsMode'])
#     env = IsaacPandaPushGoalSB3Wrapper(
#         env=envs,
#         obs_mode=config['Model']['obsMode'],
#         n_views=config['Model']['numViews'],
#         latent_rep_model=latent_rep_model,
#         latent_classifier=None,
#         reward_cfg=config['Reward']['GT'],
#         smorl=(config['Model']['method'] == 'SMORL'),
#     )
#     return env



# def evaluate_policy(policy, env, args, logger, num_eval_episodes=100, exe_steps=1, stat_save_path=None):
#     with torch.no_grad():
#         policy.diffusion_model.eval()
#         num_eval_episodes = (num_eval_episodes // env.num_envs) * env.num_envs
#         # prepare stats variables
#         total_returns = 0
#         total_latent_rep_returns = 0
#         total_avg_obj_dist = 0
#         total_max_obj_dist = 0
#         total_goal_success_frac = 0
#         total_goals_reached = 0
#         save_stat_dict = {
#             "success": [],
#             "success_frac": [],
#             "avg_obj_dist": [],
#             "max_obj_dist": [],
#             "avg_return": [],
#         }
#         img_list = []
#         goal_img = None
#         ori_dist_list = []
#         rollout = []

#         print(f"\nEvaluating policy on {num_eval_episodes} random goals...")
#         start_time = time.time()
#         for i in tqdm(range(num_eval_episodes // env.num_envs)):
#             # prepare rollout stats variables
#             ret = np.zeros(env.num_envs)
#             latent_rep_ret = np.zeros(env.num_envs)
#             avg_obj_dist = np.ones(env.num_envs)
#             max_obj_dist = np.ones(env.num_envs)
#             goal_success_frac = np.zeros(env.num_envs)
#             goals_reached = np.zeros(env.num_envs)
#             obs = env.reset()
#             t = 0
#             while t < env.horizon:
#                 observation = obs['achieved_goal']          # (B, K, Dtok)
#                 goal = obs['desired_goal']                  # (B, K, Dtok)
#                 conditions = {0: observation, args.horizon-1: goal}

#                 action_0, samples = policy(conditions, batch_size=1, verbose=args.verbose)

#                 for step in range(exe_steps):
#                     actions = samples.actions[:, step]
#                     new_obs, rewards, dones, infos = env.step(actions)
#                     # gather stats
#                     ret += rewards
#                     avg_obj_dist = np.array([infos[i]["avg_obj_dist"] for i in range(len(infos))])
#                     max_obj_dist = np.array([infos[i]["max_obj_dist"] for i in range(len(infos))])
#                     goal_success_frac = np.array([infos[i]["goal_success_frac"] for i in range(len(infos))])
#                     # save orientation distances
#                     if env.push_t and t == env.horizon - 1:
#                         ori_dist_list.extend(np.array([info["ori_dist"] for info in infos]))
                    
#                     # save episode media and goals
#                     if i == 0:
#                         rollout.append(observation[0])
#                         img_list.append(np.moveaxis(infos[0]["image"][0], 0, -1))
#                         if t == 0:
#                             goal_img = np.moveaxis(infos[0]["goal_image"][0], 0, -1)
#                             front_bg = None
#                             side_bg = None
#                             if env.obs_mode == 'dlp':
#                                 front_img = infos[0]["image"][0]
#                                 _, front_bg = extract_dlp_features_with_bg(front_img, env.latent_rep_model, env.device)
#                                 side_img = infos[0]["image"][1]
#                                 _, side_bg = extract_dlp_features_with_bg(side_img, env.latent_rep_model, env.device)
#                         if t == env.horizon - 1:
#                             if goal_success_frac[0] == 1:
#                                 eval_vid_success = True
#                                 print("Visualized eval episode was a success")
#                             else:
#                                 eval_vid_success = False
#                                 print("Visualized eval episode was a failure")
#                         if t % 10 == 0: print(args.savepath, flush=True)
#                         logger.log(t, samples, None, rollout, goal[0], front_bg=front_bg, side_bg=side_bg)
                    
#                     # update last_obs
#                     obs = new_obs
#                     t += 1
#                     if t >= env.horizon:
#                         break
#             total_returns += np.sum(ret)
#             total_latent_rep_returns += np.sum(latent_rep_ret)
#             total_avg_obj_dist += np.sum(avg_obj_dist)
#             total_max_obj_dist += np.sum(max_obj_dist)
#             total_goal_success_frac += np.sum(goal_success_frac)
#             goals_reached[goal_success_frac == 1] = 1
#             total_goals_reached += np.sum(goals_reached)
#             save_stat_dict["success"].extend(goals_reached)
#             save_stat_dict["success_frac"].extend(goal_success_frac)
#             save_stat_dict["max_obj_dist"].extend(max_obj_dist)
#             save_stat_dict["avg_obj_dist"].extend(avg_obj_dist)
#             save_stat_dict["avg_return"].extend(ret / env.horizon)

#         print(f"Evaluation completed in {time.time() - start_time:5.2f}s")
#         if env.push_t:
#             save_stat_dict["avg_ori_dist"] = ori_dist_list

#         if stat_save_path is not None:
#             with open(stat_save_path, 'wb') as file:
#                 pickle.dump(save_stat_dict, file)
#             print(f"Saved eval stats to {stat_save_path}\n")

#         # compute overall stats
#         mean_return = total_returns / num_eval_episodes
#         mean_latent_rep_return = total_latent_rep_returns / num_eval_episodes
#         mean_avg_obj_dist = total_avg_obj_dist / num_eval_episodes
#         mean_max_obj_dist = total_max_obj_dist / num_eval_episodes
#         mean_success_frac = total_goal_success_frac / num_eval_episodes
#         succes_rate = (total_goals_reached / num_eval_episodes) * 100

#         std_success_rate = np.std(save_stat_dict["success"])
#         std_return = np.std(save_stat_dict["avg_return"])
#         std_success_frac = np.std(save_stat_dict["success_frac"])
#         std_avg_obj_dist = np.std(save_stat_dict["avg_obj_dist"])
#         std_max_obj_dist = np.std(save_stat_dict["max_obj_dist"])

#         print(f"Goal success rate: {succes_rate:3.3f}%")
#         print(f"Goal success fraction: {mean_success_frac:3.3f}")
#         print(f"Max object-goal distance: {mean_max_obj_dist:3.3f}")
#         print(f"Avg. object-goal distance: {mean_avg_obj_dist:3.3f}")
#         print(f"Avg. reward: {mean_return / env.horizon:3.3f}")

#         eval_stat_dict = {
#             "succes_rate": succes_rate,
#             "mean_success_frac": mean_success_frac,
#             "mean_avg_obj_dist": mean_avg_obj_dist,
#             "mean_max_obj_dist": mean_max_obj_dist,
#             "mean_return": mean_return,
#             "mean_latent_rep_return": mean_latent_rep_return,
#             "std_success_rate": std_success_rate,
#             "std_return": std_return,
#             "std_success_frac": std_success_frac,
#             "std_avg_obj_dist": std_avg_obj_dist,
#             "std_max_obj_dist": std_max_obj_dist,
#             "img_list": img_list,
#             "goal_img": goal_img,
#             "eval_vid_success": eval_vid_success,
#             "entity_success_array": np.array(save_stat_dict["success_frac"]) * args.num_entity,
#             "ori_dist_array": np.concatenate(ori_dist_list) if env.push_t else None,
#             "mean_ori_dist":  np.mean(save_stat_dict['avg_ori_dist']) if env.push_t else None,
#             "std_ori_dist": np.std(save_stat_dict["avg_ori_dist"]) if env.push_t else None,
#         }
#         policy.diffusion_model.train()
#         return eval_stat_dict

# def wandb_log_eval_stats(env, eval_stat_dict, args):
#     # log stats
#     wandb.log({"eval_goal_achievement_%": eval_stat_dict["succes_rate"]}, commit=False)
#     wandb.log({"mean_success_frac": eval_stat_dict["mean_success_frac"]}, commit=False)
#     wandb.log({"mean_avg_obj_dist": eval_stat_dict["mean_avg_obj_dist"]}, commit=False)
#     wandb.log({"mean_max_obj_dist": eval_stat_dict["mean_max_obj_dist"]}, commit=False)
#     wandb.log({"eval_mean_reward": eval_stat_dict["mean_return"]}, commit=False)
#     wandb.log({"std_reward": eval_stat_dict["std_return"]}, commit=False)
#     wandb.log({"std_success_rate": eval_stat_dict["std_success_rate"]}, commit=False)
#     wandb.log({"std_success_frac": eval_stat_dict["std_success_frac"]}, commit=False)
#     wandb.log({"std_avg_obj_dist": eval_stat_dict["std_avg_obj_dist"]}, commit=False)
#     wandb.log({"std_max_obj_dist": eval_stat_dict["std_max_obj_dist"]}, commit=False)
#     if env.push_t:
#         wandb.log({"mean_ori_dist": eval_stat_dict["mean_ori_dist"]}, commit=False)
#         wandb.log({"std_ori_dist": eval_stat_dict["std_ori_dist"]}, commit=False)
#     # log episode video
#     vid_save_dir = os.path.join(args.savepath, 'eval_episode_video.gif')
#     ## add number to img_list and visualize with goal image side by side
#     img_list = eval_stat_dict["img_list"]
#     images = []
#     for t, img in enumerate(img_list):
#         img = img.copy()
#         cv.putText(img, f'Timestep: {t}', (10, 10), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
#         images.append(np.concatenate([img, eval_stat_dict["goal_img"]], axis=1))
#     clip = ImageSequenceClip(images, fps=15)
#     clip.write_gif(vid_save_dir, fps=15)
#     if hasattr(args, 'vis_traj_wandb') and args.vis_traj_wandb:
#         # log goal image
#         wandb.log({f"Eval Goal Image": wandb.Image(eval_stat_dict["goal_img"])}, commit=False)
#         vid_caption = "Success" if eval_stat_dict["eval_vid_success"] else "Failure"
#         vid = wandb.Video(data_or_path=vid_save_dir, caption=vid_caption, fps=15)
#         wandb.log({f"Eval Episode Video": vid}, commit=False)
#         # log orientation distance distribution plot
#     if env.push_t:
#         hist_fig = plt.figure(1, figsize=(5, 5), clear=True)
#         plt.hist(eval_stat_dict["ori_dist_array"], bins=np.linspace(0, np.pi, num=50), edgecolor='black')
#         wandb.log({f"Distribution of Orientation Distance from Goal": wandb.Image(hist_fig)}, commit=False)
#     if "entity_success_array" in eval_stat_dict:
#         hist_fig = plt.figure(1, figsize=(5, 5), clear=True)
#         # plt.hist(eval_stat_dict["entity_success_array"], edgecolor='black')
#         plt.hist([int(i*args.num_entity) for i in eval_stat_dict["entity_success_array"]], bins=np.arange(start=-0.5, stop=7, step=1), edgecolor='black')
#         wandb.log({f"Distribution of Goal Success Fraction": wandb.Image(hist_fig)}, commit=False)

# diffuser/eval_utils.py
import os
import pickle
import numpy as np

def setup_isaac_env(args):
    """
    Only call this if you are actually evaluating in IsaacGym.
    IMPORTANT: importing IsaacGym must happen before torch is imported anywhere.
    So do NOT call this in a process where torch is already imported.
    """
    # move Isaac-related imports *inside* the function
    from isaac_panda_push_env import IsaacPandaPush  # this triggers isaacgym import

    env = IsaacPandaPush(
        num_envs=getattr(args, "num_envs", 1),
        headless=getattr(args, "headless", True),
        device=getattr(args, "device", "cuda"),
    )
    return env


def evaluate_policy(policy, env, plan_args, logger,
                    num_eval_episodes=100, exe_steps=None, stat_save_path=None):
    """
    Generic rollout evaluator. Works for any env that supports reset/step.
    """
    successes = []
    returns = []
    lengths = []

    for ep in range(num_eval_episodes):
        obs = env.reset()
        done = False
        ep_ret = 0.0
        t = 0

        while not done:
            action, _ = policy(obs)
            obs, reward, done, info = env.step(action)
            ep_ret += float(reward)
            t += 1
            if exe_steps is not None and t >= exe_steps:
                break

        # try common keys
        success = False
        if isinstance(info, dict):
            success = bool(info.get("success", info.get("is_success", False)))

        successes.append(success)
        returns.append(ep_ret)
        lengths.append(t)

    stats = {
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "avg_return": float(np.mean(returns)) if returns else 0.0,
        "avg_len": float(np.mean(lengths)) if lengths else 0.0,
    }

    if stat_save_path is not None:
        os.makedirs(os.path.dirname(stat_save_path), exist_ok=True)
        with open(stat_save_path, "wb") as f:
            pickle.dump(stats, f)

    return stats

def extract_mimicgen_task_name(h5_path):
    """
    Extract the task name from a MimicGen/robomimic HDF5 dataset.

    Maps env_name (e.g., "Threading_D0", "Hammer_Cleanup_D0") to the task key
    used in TASK_VOXEL_BOUNDS (e.g., "threading", "hammer_cleanup").

    Returns:
        task_name: str or None if not found
    """
    from robomimic.utils import file_utils as FileUtils

    try:
        env_meta = FileUtils.get_env_metadata_from_dataset(h5_path)
        env_name = env_meta.get("env_name", None) or env_meta.get("env", None)

        if env_name is None:
            return None

        # Map env_name to task key
        # Examples: "Threading_D0" -> "threading", "HammerCleanup_D0" -> "hammer_cleanup"
        # Convert CamelCase to snake_case, then remove _D0, _D1, etc. suffix
        import re

        # First convert CamelCase to snake_case (e.g., HammerCleanup -> hammer_cleanup)
        task = re.sub(r'(?<!^)(?=[A-Z])', '_', env_name).lower()

        # Remove trailing _d_0, _d_1, etc. (the underscore before D is added by camelcase conversion)
        task = re.sub(r'_d_?\d+$', '', task)

        # Clean up double underscores and trailing underscores
        task = re.sub(r'_+', '_', task).rstrip('_')

        # Known mappings (env_name patterns -> TASK_VOXEL_BOUNDS keys)
        # Also includes common variations/aliases
        task_mappings = {
            "threading": "threading",
            "threading_cleanup": "threading",  # Alias - same bounds as threading
            "hammer_cleanup": "hammer_cleanup",
            "nut_assembly": "nut_assembly",
            "pick_place": "pick_place",
            "square": "square",
            "stack": "stack",
            "stack_three": "stack_three",
            "kitchen": "kitchen",
            "coffee": "coffee",
            "coffee_preparation": "coffee_preparation",
            "mug_cleanup": "mug_cleanup",
            "three_piece_assembly": "three_piece_assembly",
        }

        # Direct match
        if task in task_mappings:
            return task_mappings[task]

        # Partial match (e.g., "threading_d0" contains "threading")
        for key in task_mappings:
            if key in task:
                return task_mappings[key]

        print(f"[extract_mimicgen_task_name] Warning: Could not map env_name '{env_name}' to known task")
        return None

    except Exception as e:
        print(f"[extract_mimicgen_task_name] Warning: Failed to extract task name: {e}")
        return None


def setup_mimicgen_env(args, use_absolute_actions=True):
    """
    Create a MimicGen / RoboSuite env using robomimic metadata from a dataset HDF5.

    We use args.calib_h5_path as the dataset file by default (since you already have it),
    but you can also set args.mimicgen_dataset_h5.

    Args:
        args: Namespace with configuration
        use_absolute_actions: If True, configure controller for absolute actions
                             (control_delta=False). The policy outputs absolute
                             pose targets [x, y, z, rotvec_x, rotvec_y, rotvec_z, gripper]
                             instead of delta commands.

    This function keeps imports local so importing eval_utils doesn't require MimicGen deps
    unless you actually call mimicgen eval.
    """
    # 1) Try to register MimicGen envs (different installs use different module names)
    for mod in ("mimicgen_envs", "mimicgen"):
        try:
            __import__(mod)
            break
        except Exception:
            pass

    # 2) Pull env metadata from a robomimic-style dataset .hdf5
    dataset_h5 = getattr(args, "mimicgen_dataset_h5", None) or getattr(args, "calib_h5_path", None)
    if dataset_h5 is None:
        raise RuntimeError(
            "setup_mimicgen_env needs args.calib_h5_path or args.mimicgen_dataset_h5 "
            "pointing to a robomimic-format dataset .hdf5 that contains env metadata."
        )

    from robomimic.utils import file_utils as FileUtils
    from robomimic.utils import env_utils as EnvUtils

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_h5)

    # 3) Optional overrides for cameras / offscreen rendering
    env_kwargs = dict(env_meta.get("env_kwargs", {}) or {})
    env_kwargs.setdefault("has_renderer", False)
    env_kwargs.setdefault("has_offscreen_renderer", True)
    env_kwargs.setdefault("use_camera_obs", True)

    # 3b) Configure action mode (absolute vs delta)
    # When use_absolute_actions=True (control_delta=False), the controller interprets actions as:
    #   [goal_x, goal_y, goal_z, rotvec_x, rotvec_y, rotvec_z, gripper]
    # When use_absolute_actions=False (control_delta=True), the controller interprets actions as:
    #   [dx, dy, dz, droll, dpitch, dyaw, gripper]
    controller_configs = env_kwargs.get("controller_configs", {})
    if controller_configs is None:
        controller_configs = {}

    # Set control_delta based on action mode
    control_delta_value = not use_absolute_actions  # True for delta, False for absolute

    if isinstance(controller_configs, dict):
        controller_configs["control_delta"] = control_delta_value
    elif isinstance(controller_configs, list):
        # Multiple robots - set for all
        for cfg in controller_configs:
            if isinstance(cfg, dict):
                cfg["control_delta"] = control_delta_value
    env_kwargs["controller_configs"] = controller_configs

    if use_absolute_actions:
        print("=" * 60)
        print("[setup_mimicgen_env] Using ABSOLUTE actions (control_delta=False)")
        print("[setup_mimicgen_env]   Action format: [goal_x, goal_y, goal_z, rotvec_x, rotvec_y, rotvec_z, gripper]")
        print("[setup_mimicgen_env]   Policy outputs absolute pose targets, NOT deltas")
        print("=" * 60)
    else:
        print("=" * 60)
        print("[setup_mimicgen_env] Using RELATIVE/DELTA actions (control_delta=True)")
        print("[setup_mimicgen_env]   Action format: [dx, dy, dz, droll, dpitch, dyaw, gripper]")
        print("=" * 60)

    cams = getattr(args, "mimicgen_cams", None)
    if cams is not None:
        env_kwargs["camera_names"] = list(cams)

    # If you want to force sizes, set these args in config and they’ll override metadata
    cam_w = getattr(args, "mimicgen_camera_width", None)
    cam_h = getattr(args, "mimicgen_camera_height", None)
    if cam_w is not None:
        env_kwargs["camera_widths"] = cam_w
    if cam_h is not None:
        env_kwargs["camera_heights"] = cam_h


            # --- FIX: robomimic ObsUtils must be initialized before env.reset() ---
    from robomimic.utils import obs_utils as ObsUtils

    # --- force the cameras you want ---
    cam_names = list(getattr(args, "mimicgen_cams", ["agentview", "sideview"]))
    env_kwargs["camera_names"] = cam_names

    # --- force depth ON (you need this for pointcloud reconstruction) ---
    # safest is per-camera lists
    env_kwargs["camera_depths"]  = [True] * len(cam_names)
    env_kwargs["camera_heights"] = [getattr(args, "mimicgen_camera_height", 256)] * len(cam_names)
    env_kwargs["camera_widths"]  = [getattr(args, "mimicgen_camera_width", 256)] * len(cam_names)

    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["use_camera_obs"] = True

    # --- initialize ObsUtils BEFORE env.reset() ever happens ---
    rgb_keys   = [f"{c}_image" for c in cam_names]
    depth_keys = [f"{c}_depth" for c in cam_names]

    obs_specs = {
        "obs": {
            "rgb": rgb_keys,
            "depth": depth_keys,
            "low_dim": [],
        }
    }
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_specs)


    # 4) Create env (robomimic APIs vary slightly across versions, so be defensive)
    env_name = env_meta.get("env_name", None)
    if env_name is None:
        # some versions store it under a different key
        env_name = env_meta.get("env", None)

    try:
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_h5)

        # Ensure key exists and merge overrides here
        env_meta.setdefault("env_kwargs", {})
        env_meta["env_kwargs"].update(env_kwargs)

        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            env_name=env_meta.get("env_name", None),
            render=False,
            render_offscreen=True,
            use_image_obs=True,
            use_depth_obs=True
        )
        o = env.reset()
        print([k for k in sorted(o.keys()) if "image" in k or "depth" in k])
        for cn in cam_names:
            dk = f"{cn}_depth"
            if dk in o:
                print(f"{dk}:", o[dk].shape, float(o[dk].min()), float(o[dk].max()))

        return env

    except TypeError:
        # older signatures
        env = EnvUtils.create_env_from_metadata(
            env_meta,
            env_name=env_name,
            render=False,
            render_offscreen=True,
            use_image_obs=True,
            use_obs_depth=True
        )

    return env

def wandb_log_eval_stats(env, eval_stat_dict, plan_args):
    # minimal logging; keep it generic
    import wandb
    wandb.log({f"eval/{k}": v for k, v in eval_stat_dict.items()})
