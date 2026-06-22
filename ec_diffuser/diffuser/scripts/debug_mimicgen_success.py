#!/usr/bin/env python3
"""
Debug script for MimicGen DLP wrapper success detection.

Two modes:
1. --mode policy: Roll out the trained diffusion policy and check success detection
2. --mode demo: Replay actual demo actions from HDF5 and check success detection

Usage:
    # Mode 1: Policy rollout
    python debug_mimicgen_success.py --mode policy \
        --checkpoint_dir /home/ellina/Desktop/Code/EC-Diffuser/data/mimicgen_stack_dlp/diffusion/mimicgen_stack/16C_dlp_adalnpint_relative_H16_T5_seed42 \
        --num_episodes 5

    # Mode 2: Demo replay (sanity check)
    python debug_mimicgen_success.py --mode demo \
        --demo_h5 /home/ellina/Desktop/Code/articubot-on-mimicgen/mimicgen_data/stack_d1/core/stack_d1.hdf5 \
        --num_demos 5

    # Mode 3: Just explore the env (no policy, random actions)
    python debug_mimicgen_success.py --mode explore \
        --demo_h5 /path/to/stack_d1.hdf5 \
        --num_episodes 3
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import argparse
import numpy as np
import torch

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EC_DIFFUSER_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
LPWM_DEV = os.path.abspath(os.path.join(EC_DIFFUSER_ROOT, "..", "..", "lpwm-dev"))

sys.path.insert(0, EC_DIFFUSER_ROOT)
sys.path.insert(0, os.path.join(EC_DIFFUSER_ROOT, "diffuser"))
if os.path.isdir(LPWM_DEV):
    sys.path.insert(0, LPWM_DEV)


def load_env_from_h5(h5_path, use_absolute_actions=False):
    """Create a MimicGen env from the HDF5 dataset metadata."""
    # Register MimicGen envs
    for mod in ("mimicgen_envs", "mimicgen"):
        try:
            __import__(mod)
            break
        except Exception:
            pass

    from robomimic.utils import file_utils as FileUtils
    from robomimic.utils import env_utils as EnvUtils
    from robomimic.utils import obs_utils as ObsUtils

    env_meta = FileUtils.get_env_metadata_from_dataset(h5_path)

    cam_names = ["agentview", "sideview"]
    env_kwargs = dict(env_meta.get("env_kwargs", {}) or {})
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["use_camera_obs"] = True
    env_kwargs["camera_names"] = cam_names
    env_kwargs["camera_depths"] = [True] * len(cam_names)
    env_kwargs["camera_heights"] = [256] * len(cam_names)
    env_kwargs["camera_widths"] = [256] * len(cam_names)

    if use_absolute_actions:
        controller_configs = env_kwargs.get("controller_configs", {})
        if isinstance(controller_configs, dict):
            controller_configs["control_delta"] = False
        env_kwargs["controller_configs"] = controller_configs
        print("[ENV] Using ABSOLUTE actions (control_delta=False)")
    else:
        print("[ENV] Using RELATIVE/DELTA actions (control_delta=True)")

    # Initialize ObsUtils
    rgb_keys = [f"{c}_image" for c in cam_names]
    depth_keys = [f"{c}_depth" for c in cam_names]
    obs_specs = {
        "obs": {
            "rgb": rgb_keys,
            "depth": depth_keys,
            "low_dim": [],
        }
    }
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_specs)

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

    return env


def check_success_in_info(info, verbose=True):
    """Check all possible success keys in the info dict."""
    success_keys = ["success", "task_success", "is_success", "task_complete", "done"]

    if verbose:
        print(f"  [INFO] Keys in info: {list(info.keys()) if info else 'None'}")

    found = {}
    if info:
        for key in success_keys:
            if key in info:
                val = info[key]
                found[key] = val
                if verbose:
                    print(f"    {key}: {val} (type: {type(val).__name__})")
                    # Handle nested dict like {'task': True}
                    if isinstance(val, dict):
                        resolved = val.get('task', any(bool(v) for v in val.values()))
                        print(f"      -> NESTED DICT! Resolved to: {resolved}")
                        found[f"{key}_resolved"] = resolved

    return found


def explore_env_info(env, num_steps=50):
    """Do a few random steps and print all info keys."""
    print("\n" + "="*60)
    print("EXPLORING ENV INFO STRUCTURE")
    print("="*60)

    obs = env.reset()
    print(f"\n[RESET] Obs keys: {list(obs.keys())}")

    for step in range(num_steps):
        action = np.random.uniform(-1, 1, size=(7,))  # MimicGen action dim is 7
        obs, reward, done, info = env.step(action)

        if step % 10 == 0 or done:
            print(f"\n[STEP {step}] reward={reward:.4f}, done={done}")
            check_success_in_info(info)

        if done:
            print(f"  Episode ended at step {step}")
            break

    return info


def mode_demo_replay(args):
    """
    Mode 2: Replay demo actions from HDF5 to check success detection.
    This is a sanity check - if demos are successful, we should see success=True.
    """
    import h5py

    print("\n" + "="*60)
    print("MODE: DEMO REPLAY")
    print(f"HDF5: {args.demo_h5}")
    print("="*60)

    # Load env
    env = load_env_from_h5(args.demo_h5, use_absolute_actions=False)

    # Open HDF5
    with h5py.File(args.demo_h5, "r") as f:
        print(f"\n[HDF5] Top-level keys: {list(f.keys())}")

        if "data" not in f:
            print("[ERROR] No 'data' group in HDF5!")
            return

        demo_keys = sorted(list(f["data"].keys()), key=lambda x: int(x.split("_")[1]))
        print(f"[HDF5] Found {len(demo_keys)} demos")

        num_demos = min(args.num_demos, len(demo_keys))
        print(f"[HDF5] Will replay {num_demos} demos")

        success_count = 0
        for demo_idx in range(num_demos):
            demo_key = demo_keys[demo_idx]
            demo_grp = f[f"data/{demo_key}"]

            print(f"\n{'='*60}")
            print(f"DEMO {demo_idx}: {demo_key}")
            print(f"{'='*60}")

            # Get demo data
            actions = np.array(demo_grp["actions"])
            states = np.array(demo_grp["states"])
            print(f"  Actions shape: {actions.shape}")
            print(f"  States shape: {states.shape}")

            # Check for dones/success in demo
            if "dones" in demo_grp:
                dones = np.array(demo_grp["dones"])
                print(f"  Dones in demo: {dones.sum()} True values")

            # Reset to initial state
            init_state = states[0]
            obs = env.reset_to({"states": init_state})
            print(f"  Reset to initial state")

            # Replay actions
            final_info = None
            final_success = None
            for t, action in enumerate(actions):
                obs, reward, done, info = env.step(action)

                # Check success at each step
                success_found = check_success_in_info(info, verbose=False)

                if t == len(actions) - 1 or done:
                    print(f"\n  [STEP {t}] FINAL STEP")
                    print(f"    reward={reward:.4f}, done={done}")
                    success_found = check_success_in_info(info, verbose=True)
                    final_info = info
                    # Check resolved values first (handles nested dicts)
                    final_success = success_found.get("is_success_resolved",
                                    success_found.get("success_resolved",
                                    success_found.get("success",
                                    success_found.get("is_success",
                                    success_found.get("task_success", None)))))

                if done:
                    print(f"  Episode terminated early at step {t}")
                    break

            # Summary for this demo
            print(f"\n  [SUMMARY] Demo {demo_key}:")
            print(f"    Final success detection: {final_success}")
            if final_success:
                success_count += 1

        print(f"\n{'='*60}")
        print(f"DEMO REPLAY SUMMARY")
        print(f"{'='*60}")
        print(f"  Demos replayed: {num_demos}")
        print(f"  Success detected: {success_count}/{num_demos}")
        if success_count < num_demos:
            print(f"  WARNING: Some demos did not report success!")
            print(f"  This suggests a problem with success detection in the env wrapper.")


def mode_explore(args):
    """
    Mode 3: Just explore the env with random actions.
    """
    print("\n" + "="*60)
    print("MODE: EXPLORE")
    print("="*60)

    env = load_env_from_h5(args.demo_h5, use_absolute_actions=False)

    for ep in range(args.num_episodes):
        print(f"\n{'='*40}")
        print(f"EPISODE {ep}")
        print(f"{'='*40}")

        explore_env_info(env, num_steps=100)


def mode_policy_rollout(args):
    """
    Mode 1: Load trained policy and roll out in env.
    """
    import diffuser.utils as utils
    from diffuser.utils.arrays import set_global_device

    print("\n" + "="*60)
    print("MODE: POLICY ROLLOUT")
    print(f"Checkpoint: {args.checkpoint_dir}")
    print("="*60)

    set_global_device(args.device)

    # Load diffusion model
    # The checkpoint structure is: loadbase/dataset/diffusion_loadpath
    # We need to figure out these from the checkpoint_dir
    checkpoint_dir = args.checkpoint_dir

    # Parse the checkpoint path
    # Expected: .../data/mimicgen_stack_dlp/diffusion/mimicgen_stack/16C_dlp_adalnpint_relative_H16_T5_seed42
    parts = checkpoint_dir.rstrip("/").split("/")
    diffusion_loadpath = parts[-1]  # 16C_dlp_adalnpint_relative_H16_T5_seed42
    dataset = parts[-2]  # mimicgen_stack
    loadbase = "/".join(parts[:-2])  # .../data/mimicgen_stack_dlp/diffusion

    print(f"  loadbase: {loadbase}")
    print(f"  dataset: {dataset}")
    print(f"  diffusion_loadpath: {diffusion_loadpath}")

    # Load the args.json to get configuration
    import json
    args_path = os.path.join(checkpoint_dir, "args.json")
    if os.path.exists(args_path):
        with open(args_path, "r") as f:
            train_args = json.load(f)
        print(f"  Loaded args from {args_path}")
    else:
        print(f"  WARNING: No args.json found at {args_path}")
        train_args = {}

    # Load diffusion model
    diffusion_experiment = utils.load_diffusion(
        loadbase, dataset, diffusion_loadpath,
        epoch=args.epoch, seed=train_args.get("seed", 42), is_diffusion=True,
        override_dataset_path=None
    )
    diffusion = diffusion_experiment.ema
    ds = diffusion_experiment.dataset

    print(f"  Diffusion model loaded")
    print(f"  Dataset observation dim: {ds.observation_dim}")
    print(f"  Dataset action dim: {ds.action_dim}")
    print(f"  Horizon: {train_args.get('horizon', 'unknown')}")

    # Create policy
    from diffuser.sampling import GoalConditionedPolicy
    policy = GoalConditionedPolicy(
        diffusion_model=diffusion,
        normalizer=ds.normalizer,
        preprocess_fns=[],
        verbose=False,
        horizon=train_args.get("horizon", 16),
    )

    # Load env
    # We need the HDF5 path from train args or use a default
    h5_path = train_args.get("calib_h5_path", args.demo_h5)
    if h5_path is None:
        print("  ERROR: Need --demo_h5 path to create env")
        return

    print(f"  Creating env from: {h5_path}")
    env = load_env_from_h5(h5_path, use_absolute_actions=train_args.get("use_absolute_actions", False))

    # Also need DLP model if using DLP wrapper
    # Prefer command line args, fall back to train_args
    dlp_cfg_path = args.dlp_cfg_path or train_args.get("dlp_cfg_path", None)
    dlp_ckpt_path = args.dlp_ckpt_path or train_args.get("dlp_ckpt_path", None)
    dataset_pkl = args.dataset_pkl or train_args.get("dataset_pkl_path", None)

    if dlp_cfg_path and dlp_ckpt_path:
        print(f"  Loading DLP from: {dlp_ckpt_path}")
        from scripts.train import load_dlp_lpwm
        dlp_model, dlp_cfg = load_dlp_lpwm(dlp_cfg_path, dlp_ckpt_path, args.device)

        # Create the wrapper
        from diffuser.envs.mimicgen_dlp_wrapper import MimicGenDLPWrapper, DatasetGoalProvider

        # Load goal provider if we have preprocessed data
        goal_provider = None
        if dataset_pkl and os.path.exists(dataset_pkl):
            goal_provider = DatasetGoalProvider(dataset_pkl)
            print(f"  Loaded goal provider with {goal_provider.num_trajectories} trajectories")

        wrapper = MimicGenDLPWrapper(
            env=env,
            dlp_model=dlp_model,
            device=args.device,
            cams=["agentview", "sideview"],
            goal_provider=goal_provider,
            random_init=False,
        )

        # Rollout with wrapper
        success_count = 0
        for ep in range(args.num_episodes):
            print(f"\n{'='*40}")
            print(f"EPISODE {ep}")
            print(f"{'='*40}")

            obs = wrapper.reset()
            done = False
            t = 0

            while not done and t < 200:
                # Get conditions
                cond = wrapper.make_cond(obs, train_args.get("horizon", 16))
                action, samples = policy(cond, batch_size=1)

                obs, reward, done, info = wrapper.step(action)

                if t % 20 == 0:
                    print(f"  [STEP {t}] reward={reward:.4f}, done={done}")
                    check_success_in_info(info, verbose=True)

                t += 1

            # Final check
            print(f"\n  [FINAL] Episode ended at step {t}")
            success_found = check_success_in_info(info, verbose=True)
            if success_found.get("success", False):
                success_count += 1
                print(f"  SUCCESS!")
            else:
                print(f"  FAILURE")

        print(f"\n{'='*60}")
        print(f"POLICY ROLLOUT SUMMARY")
        print(f"{'='*60}")
        print(f"  Episodes: {args.num_episodes}")
        print(f"  Success: {success_count}/{args.num_episodes}")

    else:
        print("  No DLP model paths found in train args")
        print("  Running without DLP wrapper (raw env)")

        success_count = 0
        for ep in range(args.num_episodes):
            print(f"\n{'='*40}")
            print(f"EPISODE {ep} (raw env, random actions)")
            print(f"{'='*40}")

            obs = env.reset()
            done = False
            t = 0

            while not done and t < 200:
                action = np.random.uniform(-1, 1, size=(7,))  # MimicGen action dim is 7
                obs, reward, done, info = env.step(action)

                if t % 20 == 0:
                    print(f"  [STEP {t}] reward={reward:.4f}, done={done}")
                    check_success_in_info(info, verbose=True)

                t += 1

            print(f"\n  [FINAL] Episode ended at step {t}")
            check_success_in_info(info, verbose=True)


def main():
    parser = argparse.ArgumentParser(description="Debug MimicGen success detection")
    parser.add_argument("--mode", type=str, required=True, choices=["policy", "demo", "explore"],
                        help="Mode: 'policy' for policy rollout, 'demo' for demo replay, 'explore' for random exploration")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Path to diffusion checkpoint directory (for policy mode)")
    parser.add_argument("--demo_h5", type=str, default=None,
                        help="Path to HDF5 demo file")
    parser.add_argument("--num_episodes", type=int, default=5,
                        help="Number of episodes to run (for policy/explore mode)")
    parser.add_argument("--num_demos", type=int, default=5,
                        help="Number of demos to replay (for demo mode)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use")
    parser.add_argument("--epoch", type=str, default="latest",
                        help="Checkpoint epoch to load")
    parser.add_argument("--dlp_cfg_path", type=str, default=None,
                        help="Path to DLP config JSON")
    parser.add_argument("--dlp_ckpt_path", type=str, default=None,
                        help="Path to DLP checkpoint")
    parser.add_argument("--dataset_pkl", type=str, default=None,
                        help="Path to preprocessed dataset pkl for goal provider")

    args = parser.parse_args()

    if args.mode == "policy":
        if args.checkpoint_dir is None:
            print("ERROR: --checkpoint_dir required for policy mode")
            return
        mode_policy_rollout(args)

    elif args.mode == "demo":
        if args.demo_h5 is None:
            print("ERROR: --demo_h5 required for demo mode")
            return
        mode_demo_replay(args)

    elif args.mode == "explore":
        if args.demo_h5 is None:
            print("ERROR: --demo_h5 required for explore mode")
            return
        mode_explore(args)


if __name__ == "__main__":
    main()
