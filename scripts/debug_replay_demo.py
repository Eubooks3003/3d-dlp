#!/usr/bin/env python3
"""
Debug script to replay demos from the preprocessed pkl file.

This helps verify:
1. init_states are correct (env resets to the right configuration)
2. Actions (absolute or relative) replay correctly
3. The demo trajectories are smooth (not jittery)

Usage:
    python debug_replay_demo.py --pkl /path/to/data.pkl --h5 /path/to/original.hdf5 --demo-idx 0

    # Replay multiple demos
    python debug_replay_demo.py --pkl /path/to/data.pkl --h5 /path/to/original.hdf5 --demo-idx 0 1 2

    # Use relative actions (original h5 actions, not converted)
    python debug_replay_demo.py --pkl /path/to/data.pkl --h5 /path/to/original.hdf5 --use-relative-actions
"""

import argparse
import pickle
import numpy as np
import time
import os

def _to_hwc_uint8(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)

    # Peel batch dim if present
    if frame.ndim == 4:
        if frame.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for video frame, got {frame.shape}")
        frame = frame[0]

    if frame.ndim == 3:
        # If it's CHW (C,H,W), move channels to the end
        if frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
            frame = np.transpose(frame, (1, 2, 0))

        # Now must be HWC with 1/3/4 channels
        if frame.shape[-1] not in (1, 3, 4):
            raise ValueError(f"Frame has invalid channel count: shape={frame.shape}")

    elif frame.ndim == 2:
        # Grayscale is fine
        pass
    else:
        raise ValueError(f"Unsupported frame shape for video: {frame.shape}")

    # Convert dtype/range
    if frame.dtype != np.uint8:
        # If it's float in [0,1], scale; otherwise just clip
        if np.issubdtype(frame.dtype, np.floating) and frame.max() <= 1.0:
            frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

    return frame


def main():
    parser = argparse.ArgumentParser(description="Replay demos from pkl file")
    parser.add_argument("--pkl", type=str, required=True,
                        help="Path to preprocessed pkl file with init_states and actions")
    parser.add_argument("--h5", type=str, required=True,
                        help="Path to original h5 file (for env metadata)")
    parser.add_argument("--demo-idx", type=int, nargs="+", default=[0],
                        help="Demo index(es) to replay (default: 0)")
    parser.add_argument("--use-relative-actions", action="store_true",
                        help="Use relative/delta actions instead of absolute (control_delta=True)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (default: 1.0)")
    parser.add_argument("--save-video", type=str, default=None,
                        help="Path to save video (e.g., demo_replay.mp4)")
    parser.add_argument("--no-render", action="store_true",
                        help="Don't render to screen (useful with --save-video)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Max steps to replay per demo (default: full trajectory)")
    parser.add_argument("--save-goal-image", type=str, default=None,
                        help="Path to save goal state image (e.g., goal_state.png)")
    parser.add_argument("--save-debug-dir", type=str, default=None,
                        help="Directory to save debug visualizations (tokens, comparisons, etc.)")
    parser.add_argument("--compare-dlp-tokens", action="store_true",
                        help="Compare DLP tokens from pkl vs live encoding (requires DLP model)")
    parser.add_argument("--dlp-ckpt", type=str, default=None,
                        help="Path to DLP checkpoint for token comparison")
    parser.add_argument("--dlp-cfg", type=str, default=None,
                        help="Path to DLP config for token comparison")
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Load preprocessed data
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Loading preprocessed data from: {args.pkl}")
    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)

    print(f"\nDataset contents:")
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        elif isinstance(v, dict):
            print(f"  {k}: dict with {len(v)} keys")
        else:
            print(f"  {k}: {type(v)}")

    # Check for required keys
    if 'init_states' not in data:
        print("\n[ERROR] 'init_states' not found in pkl file!")
        print("         Run the updated preprocessing script to include init_states.")
        return

    init_states = data['init_states']
    actions = data['actions']
    path_lengths = data['path_lengths']
    observations = data.get('observations', None)
    goals = data.get('goals', None)

    num_demos = len(init_states)
    print(f"\nFound {num_demos} demos with init_states")
    print(f"  init_states shape: {init_states.shape}")
    print(f"  actions shape: {actions.shape}")
    print(f"  path_lengths: min={path_lengths.min()}, max={path_lengths.max()}")

    # Print goal information if available
    if goals is not None:
        print(f"  goals shape: {goals.shape}")
        # Verify goals are final observations
        if observations is not None:
            for i in range(min(3, num_demos)):
                pl = path_lengths[i]
                goal_t0 = goals[i, 0]
                obs_end = observations[i, pl-1]
                match = np.allclose(goal_t0, obs_end)
                print(f"    Demo {i}: goal[0] == obs[end]: {match}")

    # Setup debug directory if specified
    debug_dir = args.save_debug_dir
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        print(f"\nDebug output will be saved to: {debug_dir}")

    # -------------------------------------------------------------------------
    # Setup environment
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Setting up environment from: {args.h5}")

    # Try to import and register mimicgen
    for mod in ("mimicgen_envs", "mimicgen"):
        try:
            __import__(mod)
            break
        except ImportError:
            pass

    from robomimic.utils import file_utils as FileUtils
    from robomimic.utils import env_utils as EnvUtils
    from robomimic.utils import obs_utils as ObsUtils
    import copy

    env_meta = FileUtils.get_env_metadata_from_dataset(args.h5)

    # Configure for absolute or relative actions
    if not args.use_relative_actions:
        print("\n[Action Mode] ABSOLUTE actions (control_delta=False)")
        print("  Action format: [goal_x, goal_y, goal_z, rotvec_x, rotvec_y, rotvec_z, gripper]")
        env_meta_copy = copy.deepcopy(env_meta)
        if 'controller_configs' in env_meta_copy.get('env_kwargs', {}):
            env_meta_copy['env_kwargs']['controller_configs']['control_delta'] = False
        env_meta = env_meta_copy
    else:
        print("\n[Action Mode] RELATIVE/DELTA actions (control_delta=True)")
        print("  Action format: [dx, dy, dz, droll, dpitch, dyaw, gripper]")

    # Set up rendering
    env_meta.setdefault('env_kwargs', {})
    env_meta['env_kwargs']['has_renderer'] = not args.no_render
    env_meta['env_kwargs']['has_offscreen_renderer'] = True
    env_meta['env_kwargs']['use_camera_obs'] = True
    env_meta['env_kwargs']['camera_names'] = ['agentview', 'sideview']
    env_meta['env_kwargs']['camera_heights'] = [256, 256]
    env_meta['env_kwargs']['camera_widths'] = [256, 256]

    # Initialize ObsUtils
    obs_specs = {
        "obs": {
            "rgb": ["agentview_image", "sideview_image"],
            "depth": [],
            "low_dim": [],
        }
    }
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_specs)

    # Create environment
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=not args.no_render,
        render_offscreen=True,
        use_image_obs=True,
    )
    print(f"\nEnvironment created: {env_meta.get('env_name', 'unknown')}")
    print(f"  Action dim: {env.action_dimension}")

    # -------------------------------------------------------------------------
    # Save goal state images if requested
    # -------------------------------------------------------------------------
    if args.save_goal_image or debug_dir:
        print(f"\n{'='*60}")
        print("Saving goal state visualizations")

        for demo_idx in args.demo_idx[:1]:  # Just first demo for goal viz
            if demo_idx >= num_demos:
                continue

            # Get the final state of this trajectory by replaying to the end
            init_state = init_states[demo_idx]
            demo_actions = actions[demo_idx]
            demo_length = int(path_lengths[demo_idx])

            print(f"\n  Replaying demo {demo_idx} to capture goal state...")

            # Reset to init state
            obs = env.reset_to({"states": init_state})
            if obs is None:
                obs = env.get_observation()

            # Replay all actions to reach goal state
            for t in range(demo_length):
                action = demo_actions[t]
                obs, _, _, _ = env.step(action)

            # Capture goal state image
            if 'agentview_image' in obs:
                goal_img = _to_hwc_uint8(obs['agentview_image'])

                # Save goal image
                if args.save_goal_image:
                    import imageio
                    imageio.imwrite(args.save_goal_image, goal_img)
                    print(f"  Saved goal state image to: {args.save_goal_image}")

                if debug_dir:
                    import imageio
                    goal_path = os.path.join(debug_dir, f"goal_state_demo{demo_idx}.png")
                    imageio.imwrite(goal_path, goal_img)
                    print(f"  Saved goal state image to: {goal_path}")

            # Also save the initial state image for comparison
            if debug_dir:
                obs = env.reset_to({"states": init_state})
                if obs is None:
                    obs = env.get_observation()
                if 'agentview_image' in obs:
                    init_img = _to_hwc_uint8(obs['agentview_image'])
                    import imageio
                    init_path = os.path.join(debug_dir, f"init_state_demo{demo_idx}.png")
                    imageio.imwrite(init_path, init_img)
                    print(f"  Saved init state image to: {init_path}")

            # Save goal tokens info if available
            if debug_dir and goals is not None:
                goal_tokens = goals[demo_idx, 0]  # (K, Dtok)
                token_path = os.path.join(debug_dir, f"goal_tokens_demo{demo_idx}.txt")
                with open(token_path, 'w') as f:
                    f.write(f"Goal tokens for demo {demo_idx}\n")
                    f.write(f"Shape: {goal_tokens.shape}\n")
                    f.write(f"Min: {goal_tokens.min():.4f}, Max: {goal_tokens.max():.4f}\n")
                    f.write(f"Mean: {goal_tokens.mean():.4f}, Std: {goal_tokens.std():.4f}\n\n")
                    f.write("Tokens (K x Dtok):\n")
                    for k in range(goal_tokens.shape[0]):
                        f.write(f"  Particle {k}: {goal_tokens[k]}\n")
                print(f"  Saved goal tokens info to: {token_path}")

    # -------------------------------------------------------------------------
    # Compare DLP tokens if requested
    # -------------------------------------------------------------------------
    if args.compare_dlp_tokens and args.dlp_ckpt and args.dlp_cfg:
        print(f"\n{'='*60}")
        print("Comparing DLP tokens: pkl vs live encoding")

        try:
            import sys
            import torch
            import json

            # Add path for DLP model
            lpwm_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sys.path.insert(0, lpwm_dir)

            from voxel_models import DLP

            # Load DLP model
            print(f"  Loading DLP model from: {args.dlp_ckpt}")
            with open(args.dlp_cfg, 'r') as f:
                dlp_cfg = json.load(f)

            dlp_model = DLP(**dlp_cfg)
            ckpt = torch.load(args.dlp_ckpt, map_location='cpu')
            dlp_model.load_state_dict(ckpt['model_state_dict'])
            dlp_model.eval()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            dlp_model = dlp_model.to(device)
            print(f"  DLP model loaded on {device}")

            # Compare for first demo
            demo_idx = args.demo_idx[0] if args.demo_idx[0] < num_demos else 0
            init_state = init_states[demo_idx]
            demo_length = int(path_lengths[demo_idx])

            # Get stored tokens from pkl
            pkl_obs_t0 = observations[demo_idx, 0]  # (K, Dtok)
            pkl_goal = goals[demo_idx, 0]           # (K, Dtok)

            # Reset env and get live observation
            obs = env.reset_to({"states": init_state})
            if obs is None:
                obs = env.get_observation()

            # Encode live observation with DLP
            # This requires the same preprocessing as used during data collection
            print(f"\n  Comparing observations at t=0 for demo {demo_idx}:")
            print(f"    PKL obs[0] shape: {pkl_obs_t0.shape}")
            print(f"    PKL obs[0] range: [{pkl_obs_t0.min():.4f}, {pkl_obs_t0.max():.4f}]")

            # Note: Full comparison requires matching the exact preprocessing pipeline
            # Here we just report the pkl token statistics for debugging
            print(f"\n  PKL goal tokens stats:")
            print(f"    Shape: {pkl_goal.shape}")
            print(f"    Range: [{pkl_goal.min():.4f}, {pkl_goal.max():.4f}]")
            print(f"    Mean: {pkl_goal.mean():.4f}, Std: {pkl_goal.std():.4f}")

            if debug_dir:
                # Save detailed token comparison
                compare_path = os.path.join(debug_dir, f"token_comparison_demo{demo_idx}.txt")
                with open(compare_path, 'w') as f:
                    f.write(f"Token comparison for demo {demo_idx}\n")
                    f.write(f"="*60 + "\n\n")
                    f.write(f"PKL observation tokens at t=0:\n")
                    f.write(f"  Shape: {pkl_obs_t0.shape}\n")
                    f.write(f"  Range: [{pkl_obs_t0.min():.4f}, {pkl_obs_t0.max():.4f}]\n")
                    f.write(f"  Mean: {pkl_obs_t0.mean():.4f}, Std: {pkl_obs_t0.std():.4f}\n\n")
                    f.write(f"PKL goal tokens:\n")
                    f.write(f"  Shape: {pkl_goal.shape}\n")
                    f.write(f"  Range: [{pkl_goal.min():.4f}, {pkl_goal.max():.4f}]\n")
                    f.write(f"  Mean: {pkl_goal.mean():.4f}, Std: {pkl_goal.std():.4f}\n\n")
                    f.write(f"Per-particle statistics:\n")
                    for k in range(pkl_obs_t0.shape[0]):
                        f.write(f"  Particle {k}:\n")
                        f.write(f"    obs: mean={pkl_obs_t0[k].mean():.4f}, std={pkl_obs_t0[k].std():.4f}\n")
                        f.write(f"    goal: mean={pkl_goal[k].mean():.4f}, std={pkl_goal[k].std():.4f}\n")
                print(f"  Saved token comparison to: {compare_path}")

        except Exception as e:
            print(f"  [ERROR] Failed to compare DLP tokens: {e}")
            import traceback
            traceback.print_exc()

    # -------------------------------------------------------------------------
    # Replay demos
    # -------------------------------------------------------------------------
    video_frames = []

    for demo_idx in args.demo_idx:
        if demo_idx >= num_demos:
            print(f"\n[WARNING] Demo index {demo_idx} out of range (max: {num_demos-1}), skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Replaying demo {demo_idx}")
        print(f"{'='*60}")

        # Get init state and actions for this demo
        init_state = init_states[demo_idx]
        demo_actions = actions[demo_idx]
        demo_length = int(path_lengths[demo_idx])

        print(f"  init_state shape: {init_state.shape}")
        print(f"  demo_length: {demo_length}")

        # Reset to init state
        print(f"\nResetting to init_state...")
        obs = env.reset_to({"states": init_state})
        if obs is None:
            obs = env.get_observation()
        print(f"  Reset successful!")

        # Show action statistics for this demo
        valid_actions = demo_actions[:demo_length]
        print(f"\nAction statistics for demo {demo_idx}:")
        action_names = ['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z', 'gripper']
        for i, name in enumerate(action_names):
            a = valid_actions[:, i]
            print(f"  {name:>8}: min={a.min():>8.4f}, max={a.max():>8.4f}, mean={a.mean():>8.4f}")

        # Replay loop
        max_steps = args.max_steps if args.max_steps else demo_length
        step_delay = 0.02 / args.speed  # ~50Hz nominal, adjusted by speed

        print(f"\nReplaying {min(max_steps, demo_length)} steps...")

        for t in range(min(max_steps, demo_length)):
            action = demo_actions[t]

            # Print first few actions for debugging
            if t < 3 or t == demo_length - 1:
                print(f"\n  Step {t}:")
                print(f"    pos:     [{action[0]:>8.4f}, {action[1]:>8.4f}, {action[2]:>8.4f}]")
                print(f"    rot:     [{action[3]:>8.4f}, {action[4]:>8.4f}, {action[5]:>8.4f}]")
                print(f"    gripper: {action[6]:>8.4f}")

            # Step environment
            obs, reward, done, info = env.step(action)

            # Render
            if not args.no_render:
                env.render()

            # Collect frames for video
            if args.save_video and 'agentview_image' in obs:
                frame = _to_hwc_uint8(obs['agentview_image'])
                video_frames.append(frame)


            # Check success
            success = info.get('success', info.get('task_success', None))
            if success:
                print(f"\n  [SUCCESS] Task completed at step {t}!")

            if done:
                print(f"\n  [DONE] Episode ended at step {t}")
                break

            time.sleep(step_delay)

        # Final status
        final_success = info.get('success', info.get('task_success', False))
        print(f"\nDemo {demo_idx} finished:")
        print(f"  Steps taken: {t+1}")
        print(f"  Final success: {final_success}")

        # Save action trajectory plot if debug_dir is set
        if debug_dir:
            try:
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(3, 3, figsize=(15, 12))
                fig.suptitle(f'Action Trajectory - Demo {demo_idx}', fontsize=14)

                valid_actions = demo_actions[:demo_length]
                t_range = np.arange(demo_length)

                # Position
                axes[0, 0].plot(t_range, valid_actions[:, 0], label='x')
                axes[0, 0].set_title('Position X')
                axes[0, 0].set_xlabel('Timestep')
                axes[0, 0].grid(True)

                axes[0, 1].plot(t_range, valid_actions[:, 1], label='y')
                axes[0, 1].set_title('Position Y')
                axes[0, 1].set_xlabel('Timestep')
                axes[0, 1].grid(True)

                axes[0, 2].plot(t_range, valid_actions[:, 2], label='z')
                axes[0, 2].set_title('Position Z')
                axes[0, 2].set_xlabel('Timestep')
                axes[0, 2].grid(True)

                # Rotation
                axes[1, 0].plot(t_range, valid_actions[:, 3], label='rot_x')
                axes[1, 0].set_title('Rotation X')
                axes[1, 0].set_xlabel('Timestep')
                axes[1, 0].grid(True)

                axes[1, 1].plot(t_range, valid_actions[:, 4], label='rot_y')
                axes[1, 1].set_title('Rotation Y')
                axes[1, 1].set_xlabel('Timestep')
                axes[1, 1].grid(True)

                axes[1, 2].plot(t_range, valid_actions[:, 5], label='rot_z')
                axes[1, 2].set_title('Rotation Z')
                axes[1, 2].set_xlabel('Timestep')
                axes[1, 2].grid(True)

                # Gripper
                axes[2, 0].plot(t_range, valid_actions[:, 6], label='gripper', color='red')
                axes[2, 0].set_title('Gripper (-1=close, 1=open)')
                axes[2, 0].set_xlabel('Timestep')
                axes[2, 0].set_ylim(-1.2, 1.2)
                axes[2, 0].grid(True)

                # 3D position trajectory
                ax3d = fig.add_subplot(3, 3, 8, projection='3d')
                ax3d.plot(valid_actions[:, 0], valid_actions[:, 1], valid_actions[:, 2])
                ax3d.scatter(valid_actions[0, 0], valid_actions[0, 1], valid_actions[0, 2], c='green', s=100, label='start')
                ax3d.scatter(valid_actions[-1, 0], valid_actions[-1, 1], valid_actions[-1, 2], c='red', s=100, label='end')
                ax3d.set_xlabel('X')
                ax3d.set_ylabel('Y')
                ax3d.set_zlabel('Z')
                ax3d.set_title('3D Position Trajectory')
                ax3d.legend()

                # Action deltas (to see jittering)
                axes[2, 2].plot(t_range[1:], np.diff(valid_actions[:, 0]), alpha=0.7, label='dx')
                axes[2, 2].plot(t_range[1:], np.diff(valid_actions[:, 1]), alpha=0.7, label='dy')
                axes[2, 2].plot(t_range[1:], np.diff(valid_actions[:, 2]), alpha=0.7, label='dz')
                axes[2, 2].set_title('Position Deltas (for jitter detection)')
                axes[2, 2].set_xlabel('Timestep')
                axes[2, 2].legend()
                axes[2, 2].grid(True)

                plt.tight_layout()
                plot_path = os.path.join(debug_dir, f"action_trajectory_demo{demo_idx}.png")
                plt.savefig(plot_path, dpi=150)
                plt.close()
                print(f"  Saved action trajectory plot to: {plot_path}")

            except Exception as e:
                print(f"  [WARNING] Could not save action plot: {e}")

    # -------------------------------------------------------------------------
    # Save video
    # -------------------------------------------------------------------------
    if args.save_video and len(video_frames) > 0:
        print(f"\n{'='*60}")
        print(f"Saving video to: {args.save_video}")
        import imageio
        os.makedirs(os.path.dirname(args.save_video) or '.', exist_ok=True)
        imageio.mimsave(args.save_video, video_frames, fps=30)
        print(f"  Saved {len(video_frames)} frames")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}\n")

    # Keep window open if rendering
    if not args.no_render:
        input("Press Enter to close...")

    env.close()


if __name__ == "__main__":
    main()
