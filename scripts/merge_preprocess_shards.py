#!/usr/bin/env python3
"""
Merge preprocessed shards from parallel ec_diffuser_voxel_preprocess.py runs.

Usage:
    python merge_preprocess_shards.py \
        --input-pattern '/path/to/output_rank*.pkl' \
        --output /path/to/output.pkl
"""
import os
import glob
import argparse
import pickle
import numpy as np


def merge_shards(input_pattern: str, output_path: str):
    """Merge multiple shard pickle files into one."""

    # Find all shard files
    shard_files = sorted(glob.glob(input_pattern))
    if not shard_files:
        raise FileNotFoundError(f"No files found matching: {input_pattern}")

    print(f"Found {len(shard_files)} shard files:")
    for f in shard_files:
        print(f"  - {f}")

    # Load all shards
    shards = []
    for f in shard_files:
        with open(f, 'rb') as fp:
            shards.append(pickle.load(fp))

    # Validate that all shards have same structure
    meta0 = shards[0]['meta']
    K, Dtok, A, G, BG = meta0['K'], meta0['Dtok'], meta0['A'], meta0['G'], meta0['BG']
    action_mode = meta0['action_mode']

    for i, shard in enumerate(shards[1:], 1):
        m = shard['meta']
        assert m['K'] == K, f"Shard {i} has K={m['K']}, expected {K}"
        assert m['Dtok'] == Dtok, f"Shard {i} has Dtok={m['Dtok']}, expected {Dtok}"
        assert m['A'] == A, f"Shard {i} has A={m['A']}, expected {A}"
        assert m['G'] == G, f"Shard {i} has G={m['G']}, expected {G}"
        assert m['BG'] == BG, f"Shard {i} has BG={m['BG']}, expected {BG}"

    # Collect episodes from all shards
    all_obs = []
    all_act = []
    all_gripper = []
    all_bg = []
    all_rewards = []
    all_terminals = []
    all_timeouts = []
    all_goals = []
    all_path_lengths = []
    all_init_states = []
    all_demo_indices = []

    for shard in shards:
        E = shard['observations'].shape[0]
        for e in range(E):
            L = int(shard['path_lengths'][e])
            all_obs.append(shard['observations'][e, :L])
            all_act.append(shard['actions'][e, :L])
            all_gripper.append(shard['gripper_state'][e, :L])
            all_bg.append(shard['bg_features'][e, :L])
            all_rewards.append(shard['rewards'][e, :L])
            all_terminals.append(shard['terminals'][e, :L])
            all_timeouts.append(shard['timeouts'][e, :L])
            all_goals.append(shard['goals'][e, :L])
            all_path_lengths.append(L)
            all_init_states.append(shard['init_states'][e])
            all_demo_indices.append(shard['meta']['demo_indices'][e])

    # Sort by demo index to maintain original order
    sort_idx = np.argsort(all_demo_indices)
    all_obs = [all_obs[i] for i in sort_idx]
    all_act = [all_act[i] for i in sort_idx]
    all_gripper = [all_gripper[i] for i in sort_idx]
    all_bg = [all_bg[i] for i in sort_idx]
    all_rewards = [all_rewards[i] for i in sort_idx]
    all_terminals = [all_terminals[i] for i in sort_idx]
    all_timeouts = [all_timeouts[i] for i in sort_idx]
    all_goals = [all_goals[i] for i in sort_idx]
    all_path_lengths = [all_path_lengths[i] for i in sort_idx]
    all_init_states = [all_init_states[i] for i in sort_idx]
    all_demo_indices = [all_demo_indices[i] for i in sort_idx]

    # Repack into padded arrays
    E = len(all_obs)
    Tmax = max(all_path_lengths)

    observations  = np.zeros((E, Tmax, K, Dtok), dtype=np.float32)
    actions       = np.zeros((E, Tmax, A),       dtype=np.float32)
    gripper_state = np.zeros((E, Tmax, G),       dtype=np.float32)
    bg_features   = np.zeros((E, Tmax, BG),      dtype=np.float32)
    rewards       = np.zeros((E, Tmax, 1),       dtype=np.float32)
    terminals     = np.zeros((E, Tmax, 1),       dtype=np.float32)
    timeouts      = np.zeros((E, Tmax, 1),       dtype=np.float32)
    goals         = np.zeros((E, Tmax, K, Dtok), dtype=np.float32)

    for e in range(E):
        L = all_path_lengths[e]
        observations[e, :L]  = all_obs[e]
        actions[e, :L]       = all_act[e]
        gripper_state[e, :L] = all_gripper[e]
        bg_features[e, :L]   = all_bg[e]
        rewards[e, :L]       = all_rewards[e]
        terminals[e, :L]     = all_terminals[e]
        timeouts[e, :L]      = all_timeouts[e]
        goals[e, :L]         = all_goals[e]

        # Pad remaining timesteps
        if L < Tmax:
            observations[e, L:]  = observations[e, L-1:L]
            goals[e, L:]         = goals[e, L-1:L]
            actions[e, L:]       = 0.0
            gripper_state[e, L:] = gripper_state[e, L-1:L]
            bg_features[e, L:]   = bg_features[e, L-1:L]
            timeouts[e, L:, 0]   = 1.0

    path_lengths = np.asarray(all_path_lengths, dtype=np.int32)
    info_goals_reached = np.ones((E,), dtype=np.float32)
    info_goal_success_frac = np.ones((E,), dtype=np.float32)

    paths_dict = {
        "meta": {
            "K": int(K),
            "Dtok": int(Dtok),
            "A": int(A),
            "G": int(G),
            "BG": int(BG),
            "Tmax": int(Tmax),
            "repr": "DLP_tokens_k24",
            "action_mode": action_mode,
            "demo_indices": np.asarray(all_demo_indices, dtype=np.int32),
            "gripper_state_format": meta0.get("gripper_state_format", "pos(3)_rot6d(6)_open(1)"),
            "bg_features_format": meta0.get("bg_features_format", f"z_bg_features({BG})"),
            "source": "merged_shards",
            "num_shards": len(shards),
        },
        "observations": observations,
        "actions": actions,
        "gripper_state": gripper_state,
        "bg_features": bg_features,
        "rewards": rewards,
        "terminals": terminals,
        "timeouts": timeouts,
        "goals": goals,
        "path_lengths": path_lengths,
        "info_goals_reached": info_goals_reached,
        "info_goal_success_frac": info_goal_success_frac,
        "init_states": np.asarray(all_init_states, dtype=np.float64),
    }

    # Save merged output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(paths_dict, f)

    print(f"\nMerged {len(shards)} shards -> {output_path}")
    print(f"E={E}, Tmax={Tmax}, K={K}, Dtok={Dtok}, A={A}, G={G}, BG={BG}")

    # Optionally delete shard files
    return shard_files


def main():
    parser = argparse.ArgumentParser(description="Merge preprocessed shards")
    parser.add_argument("--input-pattern", required=True,
                        help="Glob pattern for shard files (e.g., 'output_rank*.pkl')")
    parser.add_argument("--output", required=True,
                        help="Output merged pickle file")
    parser.add_argument("--delete-shards", action="store_true",
                        help="Delete shard files after merging")

    args = parser.parse_args()

    shard_files = merge_shards(args.input_pattern, args.output)

    if args.delete_shards:
        print("\nDeleting shard files...")
        for f in shard_files:
            os.remove(f)
            print(f"  Deleted: {f}")


if __name__ == "__main__":
    main()
