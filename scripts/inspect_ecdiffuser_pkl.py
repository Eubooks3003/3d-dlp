#!/usr/bin/env python3
"""
Inspect an EC-Diffuser replay buffer .pkl (DLP or state-based).

Usage:
  python inspect_ecdiffuser_pkl.py /home/ubuntu/pathaklab/ellina/EC-Diffuser/ecdiffuser-data/push_cubes/2C/panda_push_replay_buffer_dlp.pkl
  python inspect_ecdiffuser_pkl.py <path.pkl> --episode 0 --print-head 3
"""
import argparse
import pickle
import numpy as np

def _shape(x):
    try:
        return tuple(x.shape)
    except Exception:
        return None

def _dtype(x):
    try:
        return str(x.dtype)
    except Exception:
        return type(x).__name__

def _stats(x, max_elems=200000):
    x = np.asarray(x)
    if x.size == 0:
        return "(empty)"
    # sample if huge
    flat = x.reshape(-1)
    if flat.size > max_elems:
        idx = np.random.choice(flat.size, size=max_elems, replace=False)
        flat = flat[idx]
    finite = np.isfinite(flat)
    if not finite.any():
        return "(no finite values)"
    f = flat[finite]
    return f"min={f.min():.4g} max={f.max():.4g} mean={f.mean():.4g} std={f.std():.4g}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl", help="path to replay_buffer_dlp.pkl")
    ap.add_argument("--episode", type=int, default=0, help="which episode to inspect")
    ap.add_argument("--print-head", type=int, default=0, help="print first N timesteps for small arrays")
    ap.add_argument("--no-stats", action="store_true", help="skip min/max/mean/std")
    args = ap.parse_args()

    with open(args.pkl, "rb") as f:
        data = pickle.load(f)

    print(f"\nLoaded: {args.pkl}")
    print(f"Top-level type: {type(data)}")

    # Most EC-Diffuser buffers are dicts of arrays
    if isinstance(data, dict):
        print("\nTop-level keys:")
        for k in sorted(data.keys()):
            v = data[k]
            print(f"  - {k:12s} type={type(v).__name__} shape={_shape(v)} dtype={_dtype(v)}")
            if (not args.no_stats) and isinstance(v, (np.ndarray, list)):
                try:
                    print(f"      stats: {_stats(v)}")
                except Exception:
                    pass

        # Try to infer episode count / T
        N = None
        T = None
        for k in ("observations", "actions", "rewards", "terminals", "timeouts", "goals"):
            if k in data and hasattr(data[k], "shape") and len(data[k].shape) >= 2:
                N = data[k].shape[0]
                T = data[k].shape[1]
                break

        if N is not None:
            print(f"\nInferred: N_episodes={N}, T_max={T}")
        if "path_lengths" in data:
            pl = np.asarray(data["path_lengths"])
            print(f"path_lengths: shape={pl.shape} dtype={pl.dtype} min={pl.min()} max={pl.max()} mean={pl.mean():.2f}")

        # Episode detail
        ep = int(args.episode)
        if N is not None and (ep < 0 or ep >= N):
            print(f"\n[warn] episode {ep} out of range [0, {N-1}]")
            return

        if N is not None:
            L = None
            if "path_lengths" in data:
                L = int(np.asarray(data["path_lengths"])[ep])
            print(f"\n=== Episode {ep} ===")
            if L is not None:
                print(f"  true length L={L}")

            def show_ep(name):
                if name not in data:
                    return
                arr = np.asarray(data[name])
                if arr.ndim < 2:
                    return
                ep_arr = arr[ep]  # [T,...]
                print(f"  {name}: shape={ep_arr.shape} dtype={ep_arr.dtype}")
                if not args.no_stats:
                    print(f"    stats: {_stats(ep_arr)}")
                if args.print_head > 0:
                    n = min(args.print_head, ep_arr.shape[0])
                    # print small slices only
                    sl = ep_arr[:n]
                    # if huge last dim, show first few dims
                    if sl.ndim >= 2 and sl.shape[-1] > 20:
                        sl = sl[..., :20]
                    print(f"    head[:{n}]:\n{sl}")

            for nm in ("observations", "actions", "rewards", "terminals", "timeouts", "goals"):
                show_ep(nm)

    else:
        # Some repos store a list of paths dicts
        print("\nNon-dict buffer. Printing repr of first item(s):")
        try:
            print(repr(data[:2]))
        except Exception:
            print(repr(data))

if __name__ == "__main__":
    main()
