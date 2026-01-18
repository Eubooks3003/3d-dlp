#!/usr/bin/env python3
"""
Dump a few RGB + depth frames from a robomimic-style MimicGen HDF5 for sanity checks.

Writes:
  out_dir/demo_X_t000.png                 (RGB)
  out_dir/demo_X_t000_depth.png           (depth, colorized)
  out_dir/demo_X_t000_depth.npy           (raw depth array)

Usage:
  python dump_h5_rgbd.py --h5 /path/to/out.hdf5 --out sanity_dump --n-demos 2 --n-frames 5
  python dump_h5_rgbd.py --h5 ... --cams agentview sideview --frames 0 10 20
"""

import os, argparse
import numpy as np
import h5py

from PIL import Image

def _ensure_uint8_rgb(x):
    """
    Accepts [H,W,3] in uint8 or float.
    If float in [0,1] or [0,255], converts to uint8.
    """
    x = np.asarray(x)
    if x.dtype == np.uint8:
        return x
    x = x.astype(np.float32)
    # heuristic: if max <= 1.5 treat as [0,1]
    if np.nanmax(x) <= 1.5:
        x = x * 255.0
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x

def _depth_to_viz(depth, vmin=None, vmax=None, percentile_clip=(1, 99), eps=1e-8):
    """
    Convert depth map to a colorized uint8 RGB image (simple "magma-like" via grayscale->RGB).
    No matplotlib required.
    """
    d = np.asarray(depth).astype(np.float32)
    # squeeze possible channel dims: [H,W,1] -> [H,W]
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[..., 0]
    # handle NaNs/Infs
    finite = np.isfinite(d)
    if not np.any(finite):
        return np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8), (0.0, 1.0)

    df = d[finite]
    if vmin is None or vmax is None:
        lo, hi = np.percentile(df, list(percentile_clip))
        if vmin is None: vmin = float(lo)
        if vmax is None: vmax = float(hi)
    if vmax <= vmin + eps:
        vmax = vmin + 1.0

    dn = (np.clip(d, vmin, vmax) - vmin) / (vmax - vmin + eps)
    dn[~finite] = 0.0

    # simple colormap: stack grayscale into RGB, with slight contrast
    g = (dn * 255.0).astype(np.uint8)
    rgb = np.stack([g, g, g], axis=-1)
    return rgb, (vmin, vmax)

def _list_cameras_in_obs(obs_group):
    cams = set()
    for k in obs_group.keys():
        if k.endswith("_image"):
            cams.add(k.replace("_image", ""))
    return sorted(list(cams))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cams", nargs="*", default=None, help="Camera names to dump (default: auto-detect from first demo)")
    ap.add_argument("--n-demos", type=int, default=2)
    ap.add_argument("--n-frames", type=int, default=5, help="How many frames per demo (used if --frames not given)")
    ap.add_argument("--frames", type=int, nargs="*", default=None, help="Specific frame indices to dump (overrides --n-frames)")
    ap.add_argument("--seed", type=int, default=0, help="Seed for picking random frames if --frames not set")
    ap.add_argument("--save-npy", action="store_true", help="Also save raw depth as .npy")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    with h5py.File(args.h5, "r") as f:
        demos = sorted(list(f["data"].keys()), key=lambda x: int(x.split("_")[-1]) if "_" in x else int(x[5:]))
        demos = demos[:args.n_demos]
        if len(demos) == 0:
            raise RuntimeError("No demos found under /data")

        # auto-detect cameras from first demo if not provided
        if args.cams is None:
            obs0 = f[f"data/{demos[0]}/obs"]
            cams = _list_cameras_in_obs(obs0)
            if not cams:
                raise RuntimeError("No '*_image' keys found in obs. Check your dataset.")
        else:
            cams = list(args.cams)

        print(f"[info] demos: {demos}")
        print(f"[info] cams: {cams}")

        # choose frames
        for demo in demos:
            obs = f[f"data/{demo}/obs"]

            # find T from first available camera image
            T = None
            for cam in cams:
                k = f"{cam}_image"
                if k in obs:
                    T = obs[k].shape[0]
                    break
            if T is None:
                print(f"[warn] demo {demo}: none of cams have *_image keys, skipping")
                continue

            if args.frames is not None and len(args.frames) > 0:
                frame_ids = [t for t in args.frames if 0 <= t < T]
            else:
                # pick a few spread-out frames if possible; otherwise random
                if T <= args.n_frames:
                    frame_ids = list(range(T))
                else:
                    # spread + a couple random
                    lin = np.linspace(0, T - 1, num=min(args.n_frames, T), dtype=int).tolist()
                    frame_ids = lin

            print(f"[demo {demo}] T={T}, dumping frames: {frame_ids}")

            for t in frame_ids:
                for cam in cams:
                    rgb_key = f"{cam}_image"
                    dep_key = f"{cam}_depth"
                    if rgb_key not in obs:
                        print(f"  [skip] {demo} {cam}: missing {rgb_key}")
                        continue

                    rgb = obs[rgb_key][t]  # typically [H,W,3]
                    rgb_u8 = _ensure_uint8_rgb(rgb)

                    rgb_path = os.path.join(args.out, f"{demo}_{cam}_t{t:04d}.png")
                    Image.fromarray(rgb_u8).save(rgb_path)

                    if dep_key in obs:
                        depth = obs[dep_key][t]
                        depth_viz, (vmin, vmax) = _depth_to_viz(depth)
                        dep_path = os.path.join(args.out, f"{demo}_{cam}_t{t:04d}_depth.png")
                        Image.fromarray(depth_viz).save(dep_path)

                        if args.save_npy:
                            npy_path = os.path.join(args.out, f"{demo}_{cam}_t{t:04d}_depth.npy")
                            np.save(npy_path, np.asarray(depth))

                        # print one line stats
                        d = np.asarray(depth).astype(np.float32)
                        finite = np.isfinite(d)
                        if np.any(finite):
                            df = d[finite]
                            print(f"    {demo} {cam} t={t:04d} depth: min={df.min():.4g} max={df.max():.4g} mean={df.mean():.4g} viz_clip=({vmin:.4g},{vmax:.4g})")
                    else:
                        print(f"    {demo} {cam} t={t:04d}: no depth key ({dep_key})")

    print(f"\nWrote sanity images to: {args.out}")

if __name__ == "__main__":
    main()
