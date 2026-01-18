#!/usr/bin/env python3
"""
RoboSet -> joined (multi-view) RGB-D point clouds as PLY.

- Episodes are top-level groups: Trial0, Trial1, ...
- Each trial has:
    TrialX/data/rgb_<view> : (T,H,W,3)
    TrialX/data/d_<view>   : (T,H,W)        depth, typically in mm
    TrialX/config/calibration/Camera N/...
        intrinsics: fx, fy, ppx, ppy, width, height
        extrinsics-ish: camera_base_ori (3x3), camera_base_pos (3,)
          Interpretation (default): X_base = R * X_cam + t
          Use --invert-extrinsics if your dataset stores the inverse.
- Writes fused point cloud per timestep: out/TrialX/frame000000.ply
"""

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np


# ----------------------------
# Utils
# ----------------------------

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _as_np(x):
    return np.asarray(x)

def _read_scalar(ds):
    # Many entries are shape (1,) or scalar
    a = _as_np(ds[()])
    if a.shape == ():
        return float(a)
    if a.size == 1:
        return float(a.reshape(-1)[0])
    raise ValueError(f"Expected scalar, got shape {a.shape}")

def _maybe_decode_str(x) -> str:
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8", errors="ignore")
    if isinstance(x, np.ndarray) and x.dtype.kind in ("S", "O") and x.size == 1:
        v = x.reshape(-1)[0]
        if isinstance(v, (bytes, np.bytes_)):
            return v.decode("utf-8", errors="ignore")
        return str(v)
    return str(x)

def write_ply_xyzrgb_binary(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """
    Binary little-endian PLY writer.
    xyz: (N,3) float
    rgb: (N,3) uint8
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    assert xyz.ndim == 2 and xyz.shape[1] == 3
    assert rgb.ndim == 2 and rgb.shape[1] == 3
    assert xyz.shape[0] == rgb.shape[0]

    n = xyz.shape[0]
    header = "\n".join([
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header\n"
    ]).encode("ascii")

    dtype = np.dtype([
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ])
    out = np.empty(n, dtype=dtype)
    out["x"] = xyz[:, 0]
    out["y"] = xyz[:, 1]
    out["z"] = xyz[:, 2]
    out["red"] = rgb[:, 0]
    out["green"] = rgb[:, 1]
    out["blue"] = rgb[:, 2]

    with open(path, "wb") as f:
        f.write(header)
        out.tofile(f)

def list_trials(h5: h5py.File) -> List[str]:
    trials = []
    for k in h5.keys():
        if isinstance(h5[k], h5py.Group) and re.match(r"^Trial\d+$", k):
            trials.append(k)
    # numeric sort by trailing integer
    trials.sort(key=lambda s: int(s.replace("Trial", "")))
    return trials

def detect_views(h5: h5py.File, trial: str, verbose: bool = False) -> Dict[str, Tuple[str, str]]:
    """
    Returns dict view -> (rgb_key, depth_key) within this trial.
    Pairs rgb_<view> with d_<view> under TrialX/data/.
    """
    base = f"{trial}/data"
    if base not in h5:
        return {}

    g = h5[base]
    rgb_re = re.compile(r"^rgb_(.+)$")
    dep_re = re.compile(r"^(d|depth)_(.+)$")

    rgb = {}
    dep = {}

    for name in g.keys():
        m = rgb_re.match(name)
        if m:
            rgb[m.group(1)] = f"{base}/{name}"
            continue
        m = dep_re.match(name)
        if m:
            dep[m.group(2)] = f"{base}/{name}"
            continue

    views = {}
    for v in sorted(set(rgb.keys()) & set(dep.keys())):
        views[v] = (rgb[v], dep[v])

    if verbose:
        print(f"  [debug] found view pairs: {list(views.keys())}")

    return views

def list_calib_cameras(h5: h5py.File, trial: str) -> List[str]:
    base = f"{trial}/config/calibration"
    cams = []
    if base not in h5:
        return cams
    g = h5[base]
    for k in g.keys():
        if re.match(r"^Camera\s+\d+$", k):
            cams.append(k)
    cams.sort(key=lambda s: int(s.split()[-1]))
    return cams


# ----------------------------
# Calibration reading
# ----------------------------

@dataclass
class Calib:
    K: np.ndarray         # (3,3)
    R: np.ndarray         # (3,3)
    t: np.ndarray         # (3,)
    width: int
    height: int

def read_calib(h5: h5py.File,
               trial: str,
               cam_name: str,
               fallback_fov_deg: Optional[float],
               img_hw: Tuple[int, int],
               invert_extrinsics: bool) -> Calib:
    """
    cam_name like "Camera 1"
    img_hw: (H,W) from actual rgb/depth
    """
    H, W = img_hw
    base = f"{trial}/config/calibration/{cam_name}"

    # --- intrinsics ---
    K = None
    width = W
    height = H

    intr_base = f"{base}/intrinsics"
    if intr_base in h5:
        fx_k = f"{intr_base}/fx"
        fy_k = f"{intr_base}/fy"
        ppx_k = f"{intr_base}/ppx"
        ppy_k = f"{intr_base}/ppy"
        w_k = f"{intr_base}/width"
        h_k = f"{intr_base}/height"

        if fx_k in h5 and fy_k in h5 and ppx_k in h5 and ppy_k in h5:
            fx = _read_scalar(h5[fx_k])
            fy = _read_scalar(h5[fy_k])
            cx = _read_scalar(h5[ppx_k])
            cy = _read_scalar(h5[ppy_k])
            if w_k in h5:
                width = int(_read_scalar(h5[w_k]))
            if h_k in h5:
                height = int(_read_scalar(h5[h_k]))
            K = np.array([[fx, 0.0, cx],
                          [0.0, fy, cy],
                          [0.0, 0.0, 1.0]], dtype=np.float32)

    if K is None:
        if fallback_fov_deg is None:
            raise RuntimeError("Missing intrinsics (fx/fy/ppx/ppy). Provide --fallback-fov-deg.")
        # Simple pinhole from HFOV
        fov = np.deg2rad(float(fallback_fov_deg))
        fx = (W * 0.5) / np.tan(fov * 0.5)
        fy = fx
        cx = (W - 1) * 0.5
        cy = (H - 1) * 0.5
        K = np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float32)

    # --- extrinsics ---
    Rk = f"{base}/camera_base_ori"
    tk = f"{base}/camera_base_pos"
    if Rk not in h5 or tk not in h5:
        raise RuntimeError(f"Missing extrinsics fields under {base} (need camera_base_ori, camera_base_pos).")

    R = np.asarray(h5[Rk][()], dtype=np.float32).reshape(3, 3)
    t = np.asarray(h5[tk][()], dtype=np.float32).reshape(3)

    # Interpretation default: X_base = R * X_cam + t  (base_from_cam)
    if invert_extrinsics:
        # If file stores cam_from_base, invert to base_from_cam
        # (R,t) inverse: R^T, -R^T t
        R = R.T
        t = -R @ t

    return Calib(K=K, R=R, t=t, width=width, height=height)


# ----------------------------
# Geometry
# ----------------------------

def depth_to_points(depth_m: np.ndarray,
                    K: np.ndarray,
                    pix_stride: int = 1,
                    y_up: bool = False,
                    max_depth_m: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    depth_m: (H,W) float32 meters
    Returns:
      xyz_cam: (N,3)
      us: (N,) int
      vs: (N,) int
    """
    assert depth_m.ndim == 2
    H, W = depth_m.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    # subsample pixels
    vs = np.arange(0, H, pix_stride, dtype=np.int32)
    us = np.arange(0, W, pix_stride, dtype=np.int32)
    uu, vv = np.meshgrid(us, vs)  # (Hs,Ws)
    z = depth_m[vv, uu]           # (Hs,Ws)

    valid = z > 0
    if max_depth_m is not None:
        valid &= (z <= float(max_depth_m))

    uu = uu[valid].astype(np.float32)
    vv = vv[valid].astype(np.float32)
    z = z[valid].astype(np.float32)

    x = (uu - cx) / fx * z
    y = (vv - cy) / fy * z
    if y_up:
        y = -y  # OpenCV y-down -> y-up

    xyz = np.stack([x, y, z], axis=1)
    return xyz, uu.astype(np.int32), vv.astype(np.int32)

def transform_points(xyz: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    xyz_out = R * xyz_in + t
    """
    return (xyz @ R.T) + t.reshape(1, 3)


# ----------------------------
# Main fusion
# ----------------------------

def parse_cam_map(items: Optional[List[str]]) -> Dict[str, int]:
    """
    items like ["left=1","right=2"]
    """
    m: Dict[str, int] = {}
    if not items:
        return m
    for s in items:
        if "=" not in s:
            raise ValueError(f"--cam-map expects view=N, got {s}")
        k, v = s.split("=", 1)
        k = k.strip()
        v = int(v.strip())
        m[k] = v
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, help="Path to RoboSet .h5 file")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--episodes", nargs="*", default=None,
                    help="Episodes/trials to process (e.g., Trial0 Trial1). Default: all Trial*")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Max frames per episode (starting at 0)")
    ap.add_argument("--stride", type=int, default=1, help="Frame stride (e.g., 5 = every 5th frame)")
    ap.add_argument("--views", nargs="*", default=["left", "right", "top"],
                    help="Views to fuse (default: left right top). Omit wrist unless you know how to place it.")
    ap.add_argument("--cam-map", nargs="*", default=[],
                    help="Map view name -> calibration camera index, e.g. --cam-map left=1 right=2 top=3")
    ap.add_argument("--fallback-fov-deg", type=float, default=None,
                    help="If intrinsics missing, derive K from horizontal FOV")
    ap.add_argument("--depth-scale", type=float, default=0.001,
                    help="Scale applied to raw depth values to convert to meters (mm->m is 0.001)")
    ap.add_argument("--pix-stride", type=int, default=2,
                    help="Pixel subsampling stride for backprojection (2 is usually plenty)")
    ap.add_argument("--max-depth-m", type=float, default=3.5,
                    help="Drop points beyond this depth (meters). Set to 0 to disable.")
    ap.add_argument("--invert-extrinsics", action="store_true",
                    help="Invert camera_base_(ori,pos) interpretation if fusion looks torn apart")
    ap.add_argument("--y-up", action="store_true",
                    help="Flip camera y axis (OpenCV y-down -> y-up). Try if scene is mirrored/weird.")
    ap.add_argument("--max-points", type=int, default=300_000,
                    help="Randomly downsample fused cloud to at most this many points per frame.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    max_depth_m = None if (args.max_depth_m is None or args.max_depth_m <= 0) else float(args.max_depth_m)
    cam_map = parse_cam_map(args.cam_map)

    print(f"[info] using h5: {args.h5}")
    _ensure_dir(args.out)

    with h5py.File(args.h5, "r") as h5:
        trials = args.episodes if args.episodes is not None else list_trials(h5)
        if args.max_episodes is not None:
            trials = trials[: int(args.max_episodes)]

        print(f"[info] episodes to process: {len(trials)}")

        for i, trial in enumerate(trials):
            if trial not in h5:
                print(f"[warn] missing episode {trial}, skipping")
                continue

            views_all = detect_views(h5, trial, verbose=args.verbose)
            if not views_all:
                print(f"[warn] no rgb/depth views found under {trial}, skipping")
                continue

            calib_cams = list_calib_cameras(h5, trial)
            if args.verbose:
                print(f"  [debug] calibration cameras: {calib_cams}")

            # Pick only requested views that exist
            requested = [v for v in args.views if v in views_all]
            missing = [v for v in args.views if v not in views_all]
            if missing and args.verbose:
                print(f"  [warn] requested views missing in {trial}: {missing}")

            if not requested:
                print(f"[warn] none of requested views exist in {trial}, skipping")
                continue

            # For each view, assign a calibration camera
            view_to_camname: Dict[str, Optional[str]] = {}
            for v in requested:
                if v in cam_map:
                    idx = cam_map[v]
                    camname = f"Camera {idx}"
                    if camname not in calib_cams:
                        print(f"  [warn] view '{v}' mapped to {camname} but not present in calibration.")
                        view_to_camname[v] = None
                    else:
                        view_to_camname[v] = camname
                else:
                    # no mapping provided
                    view_to_camname[v] = None

            # If any unmapped views remain, warn loudly (we will skip them)
            for v, camname in view_to_camname.items():
                if camname is None:
                    print(f"  [warn] view '{v}' has no assigned calibration camera (set --cam-map {v}=N)")

            # Determine T from one of the datasets
            rgb0_key, _ = views_all[requested[0]]
            rgb0 = h5[rgb0_key]
            T = int(rgb0.shape[0])
            H = int(rgb0.shape[1])
            W = int(rgb0.shape[2])
            if args.max_frames is not None:
                T = min(T, int(args.max_frames))

            out_ep = os.path.join(args.out, trial)
            _ensure_dir(out_ep)

            print(f"[info] ({i+1}/{len(trials)}) episode: {trial}")
            print(f"  [info] detected views={len(requested)}, T={T}, writing frames to: {out_ep}")

            # Pre-read calibrations per view
            view_calib: Dict[str, Calib] = {}
            for v in requested:
                camname = view_to_camname[v]
                if camname is None:
                    continue
                try:
                    view_calib[v] = read_calib(
                        h5=h5,
                        trial=trial,
                        cam_name=camname,
                        fallback_fov_deg=args.fallback_fov_deg,
                        img_hw=(H, W),
                        invert_extrinsics=args.invert_extrinsics,
                    )
                    if args.verbose:
                        c = view_calib[v]
                        print(f"  cam={trial}_{v} ({camname})")
                        print(f"    K=\n{c.K}")
                        print(f"    R=\n{c.R}")
                        print(f"    t={c.t}")
                except Exception as e:
                    print(f"  [warn] failed to read calib for view '{v}' via {camname}: {e}")
                    continue

            # Process frames
            for t in range(0, T, int(args.stride)):
                xyz_all = []
                rgb_all = []

                for v in requested:
                    if v not in view_calib:
                        continue
                    rgb_key, dep_key = views_all[v]

                    rgb = np.asarray(h5[rgb_key][t])  # (H,W,3)
                    dep = np.asarray(h5[dep_key][t])  # (H,W)

                    if rgb.ndim != 3 or rgb.shape[2] != 3:
                        # try to coerce if stored differently
                        rgb = rgb.reshape((H, W, 3))

                    # Depth -> meters
                    dep_m = dep.astype(np.float32) * float(args.depth_scale)

                    # Backproject in camera frame
                    calib = view_calib[v]
                    xyz_cam, us, vs = depth_to_points(
                        depth_m=dep_m,
                        K=calib.K,
                        pix_stride=int(args.pix_stride),
                        y_up=bool(args.y_up),
                        max_depth_m=max_depth_m,
                    )
                    if xyz_cam.shape[0] == 0:
                        continue

                    # Transform into base/world frame
                    xyz = transform_points(xyz_cam, calib.R, calib.t)

                    # Colors
                    cols = rgb[vs, us].astype(np.uint8)

                    xyz_all.append(xyz)
                    rgb_all.append(cols)

                if not xyz_all:
                    continue

                xyz_f = np.concatenate(xyz_all, axis=0)
                rgb_f = np.concatenate(rgb_all, axis=0)

                # Downsample if huge
                if args.max_points is not None and xyz_f.shape[0] > int(args.max_points):
                    n = xyz_f.shape[0]
                    k = int(args.max_points)
                    idx = np.random.choice(n, size=k, replace=False)
                    xyz_f = xyz_f[idx]
                    rgb_f = rgb_f[idx]

                ply_path = os.path.join(out_ep, f"frame{t:06d}.ply")
                write_ply_xyzrgb_binary(ply_path, xyz_f, rgb_f)

            print(f"  [done] {trial}")


if __name__ == "__main__":
    main()
