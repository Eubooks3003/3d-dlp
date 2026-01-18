#!/usr/bin/env python3
import os
import argparse
import numpy as np
import h5py
from typing import Dict, Tuple, List, Optional

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def write_ply_xyzrgb(path: str, xyz: np.ndarray, rgb: np.ndarray):
    """
    xyz: (N,3) float
    rgb: (N,3) uint8
    """
    assert xyz.ndim == 2 and xyz.shape[1] == 3
    assert rgb.ndim == 2 and rgb.shape[1] == 3
    assert xyz.shape[0] == rgb.shape[0]
    n = xyz.shape[0]
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = xyz[i]
            r, g, b = rgb[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

def parse_cam_map(items: List[str]) -> Dict[str, int]:
    out = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"--cam-map expects view=N, got '{it}'")
        k, v = it.split("=", 1)
        out[k.strip()] = int(v.strip())
    return out

def list_views_for_trial(h5: h5py.File, episode: str) -> List[str]:
    # expects datasets: {episode}/data/rgb_{view} and {episode}/data/d_{view}
    base = f"{episode}/data"
    if base not in h5:
        return []
    keys = list(h5[base].keys())
    rgb_views = set()
    d_views = set()
    for k in keys:
        if k.startswith("rgb_"):
            rgb_views.add(k[len("rgb_"):])
        if k.startswith("d_"):
            d_views.add(k[len("d_"):])
    return sorted(list(rgb_views & d_views))
def rescale_K_to_image(K: np.ndarray, W: int, H: int) -> np.ndarray:
    """
    Rescale intrinsics to match the actual (H,W) of the depth/rgb arrays.

    Works even if the stored K came from a different resolution (e.g., 640x480).
    This version forces the principal point to land at the image center.
    """
    K2 = K.copy().astype(np.float32)
    cx, cy = float(K2[0, 2]), float(K2[1, 2])

    # target principal point at image center
    cx_t = (W - 1) * 0.5
    cy_t = (H - 1) * 0.5

    sx = cx_t / cx if cx > 1e-6 else 1.0
    sy = cy_t / cy if cy > 1e-6 else 1.0

    K2[0, 0] *= sx  # fx
    K2[0, 2] *= sx  # cx
    K2[1, 1] *= sy  # fy
    K2[1, 2] *= sy  # cy
    return K2

def load_calibration_camera(h5: h5py.File, episode: str, cam_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      K (3,3)
      T (4,4) where camera points p_cam -> p_base = R @ p_cam + t   (unless you invert later)
    """
    base = f"{episode}/config/calibration/{cam_name}"

    fx = float(h5[f"{base}/intrinsics/fx"][()][0] if h5[f"{base}/intrinsics/fx"].shape else h5[f"{base}/intrinsics/fx"][()])
    fy = float(h5[f"{base}/intrinsics/fy"][()][0] if h5[f"{base}/intrinsics/fy"].shape else h5[f"{base}/intrinsics/fy"][()])
    ppx = float(h5[f"{base}/intrinsics/ppx"][()][0] if h5[f"{base}/intrinsics/ppx"].shape else h5[f"{base}/intrinsics/ppx"][()])
    ppy = float(h5[f"{base}/intrinsics/ppy"][()][0] if h5[f"{base}/intrinsics/ppy"].shape else h5[f"{base}/intrinsics/ppy"][()])

    K = np.array([[fx, 0.0, ppx],
                  [0.0, fy, ppy],
                  [0.0, 0.0, 1.0]], dtype=np.float32)

    R = np.array(h5[f"{base}/camera_base_ori"][()], dtype=np.float32)  # (3,3)
    t = np.array(h5[f"{base}/camera_base_pos"][()], dtype=np.float32).reshape(3)  # (3,)

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return K, T

def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=T.dtype)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def depth_to_points_xyzrgb(
    rgb: np.ndarray,
    depth_raw: np.ndarray,
    K: np.ndarray,
    *,
    pix_stride: int = 2,
    depth_scale: float = 1000.0,   # raw units per meter (1000 -> mm)
    depth_min: float = 0.10,
    depth_max: float = 3.00,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    rgb: (H,W,3) uint8
    depth_raw: (H,W) numeric (typically uint16)
    returns xyz_cam (N,3), rgb (N,3)
    """
    H, W = depth_raw.shape[:2]
    # convert to meters
    z = depth_raw.astype(np.float32) / float(depth_scale)

    # sample pixels
    us = np.arange(0, W, pix_stride, dtype=np.int32)
    vs = np.arange(0, H, pix_stride, dtype=np.int32)
    uu, vv = np.meshgrid(us, vs)
    uu = uu.reshape(-1)
    vv = vv.reshape(-1)
    zz = z[vv, uu]

    valid = (zz > depth_min) & (zz < depth_max) & np.isfinite(zz)
    uu = uu[valid].astype(np.float32)
    vv = vv[valid].astype(np.float32)
    zz = zz[valid].astype(np.float32)

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    x = (uu - cx) / fx * zz
    y = (vv - cy) / fy * zz
    xyz = np.stack([x, y, zz], axis=1).astype(np.float32)

    cols = rgb[vv.astype(np.int32), uu.astype(np.int32), :].astype(np.uint8)
    return xyz, cols

def apply_T(T: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    # xyz: (N,3)
    R = T[:3, :3]
    t = T[:3, 3]
    return (xyz @ R.T) + t[None, :]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--episode", required=True, help="e.g., Trial0")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", required=True)

    ap.add_argument("--views", nargs="+", default=["left", "right", "top"])
    ap.add_argument("--cam-map", nargs="*", default=[],
                    help="Mapping view->calib camera index, e.g. left=1 right=2 top=3")
    ap.add_argument("--invert-extrinsics", action="store_true",
                    help="Use inverse of stored T (common if convention is base->camera)")
    ap.add_argument("--save-camera-frame", action="store_true",
                    help="Also save per-view clouds in raw camera frame for debugging")

    ap.add_argument("--pix-stride", type=int, default=2)
    ap.add_argument("--depth-scale", type=float, default=1000.0, help="raw units per meter (1000=mm)")
    ap.add_argument("--depth-min", type=float, default=0.10)
    ap.add_argument("--depth-max", type=float, default=3.00)

    args = ap.parse_args()
    cam_map = parse_cam_map(args.cam_map)

    ensure_dir(args.out)
    ep_out = os.path.join(args.out, args.episode)
    ensure_dir(ep_out)

    with h5py.File(args.h5, "r") as h5:
        if args.episode not in h5:
            raise RuntimeError(f"Episode '{args.episode}' not found. Top keys: {list(h5.keys())[:50]}")

        available_views = set(list_views_for_trial(h5, args.episode))
        views = [v for v in args.views if v in available_views]
        if not views:
            raise RuntimeError(f"No requested views found. requested={args.views}, available={sorted(list(available_views))}")

        calib_root = f"{args.episode}/config/calibration"
        if calib_root not in h5:
            raise RuntimeError(f"Missing calibration at {calib_root}")

        calib_cams = sorted(list(h5[calib_root].keys()))
        # keep only "Camera N" groups
        calib_cams = [c for c in calib_cams if c.lower().startswith("camera")]
        if not calib_cams:
            raise RuntimeError(f"No calibration cameras under {calib_root}")

        per_view_xyz_base = {}
        per_view_rgb = {}
        per_view_xyz_cam = {}

        for v in views:
            if v not in cam_map:
                print(f"[warn] view '{v}' has no assigned calibration camera (set --cam-map {v}=N). Skipping.")
                continue
            idx = cam_map[v]
            cam_name = f"Camera {idx}"
            if cam_name not in h5[calib_root]:
                print(f"[warn] view '{v}' mapped to '{cam_name}', but not found under calibration. Skipping.")
                continue

            rgb_key = f"{args.episode}/data/rgb_{v}"
            dep_key = f"{args.episode}/data/d_{v}"
            rgb_ds = h5[rgb_key]
            dep_ds = h5[dep_key]

            if args.frame < 0 or args.frame >= rgb_ds.shape[0]:
                raise RuntimeError(f"--frame {args.frame} out of range for {rgb_key} with T={rgb_ds.shape[0]}")

            rgb = np.array(rgb_ds[args.frame], dtype=np.uint8)  # (H,W,3)
            dep = np.array(dep_ds[args.frame])                  # (H,W)

            K, T = load_calibration_camera(h5, args.episode, cam_name)
            H, W = dep.shape[:2]
            K = rescale_K_to_image(K, W, H)
            if args.invert_extrinsics:
                T = inv_T(T)

            xyz_cam, cols = depth_to_points_xyzrgb(
                rgb, dep, K,
                pix_stride=args.pix_stride,
                depth_scale=args.depth_scale,
                depth_min=args.depth_min,
                depth_max=args.depth_max,
            )
            xyz_base = apply_T(T, xyz_cam)

            per_view_xyz_base[v] = xyz_base
            per_view_rgb[v] = cols
            per_view_xyz_cam[v] = xyz_cam

            # write per view
            stem = f"frame_{args.frame:06d}_{v}"
            out_path = os.path.join(ep_out, stem + ".ply")
            write_ply_xyzrgb(out_path, xyz_base, cols)

            if args.save_camera_frame:
                out_path_cam = os.path.join(ep_out, stem + "_cam.ply")
                write_ply_xyzrgb(out_path_cam, xyz_cam, cols)

            print(f"[ok] wrote {v}: {out_path}  (N={xyz_base.shape[0]})")

        if not per_view_xyz_base:
            raise RuntimeError("No per-view clouds produced (likely missing --cam-map for requested views).")

        # joined
        xyz_all = np.concatenate([per_view_xyz_base[v] for v in per_view_xyz_base.keys()], axis=0)
        rgb_all = np.concatenate([per_view_rgb[v] for v in per_view_rgb.keys()], axis=0)
        joined_path = os.path.join(ep_out, f"frame_{args.frame:06d}_joined.ply")
        write_ply_xyzrgb(joined_path, xyz_all, rgb_all)
        print(f"[ok] wrote joined: {joined_path}  (N={xyz_all.shape[0]})")

if __name__ == "__main__":
    main()
