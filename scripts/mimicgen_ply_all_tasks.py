#!/usr/bin/env python3
"""
Batch rebuild MimicGen point clouds from stored depth buffers in *_rgbd_pcd.hdf5.

This version FIXES the "billboard / no depth" issue by correctly converting MuJoCo/OpenGL
depth buffers (values near ~1.0) into metric depth using near/far + the proper inverse
projection:

    z = (2*n*f) / (f + n - (2*d - 1)*(f - n))

where MuJoCo's metric near/far are:
    n = sim.model.vis.map.znear * sim.model.stat.extent
    f = sim.model.vis.map.zfar  * sim.model.stat.extent

We build the env from rgbd_h5["data"].attrs["env_args"] (same idea as your extractor),
then use sim.model + sim.data to get camera poses and fovy.

Output:
  <task>/core/mimicgen_from_depth_pcd/<demo_x>/frame000000_fused_envcalib.ply
"""

import os
import glob
import json
import argparse
import h5py
import numpy as np
import open3d as o3d

# registers mimicgen envs (if installed)
import mimicgen.envs.robosuite  # noqa: F401
try:
    import robosuite_task_zoo  # noqa: F401
except ImportError:
    pass

import robosuite


DEFAULT_CROP_BOUNDS = {"xmin": -2.0, "xmax": 2.0, "ymin": -2.0, "ymax": 2.0, "zmin": -0.2, "zmax": 2.5}


def squeeze_hw(d):
    d = np.asarray(d)
    if d.ndim == 3:
        d = np.squeeze(d)
    if d.ndim != 2:
        d = d.reshape(d.shape[-2], d.shape[-1])
    return d


def _unwrap_env_to_sim(env):
    cur = env
    for attr in ("env", "_env"):
        if hasattr(cur, attr):
            cur = getattr(cur, attr)
    if hasattr(cur, "sim"):
        return cur.sim
    if hasattr(cur, "unwrapped") and hasattr(cur.unwrapped, "sim"):
        return cur.unwrapped.sim
    raise RuntimeError("Could not locate mujoco sim on env (need env.sim).")


def available_cameras_from_env(env):
    sim = _unwrap_env_to_sim(env)
    try:
        return list(sim.model.camera_names)
    except Exception:
        return []


def choose_cameras(env_name: str, available, requested):
    """
    Your generator policy:
      - If PickPlace*: prefer agentview + frontview (fallbacks)
      - Else: prefer agentview + sideview (fallback frontview)
      - requested cams are honored if they exist
    """
    avail = list(available)
    aset = set(avail)

    req_valid = [c for c in (requested or []) if c in aset]

    def take_unique(dst, c):
        if c in aset and c not in dst:
            dst.append(c)

    cams = []
    # honor requested first
    for c in req_valid:
        take_unique(cams, c)
        if len(cams) >= 2:
            return cams[:2]

    if env_name.startswith("PickPlace"):
        take_unique(cams, "agentview")
        take_unique(cams, "frontview")
        take_unique(cams, "sideview")   # just in case
    else:
        take_unique(cams, "agentview")
        take_unique(cams, "sideview")
        if len(cams) < 2:
            take_unique(cams, "frontview")

    # fill remaining
    for c in avail:
        if len(cams) >= 2:
            break
        take_unique(cams, c)

    return cams[:2]


def compute_K_from_fovy(fovy_deg, width, height):
    fovy = np.deg2rad(float(fovy_deg))
    fy = (height / 2.0) / np.tan(fovy / 2.0)
    fx = fy
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float32)


def get_cam_pose_and_K(sim, cam_name: str, width: int, height: int):
    model = sim.model
    try:
        cam_id = model.camera_name2id(cam_name)
    except Exception as e:
        raise KeyError(f"Camera '{cam_name}' not found. Available: {list(model.camera_names)}") from e

    sim.forward()

    cam_pos = np.array(sim.data.cam_xpos[cam_id], dtype=np.float32)              # (3,)
    cam_xmat = np.array(sim.data.cam_xmat[cam_id], dtype=np.float32).reshape(3, 3)  # (3,3)

    # Tc2w from mujoco: cam_xmat/cam_xpos are camera frame in world coordinates
    Tc2w = np.eye(4, dtype=np.float32)
    Tc2w[:3, :3] = cam_xmat
    Tc2w[:3, 3] = cam_pos

    fovy = float(model.cam_fovy[cam_id])
    K = compute_K_from_fovy(fovy, width=width, height=height)

    return K, Tc2w


def mujoco_depth_to_meters(depth_buf_01: np.ndarray, sim) -> np.ndarray:
    """
    Convert MuJoCo/OpenGL depth buffer in [0,1] to metric depth along camera Z.

    Uses:
      n = znear * extent
      f = zfar  * extent
      z = (2*n*f) / (f + n - (2*d - 1)*(f - n))
    """
    d = squeeze_hw(depth_buf_01).astype(np.float32)

    # If stored as integer buffers, normalize
    if np.issubdtype(d.dtype, np.integer):
        denom = 65535.0 if d.dtype == np.uint16 else 255.0
        d = d / denom

    # If already meters (not your case), leave it
    dmax = float(np.nanmax(d)) if np.isfinite(d).any() else 0.0
    if dmax > 1.01:
        return d

    extent = float(sim.model.stat.extent)
    n = float(sim.model.vis.map.znear) * extent
    f = float(sim.model.vis.map.zfar) * extent
    if not (n > 0 and f > n):
        raise RuntimeError(f"Bad near/far from mujoco: n={n} f={f} (extent={extent})")

    d = np.clip(d, 0.0, 1.0)
    z_ndc = 2.0 * d - 1.0
    z = (2.0 * n * f) / (f + n - z_ndc * (f - n))
    return z.astype(np.float32)


def backproject(depth_m, K, pixel_stride=1):
    d = squeeze_hw(depth_m).astype(np.float32)
    H, W = d.shape

    vv = np.arange(0, H, pixel_stride, dtype=np.int32)
    uu = np.arange(0, W, pixel_stride, dtype=np.int32)
    U, V = np.meshgrid(uu, vv)

    Z = d[V, U]
    valid = np.isfinite(Z) & (Z > 0)
    if not np.any(valid):
        return np.zeros((0, 3), np.float32), np.zeros((0, 2), np.int32)

    Uv = U[valid].astype(np.float32)
    Vv = V[valid].astype(np.float32)
    Zv = Z[valid].astype(np.float32)

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    X = (Uv - cx) / fx * Zv
    Y = (Vv - cy) / fy * Zv
    pts_cam = np.stack([X, Y, Zv], axis=-1).astype(np.float32)

    idxs = np.stack([V[valid], U[valid]], axis=-1).astype(np.int32)  # (v,u)
    return pts_cam, idxs


def apply_T(T, pts):
    ones = np.ones((pts.shape[0], 1), np.float32)
    pts_h = np.concatenate([pts, ones], axis=1)
    out = (T.astype(np.float32) @ pts_h.T).T
    return out[:, :3]


def maybe_crop(xyz, rgb, crop_bounds):
    if crop_bounds is None:
        return xyz, rgb
    b = crop_bounds
    m = (
        (xyz[:, 0] >= b["xmin"]) & (xyz[:, 0] <= b["xmax"]) &
        (xyz[:, 1] >= b["ymin"]) & (xyz[:, 1] <= b["ymax"]) &
        (xyz[:, 2] >= b["zmin"]) & (xyz[:, 2] <= b["zmax"])
    )
    if not np.any(m):
        return xyz, rgb
    return xyz[m], rgb[m]


def build_env_from_env_args(env_args_json: str):
    meta = json.loads(env_args_json)
    env_name = meta["env_name"]
    kwargs = dict(meta.get("env_kwargs", {}))

    # force headless, do NOT initialize camera obs/rendering
    kwargs["has_renderer"] = False
    kwargs["has_offscreen_renderer"] = False
    kwargs["use_camera_obs"] = False
    kwargs["camera_names"] = []
    kwargs["camera_depths"] = []

    return robosuite.make(env_name, **kwargs)


def fuse_one_frame(rgbd_h5, ep, tidx, cams, sim, cam_calib, pixel_stride):
    obs = rgbd_h5[f"data/{ep}/obs"]
    all_xyz, all_rgb = [], []

    for cam in cams:
        depth_key = f"{cam}_depth"
        img_key = f"{cam}_image"
        if depth_key not in obs or img_key not in obs:
            print(f"    [WARN] missing {depth_key} or {img_key}, skip cam={cam}")
            continue

        depth_raw = np.asarray(obs[depth_key][tidx])
        img = np.asarray(obs[img_key][tidx])

        K, Tc2w = cam_calib[cam]
        depth_m = mujoco_depth_to_meters(depth_raw, sim)

        pts_cam, idxs = backproject(depth_m, K, pixel_stride=pixel_stride)
        if pts_cam.shape[0] == 0:
            continue

        v = idxs[:, 0]
        u = idxs[:, 1]
        rgb = img[v, u, :].astype(np.float32)
        if rgb.max() > 1.5:
            rgb /= 255.0

        pts_world = apply_T(Tc2w, pts_cam)
        all_xyz.append(pts_world)
        all_rgb.append(rgb)

    if not all_xyz:
        return None, None
    return np.concatenate(all_xyz, axis=0), np.concatenate(all_rgb, axis=0)


def process_task(rgbd_h5_path, out_dir, requested_cams, camera_w, camera_h,
                 pixel_stride, max_points, seed, max_demos, max_frames, crop_bounds):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    with h5py.File(rgbd_h5_path, "r") as rgbd_h5:
        data_grp = rgbd_h5["data"]
        if "env_args" not in data_grp.attrs:
            raise KeyError(f"{rgbd_h5_path} missing data.attrs['env_args']")

        env_args_json = data_grp.attrs["env_args"]
        env = build_env_from_env_args(env_args_json)
        sim = _unwrap_env_to_sim(env)

        meta = json.loads(env_args_json)
        env_name = meta.get("env_name", "")

        avail = available_cameras_from_env(env)
        cams = choose_cameras(env_name, avail, requested_cams)

        print("  available cams:", avail)
        print("  chosen cams   :", cams)

        # cache per-cam K + Tc2w
        cam_calib = {}
        for cam in cams:
            K, Tc2w = get_cam_pose_and_K(sim, cam, width=camera_w, height=camera_h)
            cam_calib[cam] = (K, Tc2w)

        eps = sorted(
            rgbd_h5["data"].keys(),
            key=lambda k: int(k.split("_")[-1]) if k.split("_")[-1].isdigit() else k
        )

        demos_done = 0
        try:
            for ep in eps:
                if max_demos > 0 and demos_done >= max_demos:
                    return

                obs = rgbd_h5[f"data/{ep}/obs"]
                depth0 = f"{cams[0]}_depth"
                if depth0 not in obs:
                    print(f"  [WARN] {ep}: missing {depth0}, skip demo")
                    continue

                T = obs[depth0].shape[0]
                if max_frames > 0:
                    T = min(T, max_frames)

                ep_out_dir = os.path.join(out_dir, ep)
                os.makedirs(ep_out_dir, exist_ok=True)

                for tidx in range(T):
                    xyz, rgb = fuse_one_frame(
                        rgbd_h5, ep, tidx, cams=cams, sim=sim, cam_calib=cam_calib, pixel_stride=pixel_stride
                    )
                    if xyz is None or xyz.shape[0] == 0:
                        continue

                    xyz, rgb = maybe_crop(xyz, rgb, crop_bounds)

                    if max_points > 0 and xyz.shape[0] > max_points:
                        sel = rng.choice(xyz.shape[0], size=max_points, replace=False)
                        xyz, rgb = xyz[sel], rgb[sel]

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
                    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))

                    out_path = os.path.join(ep_out_dir, f"frame{tidx:06d}_fused_envcalib.ply")
                    if not o3d.io.write_point_cloud(out_path, pcd):
                        raise RuntimeError(f"Open3D failed to write: {out_path}")
                    print(f"  wrote {out_path} (N={xyz.shape[0]})")

                demos_done += 1
        finally:
            try:
                env.close()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--task-glob", default="*_d0")
    ap.add_argument("--rgbd-pattern", default="*_rgbd_pcd.hdf5")

    ap.add_argument("--out-subdir", default="core/mimicgen_from_depth_pcd")

    ap.add_argument("--cams", nargs="+", default=["agentview", "sideview"])
    ap.add_argument("--camera-height", type=int, default=256)
    ap.add_argument("--camera-width", type=int, default=256)

    ap.add_argument("--pixel-stride", type=int, default=1)
    ap.add_argument("--max-points", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--no-crop", action="store_true")
    ap.add_argument("--max-demos", type=int, default=-1)
    ap.add_argument("--max-frames-per-demo", type=int, default=-1)
    ap.add_argument("--debug-one", action="store_true")

    args = ap.parse_args()

    if args.debug_one:
        args.max_demos = 1
        args.max_frames_per_demo = 1

    crop_bounds = None if args.no_crop else DEFAULT_CROP_BOUNDS

    root = os.path.expanduser(args.root)
    task_dirs = sorted(glob.glob(os.path.join(root, args.task_glob)))
    task_dirs = [d for d in task_dirs if os.path.isdir(d)]
    if not task_dirs:
        raise RuntimeError(f"No task dirs matched: {os.path.join(root, args.task_glob)}")

    print(f"Found {len(task_dirs)} task dirs under {root}")

    for task_dir in task_dirs:
        task_name = os.path.basename(task_dir.rstrip("/"))
        core_dir = os.path.join(task_dir, "core")
        rgbd_matches = sorted(glob.glob(os.path.join(core_dir, args.rgbd_pattern)))

        print(f"\n=== TASK: {task_name} ===")
        print("  rgbd matches:", rgbd_matches)

        if len(rgbd_matches) == 0:
            print("  [SKIP] no rgbd file found")
            continue
        if len(rgbd_matches) > 1:
            raise RuntimeError(f"[{task_name}] multiple rgbd matches: {rgbd_matches} (make pattern more specific)")

        rgbd_h5_path = rgbd_matches[0]
        out_dir = os.path.join(task_dir, args.out_subdir)

        print(f"  rgbd: {rgbd_h5_path}")
        print(f"  out : {out_dir}")

        try:
            process_task(
                rgbd_h5_path=rgbd_h5_path,
                out_dir=out_dir,
                requested_cams=args.cams,
                camera_w=args.camera_width,
                camera_h=args.camera_height,
                pixel_stride=args.pixel_stride,
                max_points=args.max_points,
                seed=args.seed,
                max_demos=args.max_demos,
                max_frames=args.max_frames_per_demo,
                crop_bounds=crop_bounds,
            )
        except Exception as e:
            # do NOT hide errors; surface & move on
            print(f"  [FAIL] {task_name}: {type(e).__name__}: {e}")

    print("\nAll tasks attempted.")


if __name__ == "__main__":
    main()
