#!/usr/bin/env python3
"""
Generate fused PLY point clouds from *_rgbd_pcd.hdf5 across MimicGen tasks.

Key idea: the RGBD HDF5 often stores depth as a *normalized depth buffer* (float ~0.98..0.996),
and MuJoCo camera extrinsics `cam_xmat` have layout/convention gotchas.

Instead of guessing, we auto-try combinations on the first frame:
  - cam_xmat reshape: R or R.T
  - axis convention: no flip or OpenGL->OpenCV flip (diag(1,-1,-1))
  - depth conversion: linear OR OpenGL-style inverse OR robosuite-style conversion
We pick the combination that makes two views overlap best and produces a sane scene size.

Writes:
  - per-cam PLYs (optional)
  - fused PLY per frame

Requires:
  - mimicgen + robosuite + open3d installed (your mimicgen env)
"""

import os, glob, json, argparse
import h5py
import numpy as np
import open3d as o3d

import mimicgen.envs.robosuite  # registers MimicGen envs
try:
    import robosuite_task_zoo
except ImportError:
    pass
import robosuite


# ---------- util ----------
A_GL2CV = np.diag([1.0, -1.0, -1.0]).astype(np.float32)

DEFAULT_CAMS = ["agentview", "sideview"]
DEFAULT_CROP = {"xmin": -0.5, "xmax": 2.0, "ymin": -0.5, "ymax": 2.0, "zmin": -0.2, "zmax": 2.5}


def squeeze_hw(d):
    d = np.asarray(d)
    if d.ndim == 3:
        d = np.squeeze(d)
    if d.ndim != 2:
        d = d.reshape(d.shape[-2], d.shape[-1])
    return d


def apply_T(T, pts):
    ones = np.ones((pts.shape[0], 1), np.float32)
    pts_h = np.concatenate([pts.astype(np.float32), ones], axis=1)
    out = (T.astype(np.float32) @ pts_h.T).T
    return out[:, :3]


def maybe_crop(xyz, rgb, bounds):
    if bounds is None:
        return xyz, rgb
    b = bounds
    m = (
        (xyz[:, 0] >= b["xmin"]) & (xyz[:, 0] <= b["xmax"]) &
        (xyz[:, 1] >= b["ymin"]) & (xyz[:, 1] <= b["ymax"]) &
        (xyz[:, 2] >= b["zmin"]) & (xyz[:, 2] <= b["zmax"])
    )
    if not np.any(m):
        return xyz, rgb
    return xyz[m], rgb[m]


def compute_K_from_fovy(fovy_deg, W, H):
    fovy = np.deg2rad(float(fovy_deg))
    fy = (H / 2.0) / np.tan(fovy / 2.0)
    fx = fy * (W / H)
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float32)


def backproject(depth_z, K, pixel_stride):
    z = squeeze_hw(depth_z).astype(np.float32)
    H, W = z.shape
    vv = np.arange(0, H, pixel_stride, dtype=np.int32)
    uu = np.arange(0, W, pixel_stride, dtype=np.int32)
    U, V = np.meshgrid(uu, vv)

    Z = z[V, U]
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
    idxs = np.stack([V[valid], U[valid]], axis=-1).astype(np.int32)
    return pts_cam, idxs


def write_ply(path, xyz, rgb):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
    ok = o3d.io.write_point_cloud(path, pcd)
    if not ok:
        raise RuntimeError(f"Failed to write {path}")


# ---------- env / cameras ----------
def build_env(env_args_json: str):
    meta = json.loads(env_args_json)
    env_name = meta["env_name"]
    kwargs = dict(meta.get("env_kwargs", {}))

    # must be consistent with how dataset was generated (robomimic/mimicgen processing)
    kwargs["has_renderer"] = False
    kwargs["has_offscreen_renderer"] = False
    kwargs["use_camera_obs"] = False
    kwargs["camera_names"] = []
    kwargs["camera_depths"] = []
    return robosuite.make(env_name, **kwargs)


def list_cameras(env):
    return list(env.sim.model.camera_names)


def choose_cams(env_name, available, requested):
    avail = list(available)
    s = set(avail)
    req = [c for c in (requested or []) if c in s]
    want = ["agentview", "sideview"] if not env_name.startswith("PickPlace") else ["agentview", "frontview"]

    out = []
    for c in req:
        if c not in out:
            out.append(c)
        if len(out) == 2:
            return out
    for c in want:
        if c in s and c not in out:
            out.append(c)
        if len(out) == 2:
            return out
    for c in avail:
        if c not in out:
            out.append(c)
        if len(out) == 2:
            return out
    return out


def mujoco_near_far_meters(sim):
    # MuJoCo uses znear/zfar * extent (this is the big gotcha)
    extent = float(sim.model.stat.extent)
    znear = float(sim.model.vis.map.znear) * extent
    zfar  = float(sim.model.vis.map.zfar)  * extent
    return znear, zfar


def cam_pose_from_sim(sim, cam_name, transpose_R: bool, apply_gl2cv: bool):
    cam_id = sim.model.camera_name2id(cam_name)
    pos = np.array(sim.data.cam_xpos[cam_id], dtype=np.float32)

    # MuJoCo stores cam_xmat as 9 values; reshape+transpose ambiguity is real.
    R = np.array(sim.data.cam_xmat[cam_id], dtype=np.float32).reshape(3, 3)
    if transpose_R:
        R = R.T

    if apply_gl2cv:
        R = R @ A_GL2CV

    Tc2w = np.eye(4, dtype=np.float32)
    Tc2w[:3, :3] = R
    Tc2w[:3, 3] = pos
    return Tc2w


# ---------- depth conversions ----------
def depth_to_z(depth_raw, mode: str, near_m: float, far_m: float):
    """
    depth_raw is usually float32 ~0.98..0.996 (depth buffer in [0,1]).
    We try multiple plausible conversions.
    """
    d = squeeze_hw(depth_raw).astype(np.float32)
    dmax = float(np.nanmax(d)) if np.isfinite(d).any() else 0.0

    if mode == "as_is":
        return d

    if dmax > 1.05:
        # already metric-ish
        return d

    d = np.clip(d, 0.0, 1.0)
    n = float(near_m)
    f = float(far_m)

    if mode == "linear":
        # what your meta-based script did
        return n + d * (f - n)

    if mode == "opengl_inv_simple":
        # common inverse mapping used in many codebases (variant A)
        # z = n*f / (f - d*(f-n))
        return (n * f) / (f - d * (f - n) + 1e-12)

    if mode == "opengl_ndc":
        # OpenGL depth buffer: z_ndc = 2d-1
        # z_eye = 2 n f / (f + n - z_ndc (f - n))
        z_ndc = 2.0 * d - 1.0
        return (2.0 * n * f) / (f + n - z_ndc * (f - n) + 1e-12)

    raise ValueError(f"unknown depth mode {mode}")


# ---------- scoring / auto-select ----------
def score_fusion(ptsA, ptsB, max_n=20000):
    if ptsA.shape[0] == 0 or ptsB.shape[0] == 0:
        return 1e9

    # downsample for speed
    rng = np.random.default_rng(0)
    if ptsA.shape[0] > max_n:
        ptsA = ptsA[rng.choice(ptsA.shape[0], max_n, replace=False)]
    if ptsB.shape[0] > max_n:
        ptsB = ptsB[rng.choice(ptsB.shape[0], max_n, replace=False)]

    from scipy.spatial import cKDTree
    tA = cKDTree(ptsA)
    tB = cKDTree(ptsB)
    dAB = tB.query(ptsA, k=1)[0].mean()
    dBA = tA.query(ptsB, k=1)[0].mean()

    allp = np.concatenate([ptsA, ptsB], axis=0)
    diag = float(np.linalg.norm(allp.max(0) - allp.min(0)))

    size_pen = 0.0
    if diag < 0.2: size_pen += (0.2 - diag) * 50.0
    if diag > 8.0: size_pen += (diag - 8.0) * 5.0

    return 0.5 * (dAB + dBA) + size_pen


def build_points_for_cam(sim, obs, cam, tidx, K, near_m, far_m, depth_mode,
                         pixel_stride, crop_bounds, max_points, rng, use_Rt=False):

    depth_key = f"{cam}_depth"
    img_key   = f"{cam}_image"
    if depth_key not in obs or img_key not in obs:
        return None, None

    depth_raw = np.asarray(obs[depth_key][tidx])
    img = np.asarray(obs[img_key][tidx])

    z = depth_to_z(depth_raw, mode=depth_mode, near_m=near_m, far_m=far_m)
    pts_cam, idxs = backproject(z, K, pixel_stride=pixel_stride)
    if pts_cam.shape[0] == 0:
        return None, None

    v, u = idxs[:, 0], idxs[:, 1]
    rgb = img[v, u, :].astype(np.float32)
    if rgb.max() > 1.5:
        rgb /= 255.0
    # pts_cam[:, 1] *= -1.0
    # pts_cam[:, 2] *= -1.0
    cam_id = sim.model.camera_name2id(cam)
    pos = np.array(sim.data.cam_xpos[cam_id], dtype=np.float32)
    R = np.array(sim.data.cam_xmat[cam_id], dtype=np.float32).reshape(3, 3)  # NO transpose

    # --- CV -> MuJoCo(OpenGL) camera coords ---
    pts_gl = pts_cam.copy()
    pts_gl[:, 1] *= -1.0   # y: down -> up
    pts_gl[:, 2] *= -1.0   # z: forward -> backward

    # --- camera -> world ---
    pts_world = (R @ pts_gl.T).T + pos
    # pts_world = apply_T(np.linalg.inv(Tc2w), pts_cam)
    pts_world, rgb = maybe_crop(pts_world, rgb, crop_bounds)

    if pts_world.shape[0] == 0:
        return None, None

    if max_points > 0 and pts_world.shape[0] > max_points:
        sel = rng.choice(pts_world.shape[0], size=max_points, replace=False)
        pts_world, rgb = pts_world[sel], rgb[sel]

    return pts_world, rgb


def auto_pick_convention(env, rgbd_h5, ep, tidx, cams, H, W, pixel_stride, crop_bounds, max_points, seed):
    rng = np.random.default_rng(seed)
    sim = env.sim
    sim.forward()

    near_m, far_m = mujoco_near_far_meters(sim)

    # intrinsics from fovy
    K_by_cam = {}
    for cam in cams:
        cam_id = sim.model.camera_name2id(cam)
        fovy = float(sim.model.cam_fovy[cam_id])
        K_by_cam[cam] = compute_K_from_fovy(fovy, W=W, H=H)

    obs = rgbd_h5[f"data/{ep}/obs"]

    depth_modes = ["linear", "opengl_ndc", "opengl_inv_simple"]
    transposes = [False, True]
    axisflips = [False, True]

    best = None
    best_score = 1e18

    camA, camB = cams[0], cams[1]

    for dm in depth_modes:
        for tr in transposes:
            for ax in axisflips:
                ptsA, _ = build_points_for_cam(sim, obs, camA, tidx, K_by_cam[camA], near_m, far_m, dm,
                                            pixel_stride, crop_bounds, max_points, rng, use_Rt=tr)
                ptsB, _ = build_points_for_cam(sim, obs, camB, tidx, K_by_cam[camB], near_m, far_m, dm,
                                            pixel_stride, crop_bounds, max_points, rng, use_Rt=tr)

                if ptsA is None or ptsB is None:
                    continue

                sc = score_fusion(ptsA, ptsB)
                if sc < best_score:
                    best_score = sc
                    best = dict(depth_mode=dm, transpose_R=tr, axisflip=ax, near_m=near_m, far_m=far_m)

    if best is None:
        raise RuntimeError("Auto-pick failed: could not build points for both cams with any convention.")

    print(f"  [AUTO] picked depth_mode={best['depth_mode']} transpose_R={best['transpose_R']} axisflip={best['axisflip']}"
          f" near={best['near_m']:.4f} far={best['far_m']:.4f}")

    return best, K_by_cam


def process_task(rgbd_path, out_dir, cams_req, H, W, pixel_stride, max_points, seed,
                 max_demos, max_frames, crop_bounds, debug_per_cam):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    with h5py.File(rgbd_path, "r") as h5:
        env_args_json = h5["data"].attrs["env_args"]
        meta = json.loads(env_args_json)
        env_name = meta.get("env_name", "")

        env = build_env(env_args_json)
        try:
            avail = list_cameras(env)
            cams = choose_cams(env_name, avail, cams_req)
            if len(cams) < 2:
                raise RuntimeError(f"Need 2 cams, got {cams}")

            print("  available cams:", avail)
            print("  chosen cams   :", cams)

            eps = sorted(h5["data"].keys(), key=lambda k: int(k.split("_")[-1]) if k.split("_")[-1].isdigit() else k)

            demos_done = 0
            best = None
            K_by_cam = None

            for ep in eps:
                if max_demos > 0 and demos_done >= max_demos:
                    break

                obs = h5[f"data/{ep}/obs"]
                dk0 = f"{cams[0]}_depth"
                if dk0 not in obs:
                    continue

                T = obs[dk0].shape[0]
                if max_frames > 0:
                    T = min(T, max_frames)

                ep_out = os.path.join(out_dir, ep)
                os.makedirs(ep_out, exist_ok=True)

                # auto-pick once per task (on first ep/frame we actually process)
                if best is None:
                    best, K_by_cam = auto_pick_convention(
                        env=env, rgbd_h5=h5, ep=ep, tidx=0, cams=cams,
                        H=H, W=W, pixel_stride=pixel_stride,
                        crop_bounds=crop_bounds, max_points=min(max_points, 80000) if max_points > 0 else -1,
                        seed=seed
                    )

                sim = env.sim
                sim.forward()
                near_m = best["near_m"]
                far_m  = best["far_m"]

                for tidx in range(T):
                    print(f"  {ep} t={tidx}")

                    all_xyz, all_rgb = [], []

                    for cam in cams:
                        Tc2w = cam_pose_from_sim(sim, cam, transpose_R=best["transpose_R"], apply_gl2cv=best["axisflip"])
                        pts, rgb = build_points_for_cam(
                            sim, obs, cam, tidx,
                            K_by_cam[cam],
                            near_m, far_m,
                            best["depth_mode"],
                            pixel_stride,
                            crop_bounds,
                            max_points,
                            rng,
                            use_Rt=best["transpose_R"],
                        )

                        if pts is None:
                            continue

                        if debug_per_cam:
                            write_ply(os.path.join(ep_out, f"frame{tidx:06d}_{cam}.ply"), pts, rgb)

                        all_xyz.append(pts)
                        all_rgb.append(rgb)

                    if not all_xyz:
                        continue

                    xyz = np.concatenate(all_xyz, axis=0)
                    rgb = np.concatenate(all_rgb, axis=0)

                    fused_path = os.path.join(ep_out, f"frame{tidx:06d}_fused_envcalib.ply")
                    write_ply(fused_path, xyz, rgb)
                    print(f"  wrote {fused_path} (N={xyz.shape[0]})")

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

    ap.add_argument("--cams", nargs="+", default=DEFAULT_CAMS)
    ap.add_argument("--camera-height", type=int, default=256)
    ap.add_argument("--camera-width", type=int, default=256)

    ap.add_argument("--pixel-stride", type=int, default=1)
    ap.add_argument("--max-points", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--no-crop", action="store_true")
    ap.add_argument("--max-demos", type=int, default=-1)
    ap.add_argument("--max-frames-per-demo", type=int, default=-1)
    ap.add_argument("--debug-one", action="store_true")
    ap.add_argument("--debug-per-cam", action="store_true")

    ap.add_argument("--tasks", nargs="*", default=None,
                help="Optional explicit list of task folder names to process (e.g. hammer_cleanup_d0 kitchen_d0).")
    ap.add_argument("--task-start", default=None,
                help="If set, skip tasks until this task name is reached (inclusive).")


    args = ap.parse_args()

    if args.debug_one:
        args.max_demos = 1
        args.max_frames_per_demo = 1

    crop_bounds = None if args.no_crop else DEFAULT_CROP

    root = os.path.expanduser(args.root)
    task_dirs = sorted(glob.glob(os.path.join(root, args.task_glob)))
    task_dirs = [d for d in task_dirs if os.path.isdir(d)]
    if not task_dirs:
        raise RuntimeError("No task dirs matched")

    all_names = [os.path.basename(d.rstrip("/")) for d in task_dirs]

    if args.tasks is not None and len(args.tasks) > 0:
        wanted = set(args.tasks)
        task_dirs = [d for d in task_dirs if os.path.basename(d.rstrip("/")) in wanted]
        missing = [t for t in args.tasks if t not in all_names]
        if missing:
            print("[warn] requested tasks not found under root:", missing)

    if args.task_start is not None:
        start = args.task_start
        if start not in all_names:
            raise RuntimeError(f"--task-start {start} not found. Available: {all_names}")
        i0 = all_names.index(start)
        task_dirs = task_dirs[i0:]


    print(f"Found {len(task_dirs)} task dirs under {root}")

    for task_dir in task_dirs:
        task_name = os.path.basename(task_dir.rstrip("/"))
        core_dir = os.path.join(task_dir, "core")
        rgbd_matches = sorted(glob.glob(os.path.join(core_dir, args.rgbd_pattern)))

        print(f"\n=== TASK: {task_name} ===")
        print("  rgbd matches:", rgbd_matches)

        if len(rgbd_matches) != 1:
            print(f"  [SKIP] expected 1 rgbd file, got {len(rgbd_matches)}")
            continue

        rgbd_path = rgbd_matches[0]
        out_dir = os.path.join(task_dir, args.out_subdir)

        print(f"  rgbd: {rgbd_path}")
        print(f"  out : {out_dir}")

        process_task(
            rgbd_path=rgbd_path,
            out_dir=out_dir,
            cams_req=args.cams,
            H=args.camera_height,
            W=args.camera_width,
            pixel_stride=args.pixel_stride,
            max_points=args.max_points,
            seed=args.seed,
            max_demos=args.max_demos,
            max_frames=args.max_frames_per_demo,
            crop_bounds=crop_bounds,
            debug_per_cam=args.debug_per_cam,
        )


if __name__ == "__main__":
    main()
