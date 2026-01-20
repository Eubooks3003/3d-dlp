#!/usr/bin/env python3
"""
Rebuild MimicGen point clouds from depth + camera intrinsics/extrinsics (fixed).

Key fixes vs your version:
  - depth is often NORMALIZED -> convert using near/far
  - meta/cameras/<cam>/T_cam_world may actually be P = K[R|t] (projection), not rigid
    -> recover world->cam via inv(K)@P, then invert to get cam->world
  - disable recenter/scale for saved PLY (keep real world coordinates)
"""

import os
import h5py
import numpy as np
import open3d as o3d

# ---------------- CONFIG ----------------
H5_PATH    = "/home/ellina/Desktop/Code/articubot-on-mimicgen/mimicgen_data/stack_d1/core/stack_d1_rgbd_pcd.hdf5"
OUT_DIR    = "mimicgen_from_depth_pcd_debug"

CAMS       = ["agentview", "sideview", "robot0_eye_in_hand"]
NUM_SCENES = 5

# Set to None while debugging (cropping can hide issues!)
# CROP_BOUNDS = None
# Example crop (meters) once things look correct:
CROP_BOUNDS = {"xmin": -2.0, "xmax": 2.0, "ymin": -2.0, "ymax": 2.0, "zmin": -0.2, "zmax": 2.5}

PIXEL_STRIDE = 1     # use 2/4 if clouds are too dense
MAX_POINTS   = 200000  # optional random cap per frame (-1 disables)
# ----------------------------------------


def squeeze_hw(d):
    d = np.asarray(d)
    if d.ndim == 3:
        d = np.squeeze(d)
    if d.ndim != 2:
        d = d.reshape(d.shape[-2], d.shape[-1])
    return d


def looks_like_rigid(T: np.ndarray, tol=1e-2) -> bool:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        return False
    if not np.allclose(T[3], [0, 0, 0, 1], atol=tol):
        return False
    R = T[:3, :3]
    # reject huge norms (K-baked projection-like matrices)
    cn = np.linalg.norm(R, axis=0)
    if np.any(cn > 3.0) or np.any(cn < 0.2):
        return False
    return True


def recover_cam2world(T_cam_world_like: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    If T_cam_world_like is a proper rigid cam->world, return it.
    Otherwise treat top 3x4 as P = K [R|t] mapping WORLD->CAM in pinhole model.
      E = inv(K) @ P = [R|t] (world->cam)
      Tw2c = [[R t],[0 0 0 1]]
      Tc2w = inv(Tw2c)
    """
    T = np.asarray(T_cam_world_like, dtype=np.float64).reshape(4, 4)
    if looks_like_rigid(T):
        return T.astype(np.float32)

    P = T[:3, :4]                      # 3x4
    E = np.linalg.inv(K) @ P           # 3x4  (world->cam)
    Tw2c = np.eye(4, dtype=np.float64)
    Tw2c[:3, :4] = E
    Tc2w = np.linalg.inv(Tw2c)
    return Tc2w.astype(np.float32)


def get_near_far(cam_group) -> tuple:
    # try attrs first, then datasets if they exist
    near = cam_group.attrs.get("near", None)
    far  = cam_group.attrs.get("far", None)
    if near is None and "near" in cam_group:
        near = float(np.asarray(cam_group["near"]))
    if far is None and "far" in cam_group:
        far = float(np.asarray(cam_group["far"]))
    return (None if near is None else float(near),
            None if far  is None else float(far))


def load_cam_K_T(h5, cam):
    g = h5[f"meta/cameras/{cam}"]
    K = np.asarray(g["K"], dtype=np.float32)
    T_raw = np.asarray(g["T_cam_world"], dtype=np.float32)
    near, far = get_near_far(g)

    # recover a proper cam->world
    Tc2w = recover_cam2world(T_raw, K.astype(np.float64))
    return K, Tc2w, near, far


def depth_to_meters(depth, near, far):
    """
    MimicGen depth often stored normalized in [0,1] (float or uint16).
    Convert to meters using near/far when it looks normalized.
    """
    d = squeeze_hw(depth).astype(np.float32)

    # Handle integer depth buffers that represent normalized depth
    if np.issubdtype(depth.dtype, np.integer):
        # uint16 is common
        denom = 65535.0 if depth.dtype == np.uint16 else 255.0
        d = d / denom

    # If it looks normalized and we have near/far, convert to meters
    dmax = float(np.nanmax(d)) if np.isfinite(d).any() else 0.0
    if dmax <= 1.01 and (near is not None) and (far is not None):
        d = near + d * (far - near)

    return d


def backproject_depth(depth_m, K, pixel_stride=1):
    d = squeeze_hw(depth_m).astype(np.float32)
    H, W = d.shape

    # subsample pixels for speed
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


def maybe_crop(xyz, rgb):
    if CROP_BOUNDS is None:
        return xyz, rgb
    b = CROP_BOUNDS
    m = (
        (xyz[:, 0] >= b["xmin"]) & (xyz[:, 0] <= b["xmax"]) &
        (xyz[:, 1] >= b["ymin"]) & (xyz[:, 1] <= b["ymax"]) &
        (xyz[:, 2] >= b["zmin"]) & (xyz[:, 2] <= b["zmax"])
    )
    if not np.any(m):
        return xyz, rgb
    return xyz[m], rgb[m]


def fuse_episode_timestep(h5, ep, tidx):
    obs = h5[f"data/{ep}/obs"]
    all_xyz, all_rgb = [], []

    for cam in CAMS:
        depth_key = f"{cam}_depth"
        img_key   = f"{cam}_image"
        if depth_key not in obs or img_key not in obs:
            print(f"[WARN] {ep} t={tidx}: missing {depth_key} or {img_key}, skipping {cam}")
            continue

        depth_raw = np.asarray(obs[depth_key][tidx])
        img = np.asarray(obs[img_key][tidx])

        K, Tc2w, near, far = load_cam_K_T(h5, cam)

        depth_m = depth_to_meters(depth_raw, near, far)

        # DEBUG: how dense is the depth?
        valid_ct = int(np.isfinite(depth_m).sum())
        pos_ct   = int((depth_m > 0).sum())
        print(f"  {cam}: depth shape={depth_raw.shape} dtype={depth_raw.dtype} "
              f"depth_m[min,max]=[{np.nanmin(depth_m):.4f},{np.nanmax(depth_m):.4f}] "
              f"finite={valid_ct} >0={pos_ct} near/far={near}/{far}")

        pts_cam, idxs = backproject_depth(depth_m, K, pixel_stride=PIXEL_STRIDE)
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

        print(f"  {cam}: pts_cam={pts_cam.shape[0]} pts_world={pts_world.shape[0]}")

    if not all_xyz:
        return None, None

    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)
    return xyz, rgb


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with h5py.File(H5_PATH, "r") as h5:
        eps = sorted(h5["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        print(f"Found {len(eps)} episodes; first 5:", eps[:5])

        scenes_done = 0
        for ep in eps:
            depth_key = f"{CAMS[0]}_depth"
            if depth_key not in h5[f"data/{ep}/obs"]:
                continue
            T = h5[f"data/{ep}/obs/{depth_key}"].shape[0]

            for tidx in range(T):
                if scenes_done >= NUM_SCENES:
                    print("Done.")
                    return

                print(f"\n{ep} t={tidx}")
                xyz, rgb = fuse_episode_timestep(h5, ep, tidx)
                if xyz is None or xyz.shape[0] == 0:
                    continue

                before = xyz.shape[0]
                xyz, rgb = maybe_crop(xyz, rgb)
                after = xyz.shape[0]
                print(f"  fused points: before_crop={before} after_crop={after}")

                if MAX_POINTS > 0 and xyz.shape[0] > MAX_POINTS:
                    sel = np.random.default_rng(0).choice(xyz.shape[0], size=MAX_POINTS, replace=False)
                    xyz, rgb = xyz[sel], rgb[sel]

                print(f"  xyz range: min={xyz.min(0)} max={xyz.max(0)} N={xyz.shape[0]}")

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
                pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))

                out_path = os.path.join(OUT_DIR, f"{ep}_frame{tidx:06d}_fused_fixed.ply")
                o3d.io.write_point_cloud(out_path, pcd)
                scenes_done += 1
                print(f"[{scenes_done}/{NUM_SCENES}] wrote {out_path}")


if __name__ == "__main__":
    main()
