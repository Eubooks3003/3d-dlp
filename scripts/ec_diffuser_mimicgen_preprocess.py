#!/usr/bin/env python3
import os, re, argparse, pickle
import numpy as np
import h5py
import torch
import open3d as o3d

from voxel_models import DLP
from utils.util_func import get_config
from utils.log_utils import load_checkpoint


def read_ply_xyzrgb(path):
    pcd = o3d.io.read_point_cloud(path)
    xyz = np.asarray(pcd.points, dtype=np.float32)
    rgb = np.asarray(pcd.colors, dtype=np.float32)  # [0,1] usually
    if rgb.size == 0:
        rgb = np.zeros((xyz.shape[0], 3), dtype=np.float32)
    return xyz, rgb


def voxelize_xyzrgb(xyz, rgb, grid_dhw, bounds_xyz=None, with_occ=False, eps=1e-6):
    """
    If bounds_xyz is None -> per-frame bounds (like VoxelGridXYZ(bounds=None)):
        pmin = xyz.min(0), pmax = xyz.max(0)
    This avoids dropping the whole scene due to wrong fixed bounds.
    """
    (D, H, W) = grid_dhw

    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.float32)
    if xyz.shape[0] == 0:
        C = 4 if with_occ else 3
        return np.zeros((C, D, H, W), dtype=np.float32)

    # Remove NaNs/Infs
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    rgb = rgb[finite] if rgb.shape[0] == finite.shape[0] else rgb
    if xyz.shape[0] == 0:
        C = 4 if with_occ else 3
        return np.zeros((C, D, H, W), dtype=np.float32)

    # bounds
    if bounds_xyz is None:
        pmin = xyz.min(axis=0)
        pmax = xyz.max(axis=0)
    else:
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds_xyz
        pmin = np.array([xmin, ymin, zmin], dtype=np.float32)
        pmax = np.array([xmax, ymax, zmax], dtype=np.float32)

    span = np.maximum(pmax - pmin, eps)

    # normalize to [0,1] in THIS frame’s bounds
    p01 = (xyz - pmin[None, :]) / span[None, :]
    p01 = np.clip(p01, 0.0, 1.0)

    ix = np.clip((p01[:, 0] * (W - 1)).astype(np.int32), 0, W - 1)
    iy = np.clip((p01[:, 1] * (H - 1)).astype(np.int32), 0, H - 1)
    iz = np.clip((p01[:, 2] * (D - 1)).astype(np.int32), 0, D - 1)

    # accumulate avg rgb
    size = D * H * W
    lin = (iz * H + iy) * W + ix

    cnt = np.zeros((size,), dtype=np.int32)
    rs  = np.zeros((size,), dtype=np.float32)
    gs  = np.zeros((size,), dtype=np.float32)
    bs  = np.zeros((size,), dtype=np.float32)

    np.add.at(cnt, lin, 1)
    np.add.at(rs,  lin, rgb[:, 0])
    np.add.at(gs,  lin, rgb[:, 1])
    np.add.at(bs,  lin, rgb[:, 2])

    mask = cnt > 0
    rs[mask] /= cnt[mask]
    gs[mask] /= cnt[mask]
    bs[mask] /= cnt[mask]

    rgb_vol = np.stack([rs, gs, bs], axis=0).reshape(3, D, H, W)

    if with_occ:
        occ = mask.astype(np.float32).reshape(1, D, H, W)
        vol = np.concatenate([rgb_vol, occ], axis=0)
    else:
        vol = rgb_vol

    return vol.astype(np.float32)

def build_dlp_from_cfg(cfg, device):
    model = DLP(
        cdim=cfg["ch"],
        image_size=cfg["voxel_grid_whd"][0],
        normalize_rgb=cfg["normalize_rgb"],
        n_kp_per_patch=cfg["n_kp_per_patch"],
        patch_size=cfg["patch_size"],
        anchor_s=cfg["anchor_s"],
        n_kp_enc=cfg["n_kp_enc"],
        n_kp_prior=cfg["n_kp_prior"],
        pad_mode=cfg["pad_mode"],
        dropout=cfg["dropout"],
        features_dist=cfg.get("features_dist", "gauss"),
        learned_feature_dim=cfg["learned_feature_dim"],
        learned_bg_feature_dim=cfg.get("learned_bg_feature_dim", cfg["learned_feature_dim"]),
        n_fg_categories=cfg.get("n_fg_categories", 8),
        n_fg_classes=cfg.get("n_fg_classes", 4),
        n_bg_categories=cfg.get("n_bg_categories", 4),
        n_bg_classes=cfg.get("n_bg_classes", 4),
        scale_std=cfg["scale_std"],
        offset_std=cfg["offset_std"],
        obj_on_alpha=cfg["obj_on_alpha"],
        obj_on_beta=cfg["obj_on_beta"],
        obj_res_from_fc=cfg["obj_res_from_fc"],
        obj_ch_mult_prior=cfg.get("obj_ch_mult_prior", cfg["obj_ch_mult"]),
        obj_ch_mult=cfg["obj_ch_mult"],
        obj_base_ch=cfg["obj_base_ch"],
        obj_final_cnn_ch=cfg["obj_final_cnn_ch"],
        bg_res_from_fc=cfg["bg_res_from_fc"],
        bg_ch_mult=cfg["bg_ch_mult"],
        bg_base_ch=cfg["bg_base_ch"],
        bg_final_cnn_ch=cfg["bg_final_cnn_ch"],
        use_resblock=cfg["use_resblock"],
        num_res_blocks=cfg["num_res_blocks"],
        cnn_mid_blocks=cfg.get("cnn_mid_blocks", False),
        mlp_hidden_dim=cfg.get("mlp_hidden_dim", 256),
        pint_enc_layers=cfg["pint_enc_layers"],
        pint_enc_heads=cfg["pint_enc_heads"],
        timestep_horizon=1,
        separate_depth_features=cfg.get("separate_depth_features", False),
        depth_feature_dim=cfg.get("depth_feature_dim", 0),
        split_loss=cfg.get("split_loss", False),
        depth_loss_ratio=cfg.get("depth_loss_ratio", 1.0),
    ).to(device)
    model.eval()
    return model


def _coerce_BKD(t: torch.Tensor, K: int):
    if t is None or not torch.is_tensor(t):
        return None
    if t.dim() == 3 and t.shape[1] == K:
        return t
    if t.dim() == 2 and t.shape[1] == K:
        return t.unsqueeze(-1)
    if t.dim() == 4 and t.shape[1] == 1 and t.shape[2] == K:
        return t[:, 0]
    if t.dim() == 4 and t.shape[1] == K and t.shape[2] == 1:
        return t[:, :, 0]
    return None


def _get_first_BKD(out, keys, K):
    for k in keys:
        t = _coerce_BKD(out.get(k, None), K)
        if t is not None:
            return t, k
    return None, None


def _auto_find_features(out, K, targetF=None):
    """
    Heuristic: find any tensor shaped [B,K,F] that looks like per-particle appearance/features.
    Prefer exact match to targetF (= cfg['learned_feature_dim']) if provided.
    Exclude obvious non-features by key name and common dims (1,3).
    """
    candidates = []
    for k, v in out.items():
        if not torch.is_tensor(v):
            continue
        if v.dim() != 3:
            continue
        if v.shape[1] != K:
            continue
        F = int(v.shape[2])
        kl = k.lower()
        # exclude obvious non-features
        if any(s in kl for s in ("kp", "keypoint", "cov", "sigma", "var", "scale", "obj_on", "mask", "alpha", "beta", "offset")):
            continue
        if F in (1, 3):
            continue
        score = 0
        if targetF is not None and F == int(targetF):
            score += 1000
        # prefer "feat" / "feature" named keys
        if "feat" in kl or "feature" in kl:
            score += 100
        # prefer larger F after that
        score += F
        candidates.append((score, k, v))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, kbest, vbest = candidates[0]
    return vbest, kbest

def pack_tokens_k24(out, obj_on_thresh=0.0, scale_max=10.0, var_max=1e2):
    """
    K=24 token: [pos(3), scale(3), feat(F), obj_on(1)] where F comes from z_features.
    Uses the 24-slot object latents directly.
    """
    # --- positions: z is (B,1,24,3) ---
    z = out.get("z", None)
    if z is None:
        raise KeyError("Expected out['z'] for K=24 packing.")
    if z.dim() != 4 or z.shape[2] != 24 or z.shape[-1] != 3:
        raise RuntimeError(f"Unexpected z shape: {tuple(z.shape)} (expected (B,1,24,3))")
    pos = z[:, 0]  # (B,24,3)

    # --- scale ---
    z_scale = out.get("z_scale", None)
    if z_scale is not None and z_scale.dim() == 4 and z_scale.shape[2] == 24:
        scale = z_scale[:, 0]  # (B,24,3)
    else:
        # fallback from cov_kp if needed (but your out has z_scale)
        cov = out.get("cov_kp", None)
        if cov is not None and cov.dim() == 4 and cov.shape[1] == 64:
            # can't map 64->24 reliably; use zeros instead
            scale = torch.zeros((pos.shape[0], 24, 3), device=pos.device, dtype=pos.dtype)
        else:
            scale = torch.zeros((pos.shape[0], 24, 3), device=pos.device, dtype=pos.dtype)

        # scale_raw is what you fetched from out: mu_scale / z_scale / etc.
        scale_raw = scale

        # DLP scale is typically in log-space -> convert to positive
        scale = torch.exp(scale_raw)

        # now clamp (optional)
        scale = scale.clamp(min=1e-6, max=float(scale_max))


    # --- features ---
    feat = out.get("z_features", None)
    if feat is None:
        feat = out.get("mu_features", None)
    if feat is None:
        feat = torch.zeros((pos.shape[0], 24, 0), device=pos.device, dtype=pos.dtype)
    else:
        if feat.dim() != 4 or feat.shape[2] != 24:
            raise RuntimeError(f"Unexpected feature shape: {tuple(feat.shape)} (expected (B,1,24,F))")
        feat = feat[:, 0]  # (B,24,F)

    # --- obj_on ---
    obj_on = out.get("obj_on", None)
    if obj_on is None:
        obj_on = out.get("mu_obj_on", None)
    if obj_on is None:
        obj_on = torch.ones((pos.shape[0], 24, 1), device=pos.device, dtype=pos.dtype)
    else:
        if obj_on.dim() != 4 or obj_on.shape[2] != 24:
            raise RuntimeError(f"Unexpected obj_on shape: {tuple(obj_on.shape)} (expected (B,1,24,1))")
        obj_on = obj_on[:, 0]  # (B,24,1)

    obj_on = obj_on[..., :1]

    toks = torch.cat([pos, scale, feat, obj_on], dim=-1)  # (B,24, 7+F)
    if obj_on_thresh > 0:
        toks = toks * (obj_on > obj_on_thresh).float()
    return toks, obj_on


def _print_array_stats(name, arr, per_dim=False, dim_last=None):
    arr = np.asarray(arr)
    finite = np.isfinite(arr)
    f = arr[finite]
    print(f"{name}: shape={arr.shape} dtype={arr.dtype}")
    if f.size == 0:
        print("  (no finite values)")
        return
    print(f"  min={f.min():.6g} max={f.max():.6g} mean={f.mean():.6g} std={f.std():.6g}")
    # quick percentiles
    p = np.percentile(f, [0, 1, 5, 50, 95, 99, 100])
    print(f"  pcts [0,1,5,50,95,99,100] = {np.array2string(p, precision=4, suppress_small=False)}")
    if per_dim:
        if dim_last is None:
            dim_last = arr.shape[-1]
        flat = arr.reshape(-1, dim_last)
        stds = flat.std(axis=0)
        mins = flat.min(axis=0)
        maxs = flat.max(axis=0)
        print(f"  per-dim std: {np.array2string(stds, precision=6, suppress_small=False)}")
        print(f"  per-dim min: {np.array2string(mins, precision=6, suppress_small=False)}")
        print(f"  per-dim max: {np.array2string(maxs, precision=6, suppress_small=False)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--ply-dir", required=True)
    ap.add_argument("--ply-pattern", default=r"(?P<demo>demo_\d+)_frame(?P<t>\d+)_.*\.ply")
    ap.add_argument("--dlp-cfg", required=True)
    ap.add_argument("--dlp-ckpt", required=True)
    ap.add_argument("--out-pkl", required=True)
    ap.add_argument("--grid-dhw", default="64,64,64")
    ap.add_argument("--bounds", default="-1,1,-1,1,-1,1")
    ap.add_argument("--with-occ", action="store_true")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--obj-on-thresh", type=float, default=0.0)
    ap.add_argument("--scale-max", type=float, default=10.0)
    ap.add_argument("--var-max", type=float, default=1e2)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--debug-keys", action="store_true",
                    help="Print model output keys/shapes once per run (first batch).")
    ap.add_argument("--debug-pack", action="store_true",
                    help="Print pack_tokens chosen feat key/shapes for first batch.")
    ap.add_argument("--max-demos", type=int, default=None,
                help="Only process first N demos (debug).")

    args = ap.parse_args()

    D, H, W = map(int, args.grid_dhw.split(","))
    xmin, xmax, ymin, ymax, zmin, zmax = map(float, args.bounds.split(","))
    bounds_xyz = ((xmin, xmax), (ymin, ymax), (zmin, zmax))

    device = torch.device(args.device if ("cuda" in args.device and torch.cuda.is_available()) else "cpu")

    cfg = get_config(args.dlp_cfg)
    model = build_dlp_from_cfg(cfg, device)
    _ = load_checkpoint(args.dlp_ckpt, model, None, None, map_location=device)

    expected_c = int(cfg.get("ch", 3))
    learned_feature_dim = int(cfg.get("learned_feature_dim", 0))

    if args.with_occ and expected_c == 3:
        print("[warn] cfg/ckpt expects 3 channels but --with-occ makes 4. Disabling occ.")
        args.with_occ = False

    pat = re.compile(args.ply_pattern)
    ply_map = {}
    for fn in os.listdir(args.ply_dir):
        m = pat.match(fn)
        if not m:
            continue
        demo = m.group("demo")
        t = int(m.group("t"))
        ply_map.setdefault(demo, {})[t] = os.path.join(args.ply_dir, fn)

    demos = sorted(ply_map.keys(), key=lambda s: int(s.split("_")[-1]))
    if args.max_demos is not None:
        demos = demos[:args.max_demos]
    print(f"[ply] demos found: {len(demos)}")

    ep_obs = []
    ep_act = []
    path_lengths = []

    total_written = 0
    stop_all = False
    did_debug_keys = False

    with h5py.File(args.h5, "r") as h5:
        for demo in demos:
            if args.max_frames is not None and total_written >= args.max_frames:
                break

            act_key = f"data/{demo}/actions"
            if act_key not in h5:
                continue
            actions = np.asarray(h5[act_key], dtype=np.float32)
            T = actions.shape[0]

            obs_steps = []
            act_steps = []

            t = 0
            while t < T:
                if args.max_frames is not None and total_written >= args.max_frames:
                    stop_all = True
                    break

                remaining = None if args.max_frames is None else (args.max_frames - total_written)
                batch_cap = args.batch if remaining is None else max(0, min(args.batch, remaining))
                if batch_cap == 0:
                    stop_all = True
                    break

                tb = list(range(t, min(t + batch_cap, T)))
                vox_batch, valid_tb = [], []

                for tt in tb:
                    path = ply_map.get(demo, {}).get(tt, None)
                    if path is None:
                        continue
                    xyz, rgb = read_ply_xyzrgb(path)
                    vol = voxelize_xyzrgb(xyz, rgb, (D, H, W), bounds_xyz, with_occ=args.with_occ)
                    vox_batch.append(vol)
                    valid_tb.append(tt)

                t += batch_cap
                if len(vox_batch) == 0:
                    continue

                vox = torch.from_numpy(np.stack(vox_batch, axis=0)).to(device)  # (B,C,D,H,W)
                if vox.shape[1] != expected_c:
                    raise RuntimeError(f"Channel mismatch: vox C={vox.shape[1]} but model expects C={expected_c}.")

                with torch.no_grad():
                    try:
                        out = model(vox, deterministic=True, warmup=False, with_loss=False)
                        zs = out.get("z_scale", None)
                        ms = out.get("mu_scale", None)
                        print("z_scale", None if zs is None else (zs.min().item(), zs.max().item(), zs.std().item(), zs.shape))
                        print("mu_scale", None if ms is None else (ms.min().item(), ms.max().item(), ms.std().item(), ms.shape))

                    except TypeError:
                        try:
                            out = model(vox, warmup=False, with_loss=False)
                        except TypeError:
                            out = model(vox)

                    if args.debug_keys and not did_debug_keys:
                        did_debug_keys = True
                        print("\n[DLP out keys/shapes]")
                        for k in sorted(out.keys()):
                            v = out[k]
                            if torch.is_tensor(v):
                                print(f"  {k}: {tuple(v.shape)} dtype={v.dtype}")
                            else:
                                print(f"  {k}: {type(v)}")
                        print("")

                    toks, _ = pack_tokens_k24(
                        out,
                        obj_on_thresh=args.obj_on_thresh,
                        scale_max=args.scale_max,
                        var_max=args.var_max,
                    )

                    if args.debug_pack and not did_debug_keys:
                        print(f"[pack_tokens] selected feat_key={feat_key}")

                toks_np = toks.detach().cpu().numpy().astype(np.float32)  # (B,K,Dtok)
                obs_steps.append(toks_np)
                act_steps.append(actions[valid_tb].astype(np.float32))

                total_written += toks_np.shape[0]
                if args.max_frames is not None and total_written >= args.max_frames:
                    stop_all = True
                    break

            if len(obs_steps) == 0:
                continue

            obs_ep = np.concatenate(obs_steps, axis=0)  # (L,K,Dtok)
            act_ep = np.concatenate(act_steps, axis=0)  # (L,A)
            L = obs_ep.shape[0]

            ep_obs.append(obs_ep)
            ep_act.append(act_ep)
            path_lengths.append(L)

            print(f"[demo] {demo}: wrote {L} frames (TOTAL={total_written})")
            if stop_all:
                break

    if len(ep_obs) == 0:
        raise RuntimeError("No frames written. Check ply-pattern, ply-dir, bounds/grid, and that PLYs exist.")

    E = len(ep_obs)
    Tmax = int(max(path_lengths))
    K = int(ep_obs[0].shape[1])
    Dtok = int(ep_obs[0].shape[2])
    A = int(ep_act[0].shape[1])

    observations = np.zeros((E, Tmax, K, Dtok), dtype=np.float32)
    actions      = np.zeros((E, Tmax, A),       dtype=np.float32)
    rewards      = np.zeros((E, Tmax, 1),       dtype=np.float32)
    terminals    = np.zeros((E, Tmax, 1),       dtype=np.float32)
    timeouts     = np.zeros((E, Tmax, 1),       dtype=np.float32)
    goals        = np.zeros((E, Tmax, K, Dtok), dtype=np.float32)

    for e in range(E):
        L = int(path_lengths[e])

        # fill valid segment
        observations[e, :L] = ep_obs[e]
        actions[e, :L]      = ep_act[e]

        # define goal = final valid observation (per-particle tokens)
        g = ep_obs[e][L-1]                 # (K, Dtok)
        goals[e, :L] = g[None, :, :]       # repeat across valid timesteps

    # ---- IMPORTANT: pad by repeating last valid state/goal ----
    if L < Tmax:
        observations[e, L:] = observations[e, L-1:L]   # repeats (1,K,Dtok) across time
        goals[e, L:]        = goals[e, L-1:L]          # same goal into padding
        actions[e, L:]      = 0.0                      # ok to keep actions zero

        # optional but recommended: mark padding as terminal/timeout too
        terminals[e, L-1:, 0] = 1.0
        timeouts[e,  L-1:, 0] = 1.0
    else:
        terminals[e, L-1, 0] = 1.0
        timeouts[e,  L-1, 0] = 1.0


    path_lengths = np.asarray(path_lengths, dtype=np.int32)

    assert observations.shape[0] == path_lengths.shape[0]
    assert actions.shape[0] == path_lengths.shape[0]
    assert rewards.shape[0] == path_lengths.shape[0]
    assert terminals.shape[0] == path_lengths.shape[0]
    assert timeouts.shape[0] == path_lengths.shape[0]
    assert goals.shape[0] == path_lengths.shape[0]

    paths_dict = {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "timeouts": timeouts,
        "goals": goals,
        "path_lengths": path_lengths,
    }

    os.makedirs(os.path.dirname(args.out_pkl) or ".", exist_ok=True)
    with open(args.out_pkl, "wb") as f:
        pickle.dump(paths_dict, f)

    print(f"\nWrote: {args.out_pkl}")
    print(f"E={E}, Tmax={Tmax}, K={K}, Dtok={Dtok}, A={A}")
    print("observations:", observations.shape)
    print("actions:", actions.shape)
    print("rewards:", rewards.shape)
    print("terminals:", terminals.shape)
    print("timeouts:", timeouts.shape)
    print("goals:", goals.shape)
    print("path_lengths:", path_lengths.shape, "sum:", int(path_lengths.sum()))

    # ---- extra stats dump ----
    print("\n[STATS]")
    _print_array_stats("observations", observations, per_dim=True, dim_last=Dtok)
    _print_array_stats("goals", goals, per_dim=True, dim_last=Dtok)
    _print_array_stats("actions", actions, per_dim=True, dim_last=A)
    _print_array_stats("rewards", rewards, per_dim=False)
    _print_array_stats("terminals", terminals, per_dim=False)
    _print_array_stats("timeouts", timeouts, per_dim=False)
    _print_array_stats("path_lengths", path_lengths, per_dim=False)


if __name__ == "__main__":
    main()
