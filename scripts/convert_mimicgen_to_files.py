#!/usr/bin/env python3
import os, json, argparse
from multiprocessing import Pool
import numpy as np
from PIL import Image
import h5py
from tqdm.auto import tqdm

# ---------- small utils ----------
def _squeeze_hw(d):
    d = np.asarray(d)
    if d.ndim == 3: d = np.squeeze(d)
    if d.ndim != 2: d = d.reshape(d.shape[-2], d.shape[-1])
    return d.astype(np.float32)

def _resize_nn(depth, out_wh):
    W, H = out_wh
    return np.array(
        Image.fromarray(depth.astype(np.float32), mode="F").resize((W, H), Image.NEAREST),
        dtype=np.float32,
    )

def _list_eps(h5):
    return sorted(h5["data"].keys(), key=lambda k: int(k.split("_")[-1]))

def _episode_len(h5, ep, cam):
    base = f"data/{ep}/obs"
    k = f"{cam}_image"
    if k in h5[base]: return h5[base][k].shape[0]
    k = f"{cam}_depth"
    if k in h5[base]: return h5[base][k].shape[0]
    return int(h5[f"data/{ep}/dones"].shape[0])

def _load_K(h5, cam):
    g = h5[f"meta/cameras/{cam}"]
    K = np.array(g["K"], dtype=np.float32)
    W = int(g.attrs["width"]); H = int(g.attrs["height"])
    near = g.attrs.get("near", None); far = g.attrs.get("far", None)
    near = float(near) if near is not None else None
    far  = float(far)  if far  is not None else None
    return K, W, H, near, far

def _maybe_scale_K(K, from_wh, to_wh):
    W0, H0 = map(float, from_wh); W, H = map(float, to_wh)
    if (W, H) == (W0, H0): return K.astype(np.float32)
    sx, sy = W / W0, H / H0
    KK = K.copy().astype(np.float32)
    KK[0,0]*=sx; KK[1,1]*=sy; KK[0,2]*=sx; KK[1,2]*=sy
    return KK

def _estimate_near_far(h5, cam, W, H, pct=(1.0, 99.0),
                       ep_limit=64, t_per_ep=16, px_per_img=8192, seed=1234):
    eps = _list_eps(h5)
    if not eps: return 0.1, 10.0
    rng_ep = np.random.default_rng(seed)
    rng_px = np.random.default_rng(seed + 1)
    idxs = np.arange(len(eps)); rng_ep.shuffle(idxs)
    idxs = idxs[:min(ep_limit, len(idxs))]
    dep_key = f"{cam}_depth"
    samples = []
    for k in idxs:
        ep = eps[k]
        base = f"data/{ep}/obs"
        if dep_key not in h5[base]: continue
        D = h5[base][dep_key]
        T = D.shape[0]
        ts = np.arange(T); rng_ep.shuffle(ts)
        ts = ts[:min(t_per_ep, T)]
        for t in ts:
            try:
                dep = _squeeze_hw(D[t][()])
            except Exception:
                continue
            if dep.shape != (H, W):
                dep = _resize_nn(dep, (W, H))
            valid = np.isfinite(dep) & (dep > 0)
            if not np.any(valid): continue
            yy, xx = np.where(valid)
            take = min(px_per_img, yy.size)
            sel = rng_px.choice(yy.size, size=take, replace=False)
            samples.append(dep[yy[sel], xx[sel]])
    if not samples: return 0.1, 10.0
    v = np.concatenate(samples, 0).astype(np.float32)
    near = float(np.percentile(v, pct[0])); far  = float(np.percentile(v, pct[1]))
    if not (near < far):
        near, far = float(np.min(v)), float(np.max(v))
        if near == far: near, far = max(1e-3, near*0.9), far*1.1
    return near, far

def _read_safe(ds, idx, what, ep, cam, bad, retry=1):
    for attempt in range(retry + 1):
        try:
            return ds[idx][()]
        except Exception as e:
            if attempt == retry:
                bad.append({"ep": ep, "t": int(idx), "cam": cam, "key": what, "error": f"{type(e).__name__}: {e}"})
                return None

# ---------- worker ----------
def _worker(args):
    path, cam, out, W, H, start_end = args
    ep, t0, t1 = start_end
    rgb_key = f"{cam}_image"; dep_key = f"{cam}_depth"

    items = []; bad = []
    with h5py.File(path, "r") as h5:
        base = f"data/{ep}/obs"
        has_rgb = rgb_key in h5[base]
        has_dep = dep_key in h5[base]
        RGB = h5[base][rgb_key] if has_rgb else None
        DEP = h5[base][dep_key] if has_dep else None

        for t in range(t0, t1):
            rec = {"ep": ep, "t": t, "cam": cam}
            if has_rgb:
                rgb_raw = _read_safe(RGB, t, "rgb", ep, cam, bad, retry=1)
                if rgb_raw is not None:
                    rgb = np.asarray(rgb_raw)
                    if (rgb.shape[1] != W) or (rgb.shape[0] != H):
                        rgb = np.array(
                            Image.fromarray(rgb.astype(np.uint8), mode="RGB").resize((W, H), Image.BILINEAR),
                            dtype=np.uint8,
                        )
                    rgb_rel = os.path.join("rgb", cam, ep, f"{t:06d}.png")
                    rgb_abs = os.path.join(out, rgb_rel)
                    os.makedirs(os.path.dirname(rgb_abs), exist_ok=True)
                    Image.fromarray(rgb.astype(np.uint8)).save(rgb_abs, compress_level=3)
                    rec["rgb"] = rgb_rel

            if has_dep:
                dep_raw = _read_safe(DEP, t, "depth", ep, cam, bad, retry=1)
                if dep_raw is not None:
                    dep = _squeeze_hw(dep_raw)
                    if dep.shape != (H, W):
                        dep = _resize_nn(dep, (W, H))
                    dep_rel = os.path.join("depth", cam, ep, f"{t:06d}.npy")
                    dep_abs = os.path.join(out, dep_rel)
                    os.makedirs(os.path.dirname(dep_abs), exist_ok=True)
                    np.save(dep_abs, dep, allow_pickle=False)
                    rec["depth"] = dep_rel

            if "rgb" in rec or "depth" in rec:
                items.append(rec)

    return items, (t1 - t0), bad

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Convert MimicGen HDF5 to per-file RGBD with multi-camera and appendable index.")
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cam", nargs="+", required=True, help="one or more camera names, e.g. --cam agentview robot0_eye_in_hand")
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=1024)
    ap.add_argument("--pct-near", type=float, default=1.0)
    ap.add_argument("--pct-far",  type=float, default=99.0)
    ap.add_argument("--bad-log", default="bad_frames.jsonl")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    index_path = os.path.join(args.out, "index.json")
    bad_log_path = os.path.join(args.out, args.bad_log)

    # If index exists, load and normalize structure; else create fresh
    if os.path.isfile(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        # migrate old single-cam meta to new structure
        meta = index.get("meta", {})
        if "cameras" not in meta:
            # old format: single camera fields; wrap into cameras dict if possible
            cam_name = meta.get("camera", "unknown")
            meta = {
                "source_h5": meta.get("source_h5", ""),
                "W": meta.get("W", None),
                "H": meta.get("H", None),
                "cameras": {
                    cam_name: {
                        "W": meta.get("W", None),
                        "H": meta.get("H", None),
                        "K": meta.get("K", None),
                        "near": meta.get("near", None),
                        "far": meta.get("far", None),
                    }
                },
                "episodes": meta.get("episodes", []),
            }
        items = index.get("items", [])
    else:
        meta = {"source_h5": os.path.abspath(args.h5), "W": None, "H": None, "cameras": {}, "episodes": []}
        items = []

    # Scan episodes once (shared across cams)
    with h5py.File(args.h5, "r") as h5:
        eps = _list_eps(h5)
        if not eps:
            raise SystemExit("No episodes in /data")

        # ensure / record episodes in meta (idempotent)
        if not meta["episodes"]:
            for ep in eps:
                meta["episodes"].append({"name": ep, "length": int(_episode_len(h5, ep, args.cam[0]))})

        for cam in args.cam:
            # get K / size / near-far per camera
            K_native, W0, H0, near, far = _load_K(h5, cam)
            if args.image_size is None:
                W, H = W0, H0
                K = K_native
            else:
                W = H = int(args.image_size)
                K = _maybe_scale_K(K_native, (W0, H0), (W, H))

            if (near is None) or (far is None) or not np.isfinite(near) or not np.isfinite(far):
                near, far = _estimate_near_far(
                    h5, cam, W, H, pct=(args.pct_near, args.pct_far),
                    ep_limit=64, t_per_ep=16, px_per_img=8192, seed=1234
                )
                print(f"[info] estimated near/far for '{cam}': {near:.4f}, {far:.4f}")

            # store per-cam meta
            meta["cameras"][cam] = {
                "W": int(W), "H": int(H),
                "K": np.asarray(K, dtype=np.float32).tolist(),
                "near": float(near), "far": float(far),
            }
            # set top-level W/H only if not set (purely informational)
            if meta["W"] is None: meta["W"] = int(W)
            if meta["H"] is None: meta["H"] = int(H)

            # plan work for this cam
            work = []
            total_frames = 0
            for ep in tqdm(eps, desc=f"[{cam}] Scanning", unit="ep"):
                T = _episode_len(h5, ep, cam)
                total_frames += int(T)
                if T == 0: continue
                step = max(1, int(args.chunk_size))
                for t in range(0, T, step):
                    work.append((args.h5, cam, args.out, W, H, (ep, t, min(T, t + step))))

            # convert this cam
            bad_all = []
            if args.workers > 1:
                with Pool(processes=args.workers, maxtasksperchild=64) as pool, \
                     tqdm(total=len(work), desc=f"[{cam}] Converting", unit="chunk") as p_chunks, \
                     tqdm(total=total_frames, desc=f"[{cam}] Frames", unit="f") as p_frames:
                    for part, nframes, bad in pool.imap_unordered(_worker, work):
                        items.extend(part)
                        bad_all.extend(bad)
                        p_chunks.update(1)
                        p_frames.update(nframes)
            else:
                with tqdm(total=len(work), desc=f"[{cam}] Converting", unit="chunk") as p_chunks, \
                     tqdm(total=total_frames, desc=f"[{cam}] Frames", unit="f") as p_frames:
                    for w in work:
                        part, nframes, bad = _worker(w)
                        items.extend(part); bad_all.extend(bad)
                        p_chunks.update(1); p_frames.update(nframes)

            if bad_all:
                os.makedirs(args.out, exist_ok=True)
                with open(bad_log_path, "a") as f:
                    for rec in bad_all:
                        f.write(json.dumps(rec) + "\n")
                print(f"[warn] {cam}: skipped {len(bad_all)} bad frame reads. Appended to {bad_log_path}")

    # write / update index.json
    out_index = {"meta": meta, "items": items}
    with open(index_path, "w") as f:
        json.dump(out_index, f, indent=2)
    print(f"[OK] total frames indexed: {len(items):,}  -> {index_path}")

if __name__ == "__main__":
    main()
