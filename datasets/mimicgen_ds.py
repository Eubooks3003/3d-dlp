# datasets/mimicgen_ds.py
import os, json, math, random
from typing import Optional, Tuple, List, Dict, Sequence
import numpy as np
from PIL import Image
import h5py
import torch
from torch.utils.data import Dataset


def _resize_depth_nn(depth_hw: np.ndarray, out_wh: Tuple[int, int]) -> np.ndarray:
    W, H = out_wh
    return np.array(
        Image.fromarray(depth_hw.astype(np.float32), mode="F").resize((W, H), Image.NEAREST),
        dtype=np.float32,
    )


def _squeeze_hw(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d)
    if d.ndim == 3:
        d = np.squeeze(d)
    if d.ndim != 2:
        d = d.reshape(d.shape[-2], d.shape[-1])
    return d.astype(np.float32)


def _list_episodes(h5: h5py.File) -> List[str]:
    # ep names like ep_000000...
    eps = sorted(h5["data"].keys(), key=lambda k: int(k.split("_")[-1]))
    return eps


def _episode_len(h5: h5py.File, ep: str, cam: str) -> int:
    # fall back to any obs key if needed; use image length
    base = f"data/{ep}/obs"
    rgb_key = f"{cam}_image"
    if rgb_key in h5[base]:
        return h5[base][rgb_key].shape[0]
    # otherwise try depth
    d_key = f"{cam}_depth"
    if d_key in h5[base]:
        return h5[base][d_key].shape[0]
    # last resort: actions/dones length
    return int(h5[f"data/{ep}/dones"].shape[0])


def _load_K(h5: h5py.File, cam: str) -> Tuple[np.ndarray, int, int, Optional[float], Optional[float]]:
    g = h5[f"meta/cameras/{cam}"]
    K = np.array(g["K"], dtype=np.float32)
    w = int(g.attrs["width"])
    h = int(g.attrs["height"])
    near = g.attrs.get("near", None)
    far = g.attrs.get("far", None)
    if near is not None:
        near = float(near)
    if far is not None:
        far = float(far)
    return K, w, h, near, far


class MimicGenRGBD(Dataset):
    """
    Reads RGBD from a MimicGen/robomimic HDF5 created by your exporter.

    Structure expected:
      /data/ep_xxxxxxx/obs/{cam}_image    -> [T, H, W, 3] uint8 (or float)
      /data/ep_xxxxxxx/obs/{cam}_depth    -> [T, H, W] (or T,H,W,1) float32 meters (cam-Z)
      /meta/cameras/{cam}/K               -> [3,3] float32
      /meta/cameras/{cam} attrs:
           width, height, [near], [far], percentiles=[1,99], count, basis="percentile_camZ"

    Yields: (x4chw, idx)
      x[0:3] = RGB in 0..255 float32
      x[3]   = depth in meters (cam-Z), clamped to [near, far]
    """

    def __init__(
        self,
        h5_path: str,
        cam: str = "agentview",
        mode: str = "train",
        sample_length: int = 1,
        image_size: Optional[int] = None,
        pct: Tuple[float, float] = (1.0, 99.0),
        stat_ep_limit: int = 64,
        stat_t_per_ep: int = 16,
        stat_px_per_img: int = 8192,
        seed: int = 0,
        verbose: bool = True,
    ):
        assert sample_length == 1, "This loader currently supports single-frame samples."
        self.path = h5_path
        self.cam = cam
        self.mode = mode
        self.sample_length = sample_length
        self.image_size = int(image_size) if image_size is not None else None
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        # Read basic structure once
        with h5py.File(self.path, "r") as h5:
            if "data" not in h5:
                raise RuntimeError(f"Missing /data in {self.path}")
            eps = _list_episodes(h5)
            if not eps:
                raise RuntimeError("No episodes under /data")

            self.num_episodes = len(eps)

            # Verify camera keys exist in first episode
            base0 = f"data/{eps[0]}/obs"
            have_rgb = f"{cam}_image" in h5[base0]
            have_dep = f"{cam}_depth" in h5[base0]
            if not (have_rgb or have_dep):
                # try to infer available cameras
                all_obs_keys = list(h5[base0].keys())
                cams = sorted(
                    set(k.rsplit("_", 1)[0] for k in all_obs_keys if k.endswith("_image") or k.endswith("_depth"))
                )
                raise RuntimeError(f"Camera '{cam}' not found. Available cams: {cams}")
            # define keys EARLY (used by _compute_depth_range)
            self.rgb_key = f"{cam}_image"
            self.dep_key = f"{cam}_depth"
            # Intrinsics + native size + near/far (if present)
            K, W0, H0, near_attr, far_attr = _load_K(h5, cam)

            # Build a flat index of (ep, t)
            items: List[Tuple[str, int]] = []
            for ep in eps:
                T = _episode_len(h5, ep, cam)
                for t in range(T):
                    items.append((ep, t))
            # Split 80/10/10 over frames
            n = len(items)
            a, b = int(0.8 * n), int(0.9 * n)
            if mode == "train":
                self.items = items[:a]
            elif mode.startswith("val"):
                self.items = items[a:b]
            else:
                self.items = items[b:]

            self.total_frames_all = n
            self.total_frames = len(self.items)
            # output resolution
            if self.image_size is None:
                self.W, self.H = W0, H0
            else:
                self.W = self.H = self.image_size

            self._K_native = K.copy()
            self.K = self._maybe_scale_K(K, from_wh=(W0, H0), to_wh=(self.W, self.H))  # [3,3] for ALL items
            self.K = torch.from_numpy(self.K)

            # depth range
            if (near_attr is not None) and (far_attr is not None):
                self.depth_range = (float(near_attr), float(far_attr))
            else:
                self.depth_range = self._compute_depth_range(
                    h5,
                    pct=pct,
                    ep_limit=stat_ep_limit,
                    t_per_ep=stat_t_per_ep,
                    px_per_img=stat_px_per_img,
                )

        # build id -> position mapping for convenience
        self.id2pos: Dict[int, int] = {i: i for i in range(len(self.items))}

        if verbose:
            split_name = "train" if mode == "train" else "val" if mode.startswith("val") else "test"
            print("[MimicGenRGBD] Loaded dataset")
            print(f"  file          : {self.path}")
            print(f"  episodes      : {self.num_episodes}")
            print(f"  camera        : {self.cam}")
            print(f"  native size   : {int(self._K_native[0,2]*2+1)}x{int(self._K_native[1,2]*2+1)} (from HDF5 attrs: {self.W}x{self.H} used)")
            print(f"  output size   : {self.W}x{self.H}")
            print(f"  frames (all)  : {self.total_frames_all}")
            print(f"  frames ({split_name}): {self.total_frames}")
            n, f = self.depth_range
            print(f"  depth range   : near={n:.4f}  far={f:.4f}  (meters, cam-Z)")

    def __repr__(self) -> str:
        n, f = self.depth_range
        return (f"MimicGenRGBD(path='{os.path.basename(self.path)}', cam='{self.cam}', "
                f"mode='{self.mode}', size={self.W}x{self.H}, "
                f"episodes={self.num_episodes}, frames={self.total_frames}/{self.total_frames_all}, "
                f"depth_range=({n:.3f},{f:.3f}))")
    # ---------- sizing / intrinsics ----------

    @staticmethod
    def _maybe_scale_K(K: np.ndarray, from_wh: Tuple[int, int], to_wh: Tuple[int, int]) -> np.ndarray:
        """Scale intrinsics if you resized images from (W0,H0) -> (W,H)."""
        W0, H0 = map(float, from_wh)
        W, H = map(float, to_wh)
        if (W, H) == (W0, H0):
            return K.astype(np.float32)
        sx, sy = W / W0, H / H0
        KK = K.copy().astype(np.float32)
        KK[0, 0] *= sx  # fx
        KK[1, 1] *= sy  # fy
        KK[0, 2] *= sx  # cx
        KK[1, 2] *= sy  # cy
        return KK

    def get_intrinsics(self, idx: int) -> Dict[str, float]:
        K = self.K.numpy()
        return {"fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2])}

    # ---------- stats ----------

    def _compute_depth_range(
        self,
        h5: h5py.File,
        pct: Tuple[float, float],
        ep_limit: int,
        t_per_ep: int,
        px_per_img: int,
    ) -> Tuple[float, float]:
        """Percentiles on valid depth (meters, cam-Z)."""
        eps = _list_episodes(h5)
        self._rng.shuffle(eps)
        eps = eps[: min(ep_limit, len(eps))]
        samples: List[np.ndarray] = []

        for ep in eps:
            base = f"data/{ep}/obs"
            if self.dep_key not in h5[base]:
                continue
            D = h5[base][self.dep_key]
            T = D.shape[0]
            ts = list(range(T))
            self._rng.shuffle(ts)
            ts = ts[: min(t_per_ep, T)]

            for t in ts:
                dep = _squeeze_hw(D[t])
                # resize if needed
                if dep.shape != (self.H, self.W):
                    dep = _resize_depth_nn(dep, (self.W, self.H))
                valid = np.isfinite(dep) & (dep > 0)
                if not np.any(valid):
                    continue
                yy, xx = np.where(valid)
                take = min(px_per_img, yy.size)
                sel = self._np_rng.choice(yy.size, size=take, replace=False)
                samples.append(dep[yy[sel], xx[sel]])

        if not samples:
            return (0.1, 10.0)

        v = np.concatenate(samples, 0).astype(np.float32)
        near = float(np.percentile(v, pct[0]))
        far = float(np.percentile(v, pct[1]))
        if not (near < far):
            near, far = float(np.min(v)), float(np.max(v))
            if near == far:
                near, far = max(1e-3, near * 0.9), far * 1.1
        return (near, far)

    # ---------- dataset API ----------

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        ep, t = self.items[i]

        # open read-only each time to avoid file locking issues
        # if you know the file is closed elsewhere, you can keep it open in __init__
        with h5py.File(self.path, "r", swmr=True, libver="latest") as h5:
            base = f"data/{ep}/obs"
            # RGB
            if self.rgb_key in h5[base]:
                rgb = np.asarray(h5[base][self.rgb_key][t])
            else:
                # create black if missing
                K = self.K.numpy()
                W = int(round(K[0, 2] * 2.0 + 1))
                H = int(round(K[1, 2] * 2.0 + 1))
                rgb = np.zeros((H, W, 3), np.uint8)

            # Depth
            dep = _squeeze_hw(h5[base][self.dep_key][t]) if self.dep_key in h5[base] else None

        # resize
        if (rgb.shape[0] != self.H) or (rgb.shape[1] != self.W):
            rgb = np.asarray(Image.fromarray(rgb.astype(np.uint8), mode="RGB").resize((self.W, self.H), Image.BILINEAR))
        if dep is not None and dep.shape != (self.H, self.W):
            dep = _resize_depth_nn(dep, (self.W, self.H))

        # sanitize depth and clamp to dataset near/far
        if dep is None:
            dep = np.zeros((self.H, self.W), np.float32)
        dep = np.where(np.isfinite(dep), dep, np.inf).astype(np.float32)
        near, far = self.depth_range
        dep = np.where(np.isfinite(dep), dep, far)
        dep = np.clip(dep, near, far).astype(np.float32)

        # pack (4, H, W): RGB in 0..255 (float32), depth in meters
        x = np.zeros((4, self.H, self.W), np.float32)
        x[0:3] = rgb.astype(np.float32).transpose(2, 0, 1)
        x[3] = dep

        return torch.from_numpy(x), torch.tensor(int(i), dtype=torch.long)
