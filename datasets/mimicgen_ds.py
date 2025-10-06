# datasets/mimicgen_preproc_ds.py
import os, json, random
from typing import Optional, Tuple, List, Dict
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, List, Dict, Sequence

def _resize_depth_nn(depth_hw: np.ndarray, out_wh: Tuple[int,int]) -> np.ndarray:
    W, H = out_wh
    return np.array(
        Image.fromarray(depth_hw.astype(np.float32), mode="F").resize((W, H), Image.NEAREST),
        dtype=np.float32
    )

def _maybe_scale_K(K: np.ndarray, from_wh: Tuple[int,int], to_wh: Tuple[int,int]) -> np.ndarray:
    W0, H0 = map(float, from_wh); W, H = map(float, to_wh)
    if (W, H) == (W0, H0):
        return K.astype(np.float32)
    sx, sy = W / W0, H / H0
    KK = K.copy().astype(np.float32)
    KK[0,0] *= sx; KK[1,1] *= sy; KK[0,2] *= sx; KK[1,2] *= sy
    return KK

import json, os
import h5py
import numpy as np

class MimicGenRGBD(Dataset):
    def __init__(self,
                 root: str,
                 cams=None,                 # list or single str; default: use meta’s default camera
                 mode: str = "train",
                 image_size: Optional[int] = None,
                 seed: int = 0,
                 verbose: bool = True):
        self.root = root
        self.mode = mode
        self.image_size = int(image_size) if image_size is not None else None
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        # ---- load index.json ----
        index_path = os.path.join(root, "index.json")
        with open(index_path, "r") as f:
            j = json.load(f)
        meta = j["meta"]
        items_all = j["items"]

        # normalize cams arg
        if cams is None:
            # prefer the new multi-cam meta; fallback to legacy single camera
            if "cameras" in meta and isinstance(meta["cameras"], dict) and len(meta["cameras"]) > 0:
                target_cam = sorted(meta["cameras"].keys())[0]
            else:
                target_cam = meta.get("camera", "agentview")
            cams = [target_cam]
        elif isinstance(cams, str):
            cams = [cams]
        cams = list(cams)
        assert len(cams) == 1, "Current implementation expects a single camera; pass a one-element list."
        self.cam = cams[0]

        # ---- read intrinsics + size + near/far (new format → legacy → HDF5 fallback) ----
        cam_blk = {}
        if "cameras" in meta and isinstance(meta["cameras"], dict):
            cam_blk = meta["cameras"].get(self.cam, {})

        K_list = cam_blk.get("K", meta.get("K", None))
        W = cam_blk.get("W", meta.get("W", None))
        H = cam_blk.get("H", meta.get("H", None))
        near = cam_blk.get("near", meta.get("near", None))
        far  = cam_blk.get("far",  meta.get("far",  None))

        if K_list is None or W is None or H is None:
            # try to fetch from source HDF5 (if available)
            src = meta.get("source_h5", None)
            if src and os.path.isfile(src):
                with h5py.File(src, "r") as h5:
                    g = h5[f"meta/cameras/{self.cam}"]
                    K_list = np.array(g["K"]).tolist()
                    W = int(g.attrs["width"])
                    H = int(g.attrs["height"])
                    if near is None and "near" in g.attrs:
                        near = float(g.attrs["near"])
                    if far is None and "far" in g.attrs:
                        far = float(g.attrs["far"])
            else:
                # last resort: infer W,H from the first RGB file for this cam; make a sane K
                sample = next((it for it in items_all if it.get("cam") == self.cam and "rgb" in it), None)
                if sample:
                    rp = os.path.join(root, sample["rgb"])
                    from PIL import Image
                    im = Image.open(rp)
                    W, H = im.size
                else:
                    raise RuntimeError("Could not infer image size or intrinsics; no source_h5 and no sample RGB found.")
                cx = (W - 1) * 0.5
                cy = (H - 1) * 0.5
                fx = fy = max(W, H) * 1.0  # crude but stable fallback
                K_list = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

        self._K_native = np.array(K_list, dtype=np.float32)
        # output size
        if self.image_size is None:
            self.W, self.H = int(W), int(H)
        else:
            self.W = self.H = self.image_size

        # scale K if we resize
        self.K = _maybe_scale_K(self._K_native, (int(W), int(H)), (self.W, self.H))
        self.K = torch.from_numpy(self.K)

        # depth range
        if near is None or far is None:
            # estimate robustly from files if needed (optional: implement if you want)
            near, far = 0.1, 10.0
        self.depth_range = (float(near), float(far))

        # remember keys for reading
        self.rgb_key = "rgb"
        self.dep_key = "depth"

        # ---- filter items by camera and split ----
        items_cam = [it for it in items_all if it.get("cam") == self.cam]
        if not items_cam:
            raise RuntimeError(f"No items in index for camera '{self.cam}'. Available cams in items: "
                               f"{sorted(set(it.get('cam') for it in items_all))}")

        # deterministic split 80/10/10 over the filtered list
        items_cam.sort(key=lambda it: (it["ep"], int(it["t"])))
        n = len(items_cam)
        a, b = int(0.8 * n), int(0.9 * n)
        if mode == "train":
            self.items = items_cam[:a]
        elif mode.startswith("val"):
            self.items = items_cam[a:b]
        else:
            self.items = items_cam[b:]

        self.total_frames_all = n
        self.total_frames = len(self.items)

        if verbose:
            split_name = "train" if mode == "train" else "val" if mode.startswith("val") else "test"
            print("[MimicGenRGBD] Loaded dataset (files)")
            print(f"  root          : {self.root}")
            print(f"  camera        : {self.cam}")
            print(f"  native size   : {int(self._K_native[0,2]*2+1)}x{int(self._K_native[1,2]*2+1)} (HDF5/meta W/H={W}x{H})")
            print(f"  output size   : {self.W}x{self.H}")
            print(f"  frames (all)  : {self.total_frames_all}")
            print(f"  frames ({split_name}): {self.total_frames}")
            n_, f_ = self.depth_range
            print(f"  depth range   : near={n_:.4f}  far={f_:.4f}  (meters, cam-Z)")

    def __len__(self) -> int:
        return len(self.items)

    def get_intrinsics(self, idx: int = 0) -> Dict[str, float]:
        # intrinsics are global for this dataset (per camera); idx is ignored
        K = self.K.numpy() if isinstance(self.K, torch.Tensor) else np.asarray(self.K)
        return {
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
        }
    def __getitem__(self, i: int):
        rec = self.items[i]  # {ep, t, cam, rgb?, depth?}
        # RGB
        if "rgb" in rec:
            rp = os.path.join(self.root, rec["rgb"])
            im = Image.open(rp).convert("RGB")
            if im.size != (self.W, self.H):
                im = im.resize((self.W, self.H), Image.BILINEAR)
            rgb = np.asarray(im, dtype=np.uint8)
        else:
            rgb = np.zeros((self.H, self.W, 3), np.uint8)

        # Depth
        if "depth" in rec:
            dp = os.path.join(self.root, rec["depth"])
            dep = np.load(dp).astype(np.float32)
            if dep.shape != (self.H, self.W):
                dep = np.array(Image.fromarray(dep, mode="F").resize((self.W, self.H), Image.NEAREST), dtype=np.float32)
        else:
            dep = np.zeros((self.H, self.W), np.float32)

        # clamp
        near, far = self.depth_range
        dep = np.where(np.isfinite(dep), dep, far)
        dep = np.clip(dep, near, far).astype(np.float32)

        x = np.zeros((4, self.H, self.W), np.float32)
        x[0:3] = rgb.astype(np.float32).transpose(2, 0, 1)
        x[3] = dep
        return torch.from_numpy(x), torch.tensor(int(i), dtype=torch.long)
