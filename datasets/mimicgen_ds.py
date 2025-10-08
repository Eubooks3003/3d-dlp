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
    """
    Multi-camera RGB-D loader that preserves model compatibility by returning a single-view tensor [4,H,W].
    Ways to leverage multiple cameras:
      - multiview_mode="flatten" (default): every (ep,t,cam) is a separate sample.
      - multiview_mode="select" : build joint (ep,t) across cams, then pick one cam per sample
                                  via select_policy in {"first","round_robin","random"}.
    Returns: (x, idx) where x is float32 [4,H,W], idx is a LongTensor scalar.
    """
    def __init__(self,
                 root: str,
                 cams=None,                      # str or list[str]
                 mode: str = "train",
                 image_size: Optional[int] = None,
                 seed: int = 0,
                 verbose: bool = True,
                 multiview_mode: str = "flatten",   # "flatten" | "select"
                 select_policy: str = "first"       # "first" | "round_robin" | "random"
                 ):
        self.root = root
        self.mode = mode
        self.image_size = int(image_size) if image_size is not None else None
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        assert multiview_mode in ("flatten", "select")
        assert select_policy in ("first", "round_robin", "random")
        self.multiview_mode = multiview_mode
        self.select_policy = select_policy

        # -------- helpers to normalize index.json schemas --------
        def _get_cam(it):   # camera id string
            return it.get("cam", it.get("camera"))
        def _get_time(it):  # integer timestep
            return int(it.get("t", it.get("time")))
        self._get_cam = _get_cam
        self._get_time = _get_time

        # -------- load index.json --------
        index_path = os.path.join(root, "index.json")
        with open(index_path, "r") as f:
            j = json.load(f)
        meta = j["meta"]
        items_all = j["items"]

        # -------- discover available cameras & choose cams --------
        available_cams = sorted({ _get_cam(it) for it in items_all if _get_cam(it) is not None })
        if cams is None:
            if "cameras" in meta and isinstance(meta["cameras"], dict) and len(meta["cameras"]) > 0:
                cams = sorted(meta["cameras"].keys())
            else:
                cams = [meta.get("camera", "agentview")]
        elif isinstance(cams, str):
            cams = [cams]
        cams = [c for c in cams if c in available_cams]
        if not cams:
            raise RuntimeError(f"No items found for requested cameras. Available: {available_cams}")
        self.cams = cams

        # -------- per-cam native info (K,W,H,near,far) --------
        self._native = {}
        for cam in self.cams:
            cam_blk = {}
            if "cameras" in meta and isinstance(meta["cameras"], dict):
                cam_blk = meta["cameras"].get(cam, {})

            K_list = cam_blk.get("K", meta.get("K", None))
            W = cam_blk.get("W", meta.get("W", None))
            H = cam_blk.get("H", meta.get("H", None))
            near = cam_blk.get("near", meta.get("near", None))
            far  = cam_blk.get("far",  meta.get("far",  None))

            if K_list is None or W is None or H is None:
                src = meta.get("source_h5", None)
                if src and os.path.isfile(src):
                    with h5py.File(src, "r") as h5:
                        g = h5[f"meta/cameras/{cam}"]
                        K_list = np.array(g["K"]).tolist()
                        W = int(g.attrs["width"]); H = int(g.attrs["height"])
                        if near is None and "near" in g.attrs: near = float(g.attrs["near"])
                        if far  is None and "far"  in g.attrs: far  = float(g.attrs["far"])
                else:
                    sample = next((it for it in items_all if _get_cam(it) == cam and "rgb" in it), None)
                    if sample is None:
                        raise RuntimeError(f"[{cam}] Cannot infer W,H or K (no source_h5 and no sample RGB).")
                    im = Image.open(os.path.join(root, sample["rgb"]))
                    W, H = im.size
                    cx = (W - 1) * 0.5; cy = (H - 1) * 0.5
                    fx = fy = float(max(W, H))
                    K_list = [[fx,0,cx],[0,fy,cy],[0,0,1]]

            if near is None or far is None:
                near, far = 0.1, 10.0

            self._native[cam] = {
                "K": np.array(K_list, dtype=np.float32),
                "W": int(W), "H": int(H),
                "near": float(near), "far": float(far),
            }

        # -------- output size (shared across cams) --------
        if self.image_size is None:
            W0 = self._native[self.cams[0]]["W"]; H0 = self._native[self.cams[0]]["H"]
            self.W, self.H = int(W0), int(H0)
        else:
            self.W = self.H = int(self.image_size)

        # -------- per-cam scaled intrinsics (not returned, but handy) --------
        self.Ks = {}
        for cam in self.cams:
            nat = self._native[cam]
            self.Ks[cam] = torch.from_numpy(
                _maybe_scale_K(nat["K"], (nat["W"], nat["H"]), (self.W, self.H)).astype(np.float32)
            )

        # -------- build per-cam maps: (ep,t) -> record --------
        self.cam_maps = {}
        for cam in self.cams:
            lst = [it for it in items_all if _get_cam(it) == cam]
            self.cam_maps[cam] = { (it["ep"], _get_time(it)): it for it in lst }

        # -------- construct dataset index --------
        if self.multiview_mode == "flatten":
            # union over cams → make a single list of (ep,t,cam) tuples that actually exist
            triples = []
            for cam, m in self.cam_maps.items():
                for (ep, t) in m.keys():
                    triples.append((ep, t, cam))
            triples.sort(key=lambda x: (x[0], x[1], x[2]))  # (ep,t,cam)
            n = len(triples)
            a, b = int(0.8*n), int(0.9*n)
            if mode == "train": chosen = triples[:a]
            elif mode.startswith("val"): chosen = triples[a:b]
            else: chosen = triples[b:]
            if len(chosen) == 0 and n > 0:
                chosen = triples  # fallback so split isn't empty
            self.samples = [((ep, t), cam) for (ep, t, cam) in chosen]
            if verbose:
                per_cam = {cam: sum(1 for (_,c) in self.samples if c == cam) for cam in self.cams}
                print(f"[MimicGenRGBD] multiview_mode=flatten | cams={self.cams} | samples={len(self.samples)} | per-cam={per_cam}")
        else:
            # "select": intersect keys across cams, keep (ep,t), choose a cam per sample by policy
            key_sets = [ set((ep, self._get_time(it)) for (ep,_t), it in { (k):v for k,v in self.cam_maps[cam].items() }.items()) for cam in self.cams ]
            # The above line is a bit contorted; simpler is to rebuild using items_all again:
            joint = sorted(set.intersection(*[ set(self.cam_maps[cam].keys()) for cam in self.cams ]))
            n = len(joint)
            a, b = int(0.8*n), int(0.9*n)
            if mode == "train": keys = joint[:a]
            elif mode.startswith("val"): keys = joint[a:b]
            else: keys = joint[b:]
            if len(keys) == 0 and n > 0:
                keys = joint
            self.keys = keys
            if verbose:
                print(f"[MimicGenRGBD] multiview_mode=select | cams={self.cams} | items={len(self.keys)}")

    def __len__(self) -> int:
        return len(self.samples) if self.multiview_mode == "flatten" else len(self.keys)

    def get_intrinsics(self, cam: Optional[str] = None) -> Dict[str, float]:
        cam = self.cams[0] if (cam is None) else cam
        K = self.Ks[cam]
        K = K.numpy() if isinstance(K, torch.Tensor) else np.asarray(K)
        return {"fx": float(K[0,0]), "fy": float(K[1,1]), "cx": float(K[0,2]), "cy": float(K[1,2])}

    def _load_rgbd(self, rec: Dict, cam: str) -> torch.Tensor:
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
                dep = _resize_depth_nn(dep, (self.W, self.H))
        else:
            dep = np.zeros((self.H, self.W), np.float32)
        # clamp to per-cam range
        near, far = self._native[cam]["near"], self._native[cam]["far"]
        dep = np.where(np.isfinite(dep), dep, far)
        dep = np.clip(dep, near, far).astype(np.float32)
        # pack RGBD
        x = np.zeros((4, self.H, self.W), np.float32)
        x[0:3] = rgb.astype(np.float32).transpose(2, 0, 1)
        x[3] = dep
        return torch.from_numpy(x)

    def __getitem__(self, i: int):
        if self.multiview_mode == "flatten":
            (ep_t, cam) = self.samples[i]
            rec = self.cam_maps[cam][ep_t]
            x = self._load_rgbd(rec, cam)                    # [4,H,W]
            return x, torch.tensor(i, dtype=torch.long)
        else:
            key = self.keys[i]  # (ep,t)
            if self.select_policy == "first":
                cam = self.cams[0]
            elif self.select_policy == "round_robin":
                cam = self.cams[i % len(self.cams)]
            else:  # "random"
                cam = self._rng.choice(self.cams)
            rec = self.cam_maps[cam][key]
            x = self._load_rgbd(rec, cam)                    # [4,H,W]
            return x, torch.tensor(i, dtype=torch.long)
