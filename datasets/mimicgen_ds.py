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

class MimicGenRGBD(Dataset):
    def __init__(self,
                 root: str,
                 mode: str = "train",
                 sample_length: int = 1,
                 image_size: Optional[int] = None,
                 seed: int = 0,
                 verbose: bool = True,
                 cams: Optional[Sequence[str]] = None):   # <--- NEW
        assert sample_length == 1
        self.root = root
        self.mode = mode
        self.image_size = int(image_size) if image_size is not None else None
        self._rng = random.Random(seed)

        # ---- load index ----
        index_path = os.path.join(root, "index.json")
        with open(index_path, "r") as f:
            idx = json.load(f)
        self.meta  = idx["meta"]
        items: List[Dict] = idx["items"]

        # available cameras in this preproc
        avail_cams = sorted({it.get("cam") for it in items})
        if cams is None:
            cams = avail_cams
        elif isinstance(cams, str):
            cams = [cams]
        # validate
        missing = [c for c in cams if c not in avail_cams]
        if missing:
            raise RuntimeError(f"Requested cams {missing} not in available {avail_cams}")

        self.cams = list(cams)

        # keep only frames that match requested cams and have at least RGB or depth
        items = [r for r in items if r.get("cam") in self.cams and (("rgb" in r) or ("depth" in r))]
        if not items:
            raise RuntimeError(f"No usable frames for cams={self.cams} in {index_path}")

        # split 80/10/10 over frames
        n = len(items)
        a, b = int(0.8 * n), int(0.9 * n)
        if mode == "train":
            self.items = items[:a]
        elif mode.startswith("val"):
            self.items = items[a:b]
        else:
            self.items = items[b:]

        # sizes + intrinsics (from meta; same K for requested size)
        W0, H0 = int(self.meta["W"]), int(self.meta["H"])
        self.W, self.H = (W0, H0) if self.image_size is None else (self.image_size, self.image_size)
        self.K_native = np.array(self.meta["K"], dtype=np.float32)
        self.K = _maybe_scale_K(self.K_native, (W0, H0), (self.W, self.H))
        self.K_torch = torch.from_numpy(self.K)

        # depth range
        self.depth_range = (
            float(self.meta["near"]) if self.meta.get("near") is not None else 0.1,
            float(self.meta["far"])  if self.meta.get("far")  is not None else 10.0,
        )

        self.total_frames_all = n
        self.total_frames = len(self.items)

        if verbose:
            split = "train" if mode == "train" else "val" if mode.startswith("val") else "test"
            # per-camera counts (post split)
            per_cam = {}
            for it in self.items:
                per_cam[it["cam"]] = per_cam.get(it["cam"], 0) + 1

            print("[MimicGenRGBDPreproc] Loaded preprocessed dataset")
            print(f"  root            : {self.root}")
            print(f"  cameras (avail) : {avail_cams}")
            print(f"  cameras (used)  : {self.cams}   counts={per_cam}")
            print(f"  native size     : {W0}x{H0}")
            print(f"  output size     : {self.W}x{self.H}")
            print(f"  frames (all)    : {self.total_frames_all}")
            print(f"  frames ({split}) : {self.total_frames}")
            n_, f_ = self.depth_range
            print(f"  depth range     : near={n_:.4f}  far={f_:.4f}")

    def __len__(self) -> int:
        return len(self.items)

    def get_intrinsics(self, idx: int) -> Dict[str, float]:
        K = self.K_torch.numpy()
        return {"fx": float(K[0,0]), "fy": float(K[1,1]), "cx": float(K[0,2]), "cy": float(K[1,2])}

    def __getitem__(self, i: int):
        rec = self.items[i]

        # --- RGB ---
        rgb = None
        if "rgb" in rec:
            rp = os.path.join(self.root, rec["rgb"])
            im = Image.open(rp).convert("RGB")
            if im.size != (self.W, self.H):
                im = im.resize((self.W, self.H), Image.BILINEAR)
            rgb = np.asarray(im, dtype=np.uint8)

        # --- Depth (mmap for speed/low RAM) ---
        dep = None
        if "depth" in rec:
            dp = os.path.join(self.root, rec["depth"])
            dep = np.load(dp, mmap_mode="r").astype(np.float32)  # [H,W]
            if dep.shape != (self.H, self.W):
                dep = _resize_depth_nn(dep, (self.W, self.H))

        # fallbacks / packing
        if rgb is None:
            rgb = np.zeros((self.H, self.W, 3), np.uint8)
        if dep is None:
            dep = np.zeros((self.H, self.W), np.float32)

        # sanitize + clamp
        dep = np.where(np.isfinite(dep), dep, np.inf).astype(np.float32)
        near, far = self.depth_range
        dep = np.where(np.isfinite(dep), dep, far)
        dep = np.clip(dep, near, far).astype(np.float32)

        x = np.zeros((4, self.H, self.W), np.float32)
        x[0:3] = rgb.astype(np.float32).transpose(2, 0, 1)
        x[3] = dep

        return torch.from_numpy(x), torch.tensor(int(i), dtype=torch.long)
