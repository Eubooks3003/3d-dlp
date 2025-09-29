# datasets/blender_ds.py
import os, glob, json, random
from typing import Optional, Tuple, List, Dict
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def _K_from_angles(angle_x: float, angle_y: float, W: int, H: int) -> np.ndarray:
    """Build K from Blender's horizontal/vertical FOV angles and target size."""
    fx = 0.5 * W / np.tan(float(angle_x) * 0.5)
    fy = 0.5 * H / np.tan(float(angle_y) * 0.5)
    cx = (W - 1) * 0.5
    cy = (H - 1) * 0.5
    return np.array([[fx, 0, cx],
                     [0,  fy, cy],
                     [0,   0,  1]], dtype=np.float32)


def _resize_depth_nn(depth_hw: np.ndarray, out_wh: Tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor resize for depth maps (preserves discontinuities)."""
    W, H = out_wh
    return np.array(
        Image.fromarray(depth_hw.astype(np.float32), mode="F")
             .resize((W, H), Image.NEAREST),
        dtype=np.float32
    )


def _ray_to_camZ(depth_ray: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Convert Blender Z pass (ray distance) to camera-Z (planar) using intrinsics:
        Z = d_ray / sqrt( ((u-cx)/fx)^2 + ((v-cy)/fy)^2 + 1 )
    Keeps +inf as background.
    """
    H, W = depth_ray.shape
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    u = np.arange(W, dtype=np.float32)[None, :].repeat(H, 0)
    v = np.arange(H, dtype=np.float32)[:, None].repeat(W, 1)
    x = (u - cx) / fx
    y = (v - cy) / fy
    inv_len = 1.0 / np.sqrt(x * x + y * y + 1.0)  # 1 / ||dir||
    Z = depth_ray * inv_len
    Z = np.where(np.isfinite(depth_ray), Z, np.inf).astype(np.float32)
    return Z


class BlenderRGBD(Dataset):
    """
    Expects:
      root/
        rgb/rgb_00000.png
        depth_raw/depth_raw_00000.npy     # Blender Z-pass (ray distance, meters)
        intrinsics/intrinsics_00000.json  # contains angle_x/angle_y, etc
        [dataset_meta.json]                # auto-created

    Yields: (x4chw, idx)
      - x[0:3] = RGB in 0..255 (float32)
      - x[3]   = depth in meters, **camera-Z** (float32)

    Attributes:
      - .K  [N,3,3] intrinsics (built from angles at image_size)
      - .depth_range = (near, far)  # robust percentiles on camera-Z
      - .depth_encoding = "metric_meters_camZ"
      - .id2pos: global id -> dataset position index
      - get_intrinsics(i) -> dict(fx,fy,cx,cy)
    """
    def __init__(self,
                 root: str,
                 mode: str = "train",
                 sample_length: int = 1,
                 image_size: Optional[int] = None,
                 pct: Tuple[float, float] = (1.0, 99.0),
                 cache_meta: bool = True,
                 stat_img_limit: int = 512,
                 stat_px_per_img: int = 8192,
                 clamp_far_eps: float = 1e-3,
                 preclip_near_m: float = 0.0,
                 preclip_far_m: float = 100.0,
                    hist_bins: int = 40,
                 seed: int = 0):
        assert sample_length == 1
        self.root = root
        self.mode = mode
        self.image_size = image_size
        self.preclip_near_m = preclip_near_m
        self.preclip_far_m  = preclip_far_m
        self.depth_encoding = "metric_meters_camZ"
        self.hist_bins = int(hist_bins)
        self._rng = random.Random(seed)

        rgbs = sorted(glob.glob(os.path.join(root, "rgb", "rgb_*.png")))
        if not rgbs:
            raise RuntimeError(f"No RGB files under {os.path.join(root, 'rgb')}")

        def _id(p): return int(os.path.basename(p).split("_")[-1].split(".")[0])

        items: List[tuple] = []
        for rp in rgbs:
            i = _id(rp)
            dp = os.path.join(root, "depth_raw", f"depth_raw_{i:05d}.npy")
            jp = os.path.join(root, "intrinsics", f"intrinsics_{i:05d}.json")
            if os.path.isfile(dp) and os.path.isfile(jp):
                items.append((i, rp, dp, jp))
        items.sort(key=lambda t: t[0])
        if not items:
            raise RuntimeError("Found RGBs but no matching depth_raw/intrinsics.")

        # split 80/10/10
        n = len(items)
        a, b = int(0.8 * n), int(0.9 * n)
        self.items = items[:a] if mode == "train" else items[a:b] if mode.startswith("val") else items[b:]

        # map from global id -> position within this split
        self.id2pos: Dict[int, int] = {int(t[0]): j for j, t in enumerate(self.items)}

        # output size
        w0, h0 = Image.open(self.items[0][1]).size
        if image_size is None:
            self.W, self.H = w0, h0
        else:
            self.W = self.H = int(image_size)

        # intrinsics per item (from angles at target size)
        Ks = np.zeros((len(self.items), 3, 3), np.float32)
        self.angles: List[Tuple[float, float]] = []
        for j, (_, _, _, jp) in enumerate(self.items):
            with open(jp, "r") as f:
                meta = json.load(f)
            ax = float(meta["angle_x"]); ay = float(meta["angle_y"])
            Ks[j] = _K_from_angles(ax, ay, self.W, self.H)
            self.angles.append((ax, ay))
        self.K = torch.from_numpy(Ks)

        # dataset-level near/far (robust) computed on **camera-Z**
        self.meta_path = os.path.join(root, "dataset_meta.json")

        print("[BlenderRGBD] estimating depth range (camera-Z) ...")
        self.depth_range = self._load_or_compute_range(
            pct, cache_meta, stat_img_limit, stat_px_per_img
        )

    def __len__(self) -> int:
        return len(self.items)

    def get_intrinsics(self, idx: int) -> Dict[str, float]:
        K = self.K[idx].numpy()
        return {"fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2])}

    def __getitem__(self, i: int):
        gid, rp, dp, _ = self.items[i]
        # RGB
        im = Image.open(rp).convert("RGB")
        if im.size != (self.W, self.H):
            im = im.resize((self.W, self.H), Image.BILINEAR)
        rgb = np.asarray(im, dtype=np.uint8).astype(np.float32)  # HxWx3 in 0..255

        # Depth: load ray distance -> resize (NN) -> convert to camera-Z
        d_ray = np.load(dp).astype(np.float32)  # meters, +inf for background
        d_ray = np.nan_to_num(d_ray, nan=np.inf, posinf=np.inf, neginf=np.inf)
        if d_ray.shape[::-1] != (self.W, self.H):
            d_ray = _resize_depth_nn(d_ray, (self.W, self.H))
        # sanity: ensure exact match now
        assert d_ray.shape == (self.H, self.W), f"Depth size {d_ray.shape} != {(self.H, self.W)}"

        K = self.K[i].numpy()
        d_camZ = _ray_to_camZ(d_ray, K)

        d_camZ = self._preclip_camZ(d_camZ)

        # sanitize for downstream: inf -> far, clamp to [near, far]
        near, far = self.depth_range
        d_camZ = np.where(np.isfinite(d_camZ), d_camZ, far).astype(np.float32)
        d_camZ = np.clip(d_camZ, near, far)

        x = np.zeros((4, self.H, self.W), np.float32)
        x[0:3] = rgb.transpose(2, 0, 1)
        x[3] = d_camZ
        return torch.from_numpy(x), torch.tensor(int(gid), dtype=torch.long)

    def _preclip_camZ(self, z_cam: np.ndarray) -> np.ndarray:
        out = z_cam
        finite = np.isfinite(out) & (out > 0)
        if self.preclip_far_m is not None:
            out = np.minimum(out, float(self.preclip_far_m), dtype=np.float32)
        if self.preclip_near_m is not None:
            out = np.maximum(out, float(self.preclip_near_m), dtype=np.float32)
        return out

    # ---- helpers ----
    def _load_or_compute_range(self,
                               pct: Tuple[float, float],
                               cache: bool,
                               stat_img_limit: int,
                               stat_px_per_img: int) -> Tuple[float, float]:

        # Sample a subset of images and pixels
        idxs = list(range(len(self.items)))
        self._rng.shuffle(idxs)
        idxs = idxs[:min(stat_img_limit, len(idxs))]

        # collect samples
        vals_post = []   # after preclip (used for near/far)
        vals_pre  = []   # BEFORE preclip (for histogram/inspection)

        rng_np = np.random.default_rng(1234)
        for j in idxs:
            _, _, dp, _ = self.items[j]
            d_ray = np.load(dp).astype(np.float32)
            d_ray = np.nan_to_num(d_ray, nan=np.inf, posinf=np.inf, neginf=np.inf)
            if d_ray.shape[::-1] != (self.W, self.H):
                d_ray = _resize_depth_nn(d_ray, (self.W, self.H))

            K = self.K[j].numpy()
            d_camZ_pre = _ray_to_camZ(d_ray, K)   # meters, BEFORE preclip

            # sample valid PRE values for histogram
            vmask_pre = np.isfinite(d_camZ_pre) & (d_camZ_pre > 0)
            if np.any(vmask_pre):
                yy, xx = np.where(vmask_pre)
                take = min(stat_px_per_img, yy.size)
                sel = rng_np.choice(yy.size, size=take, replace=False)
                vals_pre.append(d_camZ_pre[yy[sel], xx[sel]])

            # apply preclip for near/far statistics
            d_camZ = self._preclip_camZ(d_camZ_pre)

            vmask = np.isfinite(d_camZ) & (d_camZ > 0)
            if not np.any(vmask):
                continue
            yy, xx = np.where(vmask)
            take = min(stat_px_per_img, yy.size)
            sel = rng_np.choice(yy.size, size=take, replace=False)
            vals_post.append(d_camZ[yy[sel], xx[sel]])

        # ---- PRINT HISTOGRAM (PRE-CLIP) ----

        # vpre = np.concatenate(vals_pre, axis=0).astype(np.float64)
        # # robust display range (ignore extreme 0.1% tails so the bars are readable)
        # lo = float(np.percentile(vpre, 0.1))
        # hi = float(np.percentile(vpre, 99.9))
        # counts, edges = np.histogram(np.clip(vpre, lo, hi), bins=self.hist_bins, range=(lo, hi))

        # # ascii bars
        # mx = counts.max() if counts.size else 1
        # barW = 60
        # print("\n[BlenderRGBD] Depth histogram (meters) BEFORE preclip:")
        # print(f"  samples={vpre.size:,}  range_used=[{lo:.3f}, {hi:.3f}]  (raw min/max={vpre.min():.3f}/{vpre.max():.3f})")
        # # key percentiles to help pick a cap
        # for p in (0.1, 1, 5, 25, 50, 75, 95, 99, 99.9):
        #     q = float(np.percentile(vpre, p))
        #     print(f"   p{p:>4}: {q:.4f} m")
        # print("  histogram:")
        # for k, c in enumerate(counts):
        #     e0, e1 = edges[k], edges[k+1]
        #     n = int(round((c / mx) * barW))
        #     print(f"   [{e0:7.3f},{e1:7.3f}) | {'#'*n} ({int(c):6d})")
        # print()

        # ---- NEAR/FAR from POST-CLIP samples ----
        if not vals_post:
            near, far = 0.1, 10.0
        else:
            v = np.concatenate(vals_post, axis=0)
            near = float(np.percentile(v, pct[0]))
            far = float(np.percentile(v, pct[1]))
            if not (near < far):
                near, far = float(np.min(v)), float(np.max(v))
                if near == far:
                    near, far = max(1e-3, near * 0.9), far * 1.1

        if cache:
            try:
                with open(self.meta_path, "w") as f:
                    json.dump({"near": near, "far": far,
                               "percentiles": list(pct),
                               "count": len(self.items),
                               "basis": "percentile_camZ"}, f, indent=2)
            except Exception:
                pass
        return (near, far)

    # ---------- PLY helper ----------
    def ply_sample(self, pos: int, trim_p: float = 0.90):
        """
        Returns (rgb, depth_camZ_inf, intr, gid):
          - rgb: HxWx3 in 0..1
          - depth_camZ_inf: HxW in meters (camera-Z). Far slab set to +inf for export.
          - intr: dict with {fx, fy, cx, cy} (same as loader uses)
          - gid: global id
        Depth is recomputed from RAW ray depth with the SAME K and (H,W) the loader uses.
        """
        gid, rp, dp, _ = self.items[pos]

        # RGB at loader size
        im = Image.open(rp).convert("RGB")
        if im.size != (self.W, self.H):
            im = im.resize((self.W, self.H), Image.BILINEAR)
        rgb = np.asarray(im, dtype=np.uint8).astype(np.float32) / 255.0

        # Intrinsics (exactly what the loader uses)
        K = self.K[pos].numpy()
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

        # RAW ray distance -> resize (NN) -> camera-Z with SAME K/size
        d_ray = np.load(dp).astype(np.float32)
        if d_ray.shape != (self.H, self.W):
            d_ray = _resize_depth_nn(d_ray, (self.W, self.H))
        assert d_ray.shape == (self.H, self.W), f"[ply_sample] Depth size {d_ray.shape} != {(self.H, self.W)}"

        u = np.arange(self.W, dtype=np.float32)[None, :].repeat(self.H, 0)
        v = np.arange(self.H, dtype=np.float32)[:, None].repeat(self.W, 1)
        inv_len = 1.0 / np.sqrt(((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2 + 1.0)
        d_camZ = d_ray * inv_len
        d_camZ = np.where(np.isfinite(d_camZ), d_camZ, np.inf).astype(np.float32)

        # Trim only extreme far slab for cleaner clouds (objects remain)
        finite = np.isfinite(d_camZ) & (d_camZ > 0)
        if np.any(finite) and 0.0 < trim_p < 1.0:
            z_cap = np.percentile(d_camZ[finite], 100.0 * trim_p)
            d_camZ[d_camZ > z_cap] = np.inf

        intr = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
        return rgb, d_camZ, intr, int(gid)
