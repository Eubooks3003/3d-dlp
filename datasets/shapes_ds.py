"""
Simple Random Colored Shapes Dataset
"""
# imports
import numpy as np
from skimage.draw import random_shapes
from skimage.draw import disk, rectangle, polygon
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset


def generate_shape_dataset(img_size=64, min_shapes=2, max_shapes=5, min_size=10, max_size=12, allow_overlap=False,
                           num_images=10_000):
    images = []
    for i in tqdm(range(num_images)):
        img, _ = random_shapes((img_size, img_size), min_shapes=min_shapes, max_shapes=max_shapes,
                               intensity_range=((0, 200),), min_size=min_size, max_size=max_size,
                               allow_overlap=allow_overlap, num_trials=100)
        img[:, :, 0][img[:, :, 0] == 255] = 0
        img[:, :, 1][img[:, :, 1] == 255] = 255
        img[:, :, 2][img[:, :, 2] == 255] = 255
        img = img / 255.0
        images.append(img)
    images = np.stack(images, axis=0)  # [num_mages, H, W, 3]
    return images


def generate_shape_dataset_torch(img_size=64, min_shapes=2, max_shapes=5, min_size=11, max_size=13, allow_overlap=False,
                                 num_images=10_000):
    images = generate_shape_dataset(img_size=img_size, min_shapes=min_shapes, max_shapes=max_shapes, min_size=min_size,
                                    max_size=max_size,
                                    allow_overlap=allow_overlap, num_images=num_images)
    # create torch dataset
    img_data_torch = images.transpose(0, 3, 1, 2)  # [num_images, 3, H, W]
    img_ds = torch.utils.data.TensorDataset(torch.tensor(img_data_torch, dtype=torch.float), torch.arange(num_images))
    return img_ds

# ---------------------------
# Existing helpers (unchanged)
# ---------------------------
def _rand_color():
    return np.random.randint(0, 256, size=3).astype(np.float32)  # 0..255

def _rand_int(a, b):
    return np.random.randint(a, b+1)

def _shape_mask(shape, H, W, min_size, max_size):
    if shape == "disk":
        r = _rand_int(min_size//2, max_size//2)
        cy = _rand_int(r, H-1-r)
        cx = _rand_int(r, W-1-r)
        rr, cc = disk((cy, cx), r, shape=(H, W))
    elif shape == "rect":
        h = _rand_int(min_size, max_size)
        w = _rand_int(min_size, max_size)
        y0 = _rand_int(0, H-h)
        x0 = _rand_int(0, W-w)
        rr, cc = rectangle(start=(y0, x0), extent=(h, w), shape=(H, W))
    else:  # triangle
        s  = _rand_int(min_size, max_size)
        cy = _rand_int(s, H-1-s)
        cx = _rand_int(s, W-1-s)
        pts = np.array([[cy - s, cx],
                        [cy + s, cx - s],
                        [cy + s, cx + s]])
        rr, cc = polygon(pts[:,0], pts[:,1], shape=(H, W))
    mask = np.zeros((H, W), dtype=bool)
    mask[rr, cc] = True
    return mask

# ---------------------------
# New: intrinsics utilities
# ---------------------------
def _focal_from_fov_deg(fov_deg, size_px):
    """Return focal length in *pixels* from horizontal/vertical fov and image size."""
    fov_rad = np.deg2rad(fov_deg)
    return 0.5 * size_px / np.tan(0.5 * fov_rad)

def _maybe_range(v, rng):
    """Pick either fixed v or sample from [lo, hi] if rng (tuple) is provided."""
    if isinstance(rng, (tuple, list)) and len(rng) == 2:
        return np.random.uniform(rng[0], rng[1])
    return float(v)

# ------------------------------------------
# Intrinsics-aware RGBD synthetic generator
# ------------------------------------------
def generate_shape_dataset_rgbd(
    img_size=64,
    min_shapes=2,
    max_shapes=5,
    min_size=10,
    max_size=12,
    allow_overlap=True,               # z-buffer handles occlusions
    num_images=10_000,
    bg_color=(0, 0, 0),
    depth_range=(0.05, 1.0),          # near, far in "scene units"
    shapes=("disk", "rect", "tri"),
    # --- intrinsics controls ---
    fovx_deg=60.0,                    # scalar or (min,max) to randomize per image
    fovy_deg=None,                    # if None -> derive from fx, fy with square pixels: fovy = fovx
    pp_jitter_px=0.0,                 # stddev of principal-point jitter (pixels)
    # --- bookkeeping ---
    depth_encoding="linear"           # kept as metadata; generator writes linear depths
):
    """
    Returns:
      imgs: np.ndarray [N, H, W, 4]  (RGB: 0..255 float32, D: linear in [near,far])
      Ks:   np.ndarray [N, 3, 3]     (intrinsics per image, in pixels)
    """
    H = W = img_size
    imgs = np.empty((num_images, H, W, 4), dtype=np.float32)
    Ks   = np.empty((num_images, 3, 3), dtype=np.float32)

    for n in tqdm(range(num_images)):
        # --- intrinsics per image ---
        fovx = _maybe_range(fovx_deg, fovx_deg)  # allow tuple
        if fovy_deg is None:
            fovy = fovx
        else:
            fovy = _maybe_range(fovy_deg, fovy_deg)

        fx = _focal_from_fov_deg(fovx, W)
        fy = _focal_from_fov_deg(fovy, H)

        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0
        if pp_jitter_px > 0.0:
            cx = np.clip(cx + np.random.normal(0, pp_jitter_px), 0, W - 1)
            cy = np.clip(cy + np.random.normal(0, pp_jitter_px), 0, H - 1)

        K = np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
        Ks[n] = K

        # --- render RGBD ---
        rgb   = np.zeros((H, W, 3), dtype=np.float32)
        rgb[:] = np.array(bg_color, dtype=np.float32)[None, None, :]
        depth = np.ones((H, W), dtype=np.float32) * depth_range[1]  # initialize at far plane

        k = _rand_int(min_shapes, max_shapes)
        for _ in range(k):
            shape = np.random.choice(shapes)
            mask  = _shape_mask(shape, H, W, min_size, max_size)
            z     = np.random.uniform(*depth_range)           # per-shape scalar depth (linear)
            color = _rand_color()

            # z-buffer write: nearer (smaller z) wins
            idx = mask & (z < depth)
            if np.any(idx):
                rgb[idx]   = color
                depth[idx] = z

        imgs[n, :, :, :3] = rgb
        imgs[n, :, :, 3]  = depth

    return imgs, Ks, np.array(depth_range, dtype=np.float32), depth_encoding


class ShapesRGBDDataset(Dataset):
    def __init__(self, imgs_hw4, Ks_33, depth_range, depth_encoding="linear"):
        """
        imgs_hw4: np.ndarray [N,H,W,4]  (RGB 0..255 float32, depth linear in [near,far])
        Ks_33:    np.ndarray [N,3,3]    intrinsics per image (pixels)
        depth_range: (near, far) in same metric units as depth channel
        """
        assert imgs_hw4.ndim == 4 and imgs_hw4.shape[-1] == 4
        self.N, self.H, self.W, _ = imgs_hw4.shape

        # store images as torch [N,4,H,W] (still raw RGB 0..255, depth in [near,far])
        imgs_chw = np.transpose(imgs_hw4.astype(np.float32), (0, 3, 1, 2))
        self.images = torch.from_numpy(imgs_chw)
        self.indices = torch.arange(self.N)

        # intrinsics & metadata
        self.K = torch.from_numpy(Ks_33.astype(np.float32))        # [N,3,3]
        self.depth_range = (float(depth_range[0]), float(depth_range[1]))  # <-- tuple of floats
        self.depth_encoding = str(depth_encoding)

        # exporter-friendly bookkeeping
        self.id2pos = {int(i): int(i) for i in range(self.N)}      # global id -> position
        self.items   = [(int(i), None, None, None) for i in range(self.N)]  # only .items[pos][0] is ever used

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        return self.images[i], self.indices[i]

    # --- exporter hooks ---
    def get_intrinsics(self, idx: int):
        K = self.K[int(idx)].numpy()
        return {"fx": float(K[0,0]), "fy": float(K[1,1]), "cx": float(K[0,2]), "cy": float(K[1,2])}

    def ply_sample(self, pos: int, trim_p: float = 0.90):
        """
        Returns (rgb, depth_camZ_inf, intr, gid):
          - rgb:  HxWx3 in [0,1]
          - depth_camZ_inf: HxW in meters/scene-units (linear camera-Z). Background -> +inf.
          - intr: dict {fx,fy,cx,cy}
          - gid:  global id (int)
        """
        pos = int(pos)
        gid = int(self.items[pos][0])

        # RGB to 0..1
        rgb = self.images[pos, 0:3].numpy().transpose(1, 2, 0) / 255.0  # HxWx3

        # Depth is already linear camera-Z in [near, far]
        depth = self.images[pos, 3].numpy().copy()  # HxW

        # treat the far plane as background (drop it)
        near, far = self.depth_range
        bg = depth >= (far - 1e-7)
        depth[bg] = np.inf

        # optional far trimming by percentile (kept like Blender helper)
        finite = np.isfinite(depth) & (depth > 0)
        if np.any(finite) and 0.0 < trim_p < 1.0:
            z_cap = np.percentile(depth[finite], 100.0 * trim_p)
            depth[depth > z_cap] = np.inf

        K = self.K[pos].numpy()
        intr = {"fx": float(K[0,0]), "fy": float(K[1,1]), "cx": float(K[0,2]), "cy": float(K[1,2])}
        return rgb.astype(np.float32), depth.astype(np.float32), intr, gid


def generate_shape_dataset_torch_rgbd(**kwargs):
    """
    Returns a ShapesRGBDDataset that works with your exporter as-is.
    """
    imgs, Ks, z_range, enc = generate_shape_dataset_rgbd(**kwargs)
    return ShapesRGBDDataset(imgs, Ks, z_range, enc)
