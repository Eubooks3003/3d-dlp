#!/usr/bin/env python
"""
Grid search over crop_bounds with tqdm progress bar.

Usage:
    python scripts/crop_search_with_progress.py
"""

import numpy as np
import torch
from itertools import product
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from tqdm import tqdm
from plyfile import PlyData


@dataclass
class SearchResult:
    crop_bounds: Dict[str, float]
    iou: float
    mse: float
    ctr_delta_xyz: Optional[List[float]]
    size_ratio_dhw: Optional[List[float]]


def voxel_mask(v, thresh=0.05):
    """Occupancy mask for RGB voxels."""
    if v.ndim == 3:
        v = v.unsqueeze(0)
    mag = torch.linalg.norm(v, dim=0)
    return mag > thresh


def voxel_stats(vox, thresh=0.05):
    """Compute voxel statistics."""
    v = torch.as_tensor(vox).float()
    if v.ndim == 3:
        v = v.unsqueeze(0)
    C, D, H, W = v.shape
    m = voxel_mask(v, thresh=thresh)
    idx = m.nonzero(as_tuple=False)

    if idx.numel() == 0:
        return {"empty": True, "occ_frac": 0.0}

    mn = idx.min(0).values
    mx = idx.max(0).values
    size = (mx - mn + 1).float()
    ctr = idx.float().mean(0)

    ctr_n = torch.tensor([
        (ctr[2] / (W - 1)) * 2 - 1,
        (ctr[1] / (H - 1)) * 2 - 1,
        (ctr[0] / (D - 1)) * 2 - 1,
    ])

    return {
        "empty": False,
        "occ_frac": float(m.float().mean().item()),
        "size_DHW": size.tolist(),
        "ctr_norm_xyz": ctr_n.tolist(),
    }


def voxel_alignment_metrics(vox_live, vox_gt, thresh=0.05):
    """Compute alignment metrics between live and GT voxels."""
    L = torch.as_tensor(vox_live).float()
    G = torch.as_tensor(vox_gt).float()
    if L.ndim == 3:
        L = L.unsqueeze(0)
    if G.ndim == 3:
        G = G.unsqueeze(0)

    mL = voxel_mask(L, thresh=thresh)
    mG = voxel_mask(G, thresh=thresh)

    inter = (mL & mG).sum().item()
    union = (mL | mG).sum().item()
    iou = inter / (union + 1e-8)
    mse = torch.mean((L - G) ** 2).item()

    sL = voxel_stats(L, thresh=thresh)
    sG = voxel_stats(G, thresh=thresh)

    ctr_delta = None
    size_ratio = None
    if (not sL["empty"]) and (not sG["empty"]):
        ctr_delta = (np.array(sL["ctr_norm_xyz"]) - np.array(sG["ctr_norm_xyz"])).tolist()
        size_ratio = (np.array(sL["size_DHW"]) / (np.array(sG["size_DHW"]) + 1e-8)).tolist()

    return {
        "iou": float(iou),
        "mse": float(mse),
        "ctr_delta": ctr_delta,
        "size_ratio": size_ratio,
    }


def crop_points(pts, bounds):
    """Crop points to bounds."""
    xyz = pts[:, :3]
    m = (
        (xyz[:, 0] >= bounds["xmin"]) & (xyz[:, 0] <= bounds["xmax"]) &
        (xyz[:, 1] >= bounds["ymin"]) & (xyz[:, 1] <= bounds["ymax"]) &
        (xyz[:, 2] >= bounds["zmin"]) & (xyz[:, 2] <= bounds["zmax"])
    )
    return pts[m]


def voxelize(pts, grid_dhw=(64, 64, 64)):
    """Simple voxelization with averaging."""
    if pts.shape[0] == 0:
        D, H, W = grid_dhw
        return torch.zeros(3, D, H, W)

    xyz = pts[:, :3]
    rgb = pts[:, 3:6] if pts.shape[1] >= 6 else None

    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)

    D, H, W = grid_dhw
    voxel = np.zeros((3, D, H, W), dtype=np.float32)
    counts = np.zeros((D, H, W), dtype=np.float32)

    eps = 1e-8
    idx_x = ((xyz[:, 0] - mins[0]) / (maxs[0] - mins[0] + eps) * (W - 1)).astype(int)
    idx_y = ((xyz[:, 1] - mins[1]) / (maxs[1] - mins[1] + eps) * (H - 1)).astype(int)
    idx_z = ((xyz[:, 2] - mins[2]) / (maxs[2] - mins[2] + eps) * (D - 1)).astype(int)

    idx_x = np.clip(idx_x, 0, W - 1)
    idx_y = np.clip(idx_y, 0, H - 1)
    idx_z = np.clip(idx_z, 0, D - 1)

    if rgb is not None:
        for i in range(xyz.shape[0]):
            z, y, x = idx_z[i], idx_y[i], idx_x[i]
            voxel[:, z, y, x] += rgb[i]
            counts[z, y, x] += 1

    mask = counts > 0
    for c in range(3):
        voxel[c][mask] /= counts[mask]

    return torch.from_numpy(voxel)


def run_grid_search(pts, gt_vox, param_grid, thresh=0.05):
    """Run grid search with progress bar."""

    param_names = list(param_grid.keys())
    param_values = [param_grid[k] for k in param_names]

    combinations = list(product(*param_values))
    total = len(combinations)

    print(f"Testing {total} combinations...")

    results = []

    for combo in tqdm(combinations, desc="Grid Search"):
        crop_bounds = {}
        for name, val in zip(param_names, combo):
            crop_bounds[name] = float(val)

        # Crop and voxelize
        cropped = crop_points(pts, crop_bounds)
        if cropped.shape[0] < 100:
            continue

        live_vox = voxelize(cropped)

        # Compute metrics
        metrics = voxel_alignment_metrics(live_vox, gt_vox, thresh=thresh)

        result = SearchResult(
            crop_bounds=crop_bounds,
            iou=metrics["iou"],
            mse=metrics["mse"],
            ctr_delta_xyz=metrics["ctr_delta"],
            size_ratio_dhw=metrics["size_ratio"],
        )
        results.append(result)

    results.sort(key=lambda r: -r.iou)
    return results


if __name__ == "__main__":
    # === CONFIGURATION ===
    PLY_PATH = "/home/ellina/Desktop/Code/lpwm-dev/mimicgen_pc_filter_depth/demo_0_frame000000_fused.ply"
    GT_VOX_PATH = "/home/ellina/Desktop/Code/3D-DLP-data/voxel_data/voxel_shapenet_rgb/000000_voxels.pt"

    # Grid to search (adjust as needed)
    # Smaller grid = faster, larger grid = more thorough
    PARAM_GRID = {
        "xmin": [-1.4, -1.1, -0.8, -0.5, -0.2],
        "xmax": [0.3, 0.6, 0.9, 1.2],
        "ymin": [-0.6, -0.4, -0.2],
        "ymax": [0.2, 0.4, 0.6],
        "zmin": [-0.8, -0.4, 0.0, 0.4],
        "zmax": [2.0, 2.5, 3.0],
    }
    # This gives 5*4*3*3*4*3 = 2160 combinations

    # === LOAD DATA ===
    print("Loading PLY...")
    ply = PlyData.read(PLY_PATH)
    v = ply['vertex']
    pts = np.stack([v['x'], v['y'], v['z'], v['red']/255.0, v['green']/255.0, v['blue']/255.0], axis=-1).astype(np.float32)
    print(f"Loaded PLY: {pts.shape}")
    print(f"XYZ range: x=[{pts[:,0].min():.2f}, {pts[:,0].max():.2f}] "
          f"y=[{pts[:,1].min():.2f}, {pts[:,1].max():.2f}] "
          f"z=[{pts[:,2].min():.2f}, {pts[:,2].max():.2f}]")

    print("\nLoading GT voxel...")
    gt_vox = torch.load(GT_VOX_PATH, weights_only=False)
    print(f"GT voxel shape: {gt_vox.shape}")

    # === RUN SEARCH ===
    print("\n" + "="*60)
    results = run_grid_search(pts, gt_vox, PARAM_GRID)

    # === PRINT RESULTS ===
    print("\n" + "="*60)
    print("TOP 10 RESULTS:")
    print("="*60)
    for i, r in enumerate(results[:10]):
        print(f"\n{i+1}. IoU={r.iou:.4f}  MSE={r.mse:.6f}")
        print(f"   x=[{r.crop_bounds['xmin']:.2f}, {r.crop_bounds['xmax']:.2f}]  "
              f"y=[{r.crop_bounds['ymin']:.2f}, {r.crop_bounds['ymax']:.2f}]  "
              f"z=[{r.crop_bounds['zmin']:.2f}, {r.crop_bounds['zmax']:.2f}]")
        if r.ctr_delta_xyz:
            print(f"   ctr_delta(x,y,z): [{r.ctr_delta_xyz[0]:.3f}, {r.ctr_delta_xyz[1]:.3f}, {r.ctr_delta_xyz[2]:.3f}]")

    print("\n" + "="*60)
    print("BEST CROP BOUNDS:")
    print("="*60)
    best = results[0]
    print(f"""
crop_bounds = {{
    "xmin": {best.crop_bounds["xmin"]:.2f},
    "xmax": {best.crop_bounds["xmax"]:.2f},
    "ymin": {best.crop_bounds["ymin"]:.2f},
    "ymax": {best.crop_bounds["ymax"]:.2f},
    "zmin": {best.crop_bounds["zmin"]:.2f},
    "zmax": {best.crop_bounds["zmax"]:.2f},
}}
""")
