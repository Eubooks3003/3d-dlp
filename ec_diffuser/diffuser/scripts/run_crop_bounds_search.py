#!/usr/bin/env python
"""
Grid search over crop_bounds to find best alignment with GT voxels.

Usage:
    python scripts/run_crop_bounds_search.py \
        --gt-voxel /path/to/gt_voxel.pt \
        --env-config /path/to/env_config.json \
        --output results.json

Or run interactively in your training script after setting up the wrapper.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from itertools import product
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffuser.envs.voxel_alignment_search import (
    VoxelAlignmentSearch,
    voxel_alignment_metrics,
    voxel_stats,
    _as_CDHW,
    SearchResult,
)


def run_grid_search_with_wrapper(
    wrapper,
    raw_obs: dict,
    gt_vox: torch.Tensor,
    param_grid: Dict[str, List[float]] = None,
    output_path: str = None,
    verbose: bool = True,
) -> List[SearchResult]:
    """
    Run grid search over crop_bounds using a MimicGenDLPWrapper.

    Args:
        wrapper: MimicGenDLPWrapper instance
        raw_obs: Raw observation dict from env
        gt_vox: Ground truth voxel tensor [3, 64, 64, 64]
        param_grid: Dict mapping param name to list of values
        output_path: Optional path to save results JSON
        verbose: Print progress

    Returns:
        List of SearchResult sorted by IoU (best first)
    """
    searcher = VoxelAlignmentSearch(
        wrapper=wrapper,
        gt_vox=gt_vox,
        thresh=0.05,
        verbose=verbose,
    )

    # Default grid if not provided
    if param_grid is None:
        base = dict(wrapper.crop_bounds)
        param_grid = {
            "xmin": np.linspace(base["xmin"] - 0.5, base["xmin"] + 0.5, 5).tolist(),
            "xmax": np.linspace(base["xmax"] - 0.5, base["xmax"] + 0.5, 5).tolist(),
            "ymin": np.linspace(base["ymin"] - 0.3, base["ymin"] + 0.3, 3).tolist(),
            "ymax": np.linspace(base["ymax"] - 0.3, base["ymax"] + 0.3, 3).tolist(),
            "zmin": np.linspace(base["zmin"] - 0.5, base["zmin"] + 0.5, 5).tolist(),
        }

    results = searcher.grid_search(raw_obs=raw_obs, param_ranges=param_grid)

    if output_path:
        searcher.save_results(output_path)

    return results


def run_coarse_to_fine_search(
    wrapper,
    raw_obs: dict,
    gt_vox: torch.Tensor,
    n_rounds: int = 3,
    output_path: str = None,
    verbose: bool = True,
) -> List[SearchResult]:
    """
    Run coarse-to-fine search that narrows in on best crop_bounds.

    Args:
        wrapper: MimicGenDLPWrapper instance
        raw_obs: Raw observation dict from env
        gt_vox: Ground truth voxel tensor
        n_rounds: Number of refinement rounds
        output_path: Optional path to save results JSON
        verbose: Print progress

    Returns:
        List of SearchResult sorted by IoU (best first)
    """
    searcher = VoxelAlignmentSearch(
        wrapper=wrapper,
        gt_vox=gt_vox,
        thresh=0.05,
        verbose=verbose,
    )

    results = searcher.adaptive_search(
        raw_obs=raw_obs,
        n_rounds=n_rounds,
        n_samples_per_round=30,
    )

    if output_path:
        searcher.save_results(output_path)

    return results


def analyze_current_alignment(
    wrapper,
    raw_obs: dict,
    gt_vox: torch.Tensor,
) -> dict:
    """
    Analyze current alignment without doing a search.

    Returns dict with metrics and interpretation.
    """
    # Get current voxel
    _, _, live_vox = wrapper.encode_tokens(raw_obs)
    live_vox = torch.from_numpy(live_vox)

    metrics = voxel_alignment_metrics(live_vox, gt_vox, thresh=0.05)

    # Create result for interpretation
    live_stats = voxel_stats(live_vox, thresh=0.05)
    gt_stats = voxel_stats(gt_vox, thresh=0.05)

    result = SearchResult(
        crop_bounds=dict(wrapper.crop_bounds),
        iou=metrics["iou"],
        mse=metrics["mse"],
        ctr_delta_xyz=metrics["ctr_delta_norm_xyz_live_minus_gt"],
        size_ratio_dhw=metrics["size_ratio_DHW_live_over_gt"],
        live_occ_frac=live_stats.get("occ_frac", 0),
        gt_occ_frac=gt_stats.get("occ_frac", 0),
    )

    searcher = VoxelAlignmentSearch(gt_vox=gt_vox, verbose=False)
    interpretation = searcher.interpret_alignment(result)

    return {
        "metrics": metrics,
        "result": asdict(result),
        "interpretation": interpretation,
        "current_crop_bounds": dict(wrapper.crop_bounds),
    }


# =============================================================================
# Standalone mode: Create a simplified voxelizer for grid search without full env
# =============================================================================

class StandaloneVoxelizer:
    """
    Simplified voxelizer that mimics MimicGenDLPWrapper's voxelization
    but works on raw point clouds without the full environment.

    Use this if you have saved point clouds and want to test different
    crop_bounds without running the simulator.
    """

    def __init__(
        self,
        crop_bounds: Dict[str, float],
        grid_dhw=(64, 64, 64),
        normalize_to_unit_cube: bool = False,
    ):
        self.crop_bounds = crop_bounds
        self.grid_dhw = grid_dhw
        self.normalize_to_unit_cube = normalize_to_unit_cube

    def crop_world(self, xyz, rgb=None):
        """Crop points to crop_bounds."""
        b = self.crop_bounds
        m = (
            (xyz[:, 0] >= b["xmin"]) & (xyz[:, 0] <= b["xmax"]) &
            (xyz[:, 1] >= b["ymin"]) & (xyz[:, 1] <= b["ymax"]) &
            (xyz[:, 2] >= b["zmin"]) & (xyz[:, 2] <= b["zmax"])
        )
        if not np.any(m):
            return (xyz, rgb) if rgb is not None else xyz
        xyz2 = xyz[m]
        if rgb is None:
            return xyz2
        return xyz2, rgb[m]

    def voxelize(self, pts: np.ndarray) -> torch.Tensor:
        """
        Voxelize point cloud.

        Args:
            pts: [N, 6] array with xyz and rgb

        Returns:
            [3, D, H, W] voxel tensor
        """
        xyz = pts[:, :3]
        rgb = pts[:, 3:6] if pts.shape[1] >= 6 else None

        # Crop
        if rgb is not None:
            xyz, rgb = self.crop_world(xyz, rgb)
        else:
            xyz = self.crop_world(xyz)

        if xyz.shape[0] == 0:
            D, H, W = self.grid_dhw
            return torch.zeros(3, D, H, W)

        # Normalize to unit cube
        if self.normalize_to_unit_cube:
            mins = xyz.min(axis=0)
            maxs = xyz.max(axis=0)
            center = (mins + maxs) / 2.0
            scale = (maxs - mins).max() + 1e-8
            xyz = (xyz - center) / scale * 2.0

        # Compute bounds
        mins = xyz.min(axis=0)
        maxs = xyz.max(axis=0)

        D, H, W = self.grid_dhw

        # Voxelize with averaging
        voxel = np.zeros((3, D, H, W), dtype=np.float32)
        counts = np.zeros((D, H, W), dtype=np.float32)

        # Map xyz to grid indices
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
        else:
            for i in range(xyz.shape[0]):
                z, y, x = idx_z[i], idx_y[i], idx_x[i]
                voxel[:, z, y, x] = 1.0
                counts[z, y, x] = 1

        # Average
        mask = counts > 0
        for c in range(3):
            voxel[c][mask] /= counts[mask]

        return torch.from_numpy(voxel)


def grid_search_standalone(
    pts: np.ndarray,
    gt_vox: torch.Tensor,
    base_bounds: Dict[str, float],
    param_grid: Dict[str, List[float]] = None,
    grid_dhw=(64, 64, 64),
    thresh: float = 0.05,
    verbose: bool = True,
) -> List[SearchResult]:
    """
    Grid search without MimicGenDLPWrapper - uses raw point cloud.

    Args:
        pts: [N, 6] point cloud (xyz + rgb)
        gt_vox: Ground truth voxel
        base_bounds: Starting crop bounds
        param_grid: Parameter search grid
        grid_dhw: Voxel grid dimensions
        thresh: Occupancy threshold
        verbose: Print progress

    Returns:
        List of SearchResult sorted by IoU
    """
    gt_vox = _as_CDHW(gt_vox)

    if param_grid is None:
        param_grid = {
            "xmin": np.linspace(base_bounds["xmin"] - 0.5, base_bounds["xmin"] + 0.5, 5).tolist(),
            "xmax": np.linspace(base_bounds["xmax"] - 0.5, base_bounds["xmax"] + 0.5, 5).tolist(),
            "ymin": np.linspace(base_bounds["ymin"] - 0.3, base_bounds["ymin"] + 0.3, 3).tolist(),
            "ymax": np.linspace(base_bounds["ymax"] - 0.3, base_bounds["ymax"] + 0.3, 3).tolist(),
            "zmin": np.linspace(base_bounds["zmin"] - 0.5, base_bounds["zmin"] + 0.5, 5).tolist(),
            "zmax": np.linspace(base_bounds["zmax"] - 0.5, base_bounds["zmax"] + 0.5, 5).tolist(),
        }

    param_names = list(param_grid.keys())
    param_values = [param_grid[k] for k in param_names]

    total = 1
    for v in param_values:
        total *= len(v)

    if verbose:
        print(f"[GridSearch] Testing {total} combinations...")

    results = []

    for i, combo in enumerate(product(*param_values)):
        crop_bounds = dict(base_bounds)
        for name, val in zip(param_names, combo):
            crop_bounds[name] = float(val)

        voxelizer = StandaloneVoxelizer(crop_bounds, grid_dhw)
        live_vox = voxelizer.voxelize(pts)

        metrics = voxel_alignment_metrics(live_vox, gt_vox, thresh=thresh)
        live_stats = voxel_stats(live_vox, thresh=thresh)
        gt_stats = voxel_stats(gt_vox, thresh=thresh)

        result = SearchResult(
            crop_bounds=crop_bounds,
            iou=metrics["iou"],
            mse=metrics["mse"],
            ctr_delta_xyz=metrics["ctr_delta_norm_xyz_live_minus_gt"],
            size_ratio_dhw=metrics["size_ratio_DHW_live_over_gt"],
            live_occ_frac=live_stats.get("occ_frac", 0),
            gt_occ_frac=gt_stats.get("occ_frac", 0),
        )
        results.append(result)

        if verbose and i % 100 == 0:
            print(f"  [{i+1}/{total}] Best IoU so far: {max(r.iou for r in results):.4f}")

    results.sort(key=lambda r: -r.iou)

    if verbose:
        print(f"\n[GridSearch] Top 10 results:")
        for i, r in enumerate(results[:10]):
            print(f"  {i+1}. IoU={r.iou:.4f} MSE={r.mse:.6f}")
            print(f"     bounds: x=[{r.crop_bounds['xmin']:.2f}, {r.crop_bounds['xmax']:.2f}] "
                  f"y=[{r.crop_bounds['ymin']:.2f}, {r.crop_bounds['ymax']:.2f}] "
                  f"z=[{r.crop_bounds['zmin']:.2f}, {r.crop_bounds['zmax']:.2f}]")

    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop bounds grid search")
    parser.add_argument("--gt-voxel", type=str, required=True, help="Path to GT voxel .pt file")
    parser.add_argument("--points", type=str, help="Path to point cloud .pt file (for standalone mode)")
    parser.add_argument("--output", type=str, default="crop_bounds_search_results.json")
    parser.add_argument("--mode", choices=["standalone", "analyze"], default="standalone")

    # Crop bounds
    parser.add_argument("--xmin", type=float, default=-1.7)
    parser.add_argument("--xmax", type=float, default=0.5)
    parser.add_argument("--ymin", type=float, default=-0.4)
    parser.add_argument("--ymax", type=float, default=0.4)
    parser.add_argument("--zmin", type=float, default=-0.7)
    parser.add_argument("--zmax", type=float, default=2.5)

    # Search params
    parser.add_argument("--x-range", type=float, default=0.5, help="Search range for x params")
    parser.add_argument("--y-range", type=float, default=0.3, help="Search range for y params")
    parser.add_argument("--z-range", type=float, default=0.5, help="Search range for z params")
    parser.add_argument("--n-steps", type=int, default=5, help="Number of steps per param")

    args = parser.parse_args()

    # Load GT voxel
    gt_vox = torch.load(args.gt_voxel, weights_only=False)
    print(f"Loaded GT voxel: {gt_vox.shape}")

    base_bounds = {
        "xmin": args.xmin, "xmax": args.xmax,
        "ymin": args.ymin, "ymax": args.ymax,
        "zmin": args.zmin, "zmax": args.zmax,
    }

    if args.mode == "standalone" and args.points:
        # Load point cloud
        pts = torch.load(args.points, weights_only=False)
        if isinstance(pts, torch.Tensor):
            pts = pts.numpy()
        print(f"Loaded point cloud: {pts.shape}")

        # Build param grid
        param_grid = {
            "xmin": np.linspace(base_bounds["xmin"] - args.x_range, base_bounds["xmin"] + args.x_range, args.n_steps).tolist(),
            "xmax": np.linspace(base_bounds["xmax"] - args.x_range, base_bounds["xmax"] + args.x_range, args.n_steps).tolist(),
            "ymin": np.linspace(base_bounds["ymin"] - args.y_range, base_bounds["ymin"] + args.y_range, 3).tolist(),
            "ymax": np.linspace(base_bounds["ymax"] - args.y_range, base_bounds["ymax"] + args.y_range, 3).tolist(),
            "zmin": np.linspace(base_bounds["zmin"] - args.z_range, base_bounds["zmin"] + args.z_range, args.n_steps).tolist(),
            "zmax": np.linspace(base_bounds["zmax"] - args.z_range, base_bounds["zmax"] + args.z_range, args.n_steps).tolist(),
        }

        results = grid_search_standalone(pts, gt_vox, base_bounds, param_grid)

        # Save results
        with open(args.output, "w") as f:
            json.dump({
                "results": [asdict(r) for r in results[:100]],  # top 100
                "best": asdict(results[0]),
                "base_bounds": base_bounds,
            }, f, indent=2)
        print(f"\nSaved results to {args.output}")

    elif args.mode == "analyze":
        # Just analyze GT voxel stats
        stats = voxel_stats(gt_vox, thresh=0.05)
        print("\n=== GT Voxel Analysis ===")
        print(f"Shape: {stats.get('shape', 'N/A')}")
        print(f"Occupancy: {stats.get('occ_frac', 0):.4f}")
        if not stats.get("empty", True):
            print(f"BBox min (D,H,W): {stats['mn_DHW']}")
            print(f"BBox max (D,H,W): {stats['mx_DHW']}")
            print(f"Size (D,H,W): {[round(x, 1) for x in stats['size_DHW']]}")
            print(f"Centroid (x,y,z normalized): {[round(x, 3) for x in stats['ctr_norm_xyz']]}")

    else:
        print("For standalone mode, provide --points argument with path to point cloud .pt file")
        print("For wrapper mode, use this script as a library in your training code")
