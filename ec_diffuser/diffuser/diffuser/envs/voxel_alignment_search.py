"""
Voxel Alignment Grid Search

This script performs a guided grid search over crop_bounds parameters
to find the best alignment between live voxels (from MimicGenDLPWrapper)
and ground truth voxels (from dataset).

Usage:
    # Option 1: From within your training/eval code
    from voxel_alignment_search import VoxelAlignmentSearch
    searcher = VoxelAlignmentSearch(wrapper, gt_vox)
    best_params = searcher.grid_search()

    # Option 2: Standalone script
    python voxel_alignment_search.py --gt-voxel-path /path/to/gt.pt --output results.json
"""

import numpy as np
import torch
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from itertools import product


# =============================================================================
# Voxel Statistics & Alignment Metrics
# =============================================================================

def _as_CDHW(vox) -> torch.Tensor:
    """Convert voxel to [C, D, H, W] format."""
    v = torch.as_tensor(vox)
    if v.ndim == 3:          # [D,H,W]
        v = v.unsqueeze(0)   # [1,D,H,W]
    if v.ndim == 4:          # [C,D,H,W]
        return v.float()
    if v.ndim == 5 and v.shape[0] == 1:
        return v[0].float()
    raise ValueError(f"Bad vox shape: {tuple(v.shape)}")


def voxel_mask(v_cdhw: torch.Tensor, thresh: float = 0.05) -> torch.Tensor:
    """
    Create occupancy mask for RGB voxels using magnitude across channels.
    """
    mag = torch.linalg.norm(v_cdhw, dim=0)  # [D,H,W]
    return mag > thresh


def voxel_stats(vox, thresh: float = 0.05) -> Dict[str, Any]:
    """
    Compute statistics for a voxel grid.

    Returns:
        dict with keys:
            - empty: bool
            - shape: tuple
            - occ_frac: fraction of occupied voxels
            - mn_DHW: min indices [D,H,W]
            - mx_DHW: max indices [D,H,W]
            - size_DHW: bounding box size [D,H,W]
            - ctr_DHW: centroid in [D,H,W]
            - ctr_norm_xyz: normalized centroid in [-1,1] for [x,y,z]
    """
    v = _as_CDHW(vox)
    C, D, H, W = v.shape
    m = voxel_mask(v, thresh=thresh)
    idx = m.nonzero(as_tuple=False)  # [N,3] in (D,H,W)

    if idx.numel() == 0:
        return {"empty": True, "occ_frac": 0.0}

    mn = idx.min(0).values
    mx = idx.max(0).values
    size = (mx - mn + 1).float()
    ctr = idx.float().mean(0)  # centroid in DHW

    # Normalized center in [-1,1] in xyz order for readability
    ctr_n = torch.tensor([
        (ctr[2] / (W - 1)) * 2 - 1,  # x
        (ctr[1] / (H - 1)) * 2 - 1,  # y
        (ctr[0] / (D - 1)) * 2 - 1,  # z
    ])

    return {
        "empty": False,
        "shape": (C, D, H, W),
        "occ_frac": float(m.float().mean().item()),
        "mn_DHW": mn.tolist(),
        "mx_DHW": mx.tolist(),
        "size_DHW": size.tolist(),      # (z,y,x) sizes
        "ctr_DHW": ctr.tolist(),
        "ctr_norm_xyz": ctr_n.tolist(), # (x,y,z) in [-1,1]
    }


def voxel_alignment_metrics(vox_live, vox_gt, thresh: float = 0.05) -> Dict[str, Any]:
    """
    Compute alignment metrics between live and GT voxels.

    Returns:
        dict with keys:
            - iou: Intersection over Union of occupied voxels
            - mse: Mean squared error of voxel values
            - live: stats dict for live voxel
            - gt: stats dict for GT voxel
            - ctr_delta_norm_xyz_live_minus_gt: centroid shift [x,y,z]
            - size_ratio_DHW_live_over_gt: size ratio [D,H,W]
    """
    L = _as_CDHW(vox_live)
    G = _as_CDHW(vox_gt)
    assert L.shape == G.shape, f"Shape mismatch: {L.shape} vs {G.shape}"

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
        "live": sL,
        "gt": sG,
        "ctr_delta_norm_xyz_live_minus_gt": ctr_delta,   # (+x means live is shifted right)
        "size_ratio_DHW_live_over_gt": size_ratio,       # (>1 means live bigger in that axis)
    }


# =============================================================================
# Grid Search Result
# =============================================================================

@dataclass
class SearchResult:
    """Result from a single grid search evaluation."""
    crop_bounds: Dict[str, float]
    iou: float
    mse: float
    ctr_delta_xyz: Optional[List[float]]
    size_ratio_dhw: Optional[List[float]]
    live_occ_frac: float
    gt_occ_frac: float

    def score(self, iou_weight: float = 1.0, mse_weight: float = 0.1) -> float:
        """Composite score (higher is better)."""
        return self.iou * iou_weight - self.mse * mse_weight


# =============================================================================
# Voxel Alignment Search Class
# =============================================================================

class VoxelAlignmentSearch:
    """
    Performs grid search over crop_bounds to align live voxels with GT.

    Can work in two modes:
    1. With MimicGenDLPWrapper: uses the wrapper to generate live voxels from raw_obs
    2. Standalone: directly compares voxel tensors
    """

    def __init__(
        self,
        wrapper=None,
        gt_vox: torch.Tensor = None,
        thresh: float = 0.05,
        verbose: bool = True,
    ):
        """
        Args:
            wrapper: MimicGenDLPWrapper instance (optional)
            gt_vox: Ground truth voxel tensor [C,D,H,W] or [D,H,W]
            thresh: Threshold for voxel occupancy mask
            verbose: Print progress
        """
        self.wrapper = wrapper
        self.gt_vox = _as_CDHW(gt_vox) if gt_vox is not None else None
        self.thresh = thresh
        self.verbose = verbose
        self.results: List[SearchResult] = []

    def set_gt_voxel(self, gt_vox):
        """Set the ground truth voxel to compare against."""
        self.gt_vox = _as_CDHW(gt_vox).detach().cpu()

    def evaluate_single(
        self,
        crop_bounds: Dict[str, float],
        raw_obs=None,
        live_vox=None,
    ) -> SearchResult:
        """
        Evaluate a single crop_bounds configuration.

        Args:
            crop_bounds: dict with xmin, xmax, ymin, ymax, zmin, zmax
            raw_obs: raw observation dict (if using wrapper)
            live_vox: pre-computed live voxel (if not using wrapper)

        Returns:
            SearchResult with metrics
        """
        if self.gt_vox is None:
            raise ValueError("GT voxel not set. Call set_gt_voxel() first.")

        # Get live voxel
        if live_vox is not None:
            vox_live = _as_CDHW(live_vox)
        elif self.wrapper is not None and raw_obs is not None:
            # Update wrapper's crop bounds
            orig_bounds = dict(self.wrapper.crop_bounds)
            self.wrapper.crop_bounds = crop_bounds

            try:
                _, _, vox_np = self.wrapper.encode_tokens(raw_obs)
                vox_live = torch.from_numpy(vox_np)
            finally:
                # Restore original bounds
                self.wrapper.crop_bounds = orig_bounds
        else:
            raise ValueError("Either provide live_vox or (wrapper + raw_obs)")

        # Compute metrics
        metrics = voxel_alignment_metrics(vox_live, self.gt_vox, thresh=self.thresh)

        result = SearchResult(
            crop_bounds=crop_bounds,
            iou=metrics["iou"],
            mse=metrics["mse"],
            ctr_delta_xyz=metrics["ctr_delta_norm_xyz_live_minus_gt"],
            size_ratio_dhw=metrics["size_ratio_DHW_live_over_gt"],
            live_occ_frac=metrics["live"]["occ_frac"] if not metrics["live"]["empty"] else 0.0,
            gt_occ_frac=metrics["gt"]["occ_frac"] if not metrics["gt"]["empty"] else 0.0,
        )

        if self.verbose:
            cd = result.ctr_delta_xyz
            sr = result.size_ratio_dhw
            print(f"[ALIGN] IoU={result.iou:.4f} MSE={result.mse:.6f} "
                  f"ctrΔ(x,y,z)={None if cd is None else [round(x, 3) for x in cd]} "
                  f"size_ratio(z,y,x)={None if sr is None else [round(x, 3) for x in sr]}")

        return result

    def grid_search(
        self,
        raw_obs=None,
        param_ranges: Dict[str, List[float]] = None,
        base_bounds: Dict[str, float] = None,
    ) -> List[SearchResult]:
        """
        Perform grid search over crop_bounds parameters.

        Args:
            raw_obs: raw observation dict (required if using wrapper)
            param_ranges: dict mapping param name to list of values to try
                e.g., {"xmin": [-1.0, -0.8, -0.6], "xmax": [0.3, 0.5, 0.7]}
            base_bounds: starting crop bounds (defaults from wrapper or standard)

        Returns:
            List of SearchResult sorted by IoU (best first)
        """
        # Default base bounds
        if base_bounds is None:
            if self.wrapper is not None:
                base_bounds = dict(self.wrapper.crop_bounds)
            else:
                base_bounds = {
                    "xmin": -1.7, "xmax": 0.5,
                    "ymin": -0.4, "ymax": 0.4,
                    "zmin": -0.7, "zmax": 2.5
                }

        # Default parameter ranges (small perturbations around base)
        if param_ranges is None:
            param_ranges = {
                "xmin": [base_bounds["xmin"] + d for d in [-0.3, -0.15, 0, 0.15, 0.3]],
                "xmax": [base_bounds["xmax"] + d for d in [-0.3, -0.15, 0, 0.15, 0.3]],
                "ymin": [base_bounds["ymin"] + d for d in [-0.2, -0.1, 0, 0.1, 0.2]],
                "ymax": [base_bounds["ymax"] + d for d in [-0.2, -0.1, 0, 0.1, 0.2]],
                "zmin": [base_bounds["zmin"] + d for d in [-0.3, -0.15, 0, 0.15, 0.3]],
                "zmax": [base_bounds["zmax"] + d for d in [-0.3, -0.15, 0, 0.15, 0.3]],
            }

        # Generate all combinations
        param_names = list(param_ranges.keys())
        param_values = [param_ranges[k] for k in param_names]

        total = 1
        for v in param_values:
            total *= len(v)

        if self.verbose:
            print(f"[GridSearch] Testing {total} combinations...")

        self.results = []

        for i, combo in enumerate(product(*param_values)):
            # Build crop_bounds for this combo
            crop_bounds = dict(base_bounds)
            for name, val in zip(param_names, combo):
                crop_bounds[name] = float(val)

            if self.verbose and i % 50 == 0:
                print(f"  [{i+1}/{total}] Testing: {crop_bounds}")

            result = self.evaluate_single(crop_bounds, raw_obs=raw_obs)
            self.results.append(result)

        # Sort by IoU (descending)
        self.results.sort(key=lambda r: -r.iou)

        if self.verbose:
            print(f"\n[GridSearch] Top 5 results:")
            for i, r in enumerate(self.results[:5]):
                print(f"  {i+1}. IoU={r.iou:.4f} MSE={r.mse:.6f} bounds={r.crop_bounds}")

        return self.results

    def sweep_single_param(
        self,
        param_name: str,
        values: List[float],
        raw_obs=None,
        base_bounds: Dict[str, float] = None,
    ) -> List[SearchResult]:
        """
        Sweep a single parameter while keeping others fixed.

        Args:
            param_name: one of "xmin", "xmax", "ymin", "ymax", "zmin", "zmax"
            values: list of values to try for this parameter
            raw_obs: raw observation dict
            base_bounds: starting crop bounds

        Returns:
            List of SearchResult sorted by IoU
        """
        if base_bounds is None:
            if self.wrapper is not None:
                base_bounds = dict(self.wrapper.crop_bounds)
            else:
                raise ValueError("base_bounds required when no wrapper")

        results = []

        for val in values:
            crop_bounds = dict(base_bounds)
            crop_bounds[param_name] = float(val)

            result = self.evaluate_single(crop_bounds, raw_obs=raw_obs)
            results.append(result)

        results.sort(key=lambda r: -r.iou)
        return results

    def adaptive_search(
        self,
        raw_obs=None,
        base_bounds: Dict[str, float] = None,
        n_rounds: int = 3,
        n_samples_per_round: int = 20,
    ) -> List[SearchResult]:
        """
        Adaptive search that narrows the search space based on best results.

        Args:
            raw_obs: raw observation dict
            base_bounds: starting crop bounds
            n_rounds: number of refinement rounds
            n_samples_per_round: samples per parameter per round

        Returns:
            List of all SearchResult sorted by IoU
        """
        if base_bounds is None:
            if self.wrapper is not None:
                base_bounds = dict(self.wrapper.crop_bounds)
            else:
                base_bounds = {
                    "xmin": -1.7, "xmax": 0.5,
                    "ymin": -0.4, "ymax": 0.4,
                    "zmin": -0.7, "zmax": 2.5
                }

        # Initial wide search ranges
        search_ranges = {
            "xmin": (-2.5, 0.0),
            "xmax": (-0.5, 1.5),
            "ymin": (-1.0, 0.0),
            "ymax": (0.0, 1.0),
            "zmin": (-1.5, 0.5),
            "zmax": (1.5, 3.5),
        }

        all_results = []
        best_bounds = dict(base_bounds)

        for round_idx in range(n_rounds):
            if self.verbose:
                print(f"\n[AdaptiveSearch] Round {round_idx + 1}/{n_rounds}")

            # Sample points in current search ranges
            round_results = []

            for _ in range(n_samples_per_round * 6):  # 6 params
                crop_bounds = {}
                for param, (lo, hi) in search_ranges.items():
                    crop_bounds[param] = np.random.uniform(lo, hi)

                result = self.evaluate_single(crop_bounds, raw_obs=raw_obs)
                round_results.append(result)
                all_results.append(result)

            # Find best from this round
            round_results.sort(key=lambda r: -r.iou)
            top_k = round_results[:max(5, len(round_results) // 10)]

            # Narrow search ranges around top results
            for param in search_ranges.keys():
                param_vals = [r.crop_bounds[param] for r in top_k]
                param_mean = np.mean(param_vals)
                param_std = np.std(param_vals) + 0.05  # minimum spread

                old_lo, old_hi = search_ranges[param]
                new_lo = max(old_lo, param_mean - 2 * param_std)
                new_hi = min(old_hi, param_mean + 2 * param_std)
                search_ranges[param] = (new_lo, new_hi)

            best_bounds = top_k[0].crop_bounds

            if self.verbose:
                print(f"  Best IoU this round: {top_k[0].iou:.4f}")
                print(f"  Best bounds: {best_bounds}")

        all_results.sort(key=lambda r: -r.iou)
        self.results = all_results

        return all_results

    def save_results(self, path: str):
        """Save results to JSON file."""
        data = {
            "results": [asdict(r) for r in self.results],
            "best": asdict(self.results[0]) if self.results else None,
            "thresh": self.thresh,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        if self.verbose:
            print(f"Saved results to {path}")

    def interpret_alignment(self, result: SearchResult = None) -> str:
        """
        Generate human-readable interpretation of alignment metrics.

        Args:
            result: SearchResult to interpret (defaults to best result)

        Returns:
            String with interpretation
        """
        if result is None:
            if not self.results:
                return "No results to interpret."
            result = self.results[0]

        lines = [
            f"Alignment Analysis:",
            f"  IoU: {result.iou:.4f} ({'good' if result.iou > 0.5 else 'poor' if result.iou < 0.2 else 'moderate'})",
            f"  MSE: {result.mse:.6f}",
        ]

        if result.ctr_delta_xyz is not None:
            dx, dy, dz = result.ctr_delta_xyz
            lines.append(f"\n  Centroid Shift (live - GT in normalized coords):")
            lines.append(f"    X: {dx:+.3f} ({'live shifted RIGHT' if dx > 0.05 else 'live shifted LEFT' if dx < -0.05 else 'aligned'})")
            lines.append(f"    Y: {dy:+.3f} ({'live shifted UP' if dy > 0.05 else 'live shifted DOWN' if dy < -0.05 else 'aligned'})")
            lines.append(f"    Z: {dz:+.3f} ({'live shifted FORWARD' if dz > 0.05 else 'live shifted BACK' if dz < -0.05 else 'aligned'})")

        if result.size_ratio_dhw is not None:
            rz, ry, rx = result.size_ratio_dhw
            lines.append(f"\n  Size Ratio (live / GT):")
            lines.append(f"    X: {rx:.3f} ({'live WIDER' if rx > 1.1 else 'live NARROWER' if rx < 0.9 else 'matched'})")
            lines.append(f"    Y: {ry:.3f} ({'live TALLER' if ry > 1.1 else 'live SHORTER' if ry < 0.9 else 'matched'})")
            lines.append(f"    Z: {rz:.3f} ({'live DEEPER' if rz > 1.1 else 'live SHALLOWER' if rz < 0.9 else 'matched'})")

        lines.append(f"\n  Suggested crop_bounds adjustment:")
        if result.ctr_delta_xyz is not None:
            dx, dy, dz = result.ctr_delta_xyz
            if abs(dx) > 0.05:
                adj = "decrease" if dx > 0 else "increase"
                lines.append(f"    - {adj} xmin/xmax to shift live {'left' if dx > 0 else 'right'}")
            if abs(dy) > 0.05:
                adj = "decrease" if dy > 0 else "increase"
                lines.append(f"    - {adj} ymin/ymax to shift live {'down' if dy > 0 else 'up'}")
            if abs(dz) > 0.05:
                adj = "decrease" if dz > 0 else "increase"
                lines.append(f"    - {adj} zmin/zmax to shift live {'backward' if dz > 0 else 'forward'}")

        return "\n".join(lines)


# =============================================================================
# Standalone comparison function (no wrapper needed)
# =============================================================================

def compare_voxels_standalone(
    live_vox_path: str,
    gt_vox_path: str,
    thresh: float = 0.05,
) -> Dict[str, Any]:
    """
    Compare two voxel files directly.

    Args:
        live_vox_path: path to live voxel .pt file
        gt_vox_path: path to GT voxel .pt file
        thresh: occupancy threshold

    Returns:
        dict with alignment metrics
    """
    live_vox = torch.load(live_vox_path, weights_only=False)
    gt_vox = torch.load(gt_vox_path, weights_only=False)

    metrics = voxel_alignment_metrics(live_vox, gt_vox, thresh=thresh)

    print(f"Comparison: {live_vox_path} vs {gt_vox_path}")
    print(f"  IoU: {metrics['iou']:.4f}")
    print(f"  MSE: {metrics['mse']:.6f}")

    if metrics["ctr_delta_norm_xyz_live_minus_gt"] is not None:
        cd = metrics["ctr_delta_norm_xyz_live_minus_gt"]
        print(f"  Centroid delta (x,y,z): [{cd[0]:.3f}, {cd[1]:.3f}, {cd[2]:.3f}]")

    if metrics["size_ratio_DHW_live_over_gt"] is not None:
        sr = metrics["size_ratio_DHW_live_over_gt"]
        print(f"  Size ratio (z,y,x): [{sr[0]:.3f}, {sr[1]:.3f}, {sr[2]:.3f}]")

    return metrics


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Voxel Alignment Search")
    parser.add_argument("--live-voxel", type=str, help="Path to live voxel .pt file")
    parser.add_argument("--gt-voxel", type=str, help="Path to GT voxel .pt file")
    parser.add_argument("--gt-dir", type=str, help="Directory containing GT voxels (for batch comparison)")
    parser.add_argument("--live-dir", type=str, help="Directory containing live voxels (for batch comparison)")
    parser.add_argument("--thresh", type=float, default=0.05, help="Occupancy threshold")
    parser.add_argument("--output", type=str, default="alignment_results.json", help="Output JSON path")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to compare")

    args = parser.parse_args()

    if args.live_voxel and args.gt_voxel:
        # Single file comparison
        metrics = compare_voxels_standalone(args.live_voxel, args.gt_voxel, args.thresh)

        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved to {args.output}")

    elif args.gt_dir and args.live_dir:
        # Batch comparison
        import glob

        gt_files = sorted(glob.glob(os.path.join(args.gt_dir, "*_voxels.pt")))[:args.num_samples]
        live_files = sorted(glob.glob(os.path.join(args.live_dir, "**/*_voxels.pt"), recursive=True))[:args.num_samples]

        print(f"Found {len(gt_files)} GT files, {len(live_files)} live files")

        all_metrics = []
        for i, (gt_f, live_f) in enumerate(zip(gt_files, live_files)):
            print(f"\n--- Sample {i} ---")
            metrics = compare_voxels_standalone(live_f, gt_f, args.thresh)
            metrics["gt_file"] = gt_f
            metrics["live_file"] = live_f
            all_metrics.append(metrics)

        # Summary
        avg_iou = np.mean([m["iou"] for m in all_metrics])
        avg_mse = np.mean([m["mse"] for m in all_metrics])
        print(f"\n=== Summary ===")
        print(f"Average IoU: {avg_iou:.4f}")
        print(f"Average MSE: {avg_mse:.6f}")

        with open(args.output, "w") as f:
            json.dump({"samples": all_metrics, "avg_iou": avg_iou, "avg_mse": avg_mse}, f, indent=2)
        print(f"Saved to {args.output}")

    elif args.gt_dir:
        # Just analyze GT voxels
        import glob

        gt_files = sorted(glob.glob(os.path.join(args.gt_dir, "*_voxels.pt")))[:args.num_samples]

        print(f"Analyzing {len(gt_files)} GT voxels from {args.gt_dir}")

        for i, gt_f in enumerate(gt_files):
            gt_vox = torch.load(gt_f, weights_only=False)
            stats = voxel_stats(gt_vox, thresh=args.thresh)

            print(f"\n--- GT Sample {i}: {os.path.basename(gt_f)} ---")
            print(f"  Shape: {stats.get('shape', 'N/A')}")
            print(f"  Occupancy: {stats.get('occ_frac', 0):.4f}")
            if not stats.get("empty", True):
                print(f"  BBox min (D,H,W): {stats['mn_DHW']}")
                print(f"  BBox max (D,H,W): {stats['mx_DHW']}")
                print(f"  Size (D,H,W): {stats['size_DHW']}")
                print(f"  Centroid (x,y,z normalized): {[round(x, 3) for x in stats['ctr_norm_xyz']]}")
    else:
        parser.print_help()
