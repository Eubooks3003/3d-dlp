#!/usr/bin/env python3
"""
Compare two voxel datasets for statistical and visual differences.

Usage:
    python compare_voxel_datasets.py \
        --dataset1 /path/to/dataset1 \
        --dataset2 /path/to/dataset2 \
        --wandb --project voxel-comparison

Supports:
    - Voxel cache directories (from preprocess_mimicgen_voxels.py)
    - Direct .pt voxel files
    - Point cloud datasets (will voxelize on the fly)
"""

import os
import sys
import glob
import argparse
import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.eval_vox import log_rgb_voxels


def load_voxel_from_cache(cache_dir: str, max_samples: int = None) -> List[torch.Tensor]:
    """Load voxels from a preprocessed voxel cache directory."""
    voxels = []
    vox_files = sorted(glob.glob(os.path.join(cache_dir, "*_voxels.pt")))

    if max_samples:
        vox_files = vox_files[:max_samples]

    for vox_path in tqdm(vox_files, desc=f"Loading from {os.path.basename(cache_dir)}"):
        vox = torch.load(vox_path, map_location="cpu")
        voxels.append(vox)

    return voxels


def load_voxels_from_pt_files(pattern: str, max_samples: int = None) -> List[torch.Tensor]:
    """Load voxels from a glob pattern of .pt files."""
    voxels = []
    files = sorted(glob.glob(pattern))

    if max_samples:
        files = files[:max_samples]

    for path in tqdm(files, desc="Loading .pt files"):
        vox = torch.load(path, map_location="cpu")
        if isinstance(vox, dict) and "voxels" in vox:
            vox = vox["voxels"]
        voxels.append(vox)

    return voxels


def compute_voxel_stats(voxels: List[torch.Tensor], name: str, max_samples_for_quantile: int = 1_000_000) -> Dict:
    """Compute comprehensive statistics for a list of voxels."""
    # Stack all voxels: [N, C, D, H, W]
    all_vox = torch.stack(voxels, dim=0).float()
    N, C, D, H, W = all_vox.shape

    stats = {
        "name": name,
        "num_samples": N,
        "shape": (C, D, H, W),
    }

    # Per-channel statistics
    for c in range(C):
        channel = all_vox[:, c]  # [N, D, H, W]
        flat = channel.reshape(-1)

        # Basic stats
        stats[f"ch{c}_mean"] = float(flat.mean())
        stats[f"ch{c}_std"] = float(flat.std())
        stats[f"ch{c}_min"] = float(flat.min())
        stats[f"ch{c}_max"] = float(flat.max())

        # Percentiles - subsample if too large
        percentiles = [1, 5, 25, 50, 75, 95, 99]
        if flat.numel() > max_samples_for_quantile:
            # Subsample for quantile computation
            idx = torch.randperm(flat.numel())[:max_samples_for_quantile]
            flat_sub = flat[idx]
        else:
            flat_sub = flat

        for p in percentiles:
            stats[f"ch{c}_p{p}"] = float(torch.quantile(flat_sub, p / 100.0))

        # Non-zero statistics
        nonzero_mask = flat.abs() > 1e-6
        stats[f"ch{c}_nonzero_frac"] = float(nonzero_mask.float().mean())
        if nonzero_mask.any():
            nz = flat[nonzero_mask]
            stats[f"ch{c}_nonzero_mean"] = float(nz.mean())
            stats[f"ch{c}_nonzero_std"] = float(nz.std())

    # Overall (across all channels)
    flat_all = all_vox.reshape(-1)
    stats["overall_mean"] = float(flat_all.mean())
    stats["overall_std"] = float(flat_all.std())
    stats["overall_min"] = float(flat_all.min())
    stats["overall_max"] = float(flat_all.max())

    # Sparsity
    stats["sparsity"] = float((flat_all.abs() < 1e-6).float().mean())

    # Per-sample statistics (for distribution comparison)
    sample_means = all_vox.reshape(N, -1).mean(dim=1)
    sample_stds = all_vox.reshape(N, -1).std(dim=1)
    stats["sample_mean_mean"] = float(sample_means.mean())
    stats["sample_mean_std"] = float(sample_means.std())
    stats["sample_std_mean"] = float(sample_stds.mean())
    stats["sample_std_std"] = float(sample_stds.std())

    return stats


def _to_numpy_safe(tensor):
    """Convert tensor to numpy, falling back to list if numpy not supported."""
    try:
        return tensor.cpu().numpy()
    except RuntimeError:
        return np.array(tensor.cpu().tolist())


def compute_histogram(voxels: List[torch.Tensor], bins: int = 100,
                      range_min: float = -1.0, range_max: float = 1.0) -> Dict:
    """Compute histogram of voxel values using pure PyTorch."""
    all_vox = torch.stack(voxels, dim=0).float()
    C = all_vox.shape[1]

    histograms = {}
    bin_edges = torch.linspace(range_min, range_max, bins + 1)

    for c in range(C):
        flat = all_vox[:, c].reshape(-1)
        hist = torch.histogram(flat, bins=bin_edges)[0].float()
        hist = hist / (hist.sum() + 1e-10)  # normalize to density
        histograms[f"ch{c}"] = _to_numpy_safe(hist)

    # Overall
    flat_all = all_vox.reshape(-1)
    hist_all = torch.histogram(flat_all, bins=bin_edges)[0].float()
    hist_all = hist_all / (hist_all.sum() + 1e-10)
    histograms["overall"] = _to_numpy_safe(hist_all)
    histograms["bin_edges"] = _to_numpy_safe(bin_edges)

    return histograms


def compute_comparison_metrics(voxels1: List[torch.Tensor], voxels2: List[torch.Tensor],
                               stats1: Dict, stats2: Dict) -> Dict:
    """Compute metrics comparing two datasets."""
    metrics = {}

    # Mean/std differences
    for key in stats1:
        if isinstance(stats1[key], (int, float)) and key in stats2:
            if isinstance(stats2[key], (int, float)):
                metrics[f"diff_{key}"] = stats1[key] - stats2[key]
                if stats1[key] != 0:
                    metrics[f"reldiff_{key}"] = (stats1[key] - stats2[key]) / abs(stats1[key])

    # Sample-wise comparison (if same number of samples)
    n1, n2 = len(voxels1), len(voxels2)
    n_compare = min(n1, n2)

    if n_compare > 0:
        # Per-sample MSE
        mses = []
        maes = []
        for i in range(n_compare):
            v1 = voxels1[i].float()
            v2 = voxels2[i].float()
            if v1.shape == v2.shape:
                mse = ((v1 - v2) ** 2).mean().item()
                mae = (v1 - v2).abs().mean().item()
                mses.append(mse)
                maes.append(mae)

        if mses:
            metrics["pairwise_mse_mean"] = float(np.mean(mses))
            metrics["pairwise_mse_std"] = float(np.std(mses))
            metrics["pairwise_mae_mean"] = float(np.mean(maes))
            metrics["pairwise_mae_std"] = float(np.std(maes))

    # Histogram comparison (KL divergence approximation)
    hist1 = compute_histogram(voxels1)
    hist2 = compute_histogram(voxels2)

    for key in ["overall", "ch0", "ch1", "ch2"]:
        if key in hist1 and key in hist2:
            h1 = hist1[key] + 1e-10  # avoid log(0)
            h2 = hist2[key] + 1e-10
            # Symmetric KL
            kl_12 = np.sum(h1 * np.log(h1 / h2))
            kl_21 = np.sum(h2 * np.log(h2 / h1))
            metrics[f"kl_symmetric_{key}"] = float((kl_12 + kl_21) / 2)

            # Wasserstein (1D, approximated)
            cdf1 = np.cumsum(h1) / np.sum(h1)
            cdf2 = np.cumsum(h2) / np.sum(h2)
            metrics[f"wasserstein_{key}"] = float(np.mean(np.abs(cdf1 - cdf2)))

    return metrics


def print_comparison_report(stats1: Dict, stats2: Dict, metrics: Dict):
    """Print a formatted comparison report."""
    print("\n" + "=" * 80)
    print("VOXEL DATASET COMPARISON REPORT")
    print("=" * 80)

    print(f"\nDataset 1: {stats1['name']}")
    print(f"  Samples: {stats1['num_samples']}, Shape: {stats1['shape']}")
    print(f"  Overall mean: {stats1['overall_mean']:.6f}, std: {stats1['overall_std']:.6f}")
    print(f"  Sparsity: {stats1['sparsity']:.4f}")

    print(f"\nDataset 2: {stats2['name']}")
    print(f"  Samples: {stats2['num_samples']}, Shape: {stats2['shape']}")
    print(f"  Overall mean: {stats2['overall_mean']:.6f}, std: {stats2['overall_std']:.6f}")
    print(f"  Sparsity: {stats2['sparsity']:.4f}")

    print("\n" + "-" * 40)
    print("PER-CHANNEL COMPARISON")
    print("-" * 40)

    C = stats1['shape'][0]
    for c in range(C):
        ch_name = ["R", "G", "B"][c] if C == 3 else f"Ch{c}"
        print(f"\n[{ch_name}]")
        print(f"  Mean:  {stats1[f'ch{c}_mean']:.6f} vs {stats2[f'ch{c}_mean']:.6f}  (diff: {metrics.get(f'diff_ch{c}_mean', 0):.6f})")
        print(f"  Std:   {stats1[f'ch{c}_std']:.6f} vs {stats2[f'ch{c}_std']:.6f}  (diff: {metrics.get(f'diff_ch{c}_std', 0):.6f})")
        print(f"  Range: [{stats1[f'ch{c}_min']:.4f}, {stats1[f'ch{c}_max']:.4f}] vs [{stats2[f'ch{c}_min']:.4f}, {stats2[f'ch{c}_max']:.4f}]")
        print(f"  Nonzero frac: {stats1[f'ch{c}_nonzero_frac']:.4f} vs {stats2[f'ch{c}_nonzero_frac']:.4f}")

    print("\n" + "-" * 40)
    print("DISTRIBUTION METRICS")
    print("-" * 40)

    if "pairwise_mse_mean" in metrics:
        print(f"\nPairwise MSE: {metrics['pairwise_mse_mean']:.6f} +/- {metrics['pairwise_mse_std']:.6f}")
        print(f"Pairwise MAE: {metrics['pairwise_mae_mean']:.6f} +/- {metrics['pairwise_mae_std']:.6f}")

    for key in ["overall", "ch0", "ch1", "ch2"]:
        if f"kl_symmetric_{key}" in metrics:
            print(f"\n{key}:")
            print(f"  Symmetric KL: {metrics[f'kl_symmetric_{key}']:.6f}")
            print(f"  Wasserstein:  {metrics[f'wasserstein_{key}']:.6f}")

    print("\n" + "=" * 80)


def log_comparison_to_wandb(voxels1: List[torch.Tensor], voxels2: List[torch.Tensor],
                            stats1: Dict, stats2: Dict, metrics: Dict,
                            num_samples: int = 5):
    """Log comparison visualizations to wandb."""
    import wandb

    # Log scalar metrics
    wandb.log({f"stats/ds1/{k}": v for k, v in stats1.items() if isinstance(v, (int, float))})
    wandb.log({f"stats/ds2/{k}": v for k, v in stats2.items() if isinstance(v, (int, float))})
    wandb.log({f"metrics/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})

    # Log sample voxels side by side
    n_compare = min(num_samples, len(voxels1), len(voxels2))

    for i in range(n_compare):
        v1 = voxels1[i]
        v2 = voxels2[i]

        # Dataset 1 sample
        log_rgb_voxels(
            name=f"samples/ds1_sample_{i}",
            rgb_vol=v1,
            alpha_vol=None,
            KPx=None,
            step=i,
            mode="splat",
            topk=60000,
            alpha_thresh=0.05,
            pad=2.0,
            show_axes=True,
        )

        # Dataset 2 sample
        log_rgb_voxels(
            name=f"samples/ds2_sample_{i}",
            rgb_vol=v2,
            alpha_vol=None,
            KPx=None,
            step=i,
            mode="splat",
            topk=60000,
            alpha_thresh=0.05,
            pad=2.0,
            show_axes=True,
        )

        # Difference visualization (if same shape)
        if v1.shape == v2.shape:
            diff = (v1 - v2).abs()
            # Scale diff for visibility
            diff_scaled = diff / (diff.max() + 1e-8)
            log_rgb_voxels(
                name=f"samples/diff_sample_{i}",
                rgb_vol=diff_scaled,
                alpha_vol=None,
                KPx=None,
                step=i,
                mode="splat",
                topk=60000,
                alpha_thresh=0.01,  # lower threshold to see small diffs
                pad=2.0,
                show_axes=True,
            )

    # Log histograms
    hist1 = compute_histogram(voxels1)
    hist2 = compute_histogram(voxels2)

    try:
        import matplotlib.pyplot as plt

        for key in ["overall", "ch0", "ch1", "ch2"]:
            if key in hist1 and key in hist2:
                fig, ax = plt.subplots(figsize=(10, 6))
                bin_centers = (hist1["bin_edges"][:-1] + hist1["bin_edges"][1:]) / 2

                ax.plot(bin_centers, hist1[key], label=f"Dataset 1", alpha=0.7)
                ax.plot(bin_centers, hist2[key], label=f"Dataset 2", alpha=0.7)
                ax.fill_between(bin_centers, hist1[key], alpha=0.3)
                ax.fill_between(bin_centers, hist2[key], alpha=0.3)

                ax.set_xlabel("Voxel Value")
                ax.set_ylabel("Density")
                ax.set_title(f"Distribution Comparison - {key}")
                ax.legend()
                ax.grid(True, alpha=0.3)

                wandb.log({f"histograms/{key}": wandb.Image(fig)})
                plt.close(fig)
    except ImportError:
        print("[warn] matplotlib not available, skipping histogram plots")


def load_voxels_generic(directory: str, max_samples: int = None) -> List[torch.Tensor]:
    """Load voxels from any directory with *_voxels.pt or *voxels*.pt files."""
    voxels = []

    # Try different patterns
    patterns = [
        os.path.join(directory, "*_voxels.pt"),      # 000000_voxels.pt
        os.path.join(directory, "frame_*_voxels.pt"), # frame_000000_voxels.pt
        os.path.join(directory, "*voxels*.pt"),      # any voxels file
    ]

    vox_files = []
    for pattern in patterns:
        vox_files = sorted(glob.glob(pattern))
        if vox_files:
            break

    if not vox_files:
        return []

    if max_samples:
        vox_files = vox_files[:max_samples]

    for vox_path in tqdm(vox_files, desc=f"Loading from {os.path.basename(directory)}"):
        vox = torch.load(vox_path, map_location="cpu")
        voxels.append(vox)

    return voxels


def load_dataset(path: str, max_samples: int = None) -> Tuple[List[torch.Tensor], str]:
    """Auto-detect dataset type and load voxels."""
    name = os.path.basename(path.rstrip("/"))

    # Check if it's a voxel cache directory
    if os.path.isdir(path):
        manifest_path = os.path.join(path, "manifest.json")
        if os.path.exists(manifest_path):
            print(f"[info] Detected voxel cache directory: {path}")
            voxels = load_voxel_from_cache(path, max_samples)
            return voxels, name

        # Check for *_voxels.pt pattern (generic loader handles multiple patterns)
        vox_files = glob.glob(os.path.join(path, "*voxels*.pt"))
        if vox_files:
            print(f"[info] Detected voxel .pt files in directory: {path}")
            voxels = load_voxels_generic(path, max_samples)
            return voxels, name

        # Check for any .pt files
        pt_files = glob.glob(os.path.join(path, "*.pt"))
        if pt_files:
            print(f"[info] Loading .pt files from directory: {path}")
            voxels = load_voxels_from_pt_files(os.path.join(path, "*.pt"), max_samples)
            return voxels, name

    # Single file
    if os.path.isfile(path) and path.endswith(".pt"):
        print(f"[info] Loading single .pt file: {path}")
        vox = torch.load(path, map_location="cpu")
        if isinstance(vox, list):
            return vox[:max_samples] if max_samples else vox, name
        return [vox], name

    # Glob pattern
    if "*" in path:
        print(f"[info] Loading from glob pattern: {path}")
        voxels = load_voxels_from_pt_files(path, max_samples)
        return voxels, os.path.dirname(path)

    raise ValueError(f"Could not determine how to load dataset from: {path}")


def main():
    ap = argparse.ArgumentParser(description="Compare two voxel datasets")
    ap.add_argument("--dataset1", "-d1", type=str, required=True,
                    help="Path to first dataset (voxel cache dir, .pt file, or glob pattern)")
    ap.add_argument("--dataset2", "-d2", type=str, required=True,
                    help="Path to second dataset")
    ap.add_argument("--name1", type=str, default=None, help="Name for dataset 1")
    ap.add_argument("--name2", type=str, default=None, help="Name for dataset 2")
    ap.add_argument("--max-samples", type=int, default=100,
                    help="Maximum samples to load per dataset (default: 100)")
    ap.add_argument("--num-vis-samples", type=int, default=5,
                    help="Number of samples to visualize in wandb (default: 5)")
    ap.add_argument("--wandb", action="store_true", help="Log to wandb")
    ap.add_argument("--project", type=str, default="voxel-comparison",
                    help="Wandb project name")
    ap.add_argument("--name", type=str, default=None, help="Wandb run name")
    ap.add_argument("--save-report", type=str, default=None,
                    help="Save comparison report to JSON file")

    args = ap.parse_args()

    # Load datasets
    print("\nLoading datasets...")
    voxels1, auto_name1 = load_dataset(args.dataset1, args.max_samples)
    voxels2, auto_name2 = load_dataset(args.dataset2, args.max_samples)

    name1 = args.name1 or auto_name1
    name2 = args.name2 or auto_name2

    print(f"\nDataset 1 ({name1}): {len(voxels1)} samples")
    print(f"Dataset 2 ({name2}): {len(voxels2)} samples")

    if len(voxels1) == 0 or len(voxels2) == 0:
        print("[error] One or both datasets are empty!")
        sys.exit(1)

    # Check shapes
    shape1 = voxels1[0].shape
    shape2 = voxels2[0].shape
    print(f"Shape 1: {shape1}")
    print(f"Shape 2: {shape2}")

    if shape1 != shape2:
        print("[warn] Datasets have different voxel shapes! Some comparisons may not be valid.")

    # Compute statistics
    print("\nComputing statistics...")
    stats1 = compute_voxel_stats(voxels1, name1)
    stats2 = compute_voxel_stats(voxels2, name2)

    # Compute comparison metrics
    print("Computing comparison metrics...")
    metrics = compute_comparison_metrics(voxels1, voxels2, stats1, stats2)

    # Print report
    print_comparison_report(stats1, stats2, metrics)

    # Save report if requested
    if args.save_report:
        report = {
            "dataset1": stats1,
            "dataset2": stats2,
            "metrics": metrics,
        }
        with open(args.save_report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved report to: {args.save_report}")

    # Log to wandb
    if args.wandb:
        import wandb

        run_name = args.name or f"compare-{name1}-vs-{name2}"
        wandb.init(project=args.project, name=run_name, config={
            "dataset1": args.dataset1,
            "dataset2": args.dataset2,
            "max_samples": args.max_samples,
        })

        print("\nLogging to wandb...")
        log_comparison_to_wandb(voxels1, voxels2, stats1, stats2, metrics,
                                num_samples=args.num_vis_samples)

        wandb.finish()
        print("Wandb logging complete.")

    print("\nDone!")


if __name__ == "__main__":
    main()
