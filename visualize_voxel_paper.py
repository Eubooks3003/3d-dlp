#!/usr/bin/env python3
"""
Visualization script for paper figures - 3D Voxel DLP
Generates bounding boxes and segmentation masks in 3D, similar to the 2D visualizations in train_dlp.py

Saves Plotly JSON files that can be loaded and rendered with custom styling.

Usage:
    python visualize_voxel_paper.py --config path/to/config.json --ckpt path/to/checkpoint.pth
"""
import os
import sys
import glob
import json
import argparse
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import plotly.graph_objects as go
import plotly.io as pio

# Fix for loading checkpoints saved with NumPy 2.0+ on older NumPy
if 'numpy._core' not in sys.modules:
    import numpy.core
    sys.modules['numpy._core'] = numpy.core
    for submod in ['multiarray', 'umath', '_multiarray_umath', 'numeric']:
        full_old = f'numpy.core.{submod}'
        full_new = f'numpy._core.{submod}'
        if full_old in sys.modules and full_new not in sys.modules:
            sys.modules[full_new] = sys.modules[full_old]
        elif hasattr(numpy.core, submod) and full_new not in sys.modules:
            sys.modules[full_new] = getattr(numpy.core, submod)

from utils.util_func import get_config
from datasets.point_cloud_datasets.get_dataset import get_point_cloud_dataset, pc_collate
from voxel_models import DLP
from eval.eval_vox import (
    log_rgb_voxels as log_rgb_voxels_eval, filter_topk_kps_3d, print_vol_stats,
    _np, _as_CDHW, _as_DHW, _kp_norm_to_index
)


def save_plotly_json(fig: go.Figure, filepath: str):
    """Save a plotly figure as JSON that can be loaded later."""
    fig_json = pio.to_json(fig)
    with open(filepath, 'w') as f:
        f.write(fig_json)
    print(f"  Saved: {filepath}")


# ----------------------------------------------------------------------
# 3D Bounding Box Utilities
# ----------------------------------------------------------------------

def get_3d_bbox_from_kp_scale(
    kp: np.ndarray,
    scale: np.ndarray,
    D: int, H: int, W: int,
    kp_range: Tuple[float, float] = (-1, 1),
    scale_normalized: bool = True,
    kp_order: Tuple[str, str, str] = ("z", "y", "x")
) -> np.ndarray:
    """
    Compute 3D bounding box coordinates from keypoint position and scale.

    Args:
        kp: [n_kp, 3] keypoint positions in kp_range
        scale: [n_kp, 3] scales (normalized to [0,1] or raw voxel sizes)
        D, H, W: voxel grid dimensions (depth, height, width)
        kp_range: range of keypoint coordinates (default [-1, 1])
        scale_normalized: if True, scale is in [0, 1] range
        kp_order: axis order in kp and scale. DLP uses ("z", "y", "x").

    Returns:
        [n_kp, 6] bounding boxes as (x_min, y_min, z_min, x_max, y_max, z_max)
    """
    n_kp = kp.shape[0]

    # Map axis names to indices and sizes
    axis_to_idx = {ax: i for i, ax in enumerate(kp_order)}
    sizes = {"z": D, "y": H, "x": W}

    # Convert keypoints from normalized to voxel coordinates (use correct axis order)
    r0, r1 = kp_range
    span = r1 - r0

    x_idx = axis_to_idx["x"]
    y_idx = axis_to_idx["y"]
    z_idx = axis_to_idx["z"]

    x_center = (kp[:, x_idx] - r0) / span * (W - 1)
    y_center = (kp[:, y_idx] - r0) / span * (H - 1)
    z_center = (kp[:, z_idx] - r0) / span * (D - 1)

    # Convert scale to half-extents in voxel coordinates
    # scale * size gives full extent, then /2 for half-extent
    if scale_normalized:
        half_x = scale[:, x_idx] * W / 2
        half_y = scale[:, y_idx] * H / 2
        half_z = scale[:, z_idx] * D / 2
    else:
        half_x = scale[:, x_idx] / 2
        half_y = scale[:, y_idx] / 2
        half_z = scale[:, z_idx] / 2

    # Compute bounding box corners
    x_min = np.clip(x_center - half_x, 0, W - 1)
    x_max = np.clip(x_center + half_x, 0, W - 1)
    y_min = np.clip(y_center - half_y, 0, H - 1)
    y_max = np.clip(y_center + half_y, 0, H - 1)
    z_min = np.clip(z_center - half_z, 0, D - 1)
    z_max = np.clip(z_center + half_z, 0, D - 1)

    bboxes = np.stack([x_min, y_min, z_min, x_max, y_max, z_max], axis=-1)
    return bboxes


def compute_3d_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute 3D IoU between two bounding boxes."""
    x1_min, y1_min, z1_min, x1_max, y1_max, z1_max = box1
    x2_min, y2_min, z2_min, x2_max, y2_max, z2_max = box2

    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    zi_min = max(z1_min, z2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    zi_max = min(z1_max, z2_max)

    if xi_max <= xi_min or yi_max <= yi_min or zi_max <= zi_min:
        return 0.0

    inter_vol = (xi_max - xi_min) * (yi_max - yi_min) * (zi_max - zi_min)
    vol1 = (x1_max - x1_min) * (y1_max - y1_min) * (z1_max - z1_min)
    vol2 = (x2_max - x2_min) * (y2_max - y2_min) * (z2_max - z2_min)
    union_vol = vol1 + vol2 - inter_vol

    if union_vol <= 0:
        return 0.0

    return inter_vol / union_vol


def nms_3d(
    bboxes: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float = 0.5,
    score_thresh: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """3D Non-Maximum Suppression."""
    if len(bboxes) == 0:
        return bboxes, np.array([], dtype=int)

    order = np.argsort(scores)[::-1]

    if score_thresh is not None:
        order = order[scores[order] > score_thresh]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        ious = np.array([compute_3d_iou(bboxes[i], bboxes[j]) for j in order[1:]])
        mask = ious < iou_thresh
        order = order[1:][mask]

    keep = np.array(keep)
    return bboxes[keep], keep


def intersection_over_smaller(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute intersection over smaller box volume (for containment suppression)."""
    x1_min, y1_min, z1_min, x1_max, y1_max, z1_max = box1
    x2_min, y2_min, z2_min, x2_max, y2_max, z2_max = box2

    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    zi_min = max(z1_min, z2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    zi_max = min(z1_max, z2_max)

    if xi_max <= xi_min or yi_max <= yi_min or zi_max <= zi_min:
        return 0.0

    inter = (xi_max - xi_min) * (yi_max - yi_min) * (zi_max - zi_min)
    v1 = (x1_max - x1_min) * (y1_max - y1_min) * (z1_max - z1_min)
    v2 = (x2_max - x2_min) * (y2_max - y2_min) * (z2_max - z2_min)
    return inter / (min(v1, v2) + 1e-8)


def nms_3d_iou_or_ios(
    bboxes: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float = 0.3,
    ios_thresh: float = 0.8,
    score_thresh: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """3D NMS with both IoU and IoS (intersection over smaller) suppression.

    This handles nested boxes better than IoU alone.
    """
    if len(bboxes) == 0:
        return bboxes, np.array([], dtype=int)

    order = np.argsort(scores)[::-1]

    if score_thresh is not None:
        order = order[scores[order] > score_thresh]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        rest = order[1:]
        ious = np.array([compute_3d_iou(bboxes[i], bboxes[j]) for j in rest])
        ios = np.array([intersection_over_smaller(bboxes[i], bboxes[j]) for j in rest])

        # Suppress if either overlap metric says "same object"
        mask = (ious < iou_thresh) & (ios < ios_thresh)
        order = rest[mask]

    keep = np.array(keep, dtype=int)
    return bboxes[keep], keep


def draw_3d_bbox_wireframe(
    fig: go.Figure,
    bbox: np.ndarray,
    color: str = "rgba(255, 0, 0, 0.8)",
    width: float = 3,
    name: str = "bbox"
):
    """Draw a 3D bounding box wireframe on a plotly figure."""
    x0, y0, z0, x1, y1, z1 = bbox

    corners = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ])

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    xs, ys, zs = [], [], []
    for i, j in edges:
        xs.extend([corners[i, 0], corners[j, 0], None])
        ys.extend([corners[i, 1], corners[j, 1], None])
        zs.extend([corners[i, 2], corners[j, 2], None])

    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='lines',
        line=dict(color=color, width=width),
        name=name,
        showlegend=True
    ))


def get_largest_connected_component(binary_mask: np.ndarray) -> np.ndarray:
    """Return a mask containing only the largest connected component (26-connectivity)."""
    try:
        from scipy import ndimage
    except ImportError:
        # If scipy not available, return original mask
        return binary_mask

    labeled, num_features = ndimage.label(binary_mask, structure=np.ones((3, 3, 3)))
    if num_features == 0:
        return binary_mask

    # Find the largest component
    component_sizes = ndimage.sum(binary_mask, labeled, range(1, num_features + 1))
    largest_label = np.argmax(component_sizes) + 1

    return labeled == largest_label


def get_bbox_from_alpha_mask_3d(
    mask: np.ndarray,
    threshold: float = 0.5,
    min_voxels: int = 50,
    use_connected_components: bool = True,
    min_thickness: float = 2.0,
    max_extent: Optional[float] = None
) -> Optional[np.ndarray]:
    """Compute 3D bounding box from an alpha mask with filtering.

    Args:
        mask: [D, H, W] alpha mask
        threshold: minimum alpha value
        min_voxels: minimum number of voxels above threshold
        use_connected_components: if True, only use largest connected component
        min_thickness: minimum extent in any dimension (filters sheet-like masks)
        max_extent: maximum extent in any dimension (filters huge boxes)

    Returns:
        [6] bbox as (x_min, y_min, z_min, x_max, y_max, z_max) or None
    """
    binary_mask = mask > threshold

    # Minimum support filter
    count = binary_mask.sum()
    if count < min_voxels:
        return None

    # Keep only largest connected component
    if use_connected_components:
        binary_mask = get_largest_connected_component(binary_mask)
        count = binary_mask.sum()
        if count < min_voxels:
            return None

    indices = np.argwhere(binary_mask)
    if len(indices) == 0:
        return None

    z_min, y_min, x_min = indices.min(axis=0)
    z_max, y_max, x_max = indices.max(axis=0)

    # Extent filters
    dx = x_max - x_min
    dy = y_max - y_min
    dz = z_max - z_min

    # Filter sheet-like boxes (too thin in any dimension)
    if min(dx, dy, dz) < min_thickness:
        return None

    # Filter boxes that are too large
    if max_extent is not None and max(dx, dy, dz) > max_extent:
        return None

    return np.array([x_min, y_min, z_min, x_max, y_max, z_max], dtype=float)


# ----------------------------------------------------------------------
# 3D Segmentation Visualization
# ----------------------------------------------------------------------

def color_map_3d(n_colors: int = 64) -> np.ndarray:
    """Generate a colormap for segmentation visualization."""
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i / max(1, n_colors - 1))[:3] for i in range(n_colors)]
    return (np.array(colors) * 255).astype(np.uint8)


def create_3d_segmentation_volume(
    alpha_masks: np.ndarray,
    scores: np.ndarray,
    score_threshold: float = 0.01,
    mask_threshold: float = 0.3,
    valid_mask: Optional[np.ndarray] = None,
    reweight_by_score: bool = True,
    per_kp_normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a colored segmentation volume from alpha masks using argmax assignment.

    Each voxel is assigned to the keypoint with the highest mask value (if above threshold
    and within the valid_mask region).

    Args:
        alpha_masks: [K, D, H, W] per-keypoint alpha volumes
        scores: [K] scores for each keypoint
        score_threshold: minimum score to consider a keypoint
        mask_threshold: minimum alpha value to assign a voxel
        valid_mask: [D, H, W] boolean mask of occupied voxels (e.g., from RGB magnitude)
        reweight_by_score: multiply alpha by score so low-quality KPs lose
        per_kp_normalize: normalize each KP volume so big-scale KPs don't dominate
    """
    # Ensure numpy
    alpha_masks = np.asarray(alpha_masks).copy()
    scores = np.asarray(scores).reshape(-1)

    K, D, H, W = alpha_masks.shape

    # Normalize each KP volume so big-scale KPs don't "win" everywhere
    if per_kp_normalize:
        denom = alpha_masks.reshape(K, -1).max(axis=1).reshape(K, 1, 1, 1) + 1e-8
        alpha_masks = alpha_masks / denom

    # Reweight by scores so low-quality/background KPs lose
    if reweight_by_score:
        alpha_masks = alpha_masks * scores.reshape(K, 1, 1, 1)

    seg_vol = np.zeros((3, D, H, W), dtype=np.float32)
    instance_vol = np.zeros((D, H, W), dtype=np.int32)

    cmap = color_map_3d(K)

    # Argmax assignment: each voxel goes to the keypoint with highest alpha
    winner = np.argmax(alpha_masks, axis=0)  # [D, H, W]
    max_vals = np.max(alpha_masks, axis=0)   # [D, H, W]

    # Apply threshold: only assign where max value exceeds threshold
    valid = max_vals > mask_threshold

    # Restrict to occupied voxels if valid_mask provided
    if valid_mask is not None:
        valid = valid & valid_mask

    for ki in range(K):
        if ki >= len(scores) or scores[ki] <= score_threshold:
            continue
        voxel_mask = (winner == ki) & valid
        color = cmap[ki % len(cmap)] / 255.0

        for c in range(3):
            seg_vol[c][voxel_mask] = color[c]

        instance_vol[voxel_mask] = ki + 1

    return seg_vol, instance_vol


# ----------------------------------------------------------------------
# Core Visualization Functions
# ----------------------------------------------------------------------

def lock_scene(fig: go.Figure, D: int, H: int, W: int, camera: Optional[Dict] = None,
               aspect_ratio: Optional[tuple] = None):
    """Force a shared scene box for consistent zoom/scale across figures.

    Args:
        fig: Plotly figure to update
        D, H, W: Voxel grid dimensions (depth, height, width)
        camera: Camera settings dict
        aspect_ratio: Optional (x, y, z) tuple for manual aspect ratios.
                     If None, uses "cube" mode (equal axes).
                     If provided, uses "manual" mode with the given ratios.
                     Use (1, 1, z_ratio) where z_ratio < 1 to compress the z-axis
                     for datasets like ShapeNet where z has smaller range.
    """
    if camera is None:
        camera = dict(
            eye=dict(x=1.5, y=1.5, z=1.2),
            up=dict(x=0, y=0, z=1),
        )

    scene_dict = dict(
        xaxis=dict(range=[0, W-1], autorange=False),
        yaxis=dict(range=[0, H-1], autorange=False),
        zaxis=dict(range=[0, D-1], autorange=False),
        camera=camera,
    )

    if aspect_ratio is not None:
        scene_dict["aspectmode"] = "manual"
        scene_dict["aspectratio"] = dict(x=aspect_ratio[0], y=aspect_ratio[1], z=aspect_ratio[2])
    else:
        scene_dict["aspectmode"] = "cube"

    fig.update_layout(
        scene=scene_dict,
        margin=dict(l=0, r=0, t=0, b=0),
    )


def split_fg_bg_by_color(rgb_vol, alpha_thresh: float = 0.05, sat_thresh: float = 0.1):
    """Split an RGB volume into foreground (colorful) and background (grey) using HSV saturation.

    A voxel is considered 'grey' (background) if its HSV saturation is below sat_thresh.
    This correctly handles dark colored voxels (low brightness but high saturation).

    Returns:
        fg_vol: [3, D, H, W] foreground (colorful voxels), zeroed where grey
        bg_vol: [3, D, H, W] background (grey voxels), zeroed where colorful
    """
    if isinstance(rgb_vol, torch.Tensor):
        rgb_vol = rgb_vol.detach().cpu().numpy()
    vol = rgb_vol.copy()
    if vol.min() < 0:
        vol = (vol + 1) * 0.5
    vol = np.clip(vol, 0, 1)

    mag = np.sqrt((vol ** 2).sum(axis=0))  # [D, H, W]
    occupied = mag > alpha_thresh

    # HSV saturation: (max - min) / max per voxel
    cmax = vol.max(axis=0)   # [D, H, W]
    cmin = vol.min(axis=0)   # [D, H, W]
    sat = np.where(cmax > 1e-8, (cmax - cmin) / cmax, 0.0)

    grey_mask = occupied & (sat < sat_thresh)
    color_mask = occupied & (sat >= sat_thresh)

    fg_vol = vol.copy()
    fg_vol[:, grey_mask] = 0.0

    bg_vol = vol.copy()
    bg_vol[:, color_mask] = 0.0

    return fg_vol, bg_vol


def create_rgb_voxel_figure(
    rgb_vol,
    alpha_thresh: float = 0.05,
    topk: int = 60000,
    marker_size: int = 2
) -> go.Figure:
    """Create a plotly figure with RGB voxels."""
    # Convert to numpy if tensor
    if isinstance(rgb_vol, torch.Tensor):
        rgb_vol = rgb_vol.detach().cpu().numpy()

    if rgb_vol.ndim == 5:
        rgb_vol = rgb_vol[0]

    # Normalize first, then compute magnitude (fixes occupancy inconsistency)
    if rgb_vol.min() < 0:
        rgb_vol = (rgb_vol + 1) * 0.5
    rgb_vol = np.clip(rgb_vol, 0, 1)

    mag = np.sqrt((rgb_vol ** 2).sum(axis=0))

    mask = mag > alpha_thresh
    idx = np.argwhere(mask)

    if len(idx) == 0:
        return go.Figure()

    if len(idx) > topk:
        scores = mag[mask]
        sel = np.argpartition(scores, -topk)[-topk:]
        idx = idx[sel]

    z_i, y_i, x_i = idx[:, 0], idx[:, 1], idx[:, 2]
    r = (rgb_vol[0, z_i, y_i, x_i] * 255).astype(np.uint8)
    g = (rgb_vol[1, z_i, y_i, x_i] * 255).astype(np.uint8)
    b = (rgb_vol[2, z_i, y_i, x_i] * 255).astype(np.uint8)

    colors = [f"rgba({r[i]},{g[i]},{b[i]},0.8)" for i in range(len(r))]

    fig = go.Figure(data=[
        go.Scatter3d(
            x=x_i.astype(float).tolist(),
            y=y_i.astype(float).tolist(),
            z=z_i.astype(float).tolist(),
            mode='markers',
            marker=dict(size=marker_size, color=colors),
            name='RGB voxels'
        )
    ])

    return fig


def add_keypoints_to_fig(
    fig: go.Figure,
    kp,
    D: int, H: int, W: int,
    obj_on = None,
    half: float = 2.0,
    kp_order: Tuple[str, str, str] = ("z", "y", "x")
):
    """Add keypoint crosses to a figure.

    Args:
        kp_order: The order of axes in kp[:, :3]. DLP uses ("z", "y", "x").
    """
    # Ensure numpy
    if isinstance(kp, torch.Tensor):
        kp = kp.detach().cpu().numpy()
    if obj_on is not None and isinstance(obj_on, torch.Tensor):
        obj_on = obj_on.detach().cpu().numpy()

    # Map axis names to indices and sizes
    axis_to_idx = {ax: i for i, ax in enumerate(kp_order)}
    sizes = {"z": D, "y": H, "x": W}

    # Convert from normalized [-1,1] to voxel coords using correct axis order
    x_idx = axis_to_idx["x"]
    y_idx = axis_to_idx["y"]
    z_idx = axis_to_idx["z"]

    x = (kp[:, x_idx] + 1) * 0.5 * (W - 1)
    y = (kp[:, y_idx] + 1) * 0.5 * (H - 1)
    z = (kp[:, z_idx] + 1) * 0.5 * (D - 1)
    kp_voxel = np.stack([x, y, z], axis=-1)

    if obj_on is None:
        # All same color
        xs, ys, zs = [], [], []
        for x, y, z in kp_voxel:
            xs += [x - half, x + half, None, x, x, None, x, x, None]
            ys += [y, y, None, y - half, y + half, None, y, y, None]
            zs += [z, z, None, z, z, None, z - half, z + half, None]

        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='lines',
            line=dict(width=6, color='red'),
            name='KPs'
        ))
    else:
        # Color-coded by obj_on
        obj_on = obj_on.flatten()
        for i, ((x, y, z), on_val) in enumerate(zip(kp_voxel, obj_on)):
            on_val = float(np.clip(on_val, 0, 1))
            if on_val < 0.5:
                r, g, b = 255, int(255 * (on_val * 2)), 0
            else:
                r, g, b = int(255 * (1 - (on_val - 0.5) * 2)), 255, 0
            color = f"rgba({r},{g},{b},0.9)"

            xs = [x - half, x + half, None, x, x, None, x, x, None]
            ys = [y, y, None, y - half, y + half, None, y, y, None]
            zs = [z, z, None, z, z, None, z - half, z + half, None]

            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode='lines',
                line=dict(width=6, color=color),
                showlegend=False
            ))


def add_bboxes_to_fig(
    fig: go.Figure,
    bboxes: np.ndarray,
    scores_aligned: np.ndarray,
    kp_indices: np.ndarray,
    cmap: np.ndarray
):
    """Add bounding boxes to a figure.

    Args:
        bboxes: [N, 6] bounding boxes
        scores_aligned: [N] scores aligned with bboxes (not indexed by kp_indices)
        kp_indices: [N] the keypoint index each bbox corresponds to (for coloring)
        cmap: colormap array
    """
    for i, (bbox, kp_idx, sc) in enumerate(zip(bboxes, kp_indices, scores_aligned)):
        color = f"rgba({cmap[kp_idx % len(cmap), 0]}, {cmap[kp_idx % len(cmap), 1]}, {cmap[kp_idx % len(cmap), 2]}, 0.9)"
        draw_3d_bbox_wireframe(
            fig, bbox, color=color, width=3,
            name=f"bbox_kp{kp_idx} (score={float(sc):.2f})"
        )


def add_segmentation_to_fig(
    fig: go.Figure,
    seg_vol: np.ndarray,
    instance_vol: np.ndarray,
    topk: int = 60000,
    marker_size: int = 2
):
    """Add segmentation voxels to a figure."""
    mask = instance_vol > 0
    idx = np.argwhere(mask)

    if len(idx) == 0:
        return

    if len(idx) > topk:
        sel = np.random.choice(len(idx), topk, replace=False)
        idx = idx[sel]

    z_i, y_i, x_i = idx[:, 0], idx[:, 1], idx[:, 2]
    r = (seg_vol[0, z_i, y_i, x_i] * 255).astype(np.uint8)
    g = (seg_vol[1, z_i, y_i, x_i] * 255).astype(np.uint8)
    b = (seg_vol[2, z_i, y_i, x_i] * 255).astype(np.uint8)

    colors = [f"rgba({r[i]},{g[i]},{b[i]},0.8)" for i in range(len(r))]

    fig.add_trace(go.Scatter3d(
        x=x_i.astype(float).tolist(),
        y=y_i.astype(float).tolist(),
        z=z_i.astype(float).tolist(),
        mode='markers',
        marker=dict(size=marker_size, color=colors),
        name='Segmentation'
    ))


@torch.no_grad()
def generate_paper_figures(
    name: str,
    rgb_vol: np.ndarray,
    kp: np.ndarray,
    scale: np.ndarray,
    scores: np.ndarray,
    alpha_masks: Optional[np.ndarray] = None,
    obj_on: Optional[np.ndarray] = None,
    D: int = 32,
    H: int = 32,
    W: int = 32,
    iou_thresh: float = 0.5,
    score_thresh: Optional[float] = None,
    topk: int = 10,
    alpha_thresh: float = 0.05,
    mask_thresh: float = 0.3,
    min_mask_voxels: int = 50,
    obj_on_thresh: float = 0.5,
    output_dir: str = "./paper_figures",
    aspect_ratio: Optional[tuple] = None
) -> Dict[str, str]:
    """
    Generate paper visualization figures and save as Plotly JSON files.

    Returns:
        dict mapping figure names to file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = {}

    # Convert inputs to numpy
    if isinstance(rgb_vol, torch.Tensor):
        rgb_vol = rgb_vol.detach().cpu().numpy()
    if isinstance(kp, torch.Tensor):
        kp = kp.detach().cpu().numpy()
    if isinstance(scale, torch.Tensor):
        scale = scale.detach().cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    if alpha_masks is not None and isinstance(alpha_masks, torch.Tensor):
        alpha_masks = alpha_masks.detach().cpu().numpy()
    if obj_on is not None and isinstance(obj_on, torch.Tensor):
        obj_on = obj_on.detach().cpu().numpy()

    # Ensure correct shapes (batch dim already removed in main loop)
    if rgb_vol.ndim == 5:
        rgb_vol = rgb_vol[0]
    if kp.ndim == 3:
        kp = kp[0]
    if scale.ndim == 3:
        scale = scale[0]
    if scores.ndim == 2:
        scores = scores[0]
    # alpha_masks: expected [K, 1, D, H, W] - don't remove K dimension!
    # Only remove if it has 6 dims (batch still present)
    if alpha_masks is not None and alpha_masks.ndim == 6:
        alpha_masks = alpha_masks[0]
    if obj_on is not None and obj_on.ndim == 2:
        obj_on = obj_on[0]

    print(f"  [DEBUG] In generate_paper_figures: alpha_masks shape = {alpha_masks.shape if alpha_masks is not None else None}")

    cmap = color_map_3d(len(kp))

    # ------------------------------------------------------------------
    # 1. RGB only
    # ------------------------------------------------------------------
    fig_rgb = create_rgb_voxel_figure(rgb_vol, alpha_thresh=alpha_thresh)
    lock_scene(fig_rgb, D, H, W, aspect_ratio=aspect_ratio)
    fpath = os.path.join(output_dir, f"{name}_rgb.plotly.json")
    save_plotly_json(fig_rgb, fpath)
    saved_files['rgb'] = fpath

    # ------------------------------------------------------------------
    # 2. RGB + Keypoints
    # ------------------------------------------------------------------
    fig_rgb_kp = create_rgb_voxel_figure(rgb_vol, alpha_thresh=alpha_thresh)
    add_keypoints_to_fig(fig_rgb_kp, kp, D, H, W, obj_on=obj_on)
    lock_scene(fig_rgb_kp, D, H, W, aspect_ratio=aspect_ratio)
    fpath = os.path.join(output_dir, f"{name}_rgb_kp.plotly.json")
    save_plotly_json(fig_rgb_kp, fpath)
    saved_files['rgb_kp'] = fpath

    # ------------------------------------------------------------------
    # 3. RGB + Bounding Boxes (from keypoint+scale)
    # ------------------------------------------------------------------
    fig_bbox = create_rgb_voxel_figure(rgb_vol, alpha_thresh=alpha_thresh)

    bboxes = get_3d_bbox_from_kp_scale(kp, scale, D, H, W)
    bboxes_nms, kept_indices = nms_3d(bboxes, scores, iou_thresh, score_thresh)

    if len(kept_indices) > topk:
        kept_indices = kept_indices[:topk]
        bboxes_nms = bboxes_nms[:topk]

    # scores_aligned: the scores for the kept bboxes (aligned with bboxes_nms)
    kept_scores = scores[kept_indices]
    add_bboxes_to_fig(fig_bbox, bboxes_nms, kept_scores, kept_indices, cmap)

    # Add keypoint centers (using z,y,x axis order)
    kp_kept = kp[kept_indices]
    x = (kp_kept[:, 2] + 1) * 0.5 * (W - 1)  # x is at index 2 in z,y,x order
    y = (kp_kept[:, 1] + 1) * 0.5 * (H - 1)  # y is at index 1
    z = (kp_kept[:, 0] + 1) * 0.5 * (D - 1)  # z is at index 0
    fig_bbox.add_trace(go.Scatter3d(
        x=x.tolist(), y=y.tolist(), z=z.tolist(),
        mode='markers',
        marker=dict(size=5, color='red', symbol='x'),
        name='KP centers'
    ))

    lock_scene(fig_bbox, D, H, W, aspect_ratio=aspect_ratio)
    fpath = os.path.join(output_dir, f"{name}_bboxes.plotly.json")
    save_plotly_json(fig_bbox, fpath)
    saved_files['bboxes'] = fpath

    # ------------------------------------------------------------------
    # 4. RGB + Bounding Boxes from Alpha Masks
    # ------------------------------------------------------------------
    if alpha_masks is not None:
        # Handle various alpha_masks shapes
        print(f"  [DEBUG] alpha_masks shape for bboxes: {alpha_masks.shape}")

        if alpha_masks.ndim == 5:
            # [K, 1, D, H, W] -> [K, D, H, W]
            masks_3d = alpha_masks[:, 0]
        elif alpha_masks.ndim == 4:
            masks_3d = alpha_masks
        elif alpha_masks.ndim == 3:
            # [D, H, W] - single mask
            masks_3d = alpha_masks[np.newaxis, ...]  # [1, D, H, W]
        else:
            masks_3d = None

        # Check if we have per-keypoint masks
        if masks_3d is not None:
            K_masks = masks_3d.shape[0]
            K_scores = len(scores)
            if K_masks == 1 and K_scores > 1:
                print(f"  [SKIP] alpha_masks has K=1 but {K_scores} keypoints, skipping bboxes_masks")
                masks_3d = None

        if masks_3d is not None:
            fig_bbox_masks = create_rgb_voxel_figure(rgb_vol, alpha_thresh=alpha_thresh)

            mask_bboxes = []
            mask_scores_list = []
            mask_indices = []

            # Build occupancy mask from RGB volume (same as segmentation)
            vol = rgb_vol.copy()
            if vol.min() < 0:
                vol = (vol + 1) * 0.5
            mag = np.sqrt((vol ** 2).sum(axis=0))  # [D, H, W]
            occ_mask = mag > alpha_thresh

            print(f"  [DEBUG] masks_3d.shape={masks_3d.shape}, len(scores)={len(scores)}")
            for ki in range(len(masks_3d)):
                if ki >= len(scores):
                    print(f"  [DEBUG] Skipping ki={ki}, out of range for scores")
                    continue

                # obj_on gating: relaxed for paper visuals (set to 0.0 to disable)
                if obj_on is not None and obj_on_thresh > 0:
                    obj_on_flat = obj_on.flatten()
                    if ki < len(obj_on_flat) and float(obj_on_flat[ki]) < obj_on_thresh:
                        continue

                mask = masks_3d[ki]
                if mask.ndim == 4:
                    mask = mask[0]

                # Use relative threshold like segmentation
                thr = max(mask_thresh, 0.3 * mask.max())

                # Intersect with occupancy (like segmentation)
                binary = (mask > thr) & occ_mask
                count = binary.sum()
                if count < min_mask_voxels:
                    continue

                # Keep only largest connected component
                binary = get_largest_connected_component(binary)
                count = binary.sum()
                if count < min_mask_voxels:
                    continue

                indices = np.argwhere(binary)
                if len(indices) == 0:
                    continue

                z_min, y_min, x_min = indices.min(axis=0)
                z_max, y_max, x_max = indices.max(axis=0)

                bbox = np.array([x_min, y_min, z_min, x_max, y_max, z_max], dtype=float)
                mask_bboxes.append(bbox)
                mask_scores_list.append(scores[ki])
                mask_indices.append(ki)

            if mask_bboxes:
                mask_bboxes = np.array(mask_bboxes)
                mask_scores_arr = np.array(mask_scores_list)
                mask_indices = np.array(mask_indices)

                bboxes_nms, kept_local = nms_3d_iou_or_ios(
                    mask_bboxes, mask_scores_arr,
                    iou_thresh=iou_thresh, ios_thresh=0.8, score_thresh=score_thresh
                )
                kept_indices = mask_indices[kept_local]
                kept_scores = mask_scores_arr[kept_local]

                if len(kept_indices) > topk:
                    kept_indices = kept_indices[:topk]
                    kept_scores = kept_scores[:topk]
                    bboxes_nms = bboxes_nms[:topk]

                # scores_aligned with bboxes_nms, kp_indices for coloring
                add_bboxes_to_fig(fig_bbox_masks, bboxes_nms, kept_scores, kept_indices, cmap)

            lock_scene(fig_bbox_masks, D, H, W, aspect_ratio=aspect_ratio)
            fpath = os.path.join(output_dir, f"{name}_bboxes_masks.plotly.json")
            save_plotly_json(fig_bbox_masks, fpath)
            saved_files['bboxes_masks'] = fpath

    # ------------------------------------------------------------------
    # 5. Segmentation Map
    # ------------------------------------------------------------------
    if alpha_masks is not None:
        # Handle various alpha_masks shapes:
        # Expected: [K, 1, D, H, W] or [K, D, H, W] where K = n_keypoints
        # Might be: [1, D, H, W] or [1, 1, D, H, W] if no per-kp masks

        print(f"  [DEBUG] alpha_masks shape for segmentation: {alpha_masks.shape}")

        # Determine if we have per-keypoint masks
        if alpha_masks.ndim == 5:
            # [K, 1, D, H, W] -> [K, D, H, W]
            masks_3d = alpha_masks[:, 0]
        elif alpha_masks.ndim == 4:
            # Could be [K, D, H, W] or [1, D, H, W]
            masks_3d = alpha_masks
        elif alpha_masks.ndim == 3:
            # [D, H, W] - single mask, skip segmentation
            print(f"  [SKIP] alpha_masks is single volume, skipping segmentation")
            masks_3d = None
        else:
            masks_3d = None

        # Check if K matches number of keypoints (scores)
        if masks_3d is not None:
            K_masks = masks_3d.shape[0]
            K_scores = len(scores)
            if K_masks == 1 and K_scores > 1:
                print(f"  [SKIP] alpha_masks has K=1 but {K_scores} keypoints, skipping segmentation")
                masks_3d = None

        if masks_3d is not None:
            fig_seg = go.Figure()

            # Build occupancy mask from RGB volume (same logic as voxel visibility)
            vol = rgb_vol.copy()
            if vol.min() < 0:  # if [-1,1] range
                vol = (vol + 1) * 0.5
            mag = np.sqrt((vol ** 2).sum(axis=0))  # [D, H, W]
            occ_mask = mag > alpha_thresh

            seg_vol, instance_vol = create_3d_segmentation_volume(
                masks_3d, scores,
                score_threshold=0.01,
                mask_threshold=mask_thresh,
                valid_mask=occ_mask,
                reweight_by_score=True,
                per_kp_normalize=True,
            )

            add_segmentation_to_fig(fig_seg, seg_vol, instance_vol)

            lock_scene(fig_seg, D, H, W, aspect_ratio=aspect_ratio)
            fpath = os.path.join(output_dir, f"{name}_segmentation.plotly.json")
            save_plotly_json(fig_seg, fpath)
            saved_files['segmentation'] = fpath

    print(f"[{name}] Saved {len(saved_files)} Plotly JSON files to {output_dir}")
    return saved_files


# ----------------------------------------------------------------------
# Dataset and Model Loading
# ----------------------------------------------------------------------

class VoxelDirDataset(torch.utils.data.Dataset):
    """Simple dataset that loads voxels from a directory of .pt files."""

    def __init__(self, voxel_dir: str, max_files: int = None):
        self.voxel_dir = voxel_dir
        self.files = []

        nested_pattern = os.path.join(voxel_dir, "demo_*", "frame*_voxels.pt")
        self.files = sorted(glob.glob(nested_pattern))

        if not self.files:
            patterns = [
                os.path.join(voxel_dir, "*_voxels.pt"),
                os.path.join(voxel_dir, "frame_*_voxels.pt"),
                os.path.join(voxel_dir, "*voxels*.pt"),
            ]
            for pattern in patterns:
                self.files = sorted(glob.glob(pattern))
                if self.files:
                    break

        if max_files is not None:
            self.files = self.files[:max_files]

        print(f"[VoxelDirDataset] Found {len(self.files)} voxel files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        vox = torch.load(self.files[idx], map_location="cpu")
        return {"voxels": vox, "path": self.files[idx]}


def load_model_from_config(cfg: Dict[str, Any], device: torch.device) -> DLP:
    """Load model from config."""
    voxel_grid_whd = cfg["voxel_grid_whd"]
    model = DLP(
        cdim=cfg['ch'],
        image_size=voxel_grid_whd[0],
        normalize_rgb=cfg['normalize_rgb'],
        n_kp_per_patch=cfg['n_kp_per_patch'],
        patch_size=cfg['patch_size'],
        anchor_s=cfg['anchor_s'],
        n_kp_enc=cfg['n_kp_enc'],
        n_kp_prior=cfg['n_kp_prior'],
        pad_mode=cfg['pad_mode'],
        dropout=cfg['dropout'],
        features_dist=cfg.get('features_dist', 'gauss'),
        learned_feature_dim=cfg['learned_feature_dim'],
        learned_bg_feature_dim=cfg.get('learned_bg_feature_dim', cfg['learned_feature_dim']),
        n_fg_categories=cfg.get('n_fg_categories', 8),
        n_fg_classes=cfg.get('n_fg_classes', 4),
        n_bg_categories=cfg.get('n_bg_categories', 4),
        n_bg_classes=cfg.get('n_bg_classes', 4),
        scale_std=cfg['scale_std'],
        offset_std=cfg['offset_std'],
        obj_on_alpha=cfg['obj_on_alpha'],
        obj_on_beta=cfg['obj_on_beta'],
        obj_res_from_fc=cfg['obj_res_from_fc'],
        obj_ch_mult_prior=cfg.get('obj_ch_mult_prior', cfg['obj_ch_mult']),
        obj_ch_mult=cfg['obj_ch_mult'],
        obj_base_ch=cfg['obj_base_ch'],
        obj_final_cnn_ch=cfg['obj_final_cnn_ch'],
        bg_res_from_fc=cfg['bg_res_from_fc'],
        bg_ch_mult=cfg['bg_ch_mult'],
        bg_base_ch=cfg['bg_base_ch'],
        bg_final_cnn_ch=cfg['bg_final_cnn_ch'],
        use_resblock=cfg['use_resblock'],
        num_res_blocks=cfg['num_res_blocks'],
        cnn_mid_blocks=cfg.get('cnn_mid_blocks', False),
        mlp_hidden_dim=cfg.get('mlp_hidden_dim', 256),
        pint_enc_layers=cfg['pint_enc_layers'],
        pint_enc_heads=cfg['pint_enc_heads'],
        timestep_horizon=1,
        separate_depth_features=cfg.get('separate_depth_features', False),
        depth_feature_dim=cfg.get('depth_feature_dim', 0),
        split_loss=cfg.get('split_loss', False),
        depth_loss_ratio=cfg.get('depth_loss_ratio', 0.5),
    ).to(device)
    model.eval()
    return model


def find_latest_checkpoint(run_save_dir: str) -> Optional[str]:
    """Find the latest checkpoint in a directory."""
    candidates = []
    for ext in ("*.pth", "*.pt", "*.ckpt", "*.bin"):
        candidates += glob.glob(os.path.join(run_save_dir, ext))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate paper visualization figures for 3D Voxel DLP (saves Plotly JSON)"
    )
    parser.add_argument("--run-dir", "-r", type=str, default=None,
                        help="Path to run folder containing hparams.json and saves/best.pt")
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Path to config JSON (optional if --run-dir provided)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to checkpoint file (optional if --run-dir provided)")
    parser.add_argument("--voxel-dir", type=str, default=None,
                        help="Load voxels from this directory")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=4,
                        help="Maximum number of samples to visualize")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save Plotly JSON files (default: run_dir/paper_figures)")
    parser.add_argument("--iou-thresh", type=float, default=0.3,
                        help="IoU threshold for 3D NMS")
    parser.add_argument("--topk", type=int, default=10,
                        help="Maximum number of bboxes to show")
    parser.add_argument("--alpha-thresh", type=float, default=0.05,
                        help="Alpha threshold for RGB voxel visibility")
    parser.add_argument("--mask-thresh", type=float, default=0.3,
                        help="Threshold for mask binarization in segmentation (like 2D uses 0.3)")
    parser.add_argument("--min-mask-voxels", type=int, default=50,
                        help="Minimum voxels above threshold to create a mask-derived bbox")
    parser.add_argument("--obj-on-thresh", type=float, default=0.5,
                        help="obj_on threshold for filtering background keypoints in mask-bboxes")
    parser.add_argument("--aspect-ratio", type=str, default=None,
                        help="Aspect ratio as 'x,y,z' (e.g., '1,1,0.5' for compressed z-axis). "
                             "Use this for datasets like ShapeNet/Shapes where z has smaller range.")
    parser.add_argument("--color-fg-bg", action="store_true", default=False,
                        help="Use color-based fg/bg split (grey=bg, colorful=fg) instead of model outputs.")
    args = parser.parse_args()

    # Parse aspect ratio if provided
    aspect_ratio = None
    if args.aspect_ratio:
        try:
            parts = [float(x.strip()) for x in args.aspect_ratio.split(',')]
            if len(parts) == 3:
                aspect_ratio = tuple(parts)
            else:
                print(f"[warning] Invalid aspect-ratio format, expected 'x,y,z', got: {args.aspect_ratio}")
        except ValueError:
            print(f"[warning] Could not parse aspect-ratio: {args.aspect_ratio}")

    # Handle --run-dir: auto-discover config and checkpoint
    if args.run_dir is not None:
        run_dir = args.run_dir

        # Find config (hparams.json)
        if args.config is None:
            hparams_path = os.path.join(run_dir, "hparams.json")
            if os.path.isfile(hparams_path):
                args.config = hparams_path
                print(f"[info] Found config: {args.config}")
            else:
                print(f"[error] No hparams.json found in {run_dir}")
                sys.exit(1)

        # Find checkpoint (saves/best.pt)
        if args.ckpt is None:
            best_pt = os.path.join(run_dir, "saves", "best.pt")
            if os.path.isfile(best_pt):
                args.ckpt = best_pt
                print(f"[info] Found checkpoint: {args.ckpt}")
            else:
                # Try other common locations
                for ckpt_path in [
                    os.path.join(run_dir, "best.pt"),
                    os.path.join(run_dir, "saves", "checkpoint.pt"),
                ]:
                    if os.path.isfile(ckpt_path):
                        args.ckpt = ckpt_path
                        print(f"[info] Found checkpoint: {args.ckpt}")
                        break
                else:
                    # Find latest checkpoint
                    ckpt_path = find_latest_checkpoint(os.path.join(run_dir, "saves"))
                    if ckpt_path:
                        args.ckpt = ckpt_path
                        print(f"[info] Found checkpoint: {args.ckpt}")
                    else:
                        print(f"[error] No checkpoint found in {run_dir}/saves/")
                        sys.exit(1)

        # Default output dir to run_dir/paper_figures
        if args.output_dir is None:
            args.output_dir = os.path.join(run_dir, "paper_figures")

    # Validate required args
    if args.config is None:
        print("[error] Must provide either --run-dir or --config")
        sys.exit(1)

    if args.output_dir is None:
        args.output_dir = "./paper_figures"

    # Auto-discover checkpoint from config dir if still not found
    if args.ckpt is None:
        config_dir = os.path.dirname(args.config)
        best_pt = os.path.join(config_dir, "best.pt")
        if os.path.isfile(best_pt):
            args.ckpt = best_pt
            print(f"[info] Auto-discovered checkpoint: {args.ckpt}")
        else:
            print(f"[error] No --ckpt provided and no best.pt found in {config_dir}")
            sys.exit(1)
    elif os.path.isdir(args.ckpt):
        best_pt = os.path.join(args.ckpt, "best.pt")
        if os.path.isfile(best_pt):
            args.ckpt = best_pt
        else:
            ckpt_path = find_latest_checkpoint(args.ckpt)
            if ckpt_path:
                args.ckpt = ckpt_path
            else:
                print(f"[error] No checkpoint found in {args.ckpt}")
                sys.exit(1)

    # Load config
    cfg = get_config(args.config)
    voxel_grid_whd = cfg["voxel_grid_whd"]
    W, H, D = voxel_grid_whd

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[info] Using device: {device}")
    print(f"[info] Voxel grid: W={W}, H={H}, D={D}")

    # Load dataset
    if args.voxel_dir is not None:
        dataset = VoxelDirDataset(args.voxel_dir, max_files=args.max_samples * args.batch_size)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    else:
        ds_name = cfg['ds']
        root = cfg['root']
        base_ds = get_point_cloud_dataset(
            ds_name, root, mode=args.split, max_points=4096, include_rgb=(cfg['ch'] == 6)
        )
        dataloader = DataLoader(base_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Load model
    model = load_model_from_config(cfg, device)

    # Load checkpoint
    print(f"[info] Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)

    state = None
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model", "ema", "ema_state_dict", "net", "weights"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
    if state is None:
        state = ckpt

    def _strip_prefix(d, prefix):
        if not any(k.startswith(prefix) for k in d.keys()):
            return d
        return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in d.items()}

    state = _strip_prefix(state, "module.")
    state = _strip_prefix(state, "model.")
    model.load_state_dict(state, strict=False)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process samples
    sample_idx = 0
    with torch.no_grad():
        for batch in dataloader:
            vox = batch["voxels"].to(device)

            # Forward pass
            model_output = model(vox, warmup=False, with_loss=True)

            # Extract relevant outputs
            gt_vol = model_output['x'][0]
            rec_vol = model_output['rec'][0]

            # Keypoints
            z_base = model_output['z_base'][0]
            mu_offset = model_output['mu_offset'][0]
            mu_tot = z_base + mu_offset

            # Scales
            mu_scale = model_output['mu_scale'][0]
            scale = torch.sigmoid(mu_scale)

            # Scores
            z_base_var = model_output['z_base_var'][0]
            unc = z_base_var[..., :3].sum(-1)
            scores = -unc
            scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

            # Alpha masks - handle various shapes
            # Expected raw shape: [B, K, 1, D, H, W] where K = n_kp
            alpha_masks = model_output.get('alpha_masks', None)
            if alpha_masks is not None:
                print(f"[DEBUG] alpha_masks raw shape: {alpha_masks.shape}")
                alpha_masks = alpha_masks[0]  # Remove batch dim -> [K, 1, D, H, W]
                print(f"[DEBUG] alpha_masks after [0]: {alpha_masks.shape}")

            # Object presence
            obj_on = model_output.get('obj_on', None)
            if obj_on is not None:
                obj_on = obj_on[0]

            # Create subfolder for this sample
            sample_output_dir = os.path.join(args.output_dir, f"sample_{sample_idx:02d}")
            os.makedirs(sample_output_dir, exist_ok=True)

            # Print stats
            print(f"\n=== Sample {sample_idx} ===")
            print(f"[DEBUG] Model output keys: {list(model_output.keys())}")
            print_vol_stats("GT", gt_vol)
            print_vol_stats("REC", rec_vol)
            print(f"[KP] mu_tot range: {mu_tot.min():.3f} to {mu_tot.max():.3f}")
            print(f"[Scale] sigmoid(mu_scale) range: {scale.min():.3f} to {scale.max():.3f}")
            print(f"[Scores] range: {scores.min():.3f} to {scores.max():.3f}")

            # Use the working log_rgb_voxels from eval_vox.py
            # GT (no keypoints)
            fig_gt = log_rgb_voxels_eval(
                name="gt",
                rgb_vol=gt_vol,
                alpha_vol=None,
                KPx=None,
                step=None,
                mode="splat",
                topk=60000,
                alpha_thresh=args.alpha_thresh,
            )
            lock_scene(fig_gt, D, H, W, aspect_ratio=aspect_ratio)
            save_plotly_json(fig_gt, os.path.join(sample_output_dir, "gt_rgb.plotly.json"))

            # Reconstruction with keypoints
            fig_rec_kp = log_rgb_voxels_eval(
                name="rec_kp",
                rgb_vol=rec_vol,
                alpha_vol=None,
                KPx=mu_tot,  # normalized [-1,1] coords
                step=None,
                mode="splat",
                topk=60000,
                alpha_thresh=args.alpha_thresh,
            )
            lock_scene(fig_rec_kp, D, H, W, aspect_ratio=aspect_ratio)
            save_plotly_json(fig_rec_kp, os.path.join(sample_output_dir, "rec_rgb_kp.plotly.json"))

            # Reconstruction without keypoints
            fig_rec = log_rgb_voxels_eval(
                name="rec",
                rgb_vol=rec_vol,
                alpha_vol=None,
                KPx=None,
                step=None,
                mode="splat",
                topk=60000,
                alpha_thresh=args.alpha_thresh,
            )
            lock_scene(fig_rec, D, H, W, aspect_ratio=aspect_ratio)
            save_plotly_json(fig_rec, os.path.join(sample_output_dir, "rec_rgb.plotly.json"))

            # Foreground / Background split
            if args.color_fg_bg:
                # Color-based split: grey voxels are background, colorful are foreground
                fg_vol, bg_vol = split_fg_bg_by_color(rec_vol, alpha_thresh=args.alpha_thresh)
                fig_fg = create_rgb_voxel_figure(fg_vol, alpha_thresh=args.alpha_thresh)
                lock_scene(fig_fg, D, H, W, aspect_ratio=aspect_ratio)
                save_plotly_json(fig_fg, os.path.join(sample_output_dir, "fg.plotly.json"))

                fig_bg = create_rgb_voxel_figure(bg_vol, alpha_thresh=args.alpha_thresh)
                lock_scene(fig_bg, D, H, W, aspect_ratio=aspect_ratio)
                save_plotly_json(fig_bg, os.path.join(sample_output_dir, "bg.plotly.json"))
            else:
                # Model-based split
                fg_only = model_output['dec_objects'][0]
                fig_fg = create_rgb_voxel_figure(fg_only, alpha_thresh=args.alpha_thresh)
                lock_scene(fig_fg, D, H, W, aspect_ratio=aspect_ratio)
                save_plotly_json(fig_fg, os.path.join(sample_output_dir, "fg.plotly.json"))

                bg_mask = model_output.get('bg_mask', None)
                bg = model_output.get('bg', None)
                if bg_mask is not None and bg is not None:
                    bg_only = (bg_mask * bg[:, :3])[0]
                    fig_bg = create_rgb_voxel_figure(bg_only, alpha_thresh=args.alpha_thresh)
                    lock_scene(fig_bg, D, H, W, aspect_ratio=aspect_ratio)
                    save_plotly_json(fig_bg, os.path.join(sample_output_dir, "bg.plotly.json"))

            # Generate bboxes and segmentation using our custom functions
            generate_paper_figures(
                name="rec",
                rgb_vol=rec_vol,
                kp=mu_tot,
                scale=scale,
                scores=scores_norm,
                alpha_masks=alpha_masks,
                obj_on=obj_on,
                D=D, H=H, W=W,
                iou_thresh=args.iou_thresh,
                topk=args.topk,
                alpha_thresh=args.alpha_thresh,
                mask_thresh=args.mask_thresh,
                min_mask_voxels=args.min_mask_voxels,
                obj_on_thresh=args.obj_on_thresh,
                output_dir=sample_output_dir,
                aspect_ratio=aspect_ratio
            )

            sample_idx += 1
            if sample_idx >= args.max_samples:
                break

    print(f"\n[done] Generated visualizations for {sample_idx} samples in {args.output_dir}")
    print(f"\nTo render figures, use a script like:")
    print("""
import json
import plotly.graph_objects as go

with open("sample_00_rec_rgb.plotly.json") as f:
    fig_dict = json.load(f)

fig = go.Figure(fig_dict)
# Customize styling as needed
fig.show()
""")


if __name__ == "__main__":
    main()
