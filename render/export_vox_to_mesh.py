#!/usr/bin/env python3
"""
export_vox_to_mesh.py

Export voxel volumes to PLY meshes for Blender rendering.
Handles RGB voxels with optional alpha, marching cubes extraction,
vertex color transfer, and keypoint export.

Usage:
    from render.export_vox_to_mesh import export_voxel_field_to_ply, export_keypoints_to_json

    # Export mesh
    export_voxel_field_to_ply(
        rgb_vol=rec_vol,           # [3, D, H, W] or [B, 3, D, H, W]
        output_path="out/example_000/rec.ply",
        alpha_vol=None,            # optional [D, H, W]
        iso_level=0.15,            # marching cubes threshold
        world_bounds=(-1, 1),      # coordinate range for mesh vertices
    )

    # Export keypoints
    export_keypoints_to_json(
        keypoints=mu_tot_b0,       # [K, 3] in normalized coords [-1, 1]
        output_path="out/example_000/kp.json",
    )
"""

import os
import json
import numpy as np

# Optional imports with fallbacks
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from skimage import measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


def _to_numpy(x):
    """Convert torch tensor or array-like to numpy array."""
    if x is None:
        return None
    if HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _squeeze_batch(vol, expected_channels=None):
    """
    Squeeze batch dimension if present.

    Args:
        vol: [C, D, H, W], [B, C, D, H, W], [D, H, W], or [B, D, H, W]
        expected_channels: if provided, use to disambiguate shapes

    Returns:
        vol with batch dimension removed
    """
    vol = _to_numpy(vol)
    if vol is None:
        return None

    if vol.ndim == 5:  # [B, C, D, H, W]
        vol = vol[0]
    elif vol.ndim == 4:
        # Could be [C, D, H, W] or [B, D, H, W]
        if expected_channels is not None and vol.shape[0] == expected_channels:
            pass  # Already [C, D, H, W]
        elif vol.shape[0] == 1:  # Likely [B=1, D, H, W]
            vol = vol[0]

    return vol


def compute_occupancy_from_rgb(rgb_vol, threshold=0.1):
    """
    Compute binary occupancy from RGB magnitude.

    Args:
        rgb_vol: [3, D, H, W] RGB values in [0,1] or [-1,1]
        threshold: magnitude threshold for occupancy

    Returns:
        occ: [D, H, W] float occupancy in [0, 1]
    """
    rgb = _to_numpy(rgb_vol)

    # Handle [-1, 1] range
    if rgb.min() < 0:
        rgb = (rgb + 1.0) * 0.5

    # Compute magnitude
    mag = np.sqrt((rgb ** 2).sum(axis=0))  # [D, H, W]

    # Soft occupancy (useful for marching cubes)
    occ = np.clip(mag / (threshold + 1e-8), 0, 1)

    return occ


def voxel_coords_to_world(voxel_coords, grid_shape, world_bounds=(-1, 1)):
    """
    Convert voxel index coordinates to world coordinates.

    Args:
        voxel_coords: [N, 3] voxel indices (z, y, x) or (x, y, z)
        grid_shape: (D, H, W) voxel grid dimensions
        world_bounds: (min, max) world coordinate range

    Returns:
        world_coords: [N, 3] world coordinates
    """
    D, H, W = grid_shape
    lo, hi = world_bounds

    # Normalize to [0, 1] then scale to [lo, hi]
    norm = voxel_coords / np.array([[D-1, H-1, W-1]], dtype=np.float32)
    world = lo + norm * (hi - lo)

    return world


def sample_vertex_colors(vertices, rgb_vol, grid_shape, world_bounds=(-1, 1)):
    """
    Sample RGB colors at vertex positions from voxel grid.

    Args:
        vertices: [V, 3] vertex positions in world coords
        rgb_vol: [3, D, H, W] RGB voxel grid
        grid_shape: (D, H, W)
        world_bounds: (min, max)

    Returns:
        colors: [V, 3] RGB colors in [0, 1]
    """
    D, H, W = grid_shape
    lo, hi = world_bounds

    # World to normalized [0, 1]
    norm = (vertices - lo) / (hi - lo)

    # Normalized to voxel indices
    voxel_coords = norm * np.array([[D-1, H-1, W-1]], dtype=np.float32)

    # Clamp to valid range
    voxel_coords = np.clip(voxel_coords, 0, [D-1, H-1, W-1])

    # Nearest neighbor sampling (could use trilinear for smoother results)
    idx = np.round(voxel_coords).astype(np.int32)
    z_idx, y_idx, x_idx = idx[:, 0], idx[:, 1], idx[:, 2]

    # Sample RGB
    rgb = _to_numpy(rgb_vol)
    if rgb.min() < 0:
        rgb = (rgb + 1.0) * 0.5
    rgb = np.clip(rgb, 0, 1)

    colors = np.stack([
        rgb[0, z_idx, y_idx, x_idx],
        rgb[1, z_idx, y_idx, x_idx],
        rgb[2, z_idx, y_idx, x_idx],
    ], axis=-1)

    return colors


def export_voxel_field_to_ply(
    rgb_vol,
    output_path,
    alpha_vol=None,
    iso_level=0.15,
    world_bounds=(-1, 1),
    include_vertex_colors=True,
    include_normals=True,
    smooth_iterations=0,
    decimate_ratio=None,
):
    """
    Export a voxel field to a PLY mesh using marching cubes.

    Args:
        rgb_vol: [3, D, H, W] or [B, 3, D, H, W] RGB voxel volume
        output_path: path to save the PLY file
        alpha_vol: optional [D, H, W] or [B, D, H, W] occupancy/alpha volume.
                   If None, occupancy is computed from RGB magnitude.
        iso_level: marching cubes threshold (0-1 for alpha, magnitude-based for RGB)
        world_bounds: (min, max) coordinate range for the output mesh
        include_vertex_colors: if True, transfer RGB colors to vertices
        include_normals: if True, compute and include vertex normals
        smooth_iterations: number of Laplacian smoothing iterations (0 = none)
        decimate_ratio: if set, decimate mesh to this fraction of faces (e.g., 0.5)

    Returns:
        dict with mesh statistics (num_vertices, num_faces, bounds)
    """
    if not HAS_SKIMAGE:
        raise ImportError("scikit-image is required for marching cubes. Install with: pip install scikit-image")
    if not HAS_TRIMESH:
        raise ImportError("trimesh is required for mesh export. Install with: pip install trimesh")

    # Prepare volumes
    rgb = _squeeze_batch(rgb_vol, expected_channels=3)
    alpha = _squeeze_batch(alpha_vol)

    if rgb.shape[0] != 3:
        raise ValueError(f"rgb_vol must have 3 channels, got shape {rgb.shape}")

    D, H, W = rgb.shape[1:]
    grid_shape = (D, H, W)

    # Compute occupancy field for marching cubes
    if alpha is not None:
        occ = _to_numpy(alpha)
        if occ.ndim == 4:
            occ = occ[0]
        occ = np.clip(occ, 0, 1)
    else:
        # Compute from RGB magnitude
        occ = compute_occupancy_from_rgb(rgb, threshold=iso_level * 0.5)

    # Ensure occupancy is in [0, 1] and has some signal
    if occ.max() < iso_level:
        print(f"[WARNING] Max occupancy ({occ.max():.4f}) < iso_level ({iso_level}). Mesh will be empty.")

    # Run marching cubes
    # Note: marching_cubes expects [Z, Y, X] ordering which matches our [D, H, W]
    try:
        verts, faces, normals, values = measure.marching_cubes(
            occ,
            level=iso_level,
            spacing=(1.0, 1.0, 1.0),
            gradient_direction='descent',
            allow_degenerate=False,
        )
    except ValueError as e:
        if "Surface level" in str(e) or "No surface" in str(e):
            print(f"[WARNING] No surface found at iso_level={iso_level}. Creating empty mesh.")
            # Create minimal empty mesh
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, 'w') as f:
                f.write("ply\nformat ascii 1.0\nelement vertex 0\nelement face 0\nend_header\n")
            return {"num_vertices": 0, "num_faces": 0, "bounds": None}
        raise

    if len(verts) == 0:
        print(f"[WARNING] Marching cubes produced empty mesh at iso_level={iso_level}")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, 'w') as f:
            f.write("ply\nformat ascii 1.0\nelement vertex 0\nelement face 0\nend_header\n")
        return {"num_vertices": 0, "num_faces": 0, "bounds": None}

    # Convert voxel coords to world coords
    lo, hi = world_bounds
    world_verts = lo + (verts / np.array([[D-1, H-1, W-1]])) * (hi - lo)

    # Create trimesh
    mesh = trimesh.Trimesh(vertices=world_verts, faces=faces, process=False)

    # Add vertex colors
    if include_vertex_colors:
        colors = sample_vertex_colors(verts, rgb, grid_shape, world_bounds=(0, 1))
        # Scale to uint8
        colors_uint8 = (colors * 255).astype(np.uint8)
        # Add alpha channel (fully opaque)
        colors_rgba = np.hstack([colors_uint8, np.full((len(colors_uint8), 1), 255, dtype=np.uint8)])
        mesh.visual.vertex_colors = colors_rgba

    # Compute normals if requested (marching cubes provides them but they can be noisy)
    if include_normals:
        # Use trimesh's normal computation which is more robust
        mesh.fix_normals()

    # Optional: smooth the mesh
    if smooth_iterations > 0:
        trimesh.smoothing.filter_laplacian(mesh, iterations=smooth_iterations)

    # Optional: decimate
    if decimate_ratio is not None and 0 < decimate_ratio < 1:
        target_faces = int(len(mesh.faces) * decimate_ratio)
        if target_faces > 10:
            mesh = mesh.simplify_quadric_decimation(target_faces)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Export to PLY
    mesh.export(output_path, file_type='ply')

    stats = {
        "num_vertices": len(mesh.vertices),
        "num_faces": len(mesh.faces),
        "bounds": {
            "min": mesh.bounds[0].tolist(),
            "max": mesh.bounds[1].tolist(),
        },
        "iso_level": iso_level,
        "grid_shape": list(grid_shape),
        "world_bounds": list(world_bounds),
    }

    print(f"[export] Saved {output_path}: {stats['num_vertices']} verts, {stats['num_faces']} faces")

    return stats


def export_keypoints_to_json(
    keypoints,
    output_path,
    world_bounds=(-1, 1),
    keypoint_order="xyz",
    additional_data=None,
):
    """
    Export keypoints to JSON for Blender import.

    Args:
        keypoints: [K, 3] keypoint positions in normalized coords
        output_path: path to save the JSON file
        world_bounds: (min, max) coordinate range (for documentation/scaling)
        keypoint_order: "xyz" or "zyx" indicating coordinate order
        additional_data: optional dict of extra data to include

    Returns:
        dict with export info
    """
    kp = _to_numpy(keypoints)

    if kp.ndim == 3:  # [B, K, 3]
        kp = kp[0]

    if kp.ndim != 2 or kp.shape[1] != 3:
        raise ValueError(f"keypoints must be [K, 3], got shape {kp.shape}")

    # Build export data
    data = {
        "format_version": "1.0",
        "keypoint_order": keypoint_order,
        "world_bounds": list(world_bounds),
        "num_keypoints": len(kp),
        "keypoints": kp.tolist(),
    }

    if additional_data:
        data["additional"] = additional_data

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"[export] Saved {output_path}: {len(kp)} keypoints")

    return {"num_keypoints": len(kp), "path": output_path}


def export_meta_json(
    output_path,
    iso_level,
    grid_shape,
    world_bounds,
    **extra,
):
    """
    Export metadata JSON for Blender rendering.

    Args:
        output_path: path to save meta.json
        iso_level: marching cubes threshold used
        grid_shape: (D, H, W) voxel grid shape
        world_bounds: (min, max) coordinate bounds
        **extra: additional metadata fields
    """
    data = {
        "format_version": "1.0",
        "iso_level": iso_level,
        "grid_shape": list(grid_shape),
        "world_bounds": list(world_bounds),
    }
    data.update(extra)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    return data


def export_example_for_blender(
    output_dir,
    gt_rgb_vol=None,
    rec_rgb_vol=None,
    keypoints=None,
    gt_alpha_vol=None,
    rec_alpha_vol=None,
    iso_level=0.15,
    world_bounds=(-1, 1),
    example_name="example",
):
    """
    Convenience function to export a complete example for Blender rendering.

    Creates:
        output_dir/
            gt.ply        (if gt_rgb_vol provided)
            rec.ply       (if rec_rgb_vol provided)
            kp.json       (if keypoints provided)
            meta.json     (always)

    Args:
        output_dir: directory to save files
        gt_rgb_vol: [3, D, H, W] ground truth RGB voxels
        rec_rgb_vol: [3, D, H, W] reconstructed RGB voxels
        keypoints: [K, 3] keypoint positions
        gt_alpha_vol: optional [D, H, W] GT occupancy
        rec_alpha_vol: optional [D, H, W] reconstruction occupancy
        iso_level: marching cubes threshold
        world_bounds: coordinate bounds
        example_name: name for metadata

    Returns:
        dict with paths to all exported files
    """
    os.makedirs(output_dir, exist_ok=True)

    results = {"output_dir": output_dir}
    grid_shape = None

    # Export GT mesh
    if gt_rgb_vol is not None:
        gt_path = os.path.join(output_dir, "gt.ply")
        stats = export_voxel_field_to_ply(
            gt_rgb_vol,
            gt_path,
            alpha_vol=gt_alpha_vol,
            iso_level=iso_level,
            world_bounds=world_bounds,
        )
        results["gt_mesh"] = gt_path
        results["gt_stats"] = stats
        grid_shape = stats.get("grid_shape")

    # Export REC mesh
    if rec_rgb_vol is not None:
        rec_path = os.path.join(output_dir, "rec.ply")
        stats = export_voxel_field_to_ply(
            rec_rgb_vol,
            rec_path,
            alpha_vol=rec_alpha_vol,
            iso_level=iso_level,
            world_bounds=world_bounds,
        )
        results["rec_mesh"] = rec_path
        results["rec_stats"] = stats
        if grid_shape is None:
            grid_shape = stats.get("grid_shape")

    # Export keypoints
    if keypoints is not None:
        kp_path = os.path.join(output_dir, "kp.json")
        export_keypoints_to_json(
            keypoints,
            kp_path,
            world_bounds=world_bounds,
        )
        results["keypoints"] = kp_path

    # Export metadata
    meta_path = os.path.join(output_dir, "meta.json")
    export_meta_json(
        meta_path,
        iso_level=iso_level,
        grid_shape=grid_shape or [64, 64, 64],
        world_bounds=world_bounds,
        example_name=example_name,
    )
    results["meta"] = meta_path

    return results


# =============================================================================
# CLI for standalone usage
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export voxel volumes to PLY meshes")
    parser.add_argument("--input", "-i", required=True, help="Input .pt/.npy voxel file")
    parser.add_argument("--output", "-o", required=True, help="Output .ply file")
    parser.add_argument("--iso", type=float, default=0.15, help="Marching cubes iso level")
    parser.add_argument("--bounds", type=float, nargs=2, default=[-1, 1], help="World bounds (min max)")
    parser.add_argument("--no-colors", action="store_true", help="Don't include vertex colors")
    parser.add_argument("--smooth", type=int, default=0, help="Laplacian smoothing iterations")

    args = parser.parse_args()

    # Load input
    ext = os.path.splitext(args.input)[1].lower()
    if ext == ".pt":
        import torch
        vol = torch.load(args.input, map_location="cpu")
    elif ext == ".npy":
        vol = np.load(args.input)
    else:
        raise ValueError(f"Unsupported input format: {ext}")

    # Export
    export_voxel_field_to_ply(
        vol,
        args.output,
        iso_level=args.iso,
        world_bounds=tuple(args.bounds),
        include_vertex_colors=not args.no_colors,
        smooth_iterations=args.smooth,
    )
