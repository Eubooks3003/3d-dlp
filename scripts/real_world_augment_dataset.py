#!/usr/bin/env python3
"""
Tabletop Scene Augmentation - Label-based extraction with plane math.
"""

import numpy as np
from pathlib import Path
import struct
from dataclasses import dataclass
from typing import Dict, List, Tuple
from tqdm import tqdm


@dataclass
class PointCloud:
    xyz: np.ndarray  # (N, 3) float32
    rgb: np.ndarray  # (N, 3) uint8

    def __len__(self):
        return len(self.xyz)

    def copy(self):
        return PointCloud(self.xyz.copy(), self.rgb.copy())

    def split_components(self, voxel_size: float = 0.02, min_points: int = 500) -> List['PointCloud']:
        """Split into connected components using voxel grid connectivity."""
        if len(self) < min_points:
            return [self]

        # Voxelize
        base = self.xyz.min(axis=0)
        ijk = np.floor((self.xyz - base) / voxel_size).astype(np.int32)

        # Map voxel -> point indices
        vox2idx: Dict[tuple, List[int]] = {}
        for i, key in enumerate(map(tuple, ijk)):
            vox2idx.setdefault(key, []).append(i)

        # BFS to find connected components
        visited = set()
        components = []
        neighbors = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

        for start_vox in vox2idx.keys():
            if start_vox in visited:
                continue

            # BFS from this voxel
            stack = [start_vox]
            visited.add(start_vox)
            component_voxels = []

            while stack:
                cur = stack.pop()
                component_voxels.append(cur)
                for dx, dy, dz in neighbors:
                    nxt = (cur[0]+dx, cur[1]+dy, cur[2]+dz)
                    if nxt in vox2idx and nxt not in visited:
                        visited.add(nxt)
                        stack.append(nxt)

            # Collect point indices for this component
            indices = []
            for vox in component_voxels:
                indices.extend(vox2idx[vox])

            if len(indices) >= min_points:
                idx_arr = np.array(indices, dtype=np.int32)
                components.append(PointCloud(self.xyz[idx_arr], self.rgb[idx_arr]))

        # Sort by size (largest first)
        components.sort(key=lambda c: len(c), reverse=True)
        return components if components else [self]


@dataclass
class TablePlane:
    """Represents the table surface plane: n · x = d"""
    normal: np.ndarray  # (3,) unit normal pointing UP (toward objects)
    d: float            # plane offset
    u: np.ndarray       # (3,) first basis vector in plane
    v: np.ndarray       # (3,) second basis vector in plane

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Signed distance from points to plane (positive = above)."""
        return points @ self.normal - self.d

    def project_to_uv(self, points: np.ndarray) -> np.ndarray:
        """Project 3D points to 2D UV coordinates on the plane."""
        return np.stack([points @ self.u, points @ self.v], axis=-1)

    def uv_to_3d(self, uv: np.ndarray) -> np.ndarray:
        """Convert UV coordinates back to 3D point on plane."""
        return self.u * uv[0] + self.v * uv[1] + self.normal * self.d


# ========================== Geometry helpers ==========================

def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def fit_plane_ransac(points: np.ndarray, n_iter: int = 200, threshold: float = 0.01) -> Tuple[np.ndarray, float]:
    """Fit plane using RANSAC. Returns (normal, d) where normal · x = d."""
    best_inliers = 0
    best_normal, best_d = None, None

    for _ in range(n_iter):
        idx = np.random.choice(len(points), 3, replace=False)
        p1, p2, p3 = points[idx]

        v1, v2 = p2 - p1, p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal = normal / norm
        d = np.dot(normal, p1)

        distances = np.abs(points @ normal - d)
        inliers = np.sum(distances < threshold)

        if inliers > best_inliers:
            best_inliers = inliers
            best_normal, best_d = normal, d

    return best_normal.astype(np.float32), float(best_d)


def make_plane_frame(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Create orthonormal basis (u, v) for a plane with given normal."""
    n = normalize(normal.astype(np.float32))

    # Pick a vector not parallel to normal
    up = np.array([0, 0, 1], dtype=np.float32)
    if abs(np.dot(up, n)) > 0.95:
        up = np.array([1, 0, 0], dtype=np.float32)

    u = normalize(np.cross(up, n))
    v = normalize(np.cross(n, u))
    return u, v


# ========================== I/O ==========================

def load_ply_binary(ply_path: Path) -> PointCloud:
    """Load a binary big-endian PLY file."""
    with open(ply_path, 'rb') as f:
        header_lines = []
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            if line == 'end_header':
                break

        num_vertices = None
        for line in header_lines:
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
                break

        xyz = np.zeros((num_vertices, 3), dtype=np.float32)
        rgb = np.zeros((num_vertices, 3), dtype=np.uint8)

        for i in range(num_vertices):
            data = f.read(15)
            x, y, z = struct.unpack('>fff', data[:12])
            r, g, b = struct.unpack('BBB', data[12:15])
            xyz[i] = [x, y, z]
            rgb[i] = [r, g, b]

    return PointCloud(xyz, rgb)


def load_labels(label_path: Path) -> np.ndarray:
    with open(label_path, 'r') as f:
        lines = f.readlines()
    labels = np.array([int(line.strip()) for line in lines[1:]], dtype=np.int32)
    return labels


def save_ply_ascii(pc: PointCloud, output_path: Path):
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(pc)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(len(pc)):
            x, y, z = pc.xyz[i]
            r, g, b = pc.rgb[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


def translate(pc: PointCloud, offset: np.ndarray) -> PointCloud:
    """Translate a point cloud."""
    return PointCloud(pc.xyz + offset, pc.rgb.copy())


def move_object_on_plane(obj: PointCloud, plane: TablePlane, target_uv: np.ndarray) -> PointCloud:
    """
    Move object so its contact point (bottom-most along plane normal) lands at target_uv.
    Then snap it to sit on the plane surface.
    """
    # Find contact point (bottom-most point along normal)
    signed_dist = plane.signed_distance(obj.xyz)
    contact_idx = np.argmin(signed_dist)
    contact_point = obj.xyz[contact_idx]

    # Target 3D position on plane
    target_3d = plane.uv_to_3d(target_uv)

    # Translate so contact point goes to target
    offset = target_3d - contact_point
    moved = translate(obj, offset)

    # Snap to plane: lift so minimum distance is ~0 (small clearance)
    new_dist = plane.signed_distance(moved.xyz)
    min_dist = float(new_dist.min())
    clearance = 0.003  # 3mm above plane
    if min_dist < clearance:
        lift = clearance - min_dist
        moved = translate(moved, plane.normal * lift)

    return moved


def check_collision(obj1: PointCloud, obj2: PointCloud, margin: float = 0.02) -> bool:
    """Check if two objects' bounding boxes overlap."""
    min1, max1 = obj1.xyz.min(axis=0), obj1.xyz.max(axis=0)
    min2, max2 = obj2.xyz.min(axis=0), obj2.xyz.max(axis=0)
    min1 -= margin
    max1 += margin
    return bool(np.all(max1 >= min2) and np.all(max2 >= min1))


def find_table_label(pc: PointCloud, labels: np.ndarray) -> int:
    """Find the table label (largest medium-sized planar segment)."""
    best_label, best_count = None, 0

    for label in np.unique(labels):
        mask = labels == label
        pts = pc.xyz[mask]
        if len(pts) < 10000:
            continue

        size = pts.max(axis=0) - pts.min(axis=0)
        xy_size = max(size[0], size[1])

        # Table: 0.5m - 2.5m in XY extent
        if 0.5 <= xy_size <= 2.5 and len(pts) > best_count:
            best_label = label
            best_count = len(pts)

    return best_label


def find_background_labels(pc: PointCloud, labels: np.ndarray, table_label: int) -> set:
    """Find background labels (large segments that aren't the table)."""
    background = set()

    for label in np.unique(labels):
        if label == table_label:
            continue

        mask = labels == label
        pts = pc.xyz[mask]
        size = pts.max(axis=0) - pts.min(axis=0)

        # Background: very large (walls, floor) or huge point count
        if max(size[0], size[1], size[2]) > 2.0 or len(pts) > 200000:
            background.add(int(label))

    return background


def process_scene(scene_num: int, data_dir: Path, output_dir: Path,
                  num_augmentations: int, verbose: bool = True) -> bool:
    """Process a single scene and generate augmentations."""
    try:
        ply_path = data_dir / f"{scene_num:02d}.ply"
        label_path = data_dir / f"{scene_num:02d}.label"

        if not ply_path.exists():
            tqdm.write(f"  Scene {scene_num:02d}: PLY not found, skipping")
            return False

        pc = load_ply_binary(ply_path)
        labels = load_labels(label_path)

        if verbose:
            tqdm.write(f"  Scene {scene_num:02d}: Loaded {len(pc):,} points")

        # Auto-detect table and background
        table_label = find_table_label(pc, labels)
        if table_label is None:
            tqdm.write(f"  Scene {scene_num:02d}: No table found, skipping")
            return False

        background_labels = find_background_labels(pc, labels, table_label)

        if verbose:
            tqdm.write(f"  Scene {scene_num:02d}: Table label: {table_label}, Background: {background_labels}")

        # Extract table
        table_mask = labels == table_label
        table = PointCloud(pc.xyz[table_mask], pc.rgb[table_mask])

        # Fit plane
        normal, d = fit_plane_ransac(table.xyz)
        u, v = make_plane_frame(normal)
        plane = TablePlane(normal, d, u, v)

        # Extract objects
        objects: List[PointCloud] = []
        for label in np.unique(labels):
            if label in background_labels or label == table_label:
                continue

            mask = labels == label
            obj = PointCloud(pc.xyz[mask], pc.rgb[mask])
            components = obj.split_components(voxel_size=0.02, min_points=500)
            objects.extend(components)

        if not objects:
            tqdm.write(f"  Scene {scene_num:02d}: No objects found, skipping")
            return False

        if verbose:
            tqdm.write(f"  Scene {scene_num:02d}: Found {len(objects)} objects")

        # Orient normal toward objects
        obj_centroids = np.array([obj.xyz.mean(axis=0) for obj in objects])
        if float(np.mean(plane.signed_distance(obj_centroids))) < 0:
            normal = -plane.normal
            d = -plane.d
            u, v = make_plane_frame(normal)
            plane = TablePlane(normal, d, u, v)

        # Adjust d to table surface
        table_proj = plane.signed_distance(table.xyz)
        d_adjusted = float(np.percentile(table_proj, 98)) + plane.d
        plane = TablePlane(plane.normal, d_adjusted, plane.u, plane.v)

        # Table UV bounds
        table_uv = plane.project_to_uv(table.xyz)
        uv_min = np.percentile(table_uv, 2, axis=0)
        uv_max = np.percentile(table_uv, 98, axis=0)

        # Generate augmentations
        for aug_idx in tqdm(range(num_augmentations), desc=f"  Scene {scene_num:02d}", leave=False):
            placed_objects: List[PointCloud] = []
            margin = 0.05

            for obj in objects:
                obj_uv = plane.project_to_uv(obj.xyz)
                obj_uv_size = obj_uv.max(axis=0) - obj_uv.min(axis=0)

                valid_min = uv_min + margin + obj_uv_size / 2
                valid_max = uv_max - margin - obj_uv_size / 2

                placed = False
                for _ in range(100):
                    if np.any(valid_max <= valid_min):
                        break

                    target_uv = np.array([
                        np.random.uniform(valid_min[0], valid_max[0]),
                        np.random.uniform(valid_min[1], valid_max[1])
                    ], dtype=np.float32)

                    moved = move_object_on_plane(obj, plane, target_uv)

                    if not any(check_collision(moved, p) for p in placed_objects):
                        placed_objects.append(moved)
                        placed = True
                        break

                if not placed:
                    placed_objects.append(obj)

            # Save augmented scene
            all_xyz = [table.xyz] + [obj.xyz for obj in placed_objects]
            all_rgb = [table.rgb] + [obj.rgb for obj in placed_objects]
            augmented = PointCloud(np.vstack(all_xyz), np.vstack(all_rgb))

            out_path = output_dir / f"scene_{scene_num:02d}_aug_{aug_idx:03d}.ply"
            save_ply_ascii(augmented, out_path)

            if verbose:
                tqdm.write(f"    Saved: {out_path.name} ({len(augmented):,} points)")

        return True

    except Exception as e:
        tqdm.write(f"  Scene {scene_num:02d}: ERROR - {e}")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Augment RGBD Scenes V2 tabletop scenes')
    parser.add_argument('--data-dir', type=Path,
                        default=Path.home() / 'Downloads/rgbd-scenes-v2/pc',
                        help='Path to pc folder with PLY and label files')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Output directory (default: data-dir/../augmented)')
    parser.add_argument('--scene', type=int, default=None,
                        help='Process single scene (1-14), or all if not specified')
    parser.add_argument('--num-augmentations', '-n', type=int, default=5,
                        help='Number of augmented scenes per input scene')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Less verbose output')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.data_dir.parent / 'augmented'
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        np.random.seed(args.seed)

    # Determine scenes to process
    if args.scene is not None:
        scenes = [args.scene]
    else:
        # Find all available scenes
        scenes = []
        for i in range(1, 15):
            if (args.data_dir / f"{i:02d}.ply").exists():
                scenes.append(i)

    print(f"Processing {len(scenes)} scene(s), {args.num_augmentations} augmentations each")
    print(f"Output directory: {args.output_dir}")
    print()

    success = 0
    for scene_num in tqdm(scenes, desc="Processing scenes"):
        if process_scene(scene_num, args.data_dir, args.output_dir,
                         args.num_augmentations, verbose=not args.quiet):
            success += 1

    print(f"Done! {success}/{len(scenes)} scenes processed successfully")
    print(f"Total files: {success * args.num_augmentations}")


if __name__ == '__main__':
    main()
