# gen_tabletop_pcs.py
import os, math, random, json
import numpy as np
import trimesh as tm

# -------------------- knobs --------------------
OUT_DIR = "tabletop_pcs"
SCENES = 20000                # total point clouds to generate
OBJ_RANGE = (3, 7)          # min/max objects per scene
PTS_PER_OBJ = (2000, 6000)  # points per object (surface sampling)
NOISE_STD = 0.002           # Gaussian noise (meters)
TABLE_SIZE = (0.8, 0.6)     # table size (X by Y in meters)
TABLE_Z = 0.0               # table height
VOX_DOWNSAMPLE = 0.0        # e.g., 0.005 → 5mm voxel grid
SEED = 123

# Table sampling + occlusion
TABLE_STRIDE = 0.0075       # grid spacing (meters)
OCCLUDE_TABLE_UNDER_OBJECTS = False

# Fixed splits (ratios must sum to 1.0)
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
# ------------------------------------------------

rng = np.random.default_rng(SEED)

def makedirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    for s in SPLITS: os.makedirs(os.path.join(OUT_DIR, s), exist_ok=True)

def footprint_radius(mesh: tm.Trimesh):
    xy = mesh.vertices[:, :2]
    return float(np.linalg.norm(xy, axis=1).max())

def make_random_primitive():
    typ = random.choice(["cube", "sphere", "cylinder", "cone", "capsule"])
    s = rng.uniform(0.04, 0.12)  # ~4–12 cm
    if typ == "cube":      m = tm.creation.box(extents=[s, s, s])
    elif typ == "sphere":  m = tm.creation.icosphere(subdivisions=3, radius=s/2)
    elif typ == "cylinder":m = tm.creation.cylinder(radius=s/3, height=s)
    elif typ == "cone":    m = tm.creation.cone(radius=s/3, height=s)
    else:                  m = tm.creation.capsule(radius=s/4, height=s*0.8, count=[8,8])
    scale = np.diag(rng.uniform(0.8, 1.3, size=3))
    m.apply_transform(np.block([[scale, np.zeros((3,1))],[np.zeros((1,3)), 1.0]]))
    m.remove_degenerate_faces(); m.remove_unreferenced_vertices(); m.fix_normals()
    return m

def place_non_overlapping(meshes, table_w, table_d, max_tries=500):
    placed, occupied = [], []
    for m in meshes:
        r = footprint_radius(m) * 1.2
        ok = False
        for _ in range(max_tries):
            x = rng.uniform(-table_w/2 + r, table_w/2 - r)
            y = rng.uniform(-table_d/2 + r, table_d/2 - r)
            if all((x-x0)**2 + (y-y0)**2 >= (r+r0)**2 for (x0,y0,r0) in occupied):
                zmin = m.bounds[0,2]
                T = np.eye(4); T[:3, 3] = [x, y, TABLE_Z - zmin]
                yaw = rng.uniform(-math.pi, math.pi)
                Rz = tm.transformations.rotation_matrix(yaw, [0,0,1])
                m_placed = m.copy(); m_placed.apply_transform(Rz @ T)
                placed.append(m_placed); occupied.append((x, y, r)); ok = True; break
        if not ok: pass
    return placed

def sample_surface_points(meshes):
    pcs = []
    for m in meshes:
        n = int(rng.integers(PTS_PER_OBJ[0], PTS_PER_OBJ[1]+1))
        pts, _ = tm.sample.sample_surface_even(m, n)
        pcs.append(pts)
    if not pcs: return np.zeros((0,3), np.float32)
    P = np.concatenate(pcs, axis=0)
    if NOISE_STD > 0: P = P + rng.normal(0.0, NOISE_STD, size=P.shape)
    return P.astype(np.float32)

def sample_table_points(table_w, table_d, z, stride):
    xs = np.arange(-table_w/2, table_w/2, stride)
    ys = np.arange(-table_d/2, table_d/2, stride)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    P = np.stack([X.ravel(), Y.ravel(), np.full_like(X.ravel(), z)], axis=1)
    return P.astype(np.float32)

def occlude_table_points(table_pts, placed_meshes, margin=0.01):
    if len(table_pts) == 0 or not placed_meshes: return table_pts
    keep = np.ones(len(table_pts), dtype=bool); xy = table_pts[:, :2]
    for m in placed_meshes:
        r = footprint_radius(m) + margin
        c = m.vertices.mean(axis=0)[:2]
        keep &= (np.sum((xy - c)**2, axis=1) > (r*r))
    return table_pts[keep]

def voxel_downsample(points, voxel):
    if voxel <= 0: return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]

def fixed_split_indices(total, ratios, seed):
    # deterministic permutation then slice by counts
    rs = np.random.RandomState(seed)
    perm = rs.permutation(total)
    counts = {k: int(round(total * r)) for k, r in ratios.items()}
    # adjust rounding to sum exactly to total (give/take from 'train')
    diff = total - sum(counts.values())
    counts["train"] += diff
    starts = {}; acc = 0
    for k in ["train", "val", "test"]:
        starts[k] = (acc, acc + counts[k]); acc += counts[k]
    split_ids = {}
    for k,(a,b) in starts.items(): split_ids[k] = perm[a:b].tolist()
    return split_ids

def main():
    makedirs()
    # manifest for reproducibility (optional)
    with open(os.path.join(OUT_DIR, "settings.json"), "w") as f:
        json.dump({
            "scenes": SCENES, "obj_range": OBJ_RANGE, "pts_per_obj": PTS_PER_OBJ,
            "noise_std": NOISE_STD, "table_size": TABLE_SIZE, "table_z": TABLE_Z,
            "table_stride": TABLE_STRIDE, "occlude": OCCLUDE_TABLE_UNDER_OBJECTS,
            "voxel": VOX_DOWNSAMPLE, "seed": SEED, "splits": SPLITS
        }, f, indent=2)

    split_ids = fixed_split_indices(SCENES, SPLITS, seed=SEED)

    # reference table solid (not used for sampling)
    table_mesh = tm.creation.box(extents=[TABLE_SIZE[0], TABLE_SIZE[1], 0.02])
    table_mesh.apply_translation([0, 0, TABLE_Z - 0.01])

    for i in range(SCENES):
        k = int(rng.integers(OBJ_RANGE[0], OBJ_RANGE[1]+1))
        objs = [make_random_primitive() for _ in range(k)]
        placed = place_non_overlapping(objs, TABLE_SIZE[0], TABLE_SIZE[1])

        obj_pts = sample_surface_points(placed)
        table_pts = sample_table_points(TABLE_SIZE[0], TABLE_SIZE[1], TABLE_Z, TABLE_STRIDE)
        if OCCLUDE_TABLE_UNDER_OBJECTS:
            table_pts = occlude_table_points(table_pts, placed, margin=0.01)
        if NOISE_STD > 0:
            table_pts = table_pts + rng.normal(0.0, NOISE_STD, size=table_pts.shape)

        pts = np.concatenate([obj_pts, table_pts], axis=0)
        pts = voxel_downsample(pts, VOX_DOWNSAMPLE)

        # choose split
        if   i in split_ids["train"]: split = "train"
        elif i in split_ids["val"]:   split = "val"
        else:                         split = "test"

        out_path = os.path.join(OUT_DIR, split, f"scene_{i:05d}.ply")
        tm.points.PointCloud(pts).export(out_path)
        print(f"[{i+1}/{SCENES}] {pts.shape[0]} pts, {len(placed)} objs → {split}/{os.path.basename(out_path)}")

if __name__ == "__main__":
    main()
