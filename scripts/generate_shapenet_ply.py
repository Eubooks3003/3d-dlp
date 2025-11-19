import os, zipfile, json, random, numpy as np
import trimesh as tm
from tqdm import tqdm
from huggingface_hub import snapshot_download

HF_REPO = "ShapeNet/ShapeNetCore"

# Where to store extracted meshes permanently
EXTRACT_ROOT = os.path.expanduser("~/.cache/shapenet_core_extracted")

# Scene generation params
OUT_DIR = "tabletop_shapenet_pcs"
SCENES = 5000
OBJ_RANGE = (3, 7)
PTS_PER_OBJ = (2000, 6000)
NOISE_STD = 0.002
TABLE_SIZE = (0.8, 0.6)
TABLE_Z = 0.0
TABLE_STRIDE = 0.0075
VOX_DOWNSAMPLE = 0.0
SEED = 123
N_MODELS_PER_CLASS = 70

CATEGORIES = {
    "chair": "03001627",
    "table": "04379243",
    "mug": "03797390",
    "bottle": "02876657",
    "lamp": "03636649",
}

rng = np.random.default_rng(SEED)


# ------------------------------------------------------------
# Utility: check if a synset folder already has OBJ files
# ------------------------------------------------------------
def synset_has_objs(synset):
    root = os.path.join(EXTRACT_ROOT, synset)
    if not os.path.isdir(root):
        return False
    for _, _, files in os.walk(root):
        if any(f.endswith(".obj") for f in files):
            return True
    return False

def normalize_mesh(m, min_size=0.04, max_size=0.12):
    """
    Rescale mesh so its largest axis extent lies in [min_size, max_size] meters.
    Returns a *copy* of the mesh, or None if degenerate.
    """
    m = m.copy()
    extents = np.array(m.extents, dtype=float)
    max_extent = float(extents.max())
    if max_extent < 1e-6 or not np.isfinite(max_extent):
        return None

    target_size = float(rng.uniform(min_size, max_size))  # e.g. 4–12 cm
    scale = target_size / max_extent

    S = np.eye(4)
    S[:3, :3] *= scale
    m.apply_transform(S)
    return m

def scale_mesh(m, scale_factor=1.5):
    """Uniformly scale mesh around its centroid."""
    c = m.vertices.mean(axis=0)
    S = np.eye(4)
    S[0, 0] = S[1, 1] = S[2, 2] = scale_factor

    T1 = np.eye(4); T1[:3, 3] = -c
    T2 = np.eye(4); T2[:3, 3] = c

    m = m.copy()
    m.apply_transform(T2 @ S @ T1)
    m.remove_degenerate_faces()
    m.remove_unreferenced_vertices()
    m.fix_normals()
    return m

# ------------------------------------------------------------
# 1. DOWNLOAD SHAPENET SYNSET ZIPS (only if needed)
# ------------------------------------------------------------
def download_shapenet_zips():
    # if we already extracted all categories, no need to download
    if all(synset_has_objs(s) for s in CATEGORIES.values()):
        print("All synsets already extracted locally – skipping HuggingFace download.")
        return None

    print("Downloading synset ZIP files from HuggingFace...")
    token = os.environ.get("HF_TOKEN", None)

    patterns = [f"{syn}.zip" for syn in CATEGORIES.values()]

    path = snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        allow_patterns=patterns,   # only the 5 zips
        token=token,
    )
    return path


# ------------------------------------------------------------
# 2. EXTRACT ZIP FILE ONCE, CACHE FOREVER
# ------------------------------------------------------------
def extract_zip_once(zip_path, synset):
    out_dir = os.path.join(EXTRACT_ROOT, synset)
    if synset_has_objs(synset):
        # Already extracted and has meshes
        return out_dir

    print(f"Extracting {zip_path} -> {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)

    return out_dir


# ------------------------------------------------------------
# 3. LOAD ALL OBJ FILES FROM EXTRACTED FOLDERS
# ------------------------------------------------------------
def load_meshes_from_synset_folder(folder):
    # collect all OBJ files first
    obj_paths = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith(".obj"):
                obj_paths.append(os.path.join(root, f))

    if len(obj_paths) == 0:
        print(f"No OBJ files found in {folder}")
        return []

    # subsample paths *before* loading to save time
    if len(obj_paths) > N_MODELS_PER_CLASS:
        idx = rng.choice(len(obj_paths), N_MODELS_PER_CLASS, replace=False)
        obj_paths = [obj_paths[i] for i in idx]

    meshes = []
    desc = f"Loading meshes from {os.path.basename(folder)} ({len(obj_paths)} objs)"
    for path in tqdm(obj_paths, desc=desc):
        try:
            m = tm.load_mesh(path)
            if m.is_empty or m.vertices.shape[0] <= 20:
                continue
            m = normalize_mesh(m)
            if m is None:
                continue
            m = scale_mesh(m, scale_factor=1.3)
            meshes.append(m)
        except Exception:
            # silently skip bad meshes
            pass

    return meshes





# ------------------------------------------------------------
# TABLETOP SCENE GENERATION HELPERS  (same as before)
# ------------------------------------------------------------
def footprint_radius(mesh):
    xy = mesh.vertices[:, :2]
    return float(np.linalg.norm(xy, axis=1).max())


def place_non_overlapping(meshes, table_w, table_d, max_tries=300):
    """
    Place meshes on the table without overlap.
    Skips meshes that are too big to fit inside table bounds.
    """
    placed, occupied = [], []
    half_w = table_w / 2.0
    half_d = table_d / 2.0

    for m_in in meshes:
        # work on a copy so we don't mutate originals
        m = m_in.copy()

        # radius in XY after all mesh normalizing / scaling
        r = footprint_radius(m) * 1.2

        # compute valid sampling ranges for this object
        low_x  = -half_w + r
        high_x =  half_w - r
        low_y  = -half_d + r
        high_y =  half_d - r

        # if it simply cannot fit, skip this mesh
        if high_x <= low_x or high_y <= low_y:
            # optional: uncomment to see how many get skipped
            # print(f"Skipping mesh: too big for table (r={r:.3f})")
            continue

        placed_this = False
        for _ in range(max_tries):
            x = rng.uniform(low_x, high_x)
            y = rng.uniform(low_y, high_y)

            # check overlap in XY against previously placed objects
            if not all((x - x0) ** 2 + (y - y0) ** 2 >= (r + r0) ** 2
                       for (x0, y0, r0) in occupied):
                continue

            # random yaw around Z
            yaw = rng.uniform(-np.pi, np.pi)
            Rz = tm.transformations.rotation_matrix(yaw, [0, 0, 1])

            m_placed = m.copy()
            m_placed.apply_transform(Rz)

            # snap bottom to TABLE_Z
            zmin = m_placed.bounds[0, 2]
            T = np.eye(4)
            T[:3, 3] = [x, y, TABLE_Z - zmin]
            m_placed.apply_transform(T)

            # final safety snap
            zmin2 = m_placed.bounds[0, 2]
            if abs(zmin2 - TABLE_Z) > 1e-5:
                T_fix = np.eye(4)
                T_fix[:3, 3] = [0.0, 0.0, TABLE_Z - zmin2]
                m_placed.apply_transform(T_fix)

            placed.append(m_placed)
            occupied.append((x, y, r))
            placed_this = True
            break

        # if you care: you can also log if we failed to place after max_tries
        # if not placed_this:
        #     print("Failed to place an object (ran out of tries).")

    return placed




def sample_points(meshes):
    pcs = []
    for m in meshes:
        n = int(rng.integers(*PTS_PER_OBJ))
        pts, _ = tm.sample.sample_surface_even(m, n)
        pcs.append(pts)
    return np.concatenate(pcs, axis=0) if pcs else np.zeros((0,3))


def table_points():
    xs = np.arange(-TABLE_SIZE[0]/2, TABLE_SIZE[0]/2, TABLE_STRIDE)
    ys = np.arange(-TABLE_SIZE[1]/2, TABLE_SIZE[1]/2, TABLE_STRIDE)
    X, Y = np.meshgrid(xs, ys)
    Z = np.full_like(X, TABLE_Z)
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)


def voxel_downsample(points, voxel):
    if voxel <= 0:
        return points
    keys = np.floor(points / voxel).astype(int)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[idx]


# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------
def main():
    os.makedirs(EXTRACT_ROOT, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "train"), exist_ok=True)

    dataset_root = download_shapenet_zips()  # may be None if everything already extracted

    # 2. Extract + Load meshes
    all_meshes = []

    for name, synset in CATEGORIES.items():
        if synset_has_objs(synset):
            synset_folder = os.path.join(EXTRACT_ROOT, synset)
        else:
            if dataset_root is None:
                raise RuntimeError(
                    f"Synset {synset} not found in {EXTRACT_ROOT} and no dataset_root available."
                )
            zip_file = os.path.join(dataset_root, f"{synset}.zip")
            synset_folder = extract_zip_once(zip_file, synset)

        meshes = load_meshes_from_synset_folder(synset_folder)
        print(f"{name}: loaded {len(meshes)} meshes from {synset_folder}")


        all_meshes.extend(meshes)

    # 3. Generate scenes
    for i in tqdm(range(SCENES)):
        k = rng.integers(OBJ_RANGE[0], OBJ_RANGE[1] + 1)
        chosen = rng.choice(all_meshes, k, replace=False)

        placed = place_non_overlapping(chosen, *TABLE_SIZE)
        obj_pts = sample_points(placed)
        t_pts = table_points()

        if NOISE_STD > 0:
            t_pts += rng.normal(0, NOISE_STD, t_pts.shape)

        pts = np.concatenate([obj_pts, t_pts], axis=0)
        pts = voxel_downsample(pts, VOX_DOWNSAMPLE)

        out_path = os.path.join(OUT_DIR, "train", f"scene_{i:05d}.ply")
        tm.points.PointCloud(pts).export(out_path)

    print("All done!")


if __name__ == "__main__":
    main()
