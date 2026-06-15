<h1 align="center">3D-DLP: Self-supervised 3D Object-centric Scene Representation Learning</h1>

<p align="center">
  <a href="https://eubooks3003.github.io/3d-dlp/">Project Page</a> &nbsp;•&nbsp;
  <a href="#installation">Installation</a> &nbsp;•&nbsp;
  <a href="#data-preprocessing">Data Preprocessing</a> &nbsp;•&nbsp;
  <a href="#training">Training</a> &nbsp;•&nbsp;
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <em>Ellina Zhang, Madhavan Iyengar, Amir Zadeh, Chuan Li, David Held, Deepak Pathak, Tal Daniel</em><br>
  Carnegie Mellon University &nbsp;•&nbsp; Lambda AI &nbsp;•&nbsp; ICML 2026
</p>

---

## Abstract

We introduce **3D-DLP**, a self-supervised object-centric representation learning model that
decomposes scene-level RGB-D or voxel observations into a set of **3D latent particles**.
Building on the Deep Latent Particles (DLP) framework, each particle encodes disentangled
attributes — 3D keypoint position, bounding-box dimensions, and appearance features — and
represents a distinct entity in the scene. The model learns interpretable per-particle
segmentation maps through an end-to-end self-supervised reconstruction objective. On both
simulated and real-world datasets, the learned latent space is interpretable and controllable:
by manipulating particle positions and decoding, we can generate novel scene configurations.
Leveraging these compact 3D latent particles for downstream robotic manipulation improves
performance over baselines that either lack explicit 3D information or rely on memory-intensive
dense 3D inputs without object-centric structure.

## Contents

- [Overview](#overview)
- [Installation](#installation)
- [Repository Organization](#repository-organization)
- [Datasets](#datasets)
- [Data Preprocessing](#data-preprocessing)
- [Configuration Files](#configuration-files)
- [Training](#training)
- [Documentation](#documentation)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Overview

3D-DLP encodes a scene-level 3D observation into a fixed set of $K$ latent particles, each carrying an
explicit 3D keypoint position, a learned scale/bounding box, and appearance features, plus a single
background particle. A modality-specific decoder reconstructs the observation, and the whole model is
trained end-to-end with a self-supervised reconstruction objective — **no instance labels or masks**.

The framework covers three input modalities, selected per experiment in the [config](#configuration-files):

| Variant       | Input modality       | Description |
|---------------|----------------------|-------------|
| **3D-DLP-D**  | RGB-D images         | Particles decoded into depth-ordered object layers |
| **3D-DLP-V**  | Occupancy voxels     | Colorless 3D occupancy decomposition |
| **3D-DLP-VC** | Colored (RGB) voxels | Full colored-voxel decomposition and reconstruction |

Two components make object discovery work on dense voxel scenes: an **appearance-aware K-means
keypoint prior** (particle initialization from joint color + geometry clusters) and a **chroma
reconstruction loss**. Both are toggled through the config file.

## Installation

```bash
git clone https://github.com/Eubooks3003/3d-dlp.git
cd 3d-dlp

# Option A — conda (creates an environment named `dlp`)
conda env create -f environment.yml
conda activate dlp

# Option B — pip into an existing Python 3.8+ environment
pip install -r requirements.txt
```

The environment targets **Python 3.8**, **PyTorch 2.x**, and **CUDA 11.8**. Key dependencies include
`accelerate` (multi-GPU training), `einops`, `h5py`, `open3d` (point-cloud / voxelization),
`opencv-python`, `scikit-image`, and `imageio`.

For manual setup notes and CUDA/dependency caveats, see
[`documentation/installation.md`](documentation/installation.md).

## Repository Organization

| Path | Contents |
|------|----------|
| `train_dlp_voxel.py` | 3D-DLP training on voxels (RGB / occupancy) — main entry point |
| `train_dlp_voxel_accelerate.py` | Multi-GPU version of the voxel trainer |
| `voxel_models.py` | 3D-DLP voxel encoder / decoder |
| `models.py` | DLPv2 / DDLP model implementations |
| `modules/` | Network building blocks (2D/3D vision, point-cloud, VAE, diffusion modules) |
| `configs/` | JSON experiment configs (see [below](#configuration-files)) |
| `datasets/` | Dataset loaders and preparation scripts |
| `scripts/` | Data conversion / voxelization / K-means precompute scripts |
| `utils/` | Logging, plotting, loss functions, and helpers |
| `documentation/` | Installation, hyperparameters, example usage |
| `docs/` | Project page (GitHub Pages) |
| `assets/` | Sample assets and figures |
| `accel_conf.yml` | HuggingFace `accelerate` config for multi-GPU runs |
| `environment.yml` / `requirements.txt` | conda / pip dependency specs |

## Datasets

Dataset loaders live in [`datasets/`](datasets/) (one `*_ds.py` per dataset). The datasets used in the
paper:

| Dataset | Modality | Notes |
|---------|----------|-------|
| `GenericShapes` / `2DGenericShapes` | Synthetic point clouds / RGB-D | Procedurally generated |
| `ShapeNetScenes` | Synthetic colored scenes | Occupancy / RGB-voxel decomposition |
| `BlenderShapes` | Synthetic RGB-D | `blender_ds.py` |
| `MimicGen` | Simulated robot manipulation | Voxelized multi-view observations |
| `RLBench` | Language-conditioned manipulation | Used in the imitation-learning experiments |
| `UW RGB-D Scenes v2` | Real-world RGB-D | Tabletop scenes |

For the robot-manipulation datasets (**MimicGen**, **RLBench**), the raw demonstrations must be
converted into cached voxel grids before training — see [Data Preprocessing](#data-preprocessing).

**Custom datasets:** add a `Dataset` class under `datasets/`, register it in the dataset-resolution
helper, and create a matching JSON config in `configs/`.

## Data Preprocessing

The voxel models (3D-DLP-V / 3D-DLP-VC) train on a per-frame **64³ RGB voxel grid** built from
multi-view RGB-D. Preprocessing turns raw demonstrations into (1) fused point clouds, (2) cached
voxel grids, and — optionally — (3) cached K-means keypoint priors. Every script lives in
[`scripts/`](scripts/) and writes its outputs **back into the dataset tree**, so training just reads
the caches. Both datasets follow the same three stages:

```
raw demos ─▶ multi-view RGB-D ─▶ fused point cloud (.ply) ─▶ 64³ RGB voxel cache ─▶ (optional) K-means prior cache
```

`open3d` is required for the point-cloud / voxelization steps. Pass `--tasks` to restrict to specific
tasks (omit to process all), and re-runs skip frames that are already cached.

The fused point cloud is the geometry the voxelizer consumes — it does not have to be persisted. For
RLBench, [`scripts/rlbench_rgbd_to_voxels.py`](scripts/rlbench_rgbd_to_voxels.py) runs both stages in
one pass: it fuses the multi-view RGB-D in memory and voxelizes directly into `voxel_cache/`, writing
**no `.ply`**. The output is identical to running the two stages back to back (same frame order, file
names, and metadata), so use it when you don't need the `.ply` artifacts — or run the explicit
two-stage flow below when you do. Both paths are shown for RLBench.

### MimicGen

**Raw input.** Per-task RGB-D HDF5 files (matching `*_rgbd_pcd.hdf5`) containing the `agentview` and
`sideview` camera RGB + depth, laid out as `<root>/<task>_d0/`. These come from rendering
MimicGen / robomimic demonstrations with camera **depth** enabled (standard robomimic observation
extraction).

**1 — Fuse RGB-D → point clouds.** Back-projects each camera, fuses the two views, crops the
workspace, and writes one `.ply` per frame to `<root>/<task>_d0/core/mimicgen_from_depth_pcd/demo_*/`.
The depth-buffer → metric-depth convention is auto-detected per task.

```bash
python scripts/mimicgen_ply_all_tasks.py \
  --root /path/to/3D-DLP-mimicgen-data \
  --tasks stack_d1 coffee_d2 kitchen_d1        # --cams defaults to: agentview sideview
```

**2 — Voxelize point clouds → 64³ RGB grid.** Writes `frame*_voxels.pt` (+ meta) to
`<root>/<task>_d0/core/voxel_cache/`.

```bash
python scripts/preprocess_mimicgen_voxels.py \
  --root /path/to/3D-DLP-mimicgen-data \
  --tasks stack_d1 coffee_d2 kitchen_d1 \
  --grid_whd 64 64 64 \
  --voxel_mode avg_rgb \
  --use_task_bounds        # per-task workspace bounds (or --fixed_bounds for a shared crop; omit for per-frame)
```

### RLBench

**Raw input.** RLBench demonstrations under
`<root>/<split>/<task>/all_variations/episodes/episode<N>/`, with `front` / `overhead` /
`left_shoulder` / `right_shoulder` RGB + depth PNGs and `low_dim_obs.pkl`. Generate them with
RLBench's `tools/dataset_generator.py` (e.g. `--episodes_per_task 100 --all_variations True
--image_size 128`). The ten tasks used in the paper:

```
close_jar  open_drawer  sweep_to_dustpan_of_size  meat_off_grill  turn_tap
slide_block_to_color_target  put_item_in_drawer  reach_and_drag  push_buttons  stack_blocks
```

**One-shot (no `.ply`).** Fuse and voxelize in a single pass straight into `voxel_cache/`:

```bash
python scripts/rlbench_rgbd_to_voxels.py \
  --root /path/to/rlbench \
  --splits train_data test_data \
  --tasks close_jar open_drawer turn_tap \
  --cameras front overhead left_shoulder right_shoulder \
  --depth-scale 1000.0 \
  --grid_whd 64 64 64 \
  --voxel_mode avg_rgb
```

Or run the two explicit stages below if you also want the intermediate `.ply` files on disk.

**1 — Fuse RGB-D → point clouds.** Back-projects + fuses all cameras (depth PNGs are in millimetres,
hence `--depth-scale 1000`) and writes one `.ply` per frame to `episode<N>/fused_pcd/`.

```bash
python scripts/rlbench_ply.py \
  --root /path/to/rlbench \
  --splits train_data test_data \
  --tasks close_jar open_drawer turn_tap \
  --cameras front overhead left_shoulder right_shoulder \
  --depth-scale 1000.0 \
  --fov 60.0 \
  --max-points 200000
```

**2 — Voxelize point clouds → 64³ RGB grid.** Each frame is normalized to a unit cube; writes
`voxel_cache/*_voxels.pt` per episode.

```bash
python scripts/preprocess_rlbench_voxels.py \
  --root /path/to/rlbench \
  --splits train_data test_data \
  --tasks close_jar open_drawer turn_tap \
  --grid_whd 64 64 64 \
  --voxel_mode avg_rgb
```

### (Optional) Precompute the K-means prior

3D-DLP initializes its particles with an **appearance-aware K-means prior** — joint CIELAB-color + XYZ
clustering that is purely algorithmic, so **no trained checkpoint is required**. It can run on-the-fly
during training, but precomputing and caching it per frame makes training substantially faster. Point
the script at the same dataset root plus a config (it reads the particle count from the config):

```bash
# MimicGen — caches per-frame K-means alongside the voxel cache
python scripts/precompute_kmeans.py \
  --data-root /path/to/3D-DLP-mimicgen-data \
  --dlp-cfg configs/mimicgen_multitask.json \
  --num-gpus 1 --workers-per-gpu 4 --batch 32 \
  --tasks stack_d1 coffee_d2 kitchen_d1

# RLBench — writes kmeans_cache/*_kmeans.pt per episode
python scripts/precompute_kmeans_rlbench.py \
  --data-root /path/to/rlbench \
  --dlp-cfg configs/rlbench_multitask.json \
  --splits train_data test_data \
  --num-gpus 1 --workers-per-gpu 4 --batch 32
```

With the caches in place, train as in [Training](#training) using the matching config
(`configs/mimicgen_multitask.json` or `configs/rlbench_multitask.json`).

> **Batch wrappers.** `scripts/` also ships shell wrappers (`run_all_voxels.sh`, `run_all_kmeans.sh`,
> `preprocess_rlbench_rgbo_all.sh`, …) that run these stages across every task. They contain absolute
> paths from the authors' machines — edit the `ROOT` / `LPWM_DIR` variables at the top before use, and
> treat the per-script commands above as the canonical interface.

## Configuration Files

Every run is driven by a JSON config. The dataset flag `-d <name>` resolves to `./configs/<name>.json`;
you can also pass a full path ending in `.json`. A config sets the `modality` (image / occupancy /
rgb-voxel), dataset root, particle count, model hyperparameters, and the K-means-prior / chroma-loss
switches. Top-level configs (e.g. `shapes.json`, `blender.json`) ship in [`configs/`](configs/); new
ones can be created with [`configs/generate_config_file.py`](configs/generate_config_file.py).

Hyperparameter reference: [`documentation/hyperparameters.md`](documentation/hyperparameters.md).

## Training

The trainer takes `-d`, which is either a config **name** (resolved to `configs/<name>.json`) or a
**path** to a `.json` config:

```bash
# 3D-DLP on voxels (modality set by the config: RGB voxels / occupancy)
python train_dlp_voxel.py -d shapes
python train_dlp_voxel.py -d configs/blender.json
```

### Multi-GPU training

Multi-GPU runs use [HuggingFace Accelerate](https://huggingface.co/docs/accelerate/index) via
`train_dlp_voxel_accelerate.py` and `accel_conf.yml`:

1. Set visible GPUs, e.g. `os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"`.
2. Set `num_processes` in `accel_conf.yml` to match the GPU count.

```bash
accelerate launch --config_file ./accel_conf.yml train_dlp_voxel_accelerate.py -d <config>
```

For concurrent multi-GPU runs, copy `accel_conf.yml` and give each a distinct `main_process_port`.

## Documentation

| File | Content |
|------|---------|
| [`documentation/installation.md`](documentation/installation.md) | Manual environment setup |
| [`documentation/hyperparameters.md`](documentation/hyperparameters.md) | Hyperparameter reference and recommended values |
| [`documentation/example_usage.py`](documentation/example_usage.py) | Minimal forward / loss / sampling example |

## Citation

```bibtex
@inproceedings{zhang20263ddlp,
  title     = {3D-DLP: Self-supervised 3D Object-centric Scene Representation Learning},
  author    = {Zhang, Ellina and Iyengar, Madhavan and Zadeh, Amir and Li, Chuan and Held, David and Pathak, Deepak and Daniel, Tal},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## Acknowledgements

This codebase is built on top of [Latent Particle World Models](https://github.com/taldatech/lpwm)
(ICLR 2026 Oral) by Tal Daniel et al., which itself extends the
[Deep Latent Particles](https://taldatech.github.io/ddlp-web) (DLPv2 / DDLP) framework by Tal Daniel
and Aviv Tamar. We thank the authors for releasing their implementations, on which the 3D extensions in
this repository are based. This material is based upon work supported by ONR MURI N00014-24-1-2748.

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
