<h1 align="center">3D-DLP: Self-supervised 3D Object-centric Scene Representation Learning</h1>

<p align="center">
  <a href="https://eubooks3003.github.io/3d-dlp/">Project Page</a> &nbsp;•&nbsp;
  <a href="#installation">Installation</a> &nbsp;•&nbsp;
  <a href="#training">Training</a> &nbsp;•&nbsp;
  <a href="#evaluation">Evaluation</a> &nbsp;•&nbsp;
  <a href="#interactive-gui">GUI</a> &nbsp;•&nbsp;
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
- [Configuration Files](#configuration-files)
- [Training](#training)
- [Evaluation](#evaluation)
- [Interactive GUI](#interactive-gui)
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
`accelerate` (multi-GPU training), `einops`, `h5py`, `opencv-python`, `scikit-image`, `imageio`,
`piqa` (LPIPS/SSIM/PSNR metrics), and `ttkthemes`/`ttkwidgets` (interactive GUI).

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
| `eval/` | Evaluation scripts and metric backends (LPIPS, FVD) |
| `gui/` | `tkinter` interactive particle visualization / editing |
| `scripts/` | Data conversion and utility scripts |
| `utils/` | Logging, plotting, loss functions, and helpers |
| `documentation/` | Installation, hyperparameters, GUI guide, example usage |
| `docs/` | Project page (GitHub Pages) |
| `assets/` | Sample assets for the GUI and figures |
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

Data-preparation and conversion helpers (e.g. voxelization, format conversion) are in
[`scripts/`](scripts/).

**Custom datasets:** add a `Dataset` class under `datasets/`, register it in the dataset-resolution
helper, and create a matching JSON config in `configs/`.

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

## Evaluation

Reconstruction quality is measured with **LPIPS / SSIM / PSNR** (via `piqa`). Scripts live in
[`eval/`](eval/):

| Script | Purpose |
|--------|---------|
| `eval/eval_vox.py` | Voxel 3D-DLP reconstruction / decomposition metrics |
| `eval/eval_pc.py` | Point-cloud reconstruction metrics |
| `eval/eval_model.py` | ELBO and model-level evaluation utilities |

Each script reads an experiment config and a checkpoint, for example:

```bash
python eval/eval_vox.py --checkpoint <path/to/ckpt> --config configs/<your_config>.json
```

## Interactive GUI

A `tkinter`-based GUI to plot and edit particles and observe the effect on the reconstruction — useful
for inspecting discovered keypoints and demonstrating latent controllability (position / scale edits).

```bash
bash gui.bash
# equivalently:
PYTHONPATH=. python -m gui.interactive_gui
```

The 3D viewer and interaction components are in [`gui/`](gui/) (`gui_3d.py`, `gui_select.py`,
`gui_update.py`, …). Usage walkthrough: [`gui/gui.md`](gui/gui.md). Sample assets are under
[`assets/`](assets/).

## Documentation

| File | Content |
|------|---------|
| [`documentation/installation.md`](documentation/installation.md) | Manual environment setup |
| [`documentation/hyperparameters.md`](documentation/hyperparameters.md) | Hyperparameter reference and recommended values |
| [`documentation/gui.md`](documentation/gui.md) | Interactive GUI guide |
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
