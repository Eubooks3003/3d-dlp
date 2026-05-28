<h1 align="center">3D-DLP: Self-supervised 3D Object-centric Scene Representation Learning</h1>

<p align="center">
  <a href="https://eubooks3003.github.io/3d-dlp/">Project Page</a> &nbsp;•&nbsp;
  <a href="#citation">Citation</a> &nbsp;•&nbsp;
  <a href="#acknowledgements">Acknowledgements</a>
</p>

<p align="center">
  <em>Ellina Zhang, Madhavan Iyengar, Amir Zadeh, Chuan Li, David Held, Deepak Pathak, Tal Daniel</em><br>
  Carnegie Mellon University &nbsp;•&nbsp; Lambda AI
</p>

---

## Abstract

We introduce 3D-DLP, a self-supervised object-centric representation learning model that decomposes scene-level RGB-D or voxel observations into a set of 3D latent particles. Building on the Deep Latent Particles (DLP) framework, each particle encodes disentangled attributes — 3D keypoint position, bounding box dimensions, and appearance features — and represents a distinct entity in the scene. The model learns interpretable per-particle segmentation maps through an end-to-end self-supervised reconstruction objective. On both simulated and real-world datasets, the learned latent space is interpretable and controllable: by manipulating particle positions and decoding, we can generate novel scene configurations. Leveraging these compact 3D latent particles for downstream robotic manipulation improves performance over baselines that either lack explicit 3D information or rely on memory-intensive dense 3D inputs without object-centric structure.

## Installation

```bash
git clone https://github.com/Eubooks3003/3d-dlp.git
cd 3d-dlp
conda env create -f environment.yml
conda activate 3d-dlp
```

See [`documentation/installation.md`](documentation/installation.md) for detailed setup notes (CUDA versions, optional dependencies).

## Datasets

<!-- TODO: add dataset download links and preparation instructions. -->

## Training

```bash
# Example: train 3D-DLP on the voxel-RGB configuration
python train.py --config configs/<your_config>.json
```

Configs live in [`configs/`](configs/). Hyperparameter reference: [`documentation/hyperparameters.md`](documentation/hyperparameters.md).

## Evaluation

```bash
python eval/eval_vox.py --checkpoint <path/to/ckpt> --config configs/<your_config>.json
```

## Repository layout

```
configs/        # JSON experiment configs
modules/        # Model components (encoder, decoder, K-means, set transformer)
scripts/        # Data prep + run scripts
eval/           # Evaluation scripts (FVD, LPIPS, voxel metrics)
gui/, gui_3d/   # Interactive visualization tools
documentation/  # Setup notes, hyperparameter reference, GUI guide
docs/           # Project page (GitHub Pages)
```

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

This codebase builds on the [Deep Latent Particles](https://taldatech.github.io/ddlp-web) (DLPv2 / DDLP) framework by Tal Daniel and Aviv Tamar. We thank the authors for releasing their implementation, which the 3D extensions in this repository are derived from.

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
