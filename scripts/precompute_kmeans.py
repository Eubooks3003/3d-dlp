#!/usr/bin/env python3
"""
Fast parallel kmeans precomputation for all tasks.

Splits work across GPUs and uses multiple workers per GPU to keep utilization high.
Saves per-frame kmeans cache files that ec_diffuser_voxel_preprocess.py will pick up.

Usage:
    python scripts/precompute_kmeans.py \
        --data-root /path/to/3D-DLP-mimicgen-data \
        --dlp-cfg /path/to/hparams.json \
        --dlp-ckpt /path/to/saves/best.pt \
        --workers-per-gpu 2
"""
import os
import re
import sys
import argparse
import torch
import torch.multiprocessing as mp

import numpy.core as _core
sys.modules['numpy._core'] = _core
sys.modules['numpy._core.multiarray'] = _core.multiarray

from voxel_models import DLP
from utils.util_func import get_config
from utils.log_utils import load_checkpoint


TASKS = [
    "coffee_d0",
    "coffee_preparation_d0",
    "hammer_cleanup_d0",
    "kitchen_d0",
    "mug_cleanup_d0",
    "nut_assembly_d0",
    "pick_place_d0",
    "square_d0",
    "stack_d0",
    "stack_three_d0",
    "threading_d0",
    "three_piece_assembly_d0",
]


def load_voxel_dense(path):
    """Load compressed or dense voxel, return dense [C,D,H,W] float32."""
    data = torch.load(path, weights_only=False)
    if isinstance(data, dict) and data.get("compressed"):
        shape = data["shape"]
        coords = data["coords"].long()
        values = data["values"].float()
        vox = torch.zeros(shape, dtype=torch.float32)
        vox[:, coords[:, 0], coords[:, 1], coords[:, 2]] = values.T
        return vox
    if isinstance(data, torch.Tensor):
        return data.float()
    raise RuntimeError(f"Unexpected format in {path}: {type(data)}")


def scan_voxel_cache(voxel_cache_dir):
    """Returns {demo_name: {frame_idx: path}}."""
    voxel_map = {}
    frame_pat = re.compile(r"frame(\d+)_voxels\.pt")
    for demo_dir in sorted(os.listdir(voxel_cache_dir)):
        demo_path = os.path.join(voxel_cache_dir, demo_dir)
        if not os.path.isdir(demo_path) or not demo_dir.startswith("demo_"):
            continue
        for fn in os.listdir(demo_path):
            m = frame_pat.match(fn)
            if m:
                frame_idx = int(m.group(1))
                voxel_map.setdefault(demo_dir, {})[frame_idx] = os.path.join(demo_path, fn)
    return voxel_map


def build_frame_list(data_root, tasks):
    """Build flat list of (task, demo, frame_idx, voxel_path, kmeans_out_path)."""
    work = []
    for task in tasks:
        vox_dir = os.path.join(data_root, task, "voxel_cache", "voxel")
        km_dir = os.path.join(data_root, task, "voxel_cache", "kmeans_cache")
        if not os.path.isdir(vox_dir):
            print(f"[skip] {task}: no voxel cache at {vox_dir}")
            continue
        voxel_map = scan_voxel_cache(vox_dir)
        for demo in sorted(voxel_map.keys(), key=lambda s: int(s.split("_")[-1])):
            demo_km_dir = os.path.join(km_dir, demo)
            for frame_idx in sorted(voxel_map[demo].keys()):
                km_path = os.path.join(demo_km_dir, f"frame{frame_idx}_kmeans.pt")
                if os.path.exists(km_path):
                    continue  # already cached
                work.append((task, demo, frame_idx, voxel_map[demo][frame_idx], km_path))
    return work


def worker_fn(gpu_id, work_items, cfg_path, ckpt_path, batch_size):
    """Worker process: loads model on assigned GPU and processes its work items."""
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    cfg = get_config(cfg_path)
    model = DLP(
        cdim=cfg["ch"],
        image_size=cfg["voxel_grid_whd"][0],
        normalize_rgb=cfg["normalize_rgb"],
        n_kp_per_patch=cfg["n_kp_per_patch"],
        patch_size=cfg["patch_size"],
        anchor_s=cfg["anchor_s"],
        n_kp_enc=cfg["n_kp_enc"],
        n_kp_prior=cfg["n_kp_prior"],
        pad_mode=cfg["pad_mode"],
        dropout=cfg["dropout"],
        features_dist=cfg.get("features_dist", "gauss"),
        learned_feature_dim=cfg["learned_feature_dim"],
        learned_bg_feature_dim=cfg.get("learned_bg_feature_dim", cfg["learned_feature_dim"]),
        n_fg_categories=cfg.get("n_fg_categories", 8),
        n_fg_classes=cfg.get("n_fg_classes", 4),
        n_bg_categories=cfg.get("n_bg_categories", 4),
        n_bg_classes=cfg.get("n_bg_classes", 4),
        scale_std=cfg["scale_std"],
        offset_std=cfg["offset_std"],
        obj_on_alpha=cfg["obj_on_alpha"],
        obj_on_beta=cfg["obj_on_beta"],
        obj_res_from_fc=cfg["obj_res_from_fc"],
        obj_ch_mult_prior=cfg.get("obj_ch_mult_prior", cfg["obj_ch_mult"]),
        obj_ch_mult=cfg["obj_ch_mult"],
        obj_base_ch=cfg["obj_base_ch"],
        obj_final_cnn_ch=cfg["obj_final_cnn_ch"],
        bg_res_from_fc=cfg["bg_res_from_fc"],
        bg_ch_mult=cfg["bg_ch_mult"],
        bg_base_ch=cfg["bg_base_ch"],
        bg_final_cnn_ch=cfg["bg_final_cnn_ch"],
        use_resblock=cfg["use_resblock"],
        num_res_blocks=cfg["num_res_blocks"],
        cnn_mid_blocks=cfg.get("cnn_mid_blocks", False),
        mlp_hidden_dim=cfg.get("mlp_hidden_dim", 256),
        pint_enc_layers=cfg["pint_enc_layers"],
        pint_enc_heads=cfg["pint_enc_heads"],
        timestep_horizon=1,
        separate_depth_features=cfg.get("separate_depth_features", False),
        depth_feature_dim=cfg.get("depth_feature_dim", 0),
        split_loss=cfg.get("split_loss", False),
        depth_loss_ratio=cfg.get("depth_loss_ratio", 1.0),
    ).to(device)
    model.eval()
    _ = load_checkpoint(ckpt_path, model, None, None, map_location=device)

    prior = model.prior_module
    print(f"[GPU {gpu_id}] Model loaded, processing {len(work_items)} frames")

    from tqdm import tqdm
    pbar = tqdm(range(0, len(work_items), batch_size), desc=f"GPU {gpu_id}", unit="batch")
    done = 0

    for i in pbar:
        batch = work_items[i:i + batch_size]

        # Load voxels
        vox_list = []
        for _, _, _, vox_path, _ in batch:
            vox_list.append(load_voxel_dense(vox_path).to(device))
        vox = torch.stack(vox_list, dim=0)

        # Run kmeans prior only
        with torch.no_grad():
            kp, cov = prior.encode_prior(vox)

        # Save per-frame
        for j, (task, demo, frame_idx, _, km_path) in enumerate(batch):
            os.makedirs(os.path.dirname(km_path), exist_ok=True)
            torch.save({"kp": kp[j].cpu(), "cov": cov[j].cpu()}, km_path)

        done += len(batch)
        pbar.set_postfix(done=done, total=len(work_items))


def main():
    mp.set_start_method("spawn", force=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dlp-cfg", required=True)
    ap.add_argument("--dlp-ckpt", required=True)
    ap.add_argument("--batch", type=int, default=1,
                    help="Batch size per worker (kmeans is sequential per sample, so 1 is fine)")
    ap.add_argument("--num-gpus", type=int, default=None,
                    help="Number of GPUs to use (default: all available)")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="Subset of tasks to process (default: all)")
    args = ap.parse_args()

    num_gpus = args.num_gpus or torch.cuda.device_count()
    tasks = args.tasks if args.tasks else TASKS
    print(f"Using {num_gpus} GPUs, {len(tasks)} tasks")

    # Build work list (skips already-cached frames)
    work = build_frame_list(args.data_root, tasks)
    print(f"Total frames to compute: {len(work)}")
    if not work:
        print("Nothing to do — all frames already cached!")
        return

    # Split work evenly across GPUs
    chunks = [[] for _ in range(num_gpus)]
    for i, item in enumerate(work):
        chunks[i % num_gpus].append(item)

    for g in range(num_gpus):
        print(f"  GPU {g}: {len(chunks[g])} frames")

    # Launch workers
    processes = []
    for gpu_id in range(num_gpus):
        if not chunks[gpu_id]:
            continue
        p = mp.Process(
            target=worker_fn,
            args=(gpu_id, chunks[gpu_id], args.dlp_cfg, args.dlp_ckpt, args.batch),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\nAll done!")


if __name__ == "__main__":
    main()
