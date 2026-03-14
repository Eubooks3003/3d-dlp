#!/usr/bin/env python3
"""
Fast parallel kmeans precomputation for all tasks.

Launches N worker processes per GPU, each with its own model replica +
background I/O threads for voxel loading. On a GH200 (96GB), 4-8 workers
on a single GPU keeps utilization high since per-frame kmeans uses <1GB.

Usage (single GH200):
    PYTHONPATH=. python scripts/precompute_kmeans.py \
        --data-root /path/to/3D-DLP-mimicgen-data \
        --dlp-cfg  /path/to/hparams.json \
        --dlp-ckpt /path/to/saves/best.pt \
        --workers-per-gpu 4

Usage (2x 4090):
    PYTHONPATH=. python scripts/precompute_kmeans.py \
        --data-root /path/to/3D-DLP-mimicgen-data \
        --dlp-cfg  /path/to/hparams.json \
        --dlp-ckpt /path/to/saves/best.pt \
        --num-gpus 2 --workers-per-gpu 2
"""
import os
import re
import sys
import argparse
import torch
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

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
    """Build flat list of (task, demo, frame_idx, voxel_path, kmeans_out_path).
    Skips frames that already have a cached kmeans file."""
    work = []
    cached = 0
    for task in tasks:
        vox_dir = os.path.join(data_root, task, "voxel_cache", "voxel")
        km_dir = os.path.join(data_root, task, "voxel_cache", "kmeans_cache")
        if not os.path.isdir(vox_dir):
            print(f"[skip] {task}: no voxel cache at {vox_dir}")
            continue
        voxel_map = scan_voxel_cache(vox_dir)
        task_total, task_cached = 0, 0
        for demo in sorted(voxel_map.keys(), key=lambda s: int(s.split("_")[-1])):
            demo_km_dir = os.path.join(km_dir, demo)
            for frame_idx in sorted(voxel_map[demo].keys()):
                task_total += 1
                km_path = os.path.join(demo_km_dir, f"frame{frame_idx}_kmeans.pt")
                if os.path.exists(km_path):
                    task_cached += 1
                    cached += 1
                else:
                    work.append((task, demo, frame_idx, voxel_map[demo][frame_idx], km_path))
        print(f"  {task}: {task_total - task_cached}/{task_total} to compute")
    return work, cached


def build_model(cfg_path, ckpt_path, device):
    """Build DLP model and return just the prior module."""
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
    return model.prior_module


def worker_fn(worker_id, gpu_id, work_items, cfg_path, ckpt_path, progress_queue):
    """Worker process: loads model on assigned GPU, prefetches I/O, processes frames."""
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    prior = build_model(cfg_path, ckpt_path, device)
    print(f"[Worker {worker_id} / GPU {gpu_id}] Model loaded, {len(work_items)} frames")

    # Prefetch voxels in background threads while GPU computes
    PREFETCH = 8

    def _load(item):
        _, _, _, vox_path, _ = item
        return load_voxel_dense(vox_path)

    with ThreadPoolExecutor(max_workers=2) as io_pool:
        futures = [io_pool.submit(_load, item) for item in work_items[:PREFETCH]]
        submitted = PREFETCH

        for i, item in enumerate(work_items):
            # Get prefetched voxel
            vox = futures[0].result().unsqueeze(0).to(device)
            futures.pop(0)

            # Submit next prefetch
            if submitted < len(work_items):
                futures.append(io_pool.submit(_load, work_items[submitted]))
                submitted += 1

            # Run kmeans
            task, demo, frame_idx, _, km_path = item
            with torch.no_grad():
                kp, cov = prior.encode_prior(vox)

            # Save
            os.makedirs(os.path.dirname(km_path), exist_ok=True)
            torch.save({"kp": kp[0].cpu(), "cov": cov[0].cpu()}, km_path)

            progress_queue.put(1)


def progress_listener(queue, total):
    """Print progress from a single process."""
    from tqdm import tqdm
    pbar = tqdm(total=total, desc="Total", unit="frame")
    done = 0
    while done < total:
        n = queue.get()
        if n is None:
            break
        done += n
        pbar.update(n)
    pbar.close()


def main():
    mp.set_start_method("spawn", force=True)

    ap = argparse.ArgumentParser(description="Fast parallel kmeans precomputation")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dlp-cfg", required=True)
    ap.add_argument("--dlp-ckpt", required=True)
    ap.add_argument("--num-gpus", type=int, default=None,
                    help="Number of GPUs (default: all available)")
    ap.add_argument("--workers-per-gpu", type=int, default=4,
                    help="Worker processes per GPU (default: 4)")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="Subset of tasks (default: all)")
    args = ap.parse_args()

    num_gpus = args.num_gpus or torch.cuda.device_count()
    tasks = args.tasks if args.tasks else TASKS

    # Build work list
    work, cached = build_frame_list(args.data_root, tasks)
    print(f"\nTotal: {len(work)} to compute, {cached} already cached")
    print(f"Workers: {num_gpus} GPU(s) x {args.workers_per_gpu} workers = {num_gpus * args.workers_per_gpu} processes")
    if not work:
        print("Nothing to do!")
        return

    # Split work across all workers
    total_workers = num_gpus * args.workers_per_gpu
    chunks = [[] for _ in range(total_workers)]
    for i, item in enumerate(work):
        chunks[i % total_workers].append(item)

    # Progress queue
    progress_queue = mp.Queue()

    # Launch progress listener
    import threading
    progress_thread = threading.Thread(
        target=progress_listener, args=(progress_queue, len(work)), daemon=True
    )
    progress_thread.start()

    # Launch workers
    processes = []
    for w in range(total_workers):
        if not chunks[w]:
            continue
        gpu_id = w // args.workers_per_gpu
        p = mp.Process(
            target=worker_fn,
            args=(w, gpu_id, chunks[w], args.dlp_cfg, args.dlp_ckpt, progress_queue),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    progress_queue.put(None)
    progress_thread.join(timeout=5)
    print("\nAll done!")


if __name__ == "__main__":
    main()
