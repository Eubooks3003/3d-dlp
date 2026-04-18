#!/usr/bin/env python3
"""
Fast parallel kmeans precomputation for RLBench voxel caches.

RLBench layout (input):
    <data-root>/
      <split_dir>/                           # train_data, test_data
        <task>/
          all_variations/episodes/episode<N>/
            voxel_cache/
              000000_voxels.pt               # [4,D,H,W] RGBO (compressed sparse ok)
              000001_voxels.pt
              ...

RLBench layout (output — sibling of voxel_cache):
    <data-root>/<split>/<task>/all_variations/episodes/episode<N>/
      kmeans_cache/
        000000_kmeans.pt                     # {"kp": [K,3], "cov": [K,3,3]}
        ...

Voxels are [4,D,H,W] RGBO. The DLP prior operates on RGB only, so we slice
off the occupancy channel (vox[:3]) before the forward pass — matching the
training dataset (RLBenchMultiTaskVoxelDataset).

Usage:
    # Debug: process a handful of frames per task, one worker, verbose
    PYTHONPATH=. python scripts/precompute_kmeans_rlbench.py \
        --data-root /home/ubuntu/tal-lpwm-neurips-2026/data/rlbench \
        --dlp-cfg  configs/rlbench_multitask.json \
        --frame-stride 5 \
        --debug

    # Full run
    PYTHONPATH=. python scripts/precompute_kmeans_rlbench.py \
        --data-root /home/ubuntu/tal-lpwm-neurips-2026/data/rlbench \
        --dlp-cfg  configs/rlbench_multitask.json \
        --frame-stride 5 \
        --workers-per-gpu 4
"""
import os
import re
import sys
import time
import argparse
import tempfile
import glob
import torch
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

import numpy.core as _core
sys.modules['numpy._core'] = _core
sys.modules['numpy._core.multiarray'] = _core.multiarray


# Default set matches configs/rlbench_multitask.json
DEFAULT_TASKS = [
    "close_jar",
    "open_drawer",
    "sweep_to_dustpan_of_size",
    "meat_off_grill",
    "turn_tap",
    "slide_block_to_color_target",
    "put_item_in_drawer",
    "reach_and_drag",
    "push_buttons",
    "stack_blocks",
]
DEFAULT_SPLITS = ["train_data", "test_data"]


def load_voxel_rgb(path):
    """
    Load RLBench RGBO voxel, slice to RGB only ([3,D,H,W] float32).

    Handles both compressed-sparse dicts and dense tensors.
    """
    data = torch.load(path, weights_only=False)
    if isinstance(data, dict) and data.get("compressed"):
        shape = data["shape"]                            # [4,D,H,W] expected
        coords = data["coords"].long()
        values = data["values"].float()
        vox = torch.zeros(shape, dtype=torch.float32)
        vox[:, coords[:, 0], coords[:, 1], coords[:, 2]] = values.T
    elif isinstance(data, torch.Tensor):
        vox = data.float()
    else:
        raise RuntimeError(f"Unexpected format in {path}: {type(data)}")
    if vox.dim() != 4 or vox.shape[0] < 3:
        raise RuntimeError(f"Expected [C>=3,D,H,W] at {path}, got {tuple(vox.shape)}")
    return vox[:3].contiguous()                           # drop occupancy channel


def build_frame_list(data_root, splits, tasks, frame_stride, max_per_episode=None):
    """
    Walk the RLBench layout, return a flat work list of
      (split, task, episode, frame_idx, vox_path, km_path)
    skipping frames whose kmeans file already exists.
    """
    frame_stride = max(1, int(frame_stride))
    work, cached = [], 0
    for split in splits:
        for task in tasks:
            episodes_root = os.path.join(
                data_root, split, task, "all_variations", "episodes"
            )
            if not os.path.isdir(episodes_root):
                print(f"[skip] {split}/{task}: no episodes root at {episodes_root}")
                continue
            episode_dirs = sorted(
                glob.glob(os.path.join(episodes_root, "episode*")),
                key=lambda p: int(os.path.basename(p).replace("episode", ""))
            )
            task_new, task_total = 0, 0
            for ep_dir in episode_dirs:
                ep_idx = int(os.path.basename(ep_dir).replace("episode", ""))
                vox_dir = os.path.join(ep_dir, "voxel_cache")
                km_dir = os.path.join(ep_dir, "kmeans_cache")
                if not os.path.isdir(vox_dir):
                    continue
                km_existing = set()
                try:
                    km_existing = set(os.listdir(km_dir))
                except FileNotFoundError:
                    pass
                vox_files = sorted(glob.glob(os.path.join(vox_dir, "*_voxels.pt")))
                kept = 0
                for vp in vox_files:
                    stem = os.path.basename(vp).replace("_voxels.pt", "")
                    try:
                        frame_idx = int(stem)
                    except ValueError:
                        continue
                    if frame_idx % frame_stride != 0:
                        continue
                    if max_per_episode is not None and kept >= max_per_episode:
                        break
                    kept += 1
                    task_total += 1
                    km_fname = f"{frame_idx:06d}_kmeans.pt"
                    if km_fname in km_existing:
                        cached += 1
                        continue
                    km_path = os.path.join(km_dir, km_fname)
                    work.append((split, task, ep_idx, frame_idx, vp, km_path))
                    task_new += 1
            print(f"  [{split}] {task}: {task_new}/{task_total} to compute")
    return work, cached


def build_prior(cfg_path, ckpt_path, device):
    """Build DLP, return just the prior module (no gradients needed)."""
    from voxel_models import DLP
    from utils.util_func import get_config

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
    if ckpt_path is not None:
        from utils.log_utils import load_checkpoint
        _ = load_checkpoint(ckpt_path, model, None, None, map_location=device)
    return model.prior_module


def worker_fn(worker_id, gpu_id, work_items, cfg_path, ckpt_path, counter_file, batch_size=32):
    """Worker: load prior on gpu_id, stream through work_items in batches."""
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    prior = build_prior(cfg_path, ckpt_path, device)
    print(f"[Worker {worker_id} / GPU {gpu_id}] Model loaded, {len(work_items)} frames", flush=True)

    BATCH_SIZE = batch_size

    def _load(item):
        _, _, _, _, vp, _ = item
        return load_voxel_rgb(vp)

    def _save_batch(kp_cpu, cov_cpu, items):
        for j, (_, _, _, _, _, km_path) in enumerate(items):
            os.makedirs(os.path.dirname(km_path), exist_ok=True)
            torch.save({"kp": kp_cpu[j], "cov": cov_cpu[j]}, km_path)

    t_io = t_gpu = 0.0

    with ThreadPoolExecutor(max_workers=8) as io_pool:
        PREFETCH = min(BATCH_SIZE * 6, len(work_items))
        futures = [io_pool.submit(_load, item) for item in work_items[:PREFETCH]]
        submitted = PREFETCH
        save_future = None

        for batch_start in range(0, len(work_items), BATCH_SIZE):
            batch_items = work_items[batch_start:batch_start + BATCH_SIZE]
            actual_bs = len(batch_items)

            t0 = time.monotonic()
            vox_list = []
            for _ in range(actual_bs):
                vox_list.append(futures.pop(0).result())
                if submitted < len(work_items):
                    futures.append(io_pool.submit(_load, work_items[submitted]))
                    submitted += 1
            vox_batch = torch.stack(vox_list, dim=0).to(device)  # [B,3,D,H,W]
            t_io += time.monotonic() - t0

            if save_future is not None:
                save_future.result()

            t0 = time.monotonic()
            with torch.no_grad():
                kp, cov = prior.encode_prior(vox_batch) if hasattr(prior, "encode_prior") else prior(vox_batch)
            torch.cuda.synchronize(device)
            t_gpu += time.monotonic() - t0

            kp_cpu, cov_cpu = kp.cpu(), cov.cpu()
            save_future = io_pool.submit(_save_batch, kp_cpu, cov_cpu, batch_items)

            with open(counter_file, "ab") as f:
                f.write(b"x" * actual_bs)

            n_done = batch_start + actual_bs
            if (n_done // BATCH_SIZE) % 100 == 0:
                total_t = t_io + t_gpu
                if total_t > 0:
                    print(f"[Worker {worker_id}] {n_done}/{len(work_items)} | "
                          f"IO: {t_io:.1f}s ({100*t_io/total_t:.0f}%) | "
                          f"GPU: {t_gpu:.1f}s ({100*t_gpu/total_t:.0f}%) | "
                          f"fps: {n_done/total_t:.1f}", flush=True)

        if save_future is not None:
            save_future.result()


def run_debug(work, cfg_path, ckpt_path, max_frames=8):
    """Sequential single-GPU debug: process up to max_frames with detailed prints."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[debug] device={device}, processing up to {max_frames} frames")
    prior = build_prior(cfg_path, ckpt_path, device)

    subset = work[:max_frames]
    print(f"[debug] frames to process: {len(subset)}")
    for i, (split, task, ep, frame, vp, km) in enumerate(subset):
        print(f"[debug {i+1}/{len(subset)}] {split}/{task}/ep{ep}/frame{frame}")
        print(f"  vox: {vp}")
        print(f"  out: {km}")
        vox = load_voxel_rgb(vp).unsqueeze(0).to(device)  # [1,3,D,H,W]
        print(f"  shape {tuple(vox.shape)}  min/max/mean "
              f"{vox.min().item():.3f}/{vox.max().item():.3f}/{vox.mean().item():.4f}")
        with torch.no_grad():
            kp, cov = prior.encode_prior(vox) if hasattr(prior, "encode_prior") else prior(vox)
        print(f"  kp {tuple(kp.shape)}  cov {tuple(cov.shape)}")
        print(f"  kp[0,:3]={kp[0, :3].cpu().tolist()}")
        os.makedirs(os.path.dirname(km), exist_ok=True)
        torch.save({"kp": kp[0].cpu(), "cov": cov[0].cpu()}, km)
        print(f"  saved -> {km}\n")
    print("[debug] done")


def main():
    mp.set_start_method("forkserver", force=True)

    ap = argparse.ArgumentParser(description="Fast parallel kmeans precomputation for RLBench")
    ap.add_argument("--data-root", required=True,
                    help="e.g. /home/ubuntu/tal-lpwm-neurips-2026/data/rlbench")
    ap.add_argument("--dlp-cfg", required=True,
                    help="config json (e.g. configs/rlbench_multitask.json)")
    ap.add_argument("--dlp-ckpt", default=None,
                    help="optional checkpoint; prior is algorithmic so weights aren't required")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help=f"subset of tasks (default: {DEFAULT_TASKS})")
    ap.add_argument("--splits", nargs="*", default=None,
                    help=f"filesystem splits (default: {DEFAULT_SPLITS})")
    ap.add_argument("--frame-stride", type=int, default=5,
                    help="keep every Nth frame per episode (match training config)")
    ap.add_argument("--max-per-episode", type=int, default=None,
                    help="optional cap on frames per episode (after stride)")
    ap.add_argument("--num-gpus", type=int, default=None,
                    help="number of GPUs (default: all available)")
    ap.add_argument("--workers-per-gpu", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--debug", action="store_true",
                    help="process a few frames sequentially on GPU 0, verbose output, then exit")
    ap.add_argument("--debug-frames", type=int, default=8,
                    help="number of frames to process in debug mode")
    args = ap.parse_args()

    tasks = args.tasks if args.tasks else DEFAULT_TASKS
    splits = args.splits if args.splits else DEFAULT_SPLITS

    print(f"data-root:     {args.data_root}")
    print(f"dlp-cfg:       {args.dlp_cfg}")
    print(f"dlp-ckpt:      {args.dlp_ckpt}")
    print(f"tasks:         {tasks}")
    print(f"splits:        {splits}")
    print(f"frame-stride:  {args.frame_stride}")
    print(f"max-per-ep:    {args.max_per_episode}")
    print()

    work, cached = build_frame_list(
        args.data_root, splits, tasks,
        frame_stride=args.frame_stride,
        max_per_episode=args.max_per_episode,
    )
    print(f"\nTotal: {len(work)} to compute, {cached} already cached")
    if not work:
        print("Nothing to do!")
        return

    if args.debug:
        run_debug(work, args.dlp_cfg, args.dlp_ckpt, max_frames=args.debug_frames)
        return

    num_gpus = args.num_gpus or torch.cuda.device_count()
    total_workers = max(1, num_gpus * args.workers_per_gpu)
    print(f"Workers: {num_gpus} GPU(s) x {args.workers_per_gpu} = {total_workers} processes")

    chunks = [[] for _ in range(total_workers)]
    for i, item in enumerate(work):
        chunks[i % total_workers].append(item)

    counter_file = tempfile.mktemp(prefix="kmeans_rlbench_progress_", suffix=".bin")
    open(counter_file, "wb").close()

    processes = []
    for w in range(total_workers):
        if not chunks[w]:
            continue
        gpu_id = w // args.workers_per_gpu
        p = mp.Process(
            target=worker_fn,
            args=(w, gpu_id, chunks[w], args.dlp_cfg, args.dlp_ckpt, counter_file, args.batch),
        )
        p.start()
        processes.append(p)

    from tqdm import tqdm
    pbar = tqdm(total=len(work), desc="Total", unit="frame")
    last = 0
    while any(p.is_alive() for p in processes):
        try:
            done = os.path.getsize(counter_file)
        except OSError:
            done = last
        if done > last:
            pbar.update(done - last)
            last = done
        time.sleep(1.0)

    done = os.path.getsize(counter_file)
    if done > last:
        pbar.update(done - last)
    pbar.close()

    for p in processes:
        p.join()
    failed = sum(1 for p in processes if p.exitcode != 0)
    if failed:
        print(f"\nWARNING: {failed}/{len(processes)} workers failed. Re-run to fill in; cached files are skipped.")
    else:
        print("\nAll done!")

    os.unlink(counter_file)


if __name__ == "__main__":
    main()
