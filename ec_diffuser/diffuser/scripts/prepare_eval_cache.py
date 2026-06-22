"""
One-shot helper: build the (multitask) dataset once and freeze the bits eval
needs into <savepath>/eval_cache.pkl.

Eval-time the worker only consumes the normalizer + a handful of dims/flags;
re-fitting the normalizer per-worker on the full 10-task dataset is what makes
K=10 multitask runs swap-thrash. After this runs, eval workers can build a
stub dataset from the cache in ~5 s instead of ~3 min and ~500 MB RAM instead
of ~6-8 GB.

Usage (mirrors eval_paper_rlbench_3d.py args):
    cd /home/ellina/Desktop/EC-Diffuser
    python diffuser/scripts/prepare_eval_cache.py \\
        --config config.rlbench_multitask_keypose_dlp \\
        --dataset multitask --num_entity 16 --input_type dlp --seed 42

Idempotent: re-running overwrites <savepath>/eval_cache.pkl.
"""
import argparse
import copy
import os
import pickle
import sys
from datetime import datetime

# Make lpwm-dev importable (matches eval_paper_rlbench_3d.py).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_env_lpwm = os.environ.get("LPWM_DEV")
_cands = ([_env_lpwm] if _env_lpwm else []) + [
    os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "..", "lpwm-dev")),
]
for _p in _cands:
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)
        break

import diffuser.utils as utils

CACHE_FILENAME = "eval_cache.pkl"
CACHE_VERSION = 1


def _build_args(raw_argv):
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--out", type=str, default=None,
                     help="Output path (default: <savepath>/" + CACHE_FILENAME + ")")
    pre.add_argument("--force", action="store_true",
                     help="Overwrite an existing cache file")
    ours, rest = pre.parse_known_args(raw_argv)

    from diffuser.utils.args import ArgsParser
    sys.argv = [sys.argv[0]] + rest
    args = ArgsParser().parse_args("diffusion")
    args._out = ours.out
    args._force = ours.force
    return args


def _strip_raw_X(normalizer):
    """Drop the giant `self.X` array kept by the Normalizer constructor.

    Eval-time normalize/unnormalize only need (mins, maxs, means, stds, z, x_dim);
    keeping `self.X` means each worker carries the entire training dataset in RAM
    after unpickling. We `copy.copy` first so the in-memory normalizer used by
    other code in this script stays untouched.
    """
    snap = copy.copy(normalizer)
    if hasattr(snap, "X"):
        try:
            delattr(snap, "X")
        except Exception:
            snap.X = None
    return snap


def _snapshot_dataset_normalizer(dataset_normalizer):
    """Snapshot DatasetNormalizer in a form safe to pickle without raw data."""
    snap = copy.copy(dataset_normalizer)
    snap.normalizers = {
        key: _strip_raw_X(n) for key, n in dataset_normalizer.normalizers.items()
    }
    return snap


def main(argv):
    args = _build_args(argv)

    out_path = args._out or os.path.join(args.savepath, CACHE_FILENAME)
    if os.path.exists(out_path) and not args._force:
        print(f"[prepare_eval_cache] cache already exists: {out_path}", flush=True)
        print(f"[prepare_eval_cache] re-run with --force to overwrite", flush=True)
        return

    print(f"[prepare_eval_cache] savepath  = {args.savepath}", flush=True)
    print(f"[prepare_eval_cache] dataset   = {args.dataset}", flush=True)
    print(f"[prepare_eval_cache] config    = {args.config}", flush=True)
    print(f"[prepare_eval_cache] out       = {out_path}", flush=True)

    # Build the dataset exactly as the worker does today.
    print(f"[prepare_eval_cache] building dataset (this is the slow part)...", flush=True)
    t0 = datetime.now()
    dataset_config = utils.Config(
        args.loader,
        savepath=(args.savepath, "dataset_config.pkl"),
        dataset_path=args.override_dataset_path,
        dataset_name=args.dataset,
        horizon=args.horizon,
        obs_only=args.obs_only,
        action_only=args.action_only,
        normalizer=args.normalizer,
        particle_normalizer=args.particle_normalizer,
        preprocess_fns=args.preprocess_fns,
        use_padding=args.use_padding,
        max_path_length=args.max_path_length,
        overfit=getattr(args, "overfit", False),
        single_view=(args.input_type == "dlp" and not args.multiview),
        action_z_scale=getattr(args, "action_z_scale", 1.0),
        use_gripper_obs=getattr(args, "use_gripper_obs", False),
        use_bg_obs=getattr(args, "use_bg_obs", False),
        keypose_mode=getattr(args, "keypose_mode", False),
    )
    dataset = dataset_config()
    elapsed = (datetime.now() - t0).total_seconds()
    print(f"[prepare_eval_cache] dataset built in {elapsed:.1f}s", flush=True)

    # Pull eval-relevant attrs.
    payload = {
        "version":          CACHE_VERSION,
        "created":          datetime.now().isoformat(),
        "config":           args.config,
        "dataset_name":     args.dataset,
        "dataset_paths":    list(args.override_dataset_path) if isinstance(args.override_dataset_path, (list, tuple)) else [args.override_dataset_path],

        # Normalizer (with self.X dropped from each per-key Normalizer).
        "normalizer":       _snapshot_dataset_normalizer(dataset.normalizer),

        # Dims.
        "observation_dim":  int(dataset.observation_dim),
        "action_dim":       int(dataset.action_dim),
        "gripper_dim":      int(getattr(dataset, "gripper_dim", 0)),
        "bg_dim":           int(getattr(dataset, "bg_dim", 0)),

        # Misc flags eval_rlbench_rollouts reads off the dataset object.
        "horizon":          int(getattr(dataset, "horizon", args.horizon)),
        "action_z_scale":   float(getattr(dataset, "action_z_scale", 1.0)),
        "use_gripper_obs":  bool(getattr(dataset, "use_gripper_obs", False)),
        "use_bg_obs":       bool(getattr(dataset, "use_bg_obs", False)),
        "keypose_mode":     bool(getattr(args, "keypose_mode", False)),
    }

    print(f"[prepare_eval_cache] dims: obs={payload['observation_dim']} "
          f"act={payload['action_dim']} grip={payload['gripper_dim']} "
          f"bg={payload['bg_dim']}", flush=True)
    print(f"[prepare_eval_cache] normalizer keys: "
          f"{sorted(payload['normalizer'].normalizers.keys())}", flush=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[prepare_eval_cache] wrote {out_path}  ({size_mb:.2f} MB)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
