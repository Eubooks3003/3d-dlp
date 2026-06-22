"""
Multitask wrapper around eval_paper_rlbench_3d.py.

Spawns one CoppeliaSim subprocess per RLBench task, evaluating the same
multitask checkpoint against each task in turn (or in parallel up to
--concurrency). GPUs are assigned round-robin via CUDA_VISIBLE_DEVICES.

Each worker is its own process with its own xvfb display and CoppeliaSim
instance, sidestepping the "textures only on first launch per process"
constraint that would break in-process task switching.

Usage:
    cd EC-Diffuser/diffuser

    xvfb-run -a python diffuser/scripts/eval_paper_rlbench_multitask_3d.py \\
        --config config.rlbench_multitask_keypose_dlp \\
        --dataset multitask \\
        --num_entity 16 --input_type dlp --seed 42 \\
        --ckpt_step 980000 \\
        --n_rollouts 50 --seeds 42,123,456 \\
        --concurrency 10 --gpus 0,1

    # Subset of tasks
    ... --tasks close_jar,meat_off_grill --concurrency 2

    # Dry-run (print planned worker commands and exit)
    ... --dry_run
"""
import argparse
import glob
import importlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

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

import diffuser.utils as utils  # noqa: F401  (ensures sys.path is set up)

WORKER_SCRIPT  = os.path.join(_SCRIPT_DIR, "eval_paper_rlbench_3d.py")
PREPARE_SCRIPT = os.path.join(_SCRIPT_DIR, "prepare_eval_cache.py")
CACHE_FILENAME = "eval_cache.pkl"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_args(raw_argv):
    # allow_abbrev=False stops argparse from prefix-matching e.g. --seed onto
    # --seeds (which would silently steal ArgsParser's seed flag).
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    # Orchestrator-only flags
    pre.add_argument("--tasks", type=str, default=None,
                     help="Comma-separated subset of tasks (default: all from config.TASK_NAMES)")
    pre.add_argument("--concurrency", type=int, default=1,
                     help="Max simultaneous CoppeliaSim workers (default: 1, i.e. sequential)")
    pre.add_argument("--gpus", type=str, default=None,
                     help="Comma-separated CUDA device ids for round-robin (e.g. '0,1'). "
                          "Default: inherit current CUDA_VISIBLE_DEVICES (or '0' if unset).")
    pre.add_argument("--dry_run", action="store_true",
                     help="Print planned worker commands and exit")
    pre.add_argument("--poll_interval", type=float, default=2.0,
                     help="Seconds between subprocess status polls (default: 2.0)")
    pre.add_argument("--no_cache", action="store_true",
                     help="Skip auto-prepare and pass --no_cache to each worker "
                          "(workers rebuild dataset from raw pkls).")
    pre.add_argument("--force_prepare", action="store_true",
                     help="Rebuild <savepath>/eval_cache.pkl even if it exists.")

    # Mirror worker pre-parser flags so they're stripped from ArgsParser input
    # but forwarded to each worker.
    pre.add_argument("--ckpt_step", default="latest")
    pre.add_argument("--ckpt_path", default=None)
    pre.add_argument("--n_rollouts", type=int, default=50)
    pre.add_argument("--seeds", type=str, default="42,123,456")
    pre.add_argument("--max_steps", type=int, default=200)
    pre.add_argument("--video_episodes", type=int, default=5)
    pre.add_argument("--video_fps", type=int, default=10)
    pre.add_argument("--output_dir", type=str, default=None,
                     help="Top-level multitask output dir "
                          "(default: <savepath>/paper_eval_multitask/<timestamp>/)")

    ours, rest = pre.parse_known_args(raw_argv)

    from diffuser.utils.args import ArgsParser
    sys.argv = [sys.argv[0]] + rest
    args = ArgsParser().parse_args("diffusion")
    args._ours = ours
    args._rest_argv = rest
    return args


# ---------------------------------------------------------------------------
# Task list resolution
# ---------------------------------------------------------------------------
def _resolve_task_names(args):
    if args._ours.tasks:
        return [t.strip() for t in args._ours.tasks.split(",") if t.strip()]
    # Prefer the value already plumbed through args (from mode_to_args).
    if hasattr(args, "task_names") and args.task_names:
        return list(args.task_names)
    # Fall back to importing the config module directly.
    try:
        config_mod = importlib.import_module(args.config)
    except ImportError as e:
        raise RuntimeError(f"Could not import config module {args.config!r}: {e}")
    if hasattr(config_mod, "TASK_NAMES"):
        return list(config_mod.TASK_NAMES)
    raise RuntimeError(
        f"Could not resolve task list: pass --tasks explicitly, or expose "
        f"task_names in mode_to_args / a TASK_NAMES module-level list in {args.config}"
    )


# ---------------------------------------------------------------------------
# GPU round-robin
# ---------------------------------------------------------------------------
def _resolve_gpus(args):
    if args._ours.gpus:
        return [g.strip() for g in args._ours.gpus.split(",") if g.strip()]
    cur = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cur:
        return [g.strip() for g in cur.split(",") if g.strip()]
    return ["0"]


# ---------------------------------------------------------------------------
# Worker command construction
# ---------------------------------------------------------------------------
def _build_worker_cmd(args, task, output_subdir):
    ours = args._ours
    cmd = ["xvfb-run", "-a", sys.executable, WORKER_SCRIPT]
    cmd += list(args._rest_argv)
    cmd += ["--eval_task", task, "--output_dir", output_subdir]

    if ours.ckpt_path is not None:
        cmd += ["--ckpt_path", ours.ckpt_path]
    elif ours.ckpt_step != "latest":
        cmd += ["--ckpt_step", str(ours.ckpt_step)]
    cmd += [
        "--n_rollouts",     str(ours.n_rollouts),
        "--seeds",          ours.seeds,
        "--max_steps",      str(ours.max_steps),
        "--video_episodes", str(ours.video_episodes),
        "--video_fps",      str(ours.video_fps),
    ]
    if ours.no_cache:
        cmd += ["--no_cache"]
    return cmd


# ---------------------------------------------------------------------------
# One-time eval cache prep
# ---------------------------------------------------------------------------
def _ensure_cache(args):
    """Run prepare_eval_cache.py (synchronously) if the cache is missing or stale.

    Returns True if the cache is usable after this call, False if --no_cache.
    Raises CalledProcessError if the prepare run itself failed.
    """
    ours = args._ours
    cache_path = os.path.join(args.savepath, CACHE_FILENAME)

    if ours.no_cache:
        print(f"[multitask_eval] --no_cache set: workers will rebuild dataset from raw pkls",
              flush=True)
        return False

    if os.path.isfile(cache_path) and not ours.force_prepare:
        print(f"[multitask_eval] eval cache present: {cache_path}", flush=True)
        return True

    if ours.force_prepare:
        reason = "--force_prepare"
    else:
        reason = "missing cache"
    print(f"[multitask_eval] preparing eval cache ({reason}) -> {cache_path}", flush=True)
    print(f"[multitask_eval] this is a one-time ~3 min build; subsequent runs reuse it.",
          flush=True)

    cmd = [sys.executable, PREPARE_SCRIPT] + list(args._rest_argv)
    if ours.force_prepare:
        cmd += ["--force"]
    print(f"[multitask_eval] $ {' '.join(shlex.quote(x) for x in cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, env=os.environ.copy(), cwd=os.getcwd())
    elapsed = time.time() - t0
    if rc != 0:
        raise RuntimeError(
            f"prepare_eval_cache.py failed (rc={rc}, elapsed={elapsed:.1f}s). "
            f"Run it manually to debug, or pass --no_cache to skip."
        )
    if not os.path.isfile(cache_path):
        raise RuntimeError(
            f"prepare_eval_cache.py exited 0 but no cache at {cache_path}."
        )
    print(f"[multitask_eval] eval cache ready in {elapsed:.1f}s", flush=True)
    return True


# ---------------------------------------------------------------------------
# Per-task result parsing
# ---------------------------------------------------------------------------
def _read_worker_result(output_subdir):
    """Locate the most recent eval_*.json the worker wrote."""
    files = sorted(glob.glob(os.path.join(output_subdir, "eval_*.json")),
                   key=os.path.getmtime)
    if not files:
        return None, None
    path = files[-1]
    try:
        with open(path) as f:
            return json.load(f), path
    except Exception as e:
        return {"_parse_error": str(e)}, path


# ---------------------------------------------------------------------------
# Main orchestration loop
# ---------------------------------------------------------------------------
def main(argv):
    args = _build_args(argv)
    ours = args._ours

    task_names = _resolve_task_names(args)
    gpus = _resolve_gpus(args)
    K = max(1, int(ours.concurrency))

    # Top-level output dir.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if ours.output_dir:
        top_out = ours.output_dir
    else:
        top_out = os.path.join(args.savepath, "paper_eval_multitask", timestamp)
    os.makedirs(top_out, exist_ok=True)

    print(f"[multitask_eval] savepath        = {args.savepath}", flush=True)
    print(f"[multitask_eval] output_dir      = {top_out}", flush=True)
    print(f"[multitask_eval] tasks ({len(task_names)})  = {task_names}", flush=True)
    print(f"[multitask_eval] concurrency     = {K}", flush=True)
    print(f"[multitask_eval] gpus (round-robin) = {gpus}", flush=True)
    print(f"[multitask_eval] ckpt_step       = {ours.ckpt_step}"
          f"{'  (overridden by --ckpt_path)' if ours.ckpt_path else ''}", flush=True)
    print(f"[multitask_eval] seeds           = {ours.seeds}", flush=True)
    print(f"[multitask_eval] n_rollouts/seed = {ours.n_rollouts}"
          f"   max_steps = {ours.max_steps}", flush=True)

    # Build per-task plans.
    plans = []
    for i, task in enumerate(task_names):
        subdir = os.path.join(top_out, task)
        os.makedirs(subdir, exist_ok=True)
        gpu = gpus[i % len(gpus)]
        cmd = _build_worker_cmd(args, task, subdir)
        log_path = os.path.join(subdir, "log.txt")
        plans.append({
            "idx":       i,
            "task":      task,
            "gpu":       gpu,
            "cmd":       cmd,
            "subdir":    subdir,
            "log_path":  log_path,
        })

    print("\n[multitask_eval] planned worker commands:", flush=True)
    for p in plans:
        print(f"  [{p['idx']:2d}] task={p['task']:<30} gpu={p['gpu']}  log={p['log_path']}",
              flush=True)
        print(f"       CUDA_VISIBLE_DEVICES={p['gpu']} \\\n       "
              + " ".join(shlex.quote(x) for x in p["cmd"]), flush=True)
    print("", flush=True)

    if ours.dry_run:
        print("[multitask_eval] --dry_run: not launching workers.", flush=True)
        return

    # Build/locate the eval cache before fanning out so workers don't all race
    # on the multi-GB normalizer fit (which is what made K=10 multitask
    # swap-thrash before).
    _ensure_cache(args)

    # Persist the plan up-front so partial summary files always reference it.
    summary_path = os.path.join(top_out, "summary.json")
    summary = {
        "status":              "running",
        "savepath":            args.savepath,
        "output_dir":          top_out,
        "tasks":               task_names,
        "seeds":               [int(s) for s in ours.seeds.split(",") if s.strip()],
        "n_rollouts_per_seed": ours.n_rollouts,
        "max_steps":           ours.max_steps,
        "ckpt_step":           ours.ckpt_step,
        "ckpt_path":           ours.ckpt_path,
        "concurrency":         K,
        "gpus":                gpus,
        "dataset":             args.dataset,
        "config":              args.config,
        "timestamp_started":   timestamp,
        "per_task":            [],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Popen pool with concurrency limit.
    pending = list(plans)
    running = []  # list of {plan, proc, log_fh, t_start}
    completed = []

    def _launch(plan):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = plan["gpu"]
        log_fh = open(plan["log_path"], "w", buffering=1)
        log_fh.write(f"# task={plan['task']} gpu={plan['gpu']} "
                     f"started={datetime.now().isoformat()}\n")
        log_fh.write("# cmd: " + " ".join(shlex.quote(x) for x in plan["cmd"]) + "\n\n")
        log_fh.flush()
        proc = subprocess.Popen(
            plan["cmd"],
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=os.getcwd(),
        )
        running.append({"plan": plan, "proc": proc, "log_fh": log_fh, "t_start": time.time()})
        print(f"[multitask_eval] launched {plan['task']:<30} on GPU {plan['gpu']}  "
              f"(running={len(running)}/{K}, pending={len(pending)}, done={len(completed)})",
              flush=True)

    while pending or running:
        # Saturate up to K.
        while pending and len(running) < K:
            _launch(pending.pop(0))

        # Poll.
        time.sleep(ours.poll_interval)
        still_running = []
        for slot in running:
            rc = slot["proc"].poll()
            if rc is None:
                still_running.append(slot)
                continue
            slot["log_fh"].close()
            elapsed = time.time() - slot["t_start"]
            plan = slot["plan"]
            result, json_path = _read_worker_result(plan["subdir"])
            sr_str = "?"
            if result and isinstance(result, dict):
                if result.get("status") == "complete":
                    sr_str = f"{result.get('overall_success_rate', 0.0)*100:.1f}%"
                elif result.get("status") == "partial":
                    sr_str = f"{result.get('overall_success_rate', 0.0)*100:.1f}% (partial)"
            tag = "OK " if rc == 0 else f"FAIL(rc={rc})"
            print(f"[multitask_eval] {tag} {plan['task']:<30} "
                  f"in {elapsed:6.1f}s  success={sr_str}  log={plan['log_path']}",
                  flush=True)
            completed.append({
                "task":           plan["task"],
                "gpu":            plan["gpu"],
                "exit_code":      rc,
                "elapsed_sec":    elapsed,
                "log_path":       plan["log_path"],
                "json_path":      json_path,
                "result":         result,
            })

            # Persist partial summary after each task finishes.
            summary["per_task"] = completed
            summary["status"] = "running" if (pending or still_running) else "finalizing"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2,
                          default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
        running = still_running

    # Aggregate.
    rates_per_task = []
    for c in completed:
        r = c["result"] or {}
        if isinstance(r, dict) and r.get("status") == "complete":
            rates_per_task.append(r.get("overall_success_rate", 0.0))

    n_complete = sum(1 for c in completed
                     if c["exit_code"] == 0 and (c["result"] or {}).get("status") == "complete")
    n_failed = len(completed) - n_complete
    summary["status"] = "complete"
    summary["timestamp_finished"] = datetime.now().isoformat()
    summary["n_tasks_complete"] = n_complete
    summary["n_tasks_failed"] = n_failed
    summary["across_task_mean_success_rate"] = float(np.mean(rates_per_task)) if rates_per_task else 0.0
    summary["across_task_std_success_rate"] = float(np.std(rates_per_task, ddof=0)) if rates_per_task else 0.0
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))

    # Final report.
    print("\n" + "=" * 76, flush=True)
    print("[multitask_eval] FINAL", flush=True)
    print("=" * 76, flush=True)
    for c in completed:
        r = c["result"] or {}
        sr = r.get("overall_success_rate", 0.0) if isinstance(r, dict) else 0.0
        ok = (c["exit_code"] == 0 and isinstance(r, dict) and r.get("status") == "complete")
        flag = "  " if ok else "!!"
        print(f"  {flag} {c['task']:<30} "
              f"rc={c['exit_code']:>3}  "
              f"sr={sr*100:5.1f}%  "
              f"t={c['elapsed_sec']:6.1f}s  "
              f"gpu={c['gpu']}",
              flush=True)
    print(f"  across-task mean: "
          f"{summary['across_task_mean_success_rate']*100:5.1f}% "
          f"+/- {summary['across_task_std_success_rate']*100:.1f}%  "
          f"({n_complete}/{len(completed)} tasks complete)", flush=True)
    print(f"  summary: {summary_path}", flush=True)
    print("=" * 76, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
