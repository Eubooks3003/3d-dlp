#!/usr/bin/env python3
"""
HDF5 breakdown / schema inspector (robomimic-style friendly)

What it prints:
- Top-level keys + attributes
- Episode list under /data
- A tree view for a chosen episode (datasets show shape/dtype/compression)
- Aggregate "schema" across episodes:
    * which dataset paths exist
    * how many episodes contain each path
    * dtype(s)
    * shape(s)
    * time-length stats (min/median/max of dim0 when applicable)
- (Optional) camera-specific mismatch/missing report like your original script
"""

import argparse
import h5py
import numpy as np
from collections import defaultdict
from statistics import median

DEFAULT_CAMS = ["agentview", "robot0_eye_in_hand", "birdview", "sideview"]


def _is_dataset(obj):
    return isinstance(obj, h5py.Dataset)


def _is_group(obj):
    return isinstance(obj, h5py.Group)


def _safe_str(x):
    try:
        return str(x)
    except Exception:
        return repr(x)


def list_episode_keys(h5, data_root="data"):
    if data_root not in h5:
        return []
    eps = list(h5[data_root].keys())
    # try to sort like "demo_0", "episode_12", etc.
    def keyfun(k):
        try:
            return int(k.split("_")[-1])
        except Exception:
            return k
    return sorted(eps, key=keyfun)


def tree_print(group, prefix="", max_depth=4, depth=0):
    """
    Print a tree of groups/datasets (like `h5dump -n` but with shapes).
    """
    if depth > max_depth:
        return
    items = list(group.items())
    items.sort(key=lambda kv: kv[0])

    for name, obj in items:
        path = f"{group.name}/{name}".replace("//", "/")
        if _is_group(obj):
            print(f"{prefix}📁 {name}/")
            tree_print(obj, prefix + "  ", max_depth=max_depth, depth=depth + 1)
        else:
            ds: h5py.Dataset = obj
            comp = ds.compression if ds.compression is not None else "none"
            chunks = ds.chunks if ds.chunks is not None else "none"
            print(
                f"{prefix}🧩 {name}  shape={ds.shape}  dtype={ds.dtype}  "
                f"compression={comp}  chunks={chunks}"
            )


def collect_episode_dataset_stats(h5, data_root="data", max_episodes=None):
    """
    Aggregate dataset paths across episodes:
      stats[path] = {
        "count": #episodes containing it
        "episodes": [...],
        "dtypes": set(),
        "shapes": set(),
        "T_list": [time lengths],
      }
    """
    eps = list_episode_keys(h5, data_root=data_root)
    if max_episodes is not None:
        eps = eps[:max_episodes]

    stats = defaultdict(lambda: {
        "count": 0,
        "episodes": [],
        "dtypes": set(),
        "shapes": set(),
        "T_list": [],
    })

    per_ep_paths = {}  # ep -> set(paths)
    for ep in eps:
        base = f"{data_root}/{ep}"
        if base not in h5:
            continue

        paths_here = set()

        def visit(g: h5py.Group, rel_prefix=""):
            for k, obj in g.items():
                rel = f"{rel_prefix}/{k}".lstrip("/")
                if _is_group(obj):
                    visit(obj, rel)
                else:
                    ds: h5py.Dataset = obj
                    full_rel_path = rel  # relative to episode root
                    paths_here.add(full_rel_path)

                    stats[full_rel_path]["count"] += 1
                    stats[full_rel_path]["episodes"].append(ep)
                    stats[full_rel_path]["dtypes"].add(_safe_str(ds.dtype))
                    stats[full_rel_path]["shapes"].add(_safe_str(ds.shape))

                    # If it looks time-indexed, treat dim0 as T
                    if isinstance(ds.shape, tuple) and len(ds.shape) >= 1 and ds.shape[0] is not None:
                        try:
                            stats[full_rel_path]["T_list"].append(int(ds.shape[0]))
                        except Exception:
                            pass

        visit(h5[base], rel_prefix="")
        per_ep_paths[ep] = paths_here

    return eps, stats, per_ep_paths


def print_schema_summary(eps, stats, per_ep_paths, show_missing=True, max_lines=None):
    """
    Print per-path coverage and time-length stats.
    """
    all_paths = sorted(stats.keys())
    if max_lines is not None:
        all_paths = all_paths[:max_lines]

    print("\n=== Aggregate schema across episodes ===")
    print(f"Episodes considered: {len(eps)}")
    print(f"Unique dataset paths (relative to each episode): {len(stats)}\n")

    for p in all_paths:
        s = stats[p]
        cover = f"{s['count']}/{len(eps)}"
        dtypes = ", ".join(sorted(s["dtypes"]))
        shapes = ", ".join(sorted(s["shapes"]))
        if s["T_list"]:
            tmin = min(s["T_list"])
            tmed = int(median(s["T_list"]))
            tmax = max(s["T_list"])
            tinfo = f"T(min/med/max)={tmin}/{tmed}/{tmax}"
        else:
            tinfo = "T=n/a"

        print(f"- {p}")
        print(f"    coverage: {cover} episodes")
        print(f"    dtype(s): {dtypes}")
        print(f"    shape(s): {shapes}")
        print(f"    {tinfo}")

    if show_missing:
        print("\n=== Missing-path report (only if not present in all episodes) ===")
        for p in sorted(stats.keys()):
            if stats[p]["count"] == len(eps):
                continue
            missing = [ep for ep in eps if p not in per_ep_paths.get(ep, set())]
            if missing:
                preview = missing[:10]
                more = "" if len(missing) <= 10 else f" ... (+{len(missing)-10} more)"
                print(f"- {p}: missing in {len(missing)} episodes -> {preview}{more}")


def cam_check(h5, eps, cams, data_root="data"):
    """
    Your original camera mismatch/missing logic, but kept as an optional mode.
    """
    print("\n=== Camera check (per-episode) ===")
    diffs = 0
    for ep in eps:
        base = f"{data_root}/{ep}/obs"
        if base not in h5:
            continue

        lengths = {}
        for cam in cams:
            rgbk = f"{cam}_image"
            depk = f"{cam}_depth"
            t_rgb = h5[base][rgbk].shape[0] if rgbk in h5[base] else None
            t_dep = h5[base][depk].shape[0] if depk in h5[base] else None
            T = t_rgb if t_rgb is not None else t_dep
            lengths[cam] = T

        have_any = any(v is not None for v in lengths.values())
        if not have_any:
            continue

        present = {c: v for c, v in lengths.items() if v is not None}
        if len(set(present.values())) > 1:
            diffs += 1
            print(f"[MISMATCH] {ep}: " + ", ".join(f"{c}={present[c]}" for c in present))

        missing = [c for c, v in lengths.items() if v is None]
        if missing:
            print(f"[MISSING ] {ep}: missing {missing}")

    print(f"\nEpisodes with length mismatches across cams: {diffs}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5_path", type=str)
    ap.add_argument("--data-root", type=str, default="data",
                    help="root group that contains episodes (default: data)")
    ap.add_argument("--max-episodes", type=int, default=None,
                    help="limit episodes for faster inspection")
    ap.add_argument("--tree-episode", type=str, default=None,
                    help="print a tree for a specific episode key (e.g., demo_0). "
                         "If omitted, prints a tree for the first episode.")
    ap.add_argument("--tree-depth", type=int, default=4,
                    help="max depth for tree printing")
    ap.add_argument("--no-schema", action="store_true",
                    help="skip aggregate schema summary")
    ap.add_argument("--no-missing-report", action="store_true",
                    help="do not print missing-path report")
    ap.add_argument("--max-schema-lines", type=int, default=None,
                    help="limit number of schema paths printed (debug)")
    ap.add_argument("--cam-check", action="store_true",
                    help="run the camera mismatch/missing report (like your original script)")
    ap.add_argument("--cams", type=str, nargs="*", default=DEFAULT_CAMS,
                    help="camera base names to check (default: common robomimic cams)")
    args = ap.parse_args()

    with h5py.File(args.h5_path, "r") as h5:
        print("=== File ===")
        print(args.h5_path)
        print("\n=== Top-level keys ===")
        for k in h5.keys():
            print(f"- {k}/" if _is_group(h5[k]) else f"- {k}")

        # root attrs
        if len(h5.attrs) > 0:
            print("\n=== Root attributes ===")
            for k, v in h5.attrs.items():
                print(f"- {k}: {_safe_str(v)}")

        eps = list_episode_keys(h5, data_root=args.data_root)
        print(f"\n=== Episodes under /{args.data_root} ===")
        print(f"episodes: {len(eps)}")
        if eps:
            print("first 10:", eps[:10])

        if not eps:
            return

        # Tree view
        ep_for_tree = args.tree_episode if args.tree_episode is not None else eps[0]
        ep_path = f"{args.data_root}/{ep_for_tree}"
        if ep_path in h5:
            print(f"\n=== Tree for episode: {ep_for_tree} (/{ep_path}) ===")
            tree_print(h5[ep_path], max_depth=args.tree_depth)
        else:
            print(f"\n[WARN] requested tree episode {ep_for_tree} not found under /{args.data_root}")

        # Aggregate schema
        eps_used, stats, per_ep_paths = collect_episode_dataset_stats(
            h5, data_root=args.data_root, max_episodes=args.max_episodes
        )
        if not args.no_schema:
            print_schema_summary(
                eps_used, stats, per_ep_paths,
                show_missing=(not args.no_missing_report),
                max_lines=args.max_schema_lines
            )

        # Optional cam check
        if args.cam_check:
            cam_check(h5, eps_used, args.cams, data_root=args.data_root)


if __name__ == "__main__":
    main()
