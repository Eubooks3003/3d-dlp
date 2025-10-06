#!/usr/bin/env python3
import h5py, numpy as np, sys

h5_path = sys.argv[1]
cams = ["agentview", "robot0_eye_in_hand", "birdview", "sideview"]

with h5py.File(h5_path, "r") as h5:
    eps = sorted(h5["data"].keys(), key=lambda k: int(k.split("_")[-1]))
    print(f"episodes: {len(eps)}")
    diffs = 0
    for ep in eps:
        base = f"data/{ep}/obs"
        lengths = {}
        for cam in cams:
            rgbk = f"{cam}_image"
            depk = f"{cam}_depth"
            t_rgb = h5[base][rgbk].shape[0] if rgbk in h5[base] else None
            t_dep = h5[base][depk].shape[0] if depk in h5[base] else None
            T = t_rgb if t_rgb is not None else t_dep
            lengths[cam] = T
        # show only if at least one cam has data
        have_any = any(v is not None for v in lengths.values())
        if not have_any: 
            continue
        # check mismatch within this episode
        present = {c:v for c,v in lengths.items() if v is not None}
        if len(set(present.values())) > 1:
            diffs += 1
            print(f"[MISMATCH] {ep}: " + ", ".join(f"{c}={present[c]}" for c in present))
        # missing cams
        missing = [c for c,v in lengths.items() if v is None]
        if missing:
            print(f"[MISSING ] {ep}: missing {missing}")
    print(f"episodes with length mismatches across cams: {diffs}")
