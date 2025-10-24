import wandb, numpy as np, torch, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- utils ----
def to_np(x):
    if x is None: return None
    if torch.is_tensor(x): x = x.detach().cpu().numpy()
    return np.asarray(x)

def clean_pts(pts, mask=None, nz_eps=1e-8):
    if pts is None: return None
    if mask is not None:
        pts = pts[mask.bool()]
    finite = torch.isfinite(pts).all(dim=-1)
    pts = pts[finite]
    nonzero = (pts.abs().max(dim=-1).values > nz_eps)
    pts = pts[nonzero]
    return pts

def subsample_np(pts_np, *others, max_n=50000):
    if pts_np is None: return (None,) + tuple(None for _ in others)
    N = pts_np.shape[0]
    if N <= max_n:
        return (pts_np,) + tuple(to_np(o) if o is not None else None for o in others)
    idx = np.random.choice(N, max_n, replace=False)
    outs = [pts_np[idx]]
    for o in others:
        if o is None: outs.append(None)
        else:
            o_np = to_np(o)
            outs.append(o_np[idx] if (o_np is not None and len(o_np) == N) else None)
    return tuple(outs)

def lims_from_all(pts_list):
    P = [p for p in pts_list if p is not None and p.shape[0] > 0]
    if not P: return None
    P = np.concatenate(P, axis=0)
    lo = np.percentile(P, 1, axis=0); hi = np.percentile(P, 99, axis=0)
    pad = 0.05 * (hi - lo + 1e-6)
    lo -= pad; hi += pad
    return ([lo[0], hi[0]], [lo[1], hi[1]], [lo[2], hi[2]])

def xy(ax, pts, title, lim=None, color=None, s=1):
    if pts is None or pts.shape[0] == 0:
        ax.axis("off"); ax.set_title(f"{title} (empty)"); return
    if color is None:
        ax.scatter(pts[:,0], pts[:,1], s=s)
    else:
        ax.scatter(pts[:,0], pts[:,1], s=s, c=color, marker='.')
    if lim is not None:
        ax.set_xlim(lim[0]); ax.set_ylim(lim[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)

def log_pc_wandb(name, pts, colors=None, ids=None, step=None, max_points=50000):
    """
    Logs a W&B Object3D:
      - xyz            -> Nx3
      - xyzrgb         -> Nx6 (colors in [0,1])
      - xyzc (category)-> Nx4 (int 1..14)
    """
    pts_np, colors_np, ids_np = subsample_np(to_np(pts), colors, ids, max_n=max_points)
    if pts_np is None: return
    if ids_np is not None:
        cats = (np.maximum(ids_np, 0) % 14) + 1  # 1..14
        arr = np.concatenate([pts_np, cats[:,None].astype(np.float32)], axis=1)  # [N,4]
    elif colors_np is not None:
        arr = np.concatenate([pts_np, colors_np.astype(np.float32)], axis=1)     # [N,6]
    else:
        arr = pts_np.astype(np.float32)                                          # [N,3]
    wandb.log({name: wandb.Object3D(arr)}, step=step)

def log_pc_xy_panel(name, gt_pts, rec_pts, kp_xyz=None, step=None, max_points=60000):
    # subsample for speed in images
    gt_pts, = subsample_np(to_np(gt_pts), max_n=max_points)
    rec_pts, = subsample_np(to_np(rec_pts), max_n=max_points)
    kp_np    = to_np(kp_xyz)
    lim = lims_from_all([p for p in [gt_pts, rec_pts, kp_np] if p is not None])

    fig, axs = plt.subplots(2, 2, figsize=(8, 7), dpi=120)
    xy(axs[0,0], gt_pts, "GT XY", lim)
    xy(axs[0,1], rec_pts, "REC XY", lim)

    xy(axs[1,0], gt_pts, "GT + Particles XY", lim)
    if kp_np is not None and kp_np.size > 0:
        axs[1,0].scatter(kp_np[:,0], kp_np[:,1], s=12, c="red", marker="x")

    xy(axs[1,1], rec_pts, "REC + Particles XY", lim)
    if kp_np is not None and kp_np.size > 0:
        axs[1,1].scatter(kp_np[:,0], kp_np[:,1], s=12, c="red", marker="x")

    plt.tight_layout()
    wandb.log({name: wandb.Image(fig)}, step=step)
    plt.close(fig)


import numpy as np, torch, wandb, plotly.graph_objects as go

def downsample(x, c=None, ids=None, n=80_000):
    if x is None: return (None, None, None)
    N = x.shape[0]
    if N <= n: return x, c, ids
    idx = np.random.choice(N, n, replace=False)
    return x[idx], (c[idx] if c is not None else None), (ids[idx] if ids is not None else None)


def log_pc_plotly(name, pts, colors=None, ids=None, kps=None, step=None,
                  point_size=2, opacity=0.9):
    P = to_np(pts); C = to_np(colors); I = to_np(ids) 
    K = _to_xyz2(kps)
    if P is None or P.size == 0: 
        return
    P, C, I = downsample(P, C, I)

    # Build color
    if C is not None and C.shape[-1] == 3:
        # Plotly expects 0-255 or css strings — convert to 0-255
        rgb255 = (np.clip(C, 0, 1) * 255).astype(np.uint8)
        clr = [f"rgb({r},{g},{b})" for r,g,b in rgb255]
    elif I is not None:
        # Category colors
        pal = np.array([
            "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
            "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
            "#aec7e8","#ffbb78","#98df8a","#ff9896"
        ])
        clr = pal[I % len(pal)]
    else:
        clr = "rgba(0,0,0,0.85)"

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=P[:,0], y=P[:,1], z=P[:,2],
        mode="markers",
        marker=dict(size=point_size, opacity=opacity, color=clr),
        name=name
    ))

    # Keypoints as larger red X markers (if provided)
    if K is not None and K.size:
        fig.add_trace(go.Scatter3d(
            x=K[:,0], y=K[:,1], z=K[:,2],
            mode="markers",
            marker=dict(size=2, symbol="x", color="red"),
            name="keypoints"
        ))

    # Equal-ish aspect box
    mins = np.nanpercentile(P, 1, axis=0)
    maxs = np.nanpercentile(P, 99, axis=0)
    fig.update_scenes(xaxis=dict(range=[mins[0], maxs[0]]),
                      yaxis=dict(range=[mins[1], maxs[1]]),
                      zaxis=dict(range=[mins[2], maxs[2]]),
                      aspectmode="data")

    fig.update_layout(margin=dict(l=0,r=0,t=30,b=0), showlegend=True,
                      scene_dragmode="turntable")  # nicer orbit

    wandb.log({name: fig}, step=step)

def _np(x):
    if x is None: return None
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

def _to_xyz2(x):
    """
    Coerce pts to shape [N,3].
    Accepts None, [3], [N,3], [*,3], [3,N], [*,*,3].
    Drops rows with non-finite values.
    """
    if x is None: return None

    a = _np(x)
    if a.ndim == 1:
        if a.size == 3:
            a = a[None, :]
        elif a.size % 3 == 0:
            a = a.reshape(-1, 3)
        else:
            return None
    elif a.ndim >= 2:
        if a.shape[-1] == 3:
            a = a.reshape(-1, 3)
        elif a.shape[0] == 3 and a.ndim == 2:
            a = a.T
            if a.shape[-1] != 3: return None
        else:
            return None
    else:
        return None
    m = np.isfinite(a).all(axis=1)
    a = a[m]
    return a if a.size else None

def _downsample_xyz(x, n=80_000):
    if x is None: return None
    N = x.shape[0]
    if N <= n: return x
    idx = np.random.choice(N, n, replace=False)
    return x[idx]

def log_pc_overlay_plotly(
    name,
    gt_pts, rec_pts,
    gt_colors=None, rec_colors=None, rec_ids=None, kps=None,
    step=None,
    max_points=80_000,
    point_size_gt=2, point_size_rec=2,
    opacity_gt=0.5, opacity_rec=0.9,
    color_mode="source"  # "source" | "rec_rgb" | "rec_ids"
):
    # ---- normalize to [N,3] ----
    G = _to_xyz2(gt_pts)
    R = _to_xyz2(rec_pts)
    K = _to_xyz2(kps)

    if G is None and R is None and K is None:
        return

    # ---- optional downsample ----
    G = _downsample_xyz(G, max_points)
    R = _downsample_xyz(R, max_points)

    # ---- colors for REC ----
    if color_mode == "rec_rgb" and rec_colors is not None:
        C = _np(rec_colors)
        # if batched, reshape alongside R to [N,3]
        if C is not None:
            if C.ndim >= 2 and C.shape[-1] == 3:
                C = C.reshape(-1, 3)
                if R is not None and C.shape[0] != R.shape[0]:
                    C = None
            else:
                C = None
    else:
        C = None

    if color_mode == "rec_ids" and rec_ids is not None:
        I = _np(rec_ids)
        if I is not None:
            I = I.reshape(-1)
            if R is not None and I.shape[0] != R.shape[0]:
                I = None
    else:
        I = None

    # ---- build Plotly figure ----
    GT_CLR = "rgba(31,119,180,1.0)"   # blue
    REC_CLR = "rgba(255,127,14,1.0)"  # orange

    if C is not None:
        rgb255 = (np.clip(C, 0, 1) * 255).astype(np.uint8)
        rec_color = [f"rgb({r},{g},{b})" for r,g,b in rgb255]
        gt_color = "rgba(180,180,180,0.8)"
    elif I is not None:
        pal = np.array([
            "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
            "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
            "#aec7e8","#ffbb78","#98df8a","#ff9896"
        ])
        rec_color = pal[I % len(pal)]
        gt_color = "rgba(180,180,180,0.8)"
    else:
        rec_color = REC_CLR
        gt_color  = GT_CLR

    fig = go.Figure()

    if G is not None and G.size:
        fig.add_trace(go.Scatter3d(
            x=G[:,0], y=G[:,1], z=G[:,2],
            mode="markers",
            marker=dict(size=point_size_gt, opacity=opacity_gt, color=gt_color),
            name="GT"
        ))

    if R is not None and R.size:
        npts = R.shape[0] if hasattr(R, "shape") else len(R)
        print(f"Plotting REC points: {npts} (array size={R.size})")
        fig.add_trace(go.Scatter3d(
            x=R[:,0], y=R[:,1], z=R[:,2],
            mode="markers",
            marker=dict(size=4, opacity=1.0, color=rec_color),  # bigger & opaque to see them
            name=f"REC (N={npts})"
        ))

    if K is not None and K.size:
        fig.add_trace(go.Scatter3d(
            x=K[:,0], y=K[:,1], z=K[:,2],
            mode="markers",
            marker=dict(size=2, symbol="x", color="red"),
            name="keypoints"
        ))

    # ---- common limits (robust) ----
    stacks = [x for x in (G, R, K) if x is not None and x.size]
    P = np.vstack(stacks) if stacks else None
    if P is not None and P.size:
        mins = np.nanpercentile(P, 1, axis=0)
        maxs = np.nanpercentile(P, 99, axis=0)
        fig.update_scenes(xaxis=dict(range=[mins[0], maxs[0]]),
                          yaxis=dict(range=[mins[1], maxs[1]]),
                          zaxis=dict(range=[mins[2], maxs[2]]),
                          aspectmode="data")

    fig.update_layout(margin=dict(l=0,r=0,t=30,b=0), showlegend=True,
                      scene_dragmode="turntable")
    wandb.log({name: fig}, step=step)

def select_topk_keypoints(model_output, topk, prefer_logvar=True, eps=1e-8):
    kp = model_output.get('kp_p', None)   # [B,Kp,C]
    assert kp is not None, "model_output['kp_p'] must be present"
    B, Kp, C = kp.shape
    device = kp.device

    # ----- score source -----
    score = None
    if prefer_logvar and ('z_base_var' in model_output):
        z_var = model_output['z_base_var']
        # collapse trailing dims, keep [B, ?]
        if z_var.dim() > 2:
            while z_var.dim() > 2:
                z_var = z_var.sum(-1)
        # If shape is [B, Kv] and Kv != Kp, we can't align -> skip
        if z_var.shape[0] == B and z_var.shape[1] == Kp:
            score = -z_var  # smaller var = better
        else:
            # sizes don’t match; fall back later
            score = None

    # fallback to covariance trace if present
    if score is None and ('kp_cov' in model_output):
        cov = model_output['kp_cov']  # [B,Kp,3,3]
        tr = cov[..., 0,0] + cov[..., 1,1] + cov[..., 2,2]
        score = -tr

    if score is None:
        score = torch.zeros(B, Kp, device=device)

    # ----- optional obj_on gating (only if sizes match) -----
    obj_on = model_output.get('obj_on', None)
    if obj_on is not None:
        o = obj_on
        if o.dim() == 3 and o.size(-1) == 1:
            o = o.squeeze(-1)
        if o.shape == (B, Kp):         # only apply when it matches Kp
            score = score * o.clamp(0, 1)

    # ----- top-k -----
    k_eff = min(int(topk), Kp)
    scores_topk, idx = torch.topk(score, k=k_eff, dim=-1, largest=True, sorted=True)
    b = torch.arange(B, device=device)[:, None]
    kp_topk = kp[b, idx]

    # pack for logging
    model_output['kp_scores'] = score
    model_output['kp_topk_idx'] = idx
    model_output['kp_topk'] = kp_topk
    model_output['kp_scores_topk'] = scores_topk
    return idx, kp_topk, scores_topk


def _as_cpu_np(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu()
        return x
    return torch.tensor(x, dtype=torch.float32)

import numpy as np

def summarize_points(pts: torch.Tensor, name="eff_pts", norm_bounds=(-1, 1)):
    if pts is None or pts.numel() == 0:
        print(f"[{name}] EMPTY")
        return

    A = pts.detach().float().cpu().numpy()   # [N,3]
    finite_rows = np.isfinite(A).all(axis=1)
    A = A[finite_rows]

    if A.size == 0:
        print(f"[{name}] no finite rows")
        return

    N = A.shape[0]
    pmin = np.nanmin(A, axis=0)
    pmax = np.nanmax(A, axis=0)
    pmean = np.nanmean(A, axis=0)
    pstd  = np.nanstd(A, axis=0)
    q01, q25, q50, q75, q99 = np.nanpercentile(A, [1, 25, 50, 75, 99], axis=0)

    lo, hi = norm_bounds
    outside = np.mean((A < lo).any(axis=1) | (A > hi).any(axis=1))
    extent = (pmax - pmin)
    bbox_vol = float(np.prod(extent))

    print(f"[{name}] N={N} (finite rows)")
    print(f"[{name}] min      : {pmin}")
    print(f"[{name}] max      : {pmax}")
    print(f"[{name}] mean     : {pmean}")
    print(f"[{name}] std      : {pstd}")
    print(f"[{name}] q01/q25  : {q01} / {q25}")
    print(f"[{name}] q50/q75  : {q50} / {q75}")
    print(f"[{name}] q99      : {q99}")
    print(f"[{name}] extent   : {extent}  bbox_vol≈{bbox_vol:.4f}")
    print(f"[{name}] frac outside {norm_bounds}: {outside:.3f}")

def extract_effective_points_from_pointweights(model_output, b=0, thresh=0.5, include_bg=False):
    pts_scene = model_output["points_scene"]      # [B, Mtot, 3]
    w_points  = model_output["point_weights"]     # [B, K*M, 1]  (obj-only)

    B, Mtot, _ = pts_scene.shape
    KM = w_points.shape[1]
    Mbg = Mtot - KM

    obj_pts_scene = pts_scene[b, -KM:, :]                  # [KM,3]
    keep = (w_points[b].squeeze(-1).detach() > thresh)     # [KM]

    if include_bg:
        bg_pts = pts_scene[b, :Mbg, :]
        eff_pts = torch.cat([bg_pts, obj_pts_scene[keep]], dim=0)
    else:
        eff_pts = obj_pts_scene[keep]

    # finite-only for summaries
    mask_finite = torch.isfinite(eff_pts).all(dim=1)
    eff_f = eff_pts[mask_finite]

    print(f"[viz] b={b}: Mbg={Mbg}, KM={KM}, kept={keep.sum().item()}, "
          f"finite={mask_finite.sum().item()}, out={tuple(eff_pts.shape)}")

    return eff_pts


def _nn_dists(a: torch.Tensor, k=4, max_sample=5000):
    """
    Return per-point distances to 1st..kth nearest neighbors (excluding self).
    Subsample if too many points to keep it cheap.
    """
    if a.numel() == 0:
        return None
    x = a
    if x.shape[0] > max_sample:
        idx = torch.randperm(x.shape[0], device=x.device)[:max_sample]
        x = x[idx]
    with torch.no_grad():
        # [m,m]
        D = torch.cdist(x, x, p=2)
        D[torch.arange(D.shape[0], device=D.device), torch.arange(D.shape[0], device=D.device)] = float('inf')
        # sort along neighbors
        vals, _ = torch.topk(D, k=k, largest=False, dim=1)
    return vals  # [m,k]

def _grid_occupancy(a: torch.Tensor, bounds=(-1,1), bins=20):
    """
    Count how many voxels are occupied by at least 1 point.
    """
    if a.numel() == 0:
        return 0, 0
    lo, hi = bounds
    x = a.clamp(lo, hi)  # keep inside for indexing
    # map to [0,bins)
    g = ((x - lo) / (hi - lo) * bins).floor().clamp_min(0).clamp_max(bins-1).long()  # [N,3]
    # unique voxels
    uniq = torch.unique(g, dim=0)
    return int(uniq.shape[0]), int(bins**3)

def summarize_points_plus(pts: torch.Tensor, name="eff_pts", norm_bounds=(-1,1), bins=20):
    # your original summary:
    summarize_points(pts, name=name, norm_bounds=norm_bounds)

    # extra: NN distances & occupancy
    if pts is None or pts.numel() == 0:
        return
    mask_f = torch.isfinite(pts).all(dim=1)
    X = pts[mask_f]
    if X.numel() == 0:
        print(f"[{name}] no finite rows for extras")
        return

    # kNN distances
    knn = _nn_dists(X, k=4)
    if knn is not None:
        d1 = knn[:,0].cpu().numpy()
        d2 = knn[:,1].cpu().numpy()
        print(f"[{name}/kNN] N={len(d1)} | d1: mean={d1.mean():.4f}, med={np.median(d1):.4f}, "
              f"p1={np.percentile(d1,1):.4f}, p99={np.percentile(d1,99):.4f}")
        print(f"[{name}/kNN] d2: mean={d2.mean():.4f}, med={np.median(d2):.4f}")

        # crude “are they piled up?” signal
        tiny = (d1 < 1e-2).mean()
        small = (d1 < 5e-2).mean()
        print(f"[{name}/kNN] frac d1<1e-2: {tiny:.3f}, d1<5e-2: {small:.3f}")

    # occupancy
    occ, total = _grid_occupancy(X, bounds=norm_bounds, bins=bins)
    print(f"[{name}/occ] occupied voxels: {occ}/{total} "
          f"({occ/total:.3f}) at {bins}^3 grid")
