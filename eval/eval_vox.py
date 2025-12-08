# eval_vox.py — NumPy-safe version for PyTorch builds "without NumPy support"

import numpy as np
import torch
import wandb
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

# Try to import figure_factory (for ff.create_voxels); fall back if unavailable
try:
    import plotly.figure_factory as ff
    _HAS_FF = True
except Exception:
    _HAS_FF = False


# ----------------------------------------------------------------------
# Universal tensor → NumPy converter
# ----------------------------------------------------------------------

def _tensor_to_np(x):
    """
    Convert a torch.Tensor to a NumPy array without relying on the
    PyTorch↔NumPy C-API (which is missing in your build).
    Falls back to t.tolist() → np.array.
    """
    if not isinstance(x, torch.Tensor):
        return np.asarray(x)

    t = x.detach().cpu().contiguous()
    try:
        # This will fail on your build with "compiled without NumPy support"
        return t.numpy()
    except RuntimeError as e:
        if "compiled without NumPy support" in str(e):
            flat = t.reshape(-1).tolist()
            return np.array(flat, dtype=np.float32).reshape(*t.shape)
        raise


def _np(x):
    """Convert x (tensor or array-like) to NumPy using the safe bridge."""
    if isinstance(x, torch.Tensor):
        return _tensor_to_np(x)
    return np.asarray(x)


# ----------------------------------------------------------------------
# Basic helpers
# ----------------------------------------------------------------------

def _as_DHW(vol):
    """
    Accept [D,H,W], [1,D,H,W], or torch versions; return np [D,H,W].
    """
    if vol is None:
        return None
    V = _np(vol)
    if V.ndim == 4 and V.shape[0] == 1:  # [1,D,H,W]
        V = V[0]
    if V.ndim != 3:
        raise ValueError(f"Expected 3D volume [D,H,W] or [1,D,H,W], got {V.shape}")
    return V


def _as_CDHW(vol):
    """
    Accept [3,D,H,W] or [1,3,D,H,W]; return np [3,D,H,W].
    """
    if vol is None:
        return None
    V = _np(vol)
    if V.ndim == 5:  # [B,C,D,H,W]
        V = V[0]
    if V.ndim != 4 or V.shape[0] not in (1, 3):
        raise ValueError(f"Expected [C,D,H,W] with C in {{1,3}}, got {V.shape}")
    if V.shape[0] == 1:
        # Single-channel: treat as grayscale → 3 channels
        V = np.repeat(V, 3, axis=0)
    return V


def _kp_norm_to_index(kps, D, H, W, order=("z", "y", "x")):
    """
    kps: [K,3] in [-1,1] (torch or np), components ordered by `order`.
    Returns [K,3] voxel indices in (x,y,z) for Plotly.
    """
    k = _np(kps)
    if k.ndim == 1:
        k = k[None, :]
    size = {"x": W, "y": H, "z": D}

    def to_idx(v, n):
        return np.clip(0.5 * (v + 1.0) * (n - 1), 0, n - 1)

    x = to_idx(k[:, order.index("x")], size["x"])
    y = to_idx(k[:, order.index("y")], size["y"])
    z = to_idx(k[:, order.index("z")], size["z"])
    return np.stack([x, y, z], axis=-1)


def _robust_nonempty_box(mask_xyz):
    """
    mask_xyz: bool [X,Y,Z]
    Returns ((xmin,xmax),(ymin,ymax),(zmin,zmax)) padded by 1 voxel.
    """
    nz = np.argwhere(mask_xyz)
    if nz.size == 0:
        return (0, 1), (0, 1), (0, 1)
    mins = np.maximum(nz.min(0) - 1, 0)
    maxs = np.minimum(nz.max(0) + 1, np.array(mask_xyz.shape) - 1)
    return (int(mins[0]), int(maxs[0])), (int(mins[1]), int(maxs[1])), (int(mins[2]), int(maxs[2]))


def _mesh_from_binary(mask_xyz, color="rgba(255,127,14,1.0)", opacity=0.9, max_voxels=120000):
    """
    Minimal voxel→Mesh3d when ff.create_voxels isn't available.
    mask_xyz: bool [X,Y,Z]
    """
    X, Y, Z = mask_xyz.shape
    idx = np.argwhere(mask_xyz)
    if idx.shape[0] == 0:
        return []

    # Downsample if too many voxels
    if idx.shape[0] > max_voxels:
        sel = np.random.choice(idx.shape[0], max_voxels, replace=False)
        idx = idx[sel]

    # Cube template
    verts_template = np.array(
        [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ],
        dtype=float,
    )
    faces_template = np.array(
        [
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3],
            [3, 7, 4], [3, 4, 0],
        ],
        dtype=int,
    )

    # Keep only boundary cubes
    occ = mask_xyz
    keep = []
    for (x, y, z) in idx:
        if (
            (x == 0 or not occ[x - 1, y, z]) or (x == X - 1 or not occ[x + 1, y, z]) or
            (y == 0 or not occ[x, y - 1, z]) or (y == Y - 1 or not occ[x, y + 1, z]) or
            (z == 0 or not occ[x, y, z - 1]) or (z == Z - 1 or not occ[x, y, z + 1])
        ):
            keep.append((x, y, z))
    if not keep:
        return []

    V = []
    I = []
    base = 0
    for (x, y, z) in keep:
        V.append(verts_template + np.array([x, y, z], dtype=float))
        I.append(faces_template + base)
        base += 8
    V = np.vstack(V)
    I = np.vstack(I)
    mesh = go.Mesh3d(
        x=V[:, 0], y=V[:, 1], z=V[:, 2],
        i=I[:, 0], j=I[:, 1], k=I[:, 2],
        color=color, opacity=opacity, flatshading=True, name="voxels", showscale=False,
    )
    return [mesh]


# ----------------------------------------------------------------------
# log_vox_overlay_plotly & log_vox_isoseries
# ----------------------------------------------------------------------

def log_vox_overlay_plotly(
    name,
    gt_vol,
    rec_vol,
    kps=None,
    step=None,
    iso_levels=(0.2,),
    gt_color="rgba(31,119,180,1.0)",
    rec_color="rgba(255,127,14,1.0)",
    kp_color="#ff0000",
    kp_order=("x", "y", "z"),
    point_size_kp=6,
):
    G = _as_DHW(gt_vol)
    R = _as_DHW(rec_vol)
    fig = go.Figure()
    thr = float(np.min(iso_levels))

    # REC
    if R is not None:
        maskR = np.transpose(R, (2, 1, 0)) > thr  # (X,Y,Z)
        if maskR.any():
            if _HAS_FF:
                vox = ff.create_voxels(maskR, colorscale=[[0, rec_color], [1, rec_color]], opacity=1.0)
                for tr in vox.data:
                    tr.name = "REC"
                    fig.add_trace(tr)
            else:
                for tr in _mesh_from_binary(maskR, rec_color, 0.95):
                    tr.name = "REC"
                    fig.add_trace(tr)

    # GT
    if G is not None:
        maskG = np.transpose(G, (2, 1, 0)) > thr
        if maskG.any():
            if _HAS_FF:
                vox = ff.create_voxels(maskG, colorscale=[[0, gt_color], [1, gt_color]], opacity=0.65)
                for tr in vox.data:
                    tr.name = "GT"
                    fig.add_trace(tr)
            else:
                for tr in _mesh_from_binary(maskG, gt_color, 0.65):
                    tr.name = "GT"
                    fig.add_trace(tr)

    # Keypoints
    if kps is not None and (R is not None or G is not None):
        D, H, W = (R.shape if R is not None else G.shape)
        K = _kp_norm_to_index(kps, D, H, W, order=kp_order)
        fig.add_trace(
            go.Scatter3d(
                x=K[:, 0],
                y=K[:, 1],
                z=K[:, 2],
                mode="markers",
                marker=dict(symbol="x", size=point_size_kp, color=kp_color, line=dict(width=3, color="black")),
                name="keypoints",
            )
        )

    # Axes
    if R is not None and (R > thr).any():
        xr, yr, zr = _robust_nonempty_box(np.transpose(R, (2, 1, 0)) > thr)
    elif G is not None and (G > thr).any():
        xr, yr, zr = _robust_nonempty_box(np.transpose(G, (2, 1, 0)) > thr)
    else:
        xr, yr, zr = (0, 1), (0, 1), (0, 1)

    fig.update_scenes(
        xaxis=dict(range=[xr[0], xr[1]]),
        yaxis=dict(range=[yr[0], yr[1]]),
        zaxis=dict(range=[zr[0], zr[1]]),
        aspectmode="data",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), showlegend=True, scene_dragmode="turntable")
    wandb.log({name: fig}, step=step)


def log_vox_isoseries(
    name,
    vol,
    kps=None,
    iso_levels=(0.05, 0.1, 0.2, 0.3, 0.4),
    step=None,
    color="rgba(255,127,14,1.0)",
    kp_color="#ff0000",
    kp_order=("z", "y", "x"),
    point_size_kp=6,
):
    V = _as_DHW(vol)
    if V is None:
        return
    fig = go.Figure()
    for i, iso in enumerate(sorted(iso_levels)):
        mask = np.transpose(V, (2, 1, 0)) > float(iso)
        if not mask.any():
            continue
        alpha = float(np.clip(0.35 + 0.35 * (i / max(1, len(iso_levels) - 1)), 0.35, 0.95))
        if _HAS_FF:
            vox = ff.create_voxels(mask, colorscale=[[0, color], [1, color]], opacity=alpha)
            for tr in vox.data:
                tr.name = f"iso≥{iso:.2f}"
                fig.add_trace(tr)
        else:
            for tr in _mesh_from_binary(mask, color, alpha):
                tr.name = f"iso≥{iso:.2f}"
                fig.add_trace(tr)

    if kps is not None:
        D, H, W = V.shape
        K = _kp_norm_to_index(kps, D, H, W, order=kp_order)
        fig.add_trace(
            go.Scatter3d(
                x=K[:, 0],
                y=K[:, 1],
                z=K[:, 2],
                mode="markers",
                marker=dict(symbol="x", size=point_size_kp, color=kp_color, line=dict(width=3, color="black")),
                name="keypoints",
            )
        )

    mask_base = np.transpose(V, (2, 1, 0)) > float(min(iso_levels) if iso_levels else 0.5)
    xr, yr, zr = _robust_nonempty_box(mask_base)
    fig.update_scenes(
        xaxis=dict(range=[xr[0], xr[1]]),
        yaxis=dict(range=[yr[0], yr[1]]),
        zaxis=dict(range=[zr[0], zr[1]]),
        aspectmode="data",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), showlegend=True, scene_dragmode="turntable")
    wandb.log({name: fig}, step=step)


# ----------------------------------------------------------------------
# ELBO helpers & volume handling
# ----------------------------------------------------------------------

@torch.no_grad()
def select_kp_topk(cov_kp, post_logvar, n_keep, *, obj_on=None, warmup=False, warmup_ratio=1.0,
                   alpha=1.0, eps=1e-6):
    B, N = cov_kp.shape[:2]

    prior_var = torch.diagonal(cov_kp, -2, -1).clamp_min(eps)
    prior_trace = prior_var.sum(-1)

    post_var = torch.exp(post_logvar).clamp_min(eps)
    post_trace = post_var.sum(-1)

    score = prior_trace + alpha * post_trace

    if obj_on is not None:
        obj_on = obj_on.squeeze(-1) if obj_on.dim() == 3 else obj_on
        score = score * (1.0 / obj_on.clamp_min(1e-3))

    score = torch.where(torch.isfinite(score), score, torch.full_like(score, 1e9))
    score = score + 1e-6 * torch.randn_like(score)

    K = n_keep if not warmup else min(n_keep, max(1, int(warmup_ratio * N)))
    K = min(K, N)
    _, embed_ind = torch.topk(score, k=K, dim=-1, largest=False)
    return embed_ind, score


def gather_by_ind(x, embed_ind):
    take_shape = list(embed_ind.shape) + [1] * (x.dim() - embed_ind.dim())
    ind_exp = embed_ind.view(*take_shape).expand(-1, -1, *x.shape[2:])
    return torch.take_along_dim(x, ind_exp, dim=1)


def _as_b0_channel(vol, occ_channel=0):
    """
    Accept [B,C,D,H,W] or [B,D,H,W] or [D,H,W]; return [D,H,W] for b0.
    """
    if vol is None:
        return None
    if vol.dim() == 5:       # [B,C,D,H,W]
        vol = vol[0, occ_channel]
    elif vol.dim() == 4:     # [B,D,H,W]
        vol = vol[0]
    elif vol.dim() == 3:     # [D,H,W]
        pass
    else:
        raise ValueError(f"Unexpected vol shape {tuple(vol.shape)}")
    return vol


def extract_volumes_for_vis(model_output, *, occ_channel=0):
    if "rec" not in model_output:
        raise KeyError("model_output['rec'] is required.")
    if "x" not in model_output:
        rec_logits = _as_b0_channel(model_output["rec"], occ_channel=occ_channel)
        rec = torch.sigmoid(rec_logits)
        return None, rec

    rec_logits = _as_b0_channel(model_output["rec"], occ_channel=occ_channel)
    rec = torch.sigmoid(rec_logits)
    gt = _as_b0_channel(model_output["x"], occ_channel=occ_channel)
    return gt, rec


def print_vol_stats(tag, V):
    if V is None:
        print(f"{tag}: None")
        return
    if isinstance(V, torch.Tensor):
        v = _tensor_to_np(V.float())
    else:
        v = np.asarray(V)
    f_ok = np.isfinite(v).all()
    print(
        f"{tag}: shape={v.shape}, min={v.min():.4g}, max={v.max():.4g}, "
        f"mean={v.mean():.4g}, finite={f_ok}"
    )


# ----------------------------------------------------------------------
# log_voxel_rec_distributions
# ----------------------------------------------------------------------

def _flatten01(t, occ_channel=0):
    """
    Accepts [B,T,C,D,H,W], [B,C,D,H,W], [B,D,H,W], [D,H,W]; returns flattened CPU tensor.
    """
    if t is None:
        return None
    if t.dim() == 6:  # [B,T,C,D,H,W] or [B,T,D,H,W]
        if t.size(2) > 1:
            t = t[:, :, occ_channel]
        else:
            t = t[:, :, 0]
        t = t.reshape(-1, *t.shape[-3:])
    elif t.dim() == 5:  # [B,C,D,H,W] or [B,D,H,W]
        if t.size(1) > 1:
            t = t[:, occ_channel]
        else:
            t = t[:, 0]
        t = t.reshape(-1, *t.shape[-3:])
    elif t.dim() == 4:  # [B,D,H,W] or [D,H,W]
        if t.shape[0] != t.shape[-1]:  # assume [B,D,H,W]
            pass
    elif t.dim() == 3:  # [D,H,W]
        t = t.unsqueeze(0)
    else:
        raise ValueError(f"Unexpected shape {tuple(t.shape)}")
    return t.contiguous().view(-1).detach().cpu()


@torch.no_grad()
def log_voxel_rec_distributions(model_output, x, *, occ_channel=0, name_prefix="dist", step=None):
    if "rec" not in model_output:
        raise KeyError("model_output['rec'] (logits) required")

    rec_logits = _flatten01(model_output["rec"], occ_channel=occ_channel)
    gt_flat = _flatten01(x, occ_channel=occ_channel).float()

    rec_probs = torch.sigmoid(rec_logits).detach().cpu()

    rec_logits_np = _tensor_to_np(rec_logits)
    rec_probs_np = _tensor_to_np(rec_probs)
    gt_np = _tensor_to_np(gt_flat)

    # Stats
    def _stats(arr):
        return dict(
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
        )

    s_logits = _stats(rec_logits_np)
    s_probs = _stats(rec_probs_np)
    s_gt = dict(pos_frac=float(gt_np.mean()), count=int(gt_np.size))

    thr_grid = np.linspace(0.05, 0.95, 19)
    pred_pos_fracs = (rec_probs_np[:, None] > thr_grid[None, :]).mean(axis=0)

    eps = 1e-8
    prec, rec, iou = [], [], []
    for th in thr_grid:
        pred = rec_probs_np > th
        tp = float(np.sum(pred & (gt_np > 0.5)))
        fp = float(np.sum(pred & (gt_np < 0.5)))
        fn = float(np.sum((~pred) & (gt_np > 0.5)))
        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)
        u = tp / (tp + fp + fn + eps)
        prec.append(p)
        rec.append(r)
        iou.append(u)

    log_dict = {
        f"{name_prefix}/rec_logits_hist": wandb.Histogram(rec_logits_np),
        f"{name_prefix}/rec_probs_hist": wandb.Histogram(rec_probs_np),
        f"{name_prefix}/gt_hist": wandb.Histogram(gt_np),
        f"{name_prefix}/stats/logits_min": s_logits["min"],
        f"{name_prefix}/stats/logits_max": s_logits["max"],
        f"{name_prefix}/stats/logits_mean": s_logits["mean"],
        f"{name_prefix}/stats/probs_mean": s_probs["mean"],
        f"{name_prefix}/stats/probs_std": s_probs["std"],
        f"{name_prefix}/gt/pos_frac": s_gt["pos_frac"],
        f"{name_prefix}/gt/count": s_gt["count"],
    }

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=rec_probs_np, nbinsx=50, name="pred_probs", opacity=0.7))
    fig_hist.add_trace(go.Histogram(x=gt_np, nbinsx=2, name="gt_occ(0/1)", opacity=0.6))
    fig_hist.update_layout(barmode="overlay", title="Pred prob vs GT occupancy")
    log_dict[f"{name_prefix}/overlay_hist"] = fig_hist

    fig_sweep = go.Figure()
    fig_sweep.add_trace(go.Scatter(x=thr_grid, y=prec, mode="lines+markers", name="precision"))
    fig_sweep.add_trace(go.Scatter(x=thr_grid, y=rec, mode="lines+markers", name="recall"))
    fig_sweep.add_trace(go.Scatter(x=thr_grid, y=iou, mode="lines+markers", name="IoU"))
    fig_sweep.add_trace(
        go.Scatter(x=thr_grid, y=pred_pos_fracs, mode="lines+markers", name="pred_pos_frac", yaxis="y2")
    )
    fig_sweep.update_layout(
        title="Threshold sweep (probs→mask)",
        xaxis_title="threshold",
        yaxis=dict(title="PR/IoU"),
        yaxis2=dict(title="pred_pos_frac", overlaying="y", side="right", rangemode="tozero"),
        legend=dict(orientation="h"),
    )
    log_dict[f"{name_prefix}/threshold_sweep"] = fig_sweep

    best_i = int(np.argmax(iou))
    log_dict.update(
        {
            f"{name_prefix}/best_iou": float(iou[best_i]),
            f"{name_prefix}/best_iou_thr": float(thr_grid[best_i]),
            f"{name_prefix}/pred_pos@0.5": float(pred_pos_fracs[np.argmin(np.abs(thr_grid - 0.5))]),
        }
    )

    wandb.log(log_dict, step=step)


# ----------------------------------------------------------------------
# Covariance ellipsoids
# ----------------------------------------------------------------------

def _ellipsoid_mesh(mean_xyz, cov3, scale=1.0, nu=18, nv=18):
    w, R = np.linalg.eigh(cov3)
    w = np.clip(w, 1e-12, None)
    radii = scale * np.sqrt(w)

    u = np.linspace(0, 2 * np.pi, nu, endpoint=True)
    v = np.linspace(0, np.pi, nv, endpoint=True)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    X = radii[0] * np.cos(uu) * np.sin(vv)
    Y = radii[1] * np.sin(uu) * np.sin(vv)
    Z = radii[2] * np.cos(vv)
    P = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    P = (P @ R.T) + mean_xyz[None, :]

    def idx(i, j):
        return (i % nu) + j * nu

    faces = []
    for j in range(nv - 1):
        for i in range(nu):
            i0 = idx(i, j)
            i1 = idx(i + 1, j)
            i2 = idx(i + 1, j + 1)
            i3 = idx(i, j + 1)
            faces.append([i0, i1, i2])
            faces.append([i0, i2, i3])
    faces = np.asarray(faces, dtype=int)
    return P, faces


@torch.no_grad()
def log_cov_ellipsoids_over_voxels(
    name,
    gt_vol,
    kp_norm,
    cov_kp,
    *,
    step=None,
    kp_order=("x", "y", "z"),
    iso=0.5,
    ellip_scale=2.0,
    max_ellipsoids=128,
    color_scale="Viridis",
    show_gt=True,
):
    V = _as_DHW(gt_vol)
    if V is None:
        raise ValueError("gt_vol required for context.")
    if kp_norm is None or cov_kp is None:
        raise ValueError("kp_norm [B,N,3] and cov_kp [B,N,3,3] are required.")

    D, H, W = V.shape
    fig = go.Figure()

    if show_gt:
        maskG = np.transpose((V > float(iso)), (2, 1, 0))
        if maskG.any():
            if _HAS_FF:
                vox = ff.create_voxels(
                    maskG,
                    colorscale=[[0, "rgba(31,119,180,1.0)"], [1, "rgba(31,119,180,1.0)"]],
                    opacity=0.5,
                )
                for tr in vox.data:
                    tr.name = "GT"
                    fig.add_trace(tr)
            else:
                for tr in _mesh_from_binary(maskG, "rgba(31,119,180,1.0)", 0.5):
                    tr.name = "GT"
                    fig.add_trace(tr)

    kp0 = kp_norm[0]
    cov0 = cov_kp[0]
    KPx = _kp_norm_to_index(kp0, D, H, W, order=kp_order)
    COV = _np(cov0)
    N = int(KPx.shape[0])

    traces = np.trace(COV, axis1=-2, axis2=-1).reshape(-1)
    order = np.argsort(traces)
    if N > max_ellipsoids:
        order = order[:max_ellipsoids]
    t_sel = traces[order]
    if t_sel.size == 0:
        fig.add_annotation(text="No valid ellipsoids to plot", showarrow=False)
        wandb.log({name: fig}, step=step)
        return

    t_min = float(np.min(t_sel))
    t_max = float(np.max(t_sel))

    def _color_for(val):
        if t_max == t_min:
            frac = 0.5
        else:
            frac = float((val - t_min) / (t_max - t_min))
        rgba = sample_colorscale(color_scale, [frac])[0]
        if rgba.startswith("rgb("):
            r, g, b = [int(x) for x in rgba[4:-1].split(",")]
            return f"rgba({r},{g},{b},0.85)"
        if rgba.startswith("rgba("):
            comps = rgba[5:-1].split(",")
            comps[-1] = "0.85"
            return "rgba(" + ",".join(comps) + ")"
        return rgba

    ellip_points = []
    for idx in order:
        mu_xyz = KPx[idx]
        Sigma = COV[idx]
        if not np.isfinite(Sigma).all():
            continue
        try:
            P, F = _ellipsoid_mesh(mu_xyz, Sigma, scale=ellip_scale)
        except Exception:
            continue
        col = _color_for(traces[idx])
        fig.add_trace(
            go.Mesh3d(
                x=P[:, 0],
                y=P[:, 1],
                z=P[:, 2],
                i=F[:, 0],
                j=F[:, 1],
                k=F[:, 2],
                opacity=0.85,
                color=col,
                name=f"σ ellipsoid (tr={traces[idx]:.3g})",
                showscale=False,
                flatshading=True,
            )
        )
        ellip_points.append(P)

    fig.add_trace(
        go.Scatter3d(
            x=KPx[:, 0],
            y=KPx[:, 1],
            z=KPx[:, 2],
            mode="markers",
            marker=dict(size=3, color="black"),
            name="KP centers",
        )
    )

    pad = 2.0
    mins = []
    maxs = []

    if show_gt:
        mask_base = np.transpose((V > float(iso)), (2, 1, 0))
        if mask_base.any():
            xr, yr, zr = _robust_nonempty_box(mask_base)
            mins.append([xr[0], yr[0], zr[0]])
            maxs.append([xr[1], yr[1], zr[1]])

    if KPx is not None and KPx.size > 0:
        mins.append([float(KPx[:, 0].min()), float(KPx[:, 1].min()), float(KPx[:, 2].min())])
        maxs.append([float(KPx[:, 0].max()), float(KPx[:, 1].max()), float(KPx[:, 2].max())])

    if ellip_points:
        P_all = np.concatenate(ellip_points, axis=0)
        mins.append(P_all.min(axis=0).tolist())
        maxs.append(P_all.max(axis=0).tolist())

    if not mins:
        mins.append([0.0, 0.0, 0.0])
        maxs.append([float(W - 1), float(H - 1), float(D - 1)])

    mins = np.array(mins).min(axis=0) - pad
    maxs = np.array(maxs).max(axis=0) + pad

    fig.update_scenes(
        xaxis=dict(range=[mins[0], maxs[0]]),
        yaxis=dict(range=[mins[1], maxs[1]]),
        zaxis=dict(range=[mins[2], maxs[2]]),
        aspectmode="data",
    )

    fig.update_layout(
        title=f"Covariance ellipsoids over GT (iso≥{iso}) — color by trace(Σ)",
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=True,
        legend=dict(orientation="h"),
    )
    wandb.log({name: fig}, step=step)


# ----------------------------------------------------------------------
# RGB voxel visualizer (your main one)
# ----------------------------------------------------------------------

def _global_to_voxel_indices(kp_xyz_global, D, H, W):
    g = _np(kp_xyz_global)
    x = ((g[:, 0] + 1.0) * 0.5) * (W - 1)
    y = ((g[:, 1] + 1.0) * 0.5) * (H - 1)
    z = ((g[:, 2] + 1.0) * 0.5) * (D - 1)
    return np.stack([x, y, z], axis=1)


def _draw_kp_crosses(fig, KPx, D, H, W, *, space="voxel", half=2.0,
                     color="rgba(255,0,0,0.95)", width=4):
    KPx = _np(KPx)
    if KPx.ndim == 3:
        KPx = KPx[0]
    assert KPx.ndim == 2 and KPx.shape[1] == 3, f"KPx shape must be [K,3], got {KPx.shape}"

    if space == "global":
        KPx = _global_to_voxel_indices(KPx, D, H, W)

    KPx[:, 0] = np.clip(KPx[:, 0], 0, W - 1)
    KPx[:, 1] = np.clip(KPx[:, 1], 0, H - 1)
    KPx[:, 2] = np.clip(KPx[:, 2], 0, D - 1)

    for i, (x0, y0, z0) in enumerate(KPx.astype(float)):
        fig.add_trace(
            go.Scatter3d(
                x=[x0 - half, x0 + half],
                y=[y0, y0],
                z=[z0, z0],
                mode="lines",
                line=dict(color=color, width=width),
                showlegend=False,
                hoverinfo="text",
                text=[f"kp {i}", f"kp {i}"],
                name=f"kp {i}",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[x0, x0],
                y=[y0 - half, y0 + half],
                z=[z0, z0],
                mode="lines",
                line=dict(color=color, width=width),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[x0, x0],
                y=[y0, y0],
                z=[z0 - half, z0 + half],
                mode="lines",
                line=dict(color=color, width=width),
                showlegend=False,
                hoverinfo="skip",
            )
        )


@torch.no_grad()
def log_rgb_voxels(
    name: str,
    rgb_vol,
    alpha_vol=None,
    KPx=None,
    *,
    step=None,
    mode="splat",
    topk=60000,
    alpha_thresh=0.05,
    mesh_iso=0.2,
    pad=2.0,
    show_axes=True,
):
    RGB = _as_CDHW(rgb_vol)  # [3,D,H,W]
    if RGB.shape[0] != 3:
        raise ValueError("rgb_vol must be 3-channel [3,D,H,W] (or [1,3,D,H,W]).")
    D, H, W = RGB.shape[1:]
    ALP = None if alpha_vol is None else _as_DHW(alpha_vol)
    if ALP is not None and ALP.shape != (D, H, W):
        raise ValueError(f"alpha_vol shape {ALP.shape} != rgb spatial {(D, H, W)}")

    fig = go.Figure()

    if mode == "splat":
        alpha_thresh_local = 0.25
        rgb_mag_thresh = 0.10
        edge_percentile = 80
        use_edge_mask = True

        if ALP is not None:
            weights = ALP.clip(0, 1)
        else:
            weights = None

        mag = np.sqrt(RGB[0] ** 2 + RGB[1] ** 2 + RGB[2] ** 2)
        if weights is None:
            mask = mag >= rgb_mag_thresh
        else:
            mask = (weights >= alpha_thresh_local) & (mag >= rgb_mag_thresh)

        if use_edge_mask and weights is not None:
            try:
                import scipy.ndimage as ndi
                gx = ndi.sobel(weights, axis=2)
                gy = ndi.sobel(weights, axis=1)
                gz = ndi.sobel(weights, axis=0)
                edge = np.sqrt(gx * gx + gy * gy + gz * gz)
                if edge.any():
                    thr = np.percentile(edge[mask], edge_percentile) if mask.any() else np.percentile(edge, edge_percentile)
                    edge_mask = edge >= thr
                    mask = mask & edge_mask
            except Exception:
                pass

        idx = np.argwhere(mask)
        if idx.size == 0:
            fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="markers", name="RGB voxels"))
            wandb.log({name: fig}, step=step)
            return fig

        if idx.shape[0] > topk:
            score = weights[idx[:, 0], idx[:, 1], idx[:, 2]] if weights is not None else mag[idx[:, 0], idx[:, 1], idx[:, 2]]
            sel = np.argpartition(score, -topk)[-topk:]
            idx = idx[sel]

        z_i, y_i, x_i = idx[:, 0].astype(np.int64), idx[:, 1].astype(np.int64), idx[:, 2].astype(np.int64)
        x_f, y_f, z_f = x_i.astype(np.float32), y_i.astype(np.float32), z_i.astype(np.float32)

        r, g, b = RGB[0, z_i, y_i, x_i], RGB[1, z_i, y_i, x_i], RGB[2, z_i, y_i, x_i]
        if r.min() < 0 or g.min() < 0 or b.min() < 0:
            r = (r + 1) * 0.5
            g = (g + 1) * 0.5
            b = (b + 1) * 0.5
        Rv = np.clip((r * 255).astype(np.uint8), 0, 255)
        Gv = np.clip((g * 255).astype(np.uint8), 0, 255)
        Bv = np.clip((b * 255).astype(np.uint8), 0, 255)

        opac = weights[z_i, y_i, x_i] if weights is not None else np.ones_like(Rv, dtype=np.float32)
        color_rgba = [
            f"rgba({int(Rv[k])},{int(Gv[k])},{int(Bv[k])},{float(np.clip(opac[k], 0.0, 1.0))})"
            for k in range(len(Rv))
        ]

        fig.add_trace(
            go.Scatter3d(
                x=x_f,
                y=y_f,
                z=z_f,
                mode="markers",
                marker=dict(size=2, color=color_rgba),
                name="RGB voxels",
            )
        )
    else:
        raise ValueError(f"Unknown mode='{mode}' (only 'splat' is implemented here).")

    if KPx is not None:
        _draw_kp_crosses(fig, KPx, D, H, W, space="global", half=2.0)

    if show_axes:
        fig.update_scenes(
            xaxis=dict(range=[-pad, W - 1 + pad]),
            yaxis=dict(range=[-pad, H - 1 + pad]),
            zaxis=dict(range=[-pad, D - 1 + pad]),
            aspectmode="data",
        )
    fig.update_layout(
        title=f"RGB voxels ({mode})",
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=True,
        legend=dict(orientation="h"),
        scene=dict(xaxis_title="X (W)", yaxis_title="Y (H)", zaxis_title="Z (D)"),
    )

    wandb.log({name: fig}, step=step)
    return fig


# ----------------------------------------------------------------------
# filter_topk_kps_3d (unchanged logic, but no .numpy() used)
# ----------------------------------------------------------------------

@torch.no_grad()
def filter_topk_kps_3d(
    z_base_var,
    mu_tot,
    topk: int,
    obj_on=None,
    use_posterior_in_score: bool = False,
    eps: float = 1e-6,
):
    B = z_base_var.shape[0]
    C = z_base_var.shape[-1]
    assert C in (3, 6)

    zvar = z_base_var.view(B, -1, C)
    mu = mu_tot.view(B, -1, 3)
    K = zvar.size(1)

    prior_var = zvar[..., :3]
    if use_posterior_in_score and C == 6:
        post_var = zvar[..., 3:6].exp().clamp_min(eps)
        unc = prior_var.sum(-1) + post_var.sum(-1)
    else:
        unc = prior_var.sum(-1)

    if obj_on is not None:
        g = obj_on
        if g.dim() == 4 and g.size(-1) == 1:
            g = g.view(B, -1)
        else:
            g = g.view(B, -1)
        unc = unc * g

    k = min(topk, K)
    _, idx = torch.topk(unc, k=k, dim=1, largest=False)
    idx_exp = idx.unsqueeze(-1).expand(B, k, 3)
    topk_kp = torch.gather(mu, 1, idx_exp)
    bb_scores = -unc
    return {"indices": idx, "topk_kp": topk_kp, "unc": unc, "bb_scores": bb_scores}
