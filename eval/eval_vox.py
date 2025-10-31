import numpy as np
import plotly.graph_objects as go
import torch, wandb

# ---------- small utils ----------
def _np(x):
    if x is None: return None
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)

def _as_DHW(vol):
    """Accept [D,H,W], [1,D,H,W], or [C,D,H,W] with C==1; return [D,H,W] or None."""
    if vol is None:
        return None
    V = _np(vol)
    if V.ndim == 3:
        return V
    if V.ndim == 4:
        if V.shape[0] == 1:   # [1,D,H,W]
            return V[0]
        raise ValueError(f"Volume has channel dim {V.shape[0]} != 1")
    raise ValueError(f"Expected 3D or 4D volume, got shape {V.shape}")

def _kp_norm_to_index(K, D, H, W, order=("z","y","x")):
    """
    K: [K,3] in [-1,1] following given order.
    Map to voxel-index coordinates (0..D-1,0..H-1,0..W-1) then to scene coords centered in voxel space.
    """
    if K is None: return None
    ord_map = {ax:i for i,ax in enumerate(order)}  # where z,y,x live in K[...,i]
    z = np.clip((K[:, ord_map["z"]] + 1) * 0.5 * (D-1), 0, D-1)
    y = np.clip((K[:, ord_map["y"]] + 1) * 0.5 * (H-1), 0, H-1)
    x = np.clip((K[:, ord_map["x"]] + 1) * 0.5 * (W-1), 0, W-1)
    # Use voxel-index as the scene space (consistent with isosurface axes)
    return np.stack([x, y, z], axis=-1)  # Plotly uses x,y,z order

def _robust_box_from_vol(vol, iso):
    """Return (xrange, yrange, zrange) from nonzero iso region; fallback to full extents."""
    D,H,W = vol.shape
    mask = vol >= iso
    if not mask.any():
        return (0,W-1), (0,H-1), (0,D-1)
    zz, yy, xx = np.where(mask)
    return (float(xx.min()), float(xx.max())), (float(yy.min()), float(yy.max())), (float(zz.min()), float(zz.max()))
import numpy as np
import plotly.graph_objects as go
import wandb

# ---- try figure_factory; fall back if it pulls SciPy/Skimage and fails
try:
    import plotly.figure_factory as ff  # may import skimage/scipy under the hood
    _HAS_FF = True
except Exception:
    _HAS_FF = False

# ---- tiny universal converter ----
def _np(x):
    if x is None:
        return None
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def _as_DHW(vol):
    if vol is None: return None
    V = _np(vol)
    if V.ndim == 4 and V.shape[0] == 1:
        V = V[0]
    assert V.ndim == 3, f"Expected [D,H,W] or [1,D,H,W], got {V.shape}"
    return V

def _kp_norm_to_index(kps, D, H, W, order=("z","y","x")):
    """
    kps: [K,3] in [-1,1] (can be torch or np), with components ordered by `order`.
    returns: [K,3] as plotly (x,y,z) voxel indices
    """
    k = _np(kps)                     # <<< fix: handle CUDA tensors
    if k.ndim == 1:
        k = k[None, :]
    size = {"x": W, "y": H, "z": D}

    def to_idx(v, n):
        return np.clip(0.5 * (v + 1.0) * (n - 1), 0, n - 1)

    # map from provided order (e.g., ("z","y","x")) to (x,y,z)
    x = to_idx(k[:, order.index("x")], size["x"])
    y = to_idx(k[:, order.index("y")], size["y"])
    z = to_idx(k[:, order.index("z")], size["z"])
    return np.stack([x, y, z], axis=-1)


def _robust_nonempty_box(mask_xyz):
    nz = np.argwhere(mask_xyz)
    if nz.size == 0:
        return (0,1),(0,1),(0,1)
    mins = np.maximum(nz.min(0)-1, 0)
    maxs = np.minimum(nz.max(0)+1, np.array(mask_xyz.shape)-1)
    return (int(mins[0]), int(maxs[0])), (int(mins[1]), int(maxs[1])), (int(mins[2]), int(maxs[2]))

# ---- minimal voxel->Mesh3d when ff.create_voxels is unavailable
def _mesh_from_binary(mask_xyz, color="rgba(255,127,14,1.0)", opacity=0.9, max_voxels=120000):
    # mask_xyz: [X,Y,Z] bool
    X, Y, Z = mask_xyz.shape
    idx = np.argwhere(mask_xyz)
    if idx.shape[0] == 0:
        return []
    # (optional) downsample occupied cells to cap triangle count
    if idx.shape[0] > max_voxels:
        sel = np.random.choice(idx.shape[0], max_voxels, replace=False)
        idx = idx[sel]

    # cube template (8 verts, 12 triangles) at origin
    verts_template = np.array([
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],
        [0,0,1],[1,0,1],[1,1,1],[0,1,1],
    ], dtype=float)
    faces_template = np.array([
        [0,1,2],[0,2,3],   # z=0
        [4,6,5],[4,7,6],   # z=1
        [0,4,5],[0,5,1],   # y=0 edge
        [1,5,6],[1,6,2],   # x=1 edge
        [2,6,7],[2,7,3],   # y=1 edge
        [3,7,4],[3,4,0],   # x=0 edge
    ], dtype=int)

    # cull internal faces by checking neighbors; keep only boundary cubes
    occ = mask_xyz
    keep = []
    for (x,y,z) in idx:
        if (x==0 or not occ[x-1,y,z]) or (x==X-1 or not occ[x+1,y,z]) or \
           (y==0 or not occ[x,y-1,z]) or (y==Y-1 or not occ[x,y+1,z]) or \
           (z==0 or not occ[x,y,z-1]) or (z==Z-1 or not occ[x,y,z+1]):
            keep.append((x,y,z))
    if not keep:
        return []

    # build a single mesh
    V = []
    I = []
    base = 0
    for (x,y,z) in keep:
        V.append(verts_template + np.array([x,y,z], dtype=float))
        I.append(faces_template + base)
        base += 8
    V = np.vstack(V)
    I = np.vstack(I)
    mesh = go.Mesh3d(
        x=V[:,0], y=V[:,1], z=V[:,2],
        i=I[:,0], j=I[:,1], k=I[:,2],
        color=color, opacity=opacity, flatshading=True, name="voxels", showscale=False
    )
    return [mesh]

# ---------- public plotting APIs ----------
def log_vox_overlay_plotly(
    name, gt_vol, rec_vol, kps=None, step=None,
    iso_levels=(0.2,), gt_color="rgba(31,119,180,1.0)", rec_color="rgba(255,127,14,1.0)",
    kp_color="#ff0000", kp_order=("z","y","x"), point_size_kp=6,
):
    G = _as_DHW(gt_vol)
    R = _as_DHW(rec_vol)
    fig = go.Figure()
    thr = float(np.min(iso_levels))

    if R is not None:
        maskR = np.transpose(R, (2,1,0)) > thr  # (X,Y,Z)
        if maskR.any():
            if _HAS_FF:
                vox = ff.create_voxels(maskR, colorscale=[[0,rec_color],[1,rec_color]], opacity=1.0)
                for tr in vox.data: tr.name="REC"; fig.add_trace(tr)
            else:
                for tr in _mesh_from_binary(maskR, rec_color, 0.95): tr.name="REC"; fig.add_trace(tr)

    if G is not None:
        maskG = np.transpose(G, (2,1,0)) > thr
        if maskG.any():
            if _HAS_FF:
                vox = ff.create_voxels(maskG, colorscale=[[0,gt_color],[1,gt_color]], opacity=0.65)
                for tr in vox.data: tr.name="GT"; fig.add_trace(tr)
            else:
                for tr in _mesh_from_binary(maskG, gt_color, 0.65): tr.name="GT"; fig.add_trace(tr)

    # keypoints
    if kps is not None and (R is not None or G is not None):
        D,H,W = (R.shape if R is not None else G.shape)
        K = _kp_norm_to_index(kps, D,H,W, order=kp_order)
        fig.add_trace(go.Scatter3d(
            x=K[:,0], y=K[:,1], z=K[:,2],
            mode="markers",
            marker=dict(symbol="x", size=point_size_kp, color=kp_color, line=dict(width=3, color="black")),
            name="keypoints",
        ))

    # axes from occupied region (prefer REC)
    if R is not None and (R > thr).any():
        xr, yr, zr = _robust_nonempty_box(np.transpose(R,(2,1,0)) > thr)
    elif G is not None and (G > thr).any():
        xr, yr, zr = _robust_nonempty_box(np.transpose(G,(2,1,0)) > thr)
    else:
        xr, yr, zr = (0,1),(0,1),(0,1)

    fig.update_scenes(
        xaxis=dict(range=[xr[0], xr[1]]),
        yaxis=dict(range=[yr[0], yr[1]]),
        zaxis=dict(range=[zr[0], zr[1]]),
        aspectmode="data"
    )
    fig.update_layout(margin=dict(l=0,r=0,t=30,b=0), showlegend=True, scene_dragmode="turntable")
    wandb.log({name: fig}, step=step)

def log_vox_isoseries(
    name, vol, kps=None, iso_levels=(0.05,0.1,0.2,0.3,0.4),
    step=None, color="rgba(255,127,14,1.0)", kp_color="#ff0000",
    kp_order=("z","y","x"), point_size_kp=6,
):
    V = _as_DHW(vol)
    if V is None: return
    fig = go.Figure()
    for i, iso in enumerate(sorted(iso_levels)):
        mask = np.transpose(V, (2,1,0)) > float(iso)
        if not mask.any(): continue
        alpha = float(np.clip(0.35 + 0.35 * (i / max(1, len(iso_levels)-1)), 0.35, 0.95))
        if _HAS_FF:
            vox = ff.create_voxels(mask, colorscale=[[0,color],[1,color]], opacity=alpha)
            for tr in vox.data: tr.name=f"iso≥{iso:.2f}"; fig.add_trace(tr)
        else:
            for tr in _mesh_from_binary(mask, color, alpha): tr.name=f"iso≥{iso:.2f}"; fig.add_trace(tr)

    if kps is not None:
        D,H,W = V.shape
        K = _kp_norm_to_index(kps, D,H,W, order=kp_order)
        fig.add_trace(go.Scatter3d(
            x=K[:,0], y=K[:,1], z=K[:,2],
            mode="markers",
            marker=dict(symbol="x", size=point_size_kp, color=kp_color, line=dict(width=3, color="black")),
            name="keypoints",
        ))

    mask_base = np.transpose(V, (2,1,0)) > float(min(iso_levels) if iso_levels else 0.5)
    xr, yr, zr = _robust_nonempty_box(mask_base)
    fig.update_scenes(
        xaxis=dict(range=[xr[0], xr[1]]),
        yaxis=dict(range=[yr[0], yr[1]]),
        zaxis=dict(range=[zr[0], zr[1]]),
        aspectmode="data"
    )
    fig.update_layout(margin=dict(l=0,r=0,t=30,b=0), showlegend=True, scene_dragmode="turntable")
    wandb.log({name: fig}, step=step)


@torch.no_grad()
def select_kp_topk(cov_kp, post_logvar, n_keep, *,
                   obj_on=None, warmup=False, warmup_ratio=1.0,
                   alpha=1.0, eps=1e-6):
    """
    cov_kp:      [B, N, 3, 3]   prior covariance per kp (from 3D SSM)
    post_logvar: [B, N, 3]      posterior log-variance from your encoder
    obj_on:      [B, N] or [B, N, 1]  optional objectness (higher is better)
    n_keep:      int, target #kps to keep after filtering

    Returns:
      embed_ind: [B, K]   indices of selected kps (smallest scores)
      score:     [B, N]   per-kp scores (lower = sharper/better)
    """
    B, N = cov_kp.shape[:2]

    # strictly-positive components
    prior_var   = torch.diagonal(cov_kp, -2, -1).clamp_min(eps)   # [B, N, 3]
    prior_trace = prior_var.sum(-1)                               # [B, N]

    post_var    = torch.exp(post_logvar).clamp_min(eps)           # [B, N, 3]
    post_trace  = post_var.sum(-1)                                # [B, N]

    score = prior_trace + alpha * post_trace                      # [B, N]

    if obj_on is not None:
        obj_on = obj_on.squeeze(-1) if obj_on.dim() == 3 else obj_on
        # penalize low objectness; keep finite
        score = score * (1.0 / obj_on.clamp_min(1e-3))

    # guard against NaN/Inf and break ties from flat heatmaps
    score = torch.where(torch.isfinite(score), score, torch.full_like(score, 1e9))
    score = score + 1e-6 * torch.randn_like(score)

    K = n_keep if not warmup else min(n_keep, max(1, int(warmup_ratio * N)))
    K = min(K, N)
    _, embed_ind = torch.topk(score, k=K, dim=-1, largest=False)  # keep smallest scores
    return embed_ind, score


def gather_by_ind(x, embed_ind):
    """
    Gather along dim=1 with broadcasting.
    x:         [B, N, ...]
    embed_ind: [B, K]
    returns:   [B, K, ...]
    """
    take_shape = list(embed_ind.shape) + [1] * (x.dim() - embed_ind.dim())
    ind_exp = embed_ind.view(*take_shape).expand(-1, -1, *x.shape[2:])
    return torch.take_along_dim(x, ind_exp, dim=1)


def _as_b0_channel(vol, occ_channel=0):
    """
    Accept [B,C,D,H,W] or [B,D,H,W] or [D,H,W]; return [D,H,W] for b0.
    """
    if vol is None:
        return None
    if vol.dim() == 5:          # [B,C,D,H,W]
        vol = vol[0, occ_channel]
    elif vol.dim() == 4:        # [B,D,H,W] (or occasionally still [B,1,D,H,W] collapsed)
        vol = vol[0]
        if vol.dim() == 4:      # lingering channel
            vol = vol[occ_channel]
    elif vol.dim() == 3:        # [D,H,W]
        pass
    else:
        raise ValueError(f"Unexpected vol shape {tuple(vol.shape)}")
    return vol

def extract_volumes_for_vis(model_output, *, occ_channel=0):
    """
    Returns (gt_vol[D,H,W], rec_vol[D,H,W]) using strictly:
      - gt  = model_output['x']
      - rec = sigmoid(model_output['rec'])   # convert logits -> probs for viz
    """
    if 'rec' not in model_output:
        raise KeyError("model_output['rec'] is required.")
    if 'x' not in model_output:
        raise KeyError("model_output['x'] is required as the ground-truth volume.")

    rec_logits = _as_b0_channel(model_output['rec'], occ_channel=occ_channel)
    rec = torch.sigmoid(rec_logits)                     # probs in [0,1] for plotting
    gt  = _as_b0_channel(model_output['x'],   occ_channel=occ_channel)
    return gt, rec



def print_vol_stats(tag, V):
    if V is None:
        print(f"{tag}: None"); return
    import torch, numpy as np
    if isinstance(V, torch.Tensor):
        v = V.detach().float().cpu().numpy()
    else:
        v = np.asarray(V)
    f_ok   = np.isfinite(v).all()
    print(f"{tag}: shape={v.shape}, min={v.min():.4g}, max={v.max():.4g}, mean={v.mean():.4g}, finite={f_ok}")

import numpy as np
import torch, wandb
import plotly.graph_objects as go
import torch.nn.functional as F

def _flatten01(t, occ_channel=0):
    """
    Accepts [B,T,1,D,H,W] or [B,T,D,H,W] or [B,1,D,H,W] or [B,D,H,W].
    Returns flat 1D tensor on CPU.
    """
    if t is None:
        return None
    if t.dim() == 6:       # [B,T,C,D,H,W] or [B,T,D,H,W]
        if t.size(2) > 1:  # has C
            t = t[:, :, occ_channel]
        else:
            t = t[:, :, 0]
        t = t.reshape(-1, *t.shape[-3:])  # -> [BT, D,H,W]
    elif t.dim() == 5:     # [B,C,D,H,W] or [B,D,H,W]
        if t.size(1) > 1:
            t = t[:, occ_channel]
        else:
            t = t[:, 0]
        t = t.reshape(-1, *t.shape[-3:])  # -> [B, D,H,W]
    elif t.dim() == 4:     # [D,H,W,B?] unlikely, or already [B,D,H,W]
        pass
    elif t.dim() == 3:     # [D,H,W]
        t = t.unsqueeze(0)
    else:
        raise ValueError(f"Unexpected shape {tuple(t.shape)}")
    return t.contiguous().view(-1).detach().cpu()

@torch.no_grad()
def log_voxel_rec_distributions(model_output, x, *, occ_channel=0, name_prefix="dist", step=None):
    """
    Logs histograms + stats for rec logits/probs and GT occupancy.
    Expects model_output['rec'] as logits (NOT sigmoid'ed).
    """
    if 'rec' not in model_output:
        raise KeyError("model_output['rec'] (logits) required")

    rec_logits = _flatten01(model_output['rec'], occ_channel=occ_channel)
    gt_flat    = _flatten01(x,                        occ_channel=occ_channel).float()

    # Probs from logits
    rec_probs = torch.sigmoid(torch.from_numpy(rec_logits.numpy())).numpy()

    # --- stats ---
    def _stats(arr):
        return dict(
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
        )
    s_logits = _stats(rec_logits.numpy())
    s_probs  = _stats(rec_probs)
    s_gt     = dict(pos_frac=float(gt_flat.mean().item()), count=int(gt_flat.numel()))

    # positive fraction at common thresholds
    thr_grid = np.linspace(0.05, 0.95, 19)
    pred_pos_fracs = (rec_probs[:, None] > thr_grid[None, :]).mean(axis=0)

    # simple PR/IoU sweep
    gt_np = gt_flat.numpy()
    eps = 1e-8
    prec, rec, iou = [], [], []
    for th in thr_grid:
        pred = (rec_probs > th)
        tp = float(np.sum(pred & (gt_np > 0.5)))
        fp = float(np.sum(pred & (gt_np < 0.5)))
        fn = float(np.sum((~pred) & (gt_np > 0.5)))
        p  = tp / (tp + fp + eps)
        r  = tp / (tp + fn + eps)
        u  = tp / (tp + fp + fn + eps)
        prec.append(p); rec.append(r); iou.append(u)

    # --- histograms (wandb native + a plotly overlay for probs) ---
    log_dict = {
        f"{name_prefix}/rec_logits_hist": wandb.Histogram(rec_logits.numpy()),
        f"{name_prefix}/rec_probs_hist" : wandb.Histogram(rec_probs),
        f"{name_prefix}/gt_hist"        : wandb.Histogram(gt_np),
        f"{name_prefix}/stats/logits_min": s_logits["min"],
        f"{name_prefix}/stats/logits_max": s_logits["max"],
        f"{name_prefix}/stats/logits_mean": s_logits["mean"],
        f"{name_prefix}/stats/probs_mean": s_probs["mean"],
        f"{name_prefix}/stats/probs_std" : s_probs["std"],
        f"{name_prefix}/gt/pos_frac": s_gt["pos_frac"],
        f"{name_prefix}/gt/count": s_gt["count"],
    }

    # overlay histogram: probs vs GT (GT as 0/1 bars)
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=rec_probs, nbinsx=50, name="pred_probs", opacity=0.7))
    # put GT as two bars at 0 and 1 scaled to same total
    fig_hist.add_trace(go.Histogram(x=gt_np, nbinsx=2, name="gt_occ(0/1)", opacity=0.6))
    fig_hist.update_layout(barmode='overlay', title="Pred prob vs GT occupancy")
    log_dict[f"{name_prefix}/overlay_hist"] = fig_hist

    # threshold sweep plot
    fig_sweep = go.Figure()
    fig_sweep.add_trace(go.Scatter(x=thr_grid, y=prec, mode="lines+markers", name="precision"))
    fig_sweep.add_trace(go.Scatter(x=thr_grid, y=rec,  mode="lines+markers", name="recall"))
    fig_sweep.add_trace(go.Scatter(x=thr_grid, y=iou,  mode="lines+markers", name="IoU"))
    fig_sweep.add_trace(go.Scatter(x=thr_grid, y=pred_pos_fracs, mode="lines+markers", name="pred_pos_frac", yaxis="y2"))
    fig_sweep.update_layout(
        title="Threshold sweep (probs→mask)",
        xaxis_title="threshold",
        yaxis=dict(title="PR/IoU"),
        yaxis2=dict(title="pred_pos_frac", overlaying='y', side='right', rangemode='tozero'),
        legend=dict(orientation='h')
    )
    log_dict[f"{name_prefix}/threshold_sweep"] = fig_sweep

    # a couple of handy scalars
    best_i = int(np.argmax(iou))
    log_dict.update({
        f"{name_prefix}/best_iou": float(iou[best_i]),
        f"{name_prefix}/best_iou_thr": float(thr_grid[best_i]),
        f"{name_prefix}/pred_pos@0.5": float(pred_pos_fracs[np.argmin(np.abs(thr_grid-0.5))]),
    })

    wandb.log(log_dict, step=step)
