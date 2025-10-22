# eval/eval_vox.py
import math
import numpy as np
import torch
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# ---------------------------
# basic utils
# ---------------------------
def to_np(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x)

def clamp01(x):
    if x is None: return None
    if torch.is_tensor(x):
        return x.clamp(0, 1)
    return np.clip(x, 0, 1)

def _ensure_5d(vol):
    """
    Accepts [B,C,D,H,W] | [C,D,H,W] | [D,H,W] (float or bool).
    Returns (V, B, C) where V is np.float32 [B,C,D,H,W].
    """
    if vol is None:
        return None, 0, 0
    if torch.is_tensor(vol):
        vol = vol.detach().cpu().float()
    vol = np.asarray(vol)

    if vol.ndim == 5:
        B, C = int(vol.shape[0]), int(vol.shape[1])
    elif vol.ndim == 4:
        vol = vol[None, ...]     # [1,C,D,H,W]
        B, C = 1, int(vol.shape[1])
    elif vol.ndim == 3:
        vol = vol[None, None, ...]  # [1,1,D,H,W]
        B, C = 1, 1
    else:
        raise ValueError(f"volume must be [B,C,D,H,W] or [C,D,H,W] or [D,H,W], got {vol.shape}")
    return vol.astype(np.float32), B, C

def _center_slices(v_np):
    """
    v_np: [C,D,H,W] or [1,D,H,W]
    Returns dict with mid XY/XZ/YZ slice (as 2D arrays).
    """
    if v_np.ndim == 4:
        C, D, H, W = v_np.shape
    elif v_np.ndim == 3:
        C, D, H, W = 1, *v_np.shape
        v_np = v_np[None, ...]
    else:
        raise ValueError(f"expected [C,D,H,W] got {v_np.shape}")

    c = min(C, 1)   # show the first channel by default
    v0 = v_np[0] if C >= 1 else v_np[0]
    d2, h2, w2 = D // 2, H // 2, W // 2
    return {
        "XY@z": v0[d2, :, :],      # [H,W]
        "XZ@y": v0[:, h2, :],      # [D,W]
        "YZ@x": v0[:, :, w2],      # [D,H]
    }

def _format_slice(ax, img, title, cmap="magma"):
    ax.imshow(img, cmap=cmap, origin="lower", interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

def _normalize_for_plot(vol, per_volume=True):
    """
    Put arbitrary floats into [0,1] per-volume for nicer plotting.
    """
    v = vol.copy()
    if per_volume:
        lo = np.nanpercentile(v, 1)
        hi = np.nanpercentile(v, 99)
        if not math.isfinite(lo) or not math.isfinite(hi) or abs(hi - lo) < 1e-8:
            return np.zeros_like(v)
        v = (v - lo) / (hi - lo + 1e-12)
        v = np.clip(v, 0, 1)
    return v

# ---------------------------
# metrics
# ---------------------------
def vox_iou(pred, target, thresh=0.5):
    """
    IoU for binary occupancy volumes.
    Accepts [B,1,D,H,W] or [B,D,H,W] etc.
    """
    p, B, C = _ensure_5d(pred)
    t, _, _ = _ensure_5d(target)
    assert p.shape == t.shape, f"shapes must match, got {p.shape} vs {t.shape}"
    # use channel 0
    P = (p[:, 0] >= thresh)
    T = (t[:, 0] >= thresh)
    inter = (P & T).sum(axis=(1, 2, 3))
    union = (P | T).sum(axis=(1, 2, 3))
    iou = np.where(union > 0, inter / (union + 1e-8), 1.0)
    return float(iou.mean())

def vox_mse(pred, target):
    p, B, C = _ensure_5d(pred)
    t, _, _ = _ensure_5d(target)
    assert p.shape == t.shape
    return float(((p - t) ** 2).mean())

def vox_psnr(pred, target, data_range=1.0):
    mse = vox_mse(pred, target)
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * math.log10(data_range) - 10.0 * math.log10(mse + 1e-12))

# ---------------------------
# logging: 2D slices panel
# ---------------------------
def log_vox_center_slices(name, vol, step=None, cmap="magma", normalize=True):
    """
    vol: [B,C,D,H,W] or [C,D,H,W] or [D,H,W]  (float/bool)
    Logs a 2x2 panel with center XY/XZ/YZ of the first item in the batch.
    """
    v, B, C = _ensure_5d(vol)
    v0 = v[0]  # [C,D,H,W]
    if normalize:
        v0 = _normalize_for_plot(v0)
    sl = _center_slices(v0)

    fig, axs = plt.subplots(2, 2, figsize=(7, 6), dpi=120)
    _format_slice(axs[0,0], sl["XY@z"], "XY @ mid-z", cmap)
    _format_slice(axs[0,1], sl["XZ@y"], "XZ @ mid-y", cmap)
    _format_slice(axs[1,0], sl["YZ@x"], "YZ @ mid-x", cmap)
    axs[1,1].axis("off"); axs[1,1].set_title("")

    plt.tight_layout()
    wandb.log({name: wandb.Image(fig)}, step=step)
    plt.close(fig)

def log_vox_compare_slices(name, gt, rec, step=None, cmap="magma", normalize=True, kps=None):
    """
    Side-by-side slices for GT vs REC; optional KP overlay (in [-1,1]^3).
    """
    g, _, _ = _ensure_5d(gt)
    r, _, _ = _ensure_5d(rec)
    g0, r0 = g[0], r[0]    # [C,D,H,W]
    if normalize:
        g0 = _normalize_for_plot(g0)
        r0 = _normalize_for_plot(r0)

    g_sl = _center_slices(g0)
    r_sl = _center_slices(r0)

    fig, axs = plt.subplots(2, 3, figsize=(10, 7), dpi=120)
    _format_slice(axs[0,0], g_sl["XY@z"], "GT: XY@z", cmap)
    _format_slice(axs[0,1], g_sl["XZ@y"], "GT: XZ@y", cmap)
    _format_slice(axs[0,2], g_sl["YZ@x"], "GT: YZ@x", cmap)

    _format_slice(axs[1,0], r_sl["XY@z"], "REC: XY@z", cmap)
    _format_slice(axs[1,1], r_sl["XZ@y"], "REC: XZ@y", cmap)
    _format_slice(axs[1,2], r_sl["YZ@x"], "REC: YZ@x", cmap)

    # optional KP overlay (just on XY@z)
    if kps is not None:
        K = to_np(kps)
        if K is not None:
            # K ∈ [-1,1]^3 to pixel coords on XY slice
            # Assumes square voxels and v0 shape [C,D,H,W].
            _, D, H, W = g0.shape
            def norm_to_idx(u, N):
                # [-1,1] -> [0, N-1]
                return (0.5 * (u + 1.0) * (N - 1.0))
            x_pix = norm_to_idx(K[..., 0], W)
            y_pix = norm_to_idx(K[..., 1], H)
            axs[1,0].scatter(x_pix, y_pix, s=10, c="red", marker="x", linewidths=0.75)

    plt.tight_layout()
    wandb.log({name: wandb.Image(fig)}, step=step)
    plt.close(fig)

# ---------------------------
# logging: interactive 3D volume
# ---------------------------
def log_vox_plotly_volume(name, vol, step=None, isovalue=0.5, mode="volume", colorscale="Viridis"):
    """
    Logs an interactive 3D volume or isosurface of a single example in the batch.
      mode: "volume" (dense alpha ray-marching) or "isosurface"
      isovalue: threshold for "isosurface".
    """
    v, B, C = _ensure_5d(vol)
    V = v[0, 0]  # use first channel
    V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)

    fig = go.Figure()
    if mode == "isosurface":
        fig.add_trace(go.Isosurface(
            value=V,
            x=np.arange(V.shape[2]).flatten()[None].repeat(V.shape[0]*V.shape[1], 0).flatten(),
            y=np.tile(np.arange(V.shape[1]), V.shape[0]*V.shape[2]),
            z=np.repeat(np.arange(V.shape[0]), V.shape[1]*V.shape[2]),
            isomin=isovalue, isomax=V.max(),
            opacity=0.6,
            caps=dict(x_show=False, y_show=False, z_show=False),
            colorscale=colorscale,
            surface_count=3
        ))
    else:
        fig.add_trace(go.Volume(
            value=V.flatten(),
            x=np.arange(V.shape[2]).flatten()[None].repeat(V.shape[0]*V.shape[1], 0).flatten(),
            y=np.tile(np.arange(V.shape[1]), V.shape[0]*V.shape[2]),
            z=np.repeat(np.arange(V.shape[0]), V.shape[1]*V.shape[2]),
            opacity=0.08,  # small for ray marching
            surface_count=15,
            colorscale=colorscale
        ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(aspectmode="data"),
        showlegend=False,
        title=name
    )
    wandb.log({name: fig}, step=step)

# ---------------------------
# keypoint selection (reused)
# ---------------------------
def select_topk_keypoints_vox(model_output, topk, prefer_logvar=True, eps=1e-8):
    """
    Same contract as your PC helper; uses z_base_var (smaller is better) if present,
    falls back to kp_cov trace, else zeros.
    """
    kp = model_output.get('kp_p', None)   # [B,Kp,3]
    assert kp is not None, "model_output['kp_p'] must be present"
    B, Kp, _ = kp.shape
    device = kp.device

    score = None
    if prefer_logvar and ('z_base_var' in model_output):
        z_var = model_output['z_base_var']
        if z_var.dim() > 2:
            while z_var.dim() > 2:
                z_var = z_var.sum(-1)
        if z_var.shape[:2] == (B, Kp):
            score = -z_var

    if score is None and ('kp_cov' in model_output):
        cov = model_output['kp_cov']
        tr = cov[..., 0,0] + cov[..., 1,1] + cov[..., 2,2]
        score = -tr

    if score is None:
        score = torch.zeros(B, Kp, device=device)

    obj_on = model_output.get('obj_on', None)
    if obj_on is not None:
        o = obj_on.squeeze(-1) if (obj_on.dim() == 3 and obj_on.size(-1) == 1) else obj_on
        if o.shape == (B, Kp):
            score = score * o.clamp(0, 1)

    k_eff = min(int(topk), Kp)
    scores_topk, idx = torch.topk(score, k=k_eff, dim=-1, largest=True, sorted=True)
    b = torch.arange(B, device=device)[:, None]
    kp_topk = kp[b, idx]

    model_output['kp_scores'] = score
    model_output['kp_topk_idx'] = idx
    model_output['kp_topk'] = kp_topk
    model_output['kp_scores_topk'] = scores_topk
    return idx, kp_topk, scores_topk


# ---- eval_vox_plotly.py-like helpers ----
import numpy as np, torch, wandb, plotly.graph_objects as go

def _to_np(x):
    if x is None: return None
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

def _pick_channel(vol, channel=0):
    """
    Accepts [C,D,H,W] or [D,H,W] and returns a [D,H,W] float32.
    """
    v = _to_np(vol)
    assert v.ndim in (3,4), f"expected 3D or 4D, got {v.shape}"
    if v.ndim == 4:
        c = min(channel, v.shape[0]-1)
        v = v[c]
    v = v.astype(np.float32)
    # sanitize
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return v

def _kp_np(kps):
    """
    Accepts [K,3] or [B,K,3] and returns Nx3 np array (no scaling).
    """
    if kps is None: return None
    k = _to_np(kps)
    if k.ndim == 3:  # [B,K,3] -> pick first
        k = k[0]
    if k.ndim == 1 and k.size == 3:
        k = k[None]
    if k.ndim != 2 or k.shape[-1] != 3: 
        return None
    return k

def _denorm_kp_to_index(kp_xyz_norm, D,H,W):
    """
    Convert keypoints from normalized [-1,1] coords to voxel index space [0..D/H/W).
    """
    # grid_sample convention: x=last (W), y=H, z=D
    x = (kp_xyz_norm[:,0] + 1.0) * 0.5 * (W-1)
    y = (kp_xyz_norm[:,1] + 1.0) * 0.5 * (H-1)
    z = (kp_xyz_norm[:,2] + 1.0) * 0.5 * (D-1)
    return np.stack([x,y,z], axis=-1)

def log_vox_plotly_volume(
    name,
    vol,                 # [C,D,H,W] or [D,H,W]; float in [0,1] or any scalar field
    step=None,
    channel=0,
    mode="volume",       # "volume" | "isosurface" | "points"
    isovalue=0.5,        # used for isosurface / points threshold
    opacity=0.15,        # only for "volume"
    kps=None,            # keypoints in [-1,1]^3
    kps_size=3,
    cmap="Viridis",
):
    """
    Logs a 3D Plotly view to W&B.
    - "volume": true 3D volume raymarch (fastest preview, tunable opacity)
    - "isosurface": single surface at 'isovalue'
    - "points": plots occupied voxels (>= isovalue) as small points
    """
    V = _pick_channel(vol, channel=channel)  # [D,H,W]
    D,H,W = V.shape
    fig = go.Figure()

    if mode == "volume":
        fig.add_trace(go.Volume(
            value=V.flatten(),
            x=np.repeat(np.arange(W), D*H),
            y=np.tile(np.repeat(np.arange(H), W), D),
            z=np.repeat(np.arange(D), H*W),
            opacity=opacity,            # overall opacity
            surface_count=15,           # number of isosurfaces in the volume render
            showscale=False,
            colorscale=cmap,
        ))

    elif mode == "isosurface":
        fig.add_trace(go.Isosurface(
            value=V.flatten(),
            x=np.repeat(np.arange(W), D*H),
            y=np.tile(np.repeat(np.arange(H), W), D),
            z=np.repeat(np.arange(D), H*W),
            isomin=isovalue,
            isomax=isovalue,
            surface_count=1,
            caps=dict(x_show=False, y_show=False, z_show=False),
            showscale=False,
            colorscale=cmap,
        ))

    elif mode == "points":
        idx = np.argwhere(V >= isovalue)
        if idx.shape[0] > 120_000:  # safety downsample for speed
            sel = np.random.choice(idx.shape[0], 120_000, replace=False)
            idx = idx[sel]
        # idx is [N,3] with [z,y,x] order
        fig.add_trace(go.Scatter3d(
            x=idx[:,2], y=idx[:,1], z=idx[:,0],
            mode="markers",
            marker=dict(size=2, opacity=0.9),
            name="voxels",
        ))

    # --- optional KP overlay (in normalized [-1,1] -> index space) ---
    K = _kp_np(kps)
    if K is not None:
        K_ijk = _denorm_kp_to_index(K, D,H,W)  # [N,3] in (x,y,z) index space

        fig.add_trace(go.Scatter3d(
            x=K_ijk[:,0], y=K_ijk[:,1], z=K_ijk[:,2],
            mode="markers",
            marker=dict(size=kps_size, symbol="x", color="red"),
            name="keypoints",
        ))

    # nice equal aspectbox
    fig.update_scenes(
        xaxis=dict(range=[0, W-1]),
        yaxis=dict(range=[0, H-1]),
        zaxis=dict(range=[0, D-1]),
        aspectmode="data"
    )
    fig.update_layout(
        margin=dict(l=0,r=0,t=30,b=0),
        scene_dragmode="turntable",
        showlegend=True,
        title=name,
    )

    wandb.log({name: fig}, step=step)


# --- GT logging: same API as rec ---
def log_vox_plotly_gt_suite(prefix, gt_vol, step=None, channel=0, kps=None,
                            iso=0.5, vol_opacity=0.15):
    log_vox_plotly_volume(f"{prefix}/gt_volume",   gt_vol, step=step,
                          channel=channel, mode="volume", opacity=vol_opacity, kps=kps)
    log_vox_plotly_volume(f"{prefix}/gt_isosurf@{iso}", gt_vol, step=step,
                          channel=channel, mode="isosurface", isovalue=iso, kps=kps)
    log_vox_plotly_volume(f"{prefix}/gt_points@{iso}",  gt_vol, step=step,
                          channel=channel, mode="points", isovalue=iso, kps=kps)


def log_vox_plotly_rec_suite(prefix, rec_vol, step=None, channel=0, kps=None,
                             iso=0.5, vol_opacity=0.15):
    """
    Mirrors log_vox_plotly_gt_suite but for reconstructed volumes.
    Uses the same three modes: volume, isosurface (at `iso`), and points (at `iso`).
    """
    log_vox_plotly_volume(f"{prefix}/rec_volume",   rec_vol, step=step,
                          channel=channel, mode="volume", opacity=vol_opacity, kps=kps)
    log_vox_plotly_volume(f"{prefix}/rec_isosurf@{iso}", rec_vol, step=step,
                          channel=channel, mode="isosurface", isovalue=iso, kps=kps)
    log_vox_plotly_volume(f"{prefix}/rec_points@{iso}",  rec_vol, step=step,
                          channel=channel, mode="points", isovalue=iso, kps=kps)
