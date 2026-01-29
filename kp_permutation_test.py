#!/usr/bin/env python3
"""
Generate one KP+voxel figure per axis permutation.
Look at all 6 and pick the one where crosses sit on objects.
"""
import os, sys, json, argparse
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.io as pio
from itertools import permutations

if 'numpy._core' not in sys.modules:
    import numpy.core
    sys.modules['numpy._core'] = numpy.core
    for submod in ['multiarray', 'umath', '_multiarray_umath', 'numeric']:
        full_old = f'numpy.core.{submod}'
        full_new = f'numpy._core.{submod}'
        if full_old in sys.modules and full_new not in sys.modules:
            sys.modules[full_new] = sys.modules[full_old]
        elif hasattr(numpy.core, submod) and full_new not in sys.modules:
            sys.modules[full_new] = getattr(numpy.core, submod)

from utils.util_func import get_config
from datasets.point_cloud_datasets.get_dataset import get_point_cloud_dataset
from voxel_models import DLP
from visualize_voxel_paper import load_model_from_config, kp_norm_to_voxel_xyz


def make_fig(rgb_vol, mu_tot, D, H, W, kp_order, alpha_thresh=0.05):
    """One figure: RGB voxels + KP crosses for a given kp_order."""
    if isinstance(rgb_vol, torch.Tensor):
        rgb_vol = rgb_vol.detach().cpu().numpy()
    if rgb_vol.ndim == 5:
        rgb_vol = rgb_vol[0]
    if rgb_vol.min() < 0:
        rgb_vol = (rgb_vol + 1) * 0.5
    rgb_vol = np.clip(rgb_vol, 0, 1)

    mag = np.sqrt((rgb_vol ** 2).sum(axis=0))
    mask = mag > alpha_thresh
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return go.Figure()

    if len(idx) > 40000:
        scores = mag[mask]
        sel = np.argpartition(scores, -40000)[-40000:]
        idx = idx[sel]

    z_i, y_i, x_i = idx[:, 0], idx[:, 1], idx[:, 2]
    r = (rgb_vol[0, z_i, y_i, x_i] * 255).astype(np.uint8)
    g = (rgb_vol[1, z_i, y_i, x_i] * 255).astype(np.uint8)
    b = (rgb_vol[2, z_i, y_i, x_i] * 255).astype(np.uint8)
    colors = [f"rgba({r[i]},{g[i]},{b[i]},0.6)" for i in range(len(r))]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x_i.astype(float).tolist(),
        y=y_i.astype(float).tolist(),
        z=z_i.astype(float).tolist(),
        mode='markers', marker=dict(size=2, color=colors),
        name='voxels'
    ))

    # KP crosses
    kp = mu_tot
    if isinstance(kp, torch.Tensor):
        kp = kp.detach().cpu().numpy()
    if kp.ndim == 3:
        kp = kp[0]

    xyz = kp_norm_to_voxel_xyz(kp, D, H, W, kp_order=kp_order)
    half = 2.0
    xs, ys, zs = [], [], []
    for x, y, z in xyz:
        xs += [x-half, x+half, None, x, x, None, x, x, None]
        ys += [y, y, None, y-half, y+half, None, y, y, None]
        zs += [z, z, None, z, z, None, z-half, z+half, None]
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs, mode='lines',
        line=dict(width=6, color='red'), name='KPs'
    ))

    order_str = ','.join(kp_order)
    fig.update_layout(
        title=f"kp_order=({order_str})",
        scene=dict(
            xaxis=dict(range=[0, W-1]), yaxis=dict(range=[0, H-1]),
            zaxis=dict(range=[0, D-1]), aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", "-r", required=True)
    parser.add_argument("--output-dir", "-o", default="kp_permutation_test")
    parser.add_argument("--alpha-thresh", type=float, default=0.05)
    args = parser.parse_args()

    cfg_path = os.path.join(args.run_dir, "hparams.json")
    ckpt_path = os.path.join(args.run_dir, "saves", "best.pt")
    if not os.path.isfile(ckpt_path):
        ckpt_path = os.path.join(args.run_dir, "best.pt")

    cfg = get_config(cfg_path)
    W, H, D = cfg["voxel_grid_whd"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset (1 sample)
    ds = get_point_cloud_dataset(
        cfg['ds'], cfg['root'], mode='train', max_points=4096, include_rgb=(cfg['ch'] == 6)
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
    batch = next(iter(loader))
    vox = batch["voxels"].to(device)

    # Load model
    model = load_model_from_config(cfg, device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = None
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model", "ema", "ema_state_dict"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
    if state is None:
        state = ckpt
    def strip(d, p):
        return {(k[len(p):] if k.startswith(p) else k): v for k, v in d.items()}
    state = strip(strip(state, "module."), "model.")
    model.load_state_dict(state, strict=False)

    with torch.no_grad():
        out = model(vox, warmup=False, with_loss=True)

    mu_tot = (out['z_base'] + out['mu_offset'])[0]
    rgb_vol = out['rec'][0]

    os.makedirs(args.output_dir, exist_ok=True)

    all_orders = list(permutations(["x", "y", "z"]))
    print(f"Generating {len(all_orders)} figures...")

    for order in all_orders:
        fig = make_fig(rgb_vol, mu_tot, D, H, W, kp_order=order, alpha_thresh=args.alpha_thresh)
        name = ''.join(order)
        path = os.path.join(args.output_dir, f"kp_order_{name}.html")
        fig.write_html(path)
        print(f"  {order} -> {path}")

    print(f"\nDone. Open the 6 HTML files in {args.output_dir}/ and pick the correct one.")


if __name__ == "__main__":
    main()
