#!/usr/bin/env python3
# dlp_prior_kp_only.py
import os, json, argparse, math, random
import numpy as np
import torch
import torch.nn as nn

# -- repo deps
from utils.util_func import get_config, prepare_logdir, log_line
from datasets.point_cloud_datasets.get_dataset import get_point_cloud_dataset
from datasets.voxelize_ds_wrapper import VoxelizedDataset
from modules.modules import VoxelPatcher, Encoder, AlternativeSpatialSoftmaxKP3D

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# ------------------------- KMeans (hard) -------------------------

def kmeans_hard(x, K, init_centers=None, iters=50, tol=1e-4):
    """
    x: [N,3] points in (-1,1) global coords
    K: number of clusters
    init_centers: [M,3] (if provided). If M!=K, downselect via KMeans++ to K.
    """
    device = x.device
    x = x.float()
    if init_centers is None:
        # KMeans++ from data
        idx0 = torch.randint(0, x.shape[0], (1,), device=device)
        centers = [x[idx0]]
        while len(centers) < K:
            C = torch.cat(centers, dim=0)  # [k,3]
            d2 = torch.cdist(x, C, p=2).pow(2).min(dim=1).values
            probs = d2 / (d2.sum() + 1e-9)
            idx = torch.multinomial(probs, 1)
            centers.append(x[idx])
        centers = torch.cat(centers, dim=0)
    else:
        C0 = init_centers.float().to(device)
        # reduce/expand to exactly K via KMeans++ over the init set
        if C0.shape[0] != K:
            idx0 = torch.randint(0, C0.shape[0], (1,), device=device)
            centers = [C0[idx0]]
            while len(centers) < K:
                C = torch.cat(centers, dim=0)
                d2 = torch.cdist(C0, C, p=2).pow(2).min(dim=1).values
                probs = d2 / (d2.sum() + 1e-9)
                idx = torch.multinomial(probs, 1)
                centers.append(C0[idx])
            centers = torch.cat(centers, dim=0)
        else:
            centers = C0.clone()

    for _ in range(iters):
        d2 = torch.cdist(x, centers, p=2).pow(2)
        assign = d2.argmin(dim=1)
        new_centers = torch.zeros_like(centers)
        for k in range(K):
            m = (assign == k)
            if m.any():
                new_centers[k] = x[m].mean(dim=0)
            else:
                j = torch.randint(0, x.shape[0], (1,), device=device)
                new_centers[k] = x[j]
        shift = (new_centers - centers).norm(dim=-1).mean()
        centers = new_centers
        if shift.item() < tol:
            break
    return centers

# ------------------------- DLPPriorKPOnly -------------------------

class DLPPriorKPOnly(nn.Module):
    """
    Minimal prior that ONLY produces keypoints from voxels via two strategies:
      1) "kmeans_from_centers": cluster voxel patch centers with KMeans (init = all centers)
      2) "ssm_raw": run SSM directly on per-patch features (no probs), map to global
    """
    def __init__(self,
                 cdim=1,
                 volume_size=64,
                 patch_size=16,
                 kp_range=(-1,1),
                 n_kp_prior=64,
                 n_kp_per_patch=1,
                 kp_mode="kmeans_from_centers",  # or "ssm_raw"
                 use_patch_mass_weight=False,
                 # encoder knobs (only used in ssm_raw path)
                 ch_mult=(1,2,3), base_ch=32, num_res_blocks=2,
                 use_resblock=False, cnn_mid_blocks=False,
                 init_conv_layers=True, init_conv_fg_std=0.02, init_zero_bias=True,
                 ):
        super().__init__()
        if isinstance(volume_size, int):
            D=H=W=int(volume_size)
        else:
            D,H,W = map(int, volume_size)
        self.D,self.H,self.W = D,H,W
        self.volume_size = (D,H,W)
        self.cdim = int(cdim)
        self.patch_size = int(patch_size)
        self.kp_range = tuple(kp_range)
        self.n_kp_prior = int(n_kp_prior)
        self.n_kp_per_patch = int(n_kp_per_patch)
        self.kp_mode = str(kp_mode)
        self.use_patch_mass_weight = bool(use_patch_mass_weight)

        # patcher
        self.patcher = VoxelPatcher(cdim=cdim, volume_size=(D,H,W), patch_size=self.patch_size)
        pd,ph,pw = self.patcher.pd, self.patcher.ph, self.patcher.pw
        self.nz, self.ny, self.nx = D//pd, H//ph, W//pw
        self.num_patches = self.nx*self.ny*self.nz

        # precompute centers in index space (x,y,z)
        origins = self._precompute_patch_origins_xyz(W,H,D,pw,ph,pd)  # [N,3] in idx units (x,y,z)
        centers = origins + torch.tensor([(pw-1)/2.0, (ph-1)/2.0, (pd-1)/2.0], dtype=torch.float32).view(1,3)
        self.register_buffer("patch_origins_xyz_idx", origins)  # idx coords
        self.register_buffer("patch_centers_xyz_idx", centers)
        self.register_buffer("size_minus1", torch.tensor([W-1, H-1, D-1], dtype=torch.float32))

        # encoder + SSM only needed for ssm_raw mode
        if self.kp_mode == "ssm_raw":
            attn_res = [max(self.patch_size // 16, 1)]
            self.enc = Encoder(ch=base_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_res, dropout=0.0, resamp_with_conv=True, in_channels=cdim,
                               resolution=self.patch_size, z_channels=self.n_kp_per_patch, double_z=False,
                               padding_mode="replicate", residual=use_resblock, mid_blocks=cnn_mid_blocks)
            self.ssm = AlternativeSpatialSoftmaxKP3D(kp_range=self.kp_range)
            self._init_weights(init_conv_layers, init_conv_fg_std, init_zero_bias)

    @staticmethod
    def _precompute_patch_origins_xyz(W,H,L,pw,ph,pl):
        xs = torch.arange(0, W, step=pw, dtype=torch.float32)
        ys = torch.arange(0, H, step=ph, dtype=torch.float32)
        zs = torch.arange(0, L, step=pl, dtype=torch.float32)
        X,Y,Z = torch.meshgrid(xs, ys, zs, indexing='ij')  # [nx,ny,nz]
        return torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=-1)  # [N,3]

    def _init_weights(self, init_conv_layers, init_conv_fg_std, init_zero_bias):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if init_conv_layers:
                    nn.init.normal_(m.weight, 0.0, init_conv_fg_std)
                if init_zero_bias and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # ----- mapping -----
    def get_patch_centers(self):
        xmin,xmax = self.kp_range
        g01 = self.patch_centers_xyz_idx / self.size_minus1  # [N,3]
        return g01 * (xmax - xmin) + xmin

    def _local_to_global(self, kp_local):
        """
        kp_local: [B,N,K,3] in kp_range -> global kp_range using precomputed origins/size.
        """
        xmin,xmax = self.kp_range
        pw,ph,pd = self.patcher.pw, self.patcher.ph, self.patcher.pd
        scale = torch.tensor([pw-1.0, ph-1.0, pd-1.0], dtype=torch.float32, device=kp_local.device).view(1,1,1,3)
        lk01 = (kp_local - xmin) / (xmax - xmin)
        lk_idx = lk01 * scale
        origins = self.patch_origins_xyz_idx.to(kp_local).view(1,-1,1,3)
        idx_xyz = lk_idx + origins
        size_m1 = self.size_minus1.to(kp_local).view(1,1,1,3)
        g01 = idx_xyz / size_m1
        return g01 * (xmax - xmin) + xmin

    # ----- core -----
    def encode_prior(self, x, k=None):
        """
        x: [B,C,D,H,W] volume in any scale (we don't need normalization).
        Returns:
          kp: [B, M, 3] in global (x,y,z) ∈ kp_range
          meta: dict with useful internals
        """
        assert x.dim() == 5, f"expected [B,C,D,H,W], got {x.shape}"
        B,C,D,H,W = x.shape
        assert (D,H,W) == self.volume_size, f"volume_size mismatch {(D,H,W)} vs {self.volume_size}"

        # patchify: -> [B, C, N, pd, ph, pw]
        patches = self.patcher.vox_to_patches(x).permute(0,2,1,3,4,5).contiguous()
        B,N = patches.shape[:2]

        if self.kp_mode == "kmeans_from_centers":
            # compute optional weights per patch from mass (sum of occupancy/intensity)
            if self.use_patch_mass_weight:
                mass = patches.sum(dim=(2,3,4,5))  # [B,N]
                mass = mass / (mass.max(dim=1, keepdim=True).values + 1e-9)
            else:
                mass = None

            centers = self.get_patch_centers().to(x)  # [N,3]
            K = int(self.n_kp_prior if k is None else k)
            kps_out = []
            for b in range(B):
                if mass is not None:
                    # repeat centers by weight (discrete sampling) to bias clustering
                    M = 4096
                    probs = (mass[b] + 1e-9) / (mass[b].sum() + 1e-9)
                    idx = torch.multinomial(probs, num_samples=min(M, N), replacement=True)
                    x_pts = centers[idx]
                else:
                    x_pts = centers
                kps = kmeans_hard(x_pts, K=K, init_centers=centers, iters=50)
                kps_out.append(kps)
            kp = torch.stack(kps_out, dim=0)  # [B,K,3]
            meta = {"mode":"kmeans_from_centers", "K":K, "centers_init":centers}
            return kp, meta

        elif self.kp_mode == "ssm_raw":
            # encode each patch and run spatial softmax (no activation/probs)
            pd,ph,pw = self.patcher.pd, self.patcher.ph, self.patcher.pw
            enc_in = patches.view(B*N, C, pd, ph, pw)
            enc_out = self.enc(enc_in)
            z = enc_out[1] if isinstance(enc_out, tuple) else enc_out  # [B*N, K, pd,ph,pw]
            kp_local, cov_local = self.ssm(z, probs=False, variance=True)  # [B*N,K,3], [B*N,K,3,3]
            kp_local = kp_local.view(B, N, self.n_kp_per_patch, 3)
            kp_global = self._local_to_global(kp_local).view(B, -1, 3)     # flatten patches
            meta = {"mode":"ssm_raw", "per_patch": self.n_kp_per_patch}
            return kp_global, meta

        else:
            raise ValueError(f"Unknown kp_mode={self.kp_mode}")

# ------------------------- Viz helper -------------------------
def save_plot(out_html, vol, kps, iso=0.4, title="KP viz"):
    if not PLOTLY_OK:
        return
    vol_np = vol.detach().cpu().numpy()
    D,H,W = vol_np.shape
    xs,ys,zs = np.linspace(-1,1,W), np.linspace(-1,1,H), np.linspace(-1,1,D)
    X,Y,Z = np.meshgrid(xs,ys,zs, indexing='xy')
    surf = go.Isosurface(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=vol_np.transpose(2,1,0).flatten(),
        isomin=iso, isomax=0.95, surface_count=2, opacity=0.5,
        caps=dict(x_show=False,y_show=False,z_show=False),
        name="volume"
    )
    data = [surf]
    if kps is not None and kps.numel() > 0:
        P = kps.detach().cpu().numpy()
        data.append(go.Scatter3d(x=P[:,0], y=P[:,1], z=P[:,2],
                                 mode='markers', name='kp',
                                 marker=dict(size=6, symbol='diamond')))
    fig = go.Figure(data=data); fig.update_layout(title=title, scene_aspectmode='data', width=1000, height=800)
    fig.write_html(out_html)

# ------------------------- Tiny runner (optional) -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d","--dataset", type=str, default="voxel")
    ap.add_argument("--mode", type=str, default="kmeans_from_centers", choices=["kmeans_from_centers","ssm_raw"])
    ap.add_argument("--iso", type=float, default=0.4)
    args = ap.parse_args()

    conf_path = args.dataset if args.dataset.endswith(".json") else os.path.join("./configs", f"{args.dataset}.json")
    cfg = get_config(conf_path)

    device = torch.device(cfg.get("device","cuda") if torch.cuda.is_available() else "cpu")
    run_name = f"{cfg['ds']}_kp_only_{args.mode}"+cfg.get("run_prefix","")
    log_dir = prepare_logdir(runname=run_name, src_dir="./logs")
    figs = os.path.join(log_dir, "figures"); os.makedirs(figs, exist_ok=True)
    saves = os.path.join(log_dir, "saves");   os.makedirs(saves, exist_ok=True)

    # voxel ds
    base_ds = get_point_cloud_dataset(cfg["ds"], cfg["root"], mode="train", max_points=4096, include_rgb=True)
    sample = base_ds[0]
    vol = sample["voxels"]
    if vol.ndim == 4: vol = vol[0]
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6)
    D,H,W = vol.shape

    prior = DLPPriorKPOnly(
        cdim=1, volume_size=(D,H,W),
        patch_size=int(cfg.get("patch_size",16)),
        kp_range=tuple(cfg.get("kp_range",(-1,1))),
        n_kp_prior=int(cfg.get("n_kp_prior",64)),
        n_kp_per_patch=int(cfg.get("n_kp",1)),
        kp_mode=args.mode,
        use_patch_mass_weight=bool(cfg.get("kp_kmeans_use_mass", False)),
        ch_mult=tuple(cfg.get("obj_ch_mult",(1,2,3))),
        base_ch=int(cfg.get("obj_base_ch",32)),
        num_res_blocks=int(cfg.get("num_res_blocks",2)),
        use_resblock=bool(cfg.get("use_resblock",False)),
        cnn_mid_blocks=bool(cfg.get("cnn_mid_blocks",False)),
        init_conv_layers=bool(cfg.get("init_conv_layers",True)),
        init_conv_fg_std=float(cfg.get("init_conv_fg_std",0.02)),
        init_zero_bias=bool(cfg.get("init_zero_bias",True)),
    ).to(device)

    x = vol.unsqueeze(0).unsqueeze(0).to(device)  # [1,1,D,H,W]
    with torch.no_grad():
        kp, meta = prior.encode_prior(x)

    with open(os.path.join(saves, "kps.json"), "w") as f:
        json.dump(kp[0].detach().cpu().numpy().tolist(), f, indent=2)

    if PLOTLY_OK:
        save_plot(os.path.join(figs, "viz.html"), vol, kp[0], iso=float(cfg.get("viz_iso",0.4)),
                  title=f"KP ({args.mode})")

    log_line(log_dir, f"mode={args.mode}, kp={kp.shape}, vol={tuple(vol.shape)}")
    print("[done]")
    print(f"  KP JSON  : {os.path.join(saves, 'kps.json')}")
    if PLOTLY_OK:
        print(f"  HTML viz : {os.path.join(figs, 'viz.html')}")

if __name__ == "__main__":
    main()
