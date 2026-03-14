"""
Modules for DLP
"""
# imports
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Beta
from utils.util_func import reparameterize, spatial_transform, create_masks_fast, create_masks_with_scale, \
    modulate
# modules
from modules.vision_modules import Encoder, Decoder

"""
Basic Modules
"""

class AlternativeSpatialSoftmaxKP3D(torch.nn.Module):
    """
    3D spatial-softmax over voxel heatmaps.

    Inputs:
      heatmap: [B, K, D, H, W]  (D=z depth, H=y, W=x)

    Returns:
      if variance=False:
         kp: [B, K, 3]                    # (x, y, z) in kp_range
      if variance=True:
         (kp, cov):                       # full covariance
             kp  [B, K, 3]                # (x, y, z)
             cov [B, K, 3, 3]             # cov[i] symmetric PSD
      if probs=True (in addition to above when requested):
         sm_x [B, K, W], sm_y [B, K, H], sm_z [B, K, D]
    """
    def __init__(self, kp_range=(-1, 1)):
        super().__init__()
        self.kp_range = kp_range

    def forward(self, heatmap, probs: bool=False, variance: bool=False):
        B, K, D, H, W = heatmap.shape

        # Softmax over all voxels
        logits = heatmap.view(B, K, -1)                 # [B, K, D*H*W]
        scores = torch.softmax(logits, dim=-1)
        scores = scores.view(B, K, D, H, W)             # [B, K, D, H, W]

        # Axes in the target range
        x_axis = torch.linspace(self.kp_range[0], self.kp_range[1], W, device=heatmap.device, dtype=heatmap.dtype)
        y_axis = torch.linspace(self.kp_range[0], self.kp_range[1], H, device=heatmap.device, dtype=heatmap.dtype)
        z_axis = torch.linspace(self.kp_range[0], self.kp_range[1], D, device=heatmap.device, dtype=heatmap.dtype)

        # Per-axis marginals
        sm_x = scores.sum(dim=(2, 3))      # [B, K, W]  sum over z,y
        sm_y = scores.sum(dim=(2, 4))      # [B, K, H]  sum over z,x
        sm_z = scores.sum(dim=(3, 4))      # [B, K, D]  sum over y,x

        # Expectations (E[x], E[y], E[z])
        Ex = (sm_x * x_axis.view(1, 1, W)).sum(dim=-1)  # [B, K]
        Ey = (sm_y * y_axis.view(1, 1, H)).sum(dim=-1)  # [B, K]
        Ez = (sm_z * z_axis.view(1, 1, D)).sum(dim=-1)  # [B, K]

        kp = torch.stack([Ex, Ey, Ez], dim=-1)          # [B, K, 3]  -> (x, y, z)

        if not variance:
            if probs:
                return kp, sm_x, sm_y, sm_z
            return kp

        # Second moments E[x^2], E[y^2], E[z^2]
        Ex2 = (scores * (x_axis.view(1,1,1,1,W) ** 2)).sum(dim=(2,3,4))  # [B, K]
        Ey2 = (scores * (y_axis.view(1,1,1,H,1) ** 2)).sum(dim=(2,3,4))  # [B, K]
        Ez2 = (scores * (z_axis.view(1,1,D,1,1) ** 2)).sum(dim=(2,3,4))  # [B, K]

        # Cross-moments E[xy], E[xz], E[yz]
        Exy = (scores * (x_axis.view(1,1,1,1,W) * y_axis.view(1,1,1,H,1))).sum(dim=(2,3,4))  # [B, K]
        Exz = (scores * (x_axis.view(1,1,1,1,W) * z_axis.view(1,1,D,1,1))).sum(dim=(2,3,4))  # [B, K]
        Eyz = (scores * (y_axis.view(1,1,1,H,1) * z_axis.view(1,1,D,1,1))).sum(dim=(2,3,4))  # [B, K]

        # Covariances: Cov[a,b] = E[ab] - E[a]E[b]
        var_x = (Ex2 - Ex**2).clamp_min(1e-6)
        var_y = (Ey2 - Ey**2).clamp_min(1e-6)
        var_z = (Ez2 - Ez**2).clamp_min(1e-6)
        cov_xy = Exy - Ex*Ey
        cov_xz = Exz - Ex*Ez
        cov_yz = Eyz - Ey*Ez

        # Assemble symmetric covariance matrix [B, K, 3, 3]
        cov = torch.zeros(B, K, 3, 3, device=heatmap.device, dtype=heatmap.dtype)
        cov[:, :, 0, 0] = var_x
        cov[:, :, 1, 1] = var_y
        cov[:, :, 2, 2] = var_z
        cov[:, :, 0, 1] = cov[:, :, 1, 0] = cov_xy
        cov[:, :, 0, 2] = cov[:, :, 2, 0] = cov_xz
        cov[:, :, 1, 2] = cov[:, :, 2, 1] = cov_yz

        if probs:
            return kp, cov, sm_x, sm_y, sm_z
        return kp, cov
    
class FeatureKMeansRGB(nn.Module):
    """
    RGB-aware k-means prior:
      - Build per-voxel color features (Lab or ILR), optional XYZ appending
      - Prefilter by saliency (L*, alpha, or ||rgb||)
      - k-means in feature space, then compute μ,Σ in XYZ space per cluster
    Returns per-batch keypoints in global [-1,1] coords + covariances and meta.
    """
    def __init__(
        self,
        K: int,
        feat_mode: str = "lab",          # {"lab","ilr"}
        append_xyz: bool = False,
        saliency: str = "L",             # {"L","alpha","rgbnorm"}
        keep_top: int = 80_000,
        sample_m: int = 50_000,
        iters: int = 30,
        tol: float = 1e-4,
        ridge: float = 1e-4,
    ):
        super().__init__()
        self.K = int(K)
        self.feat_mode = feat_mode
        self.append_xyz = append_xyz
        self.saliency = saliency
        self.keep_top = int(keep_top)
        self.sample_m = int(sample_m)
        self.iters = int(iters)
        self.tol = float(tol)
        self.ridge = float(ridge)

        self.SRGB_A = 0.055
        self.SRGB_GAMMA = 2.4
        self.SRGB_LINEAR_THRESH = 0.04045
        self.SRGB_LINEAR_SCALE = 12.92
        self.SRGB2XYZ = torch.tensor([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ], dtype=torch.float32)
        self.D65_WHITE = torch.tensor([0.95047, 1.00000, 1.08883], dtype=torch.float32)
        self.DELTA = 6.0 / 29.0
        self.LAB_EPS = self.DELTA ** 3
        self.LAB_K = 1.0 / (3 * (self.DELTA ** 2))
        self.LAB_C = 4.0 / 29.0
        self.ILR_S1 = 1.0 / torch.sqrt(torch.tensor(2.0))
        self.ILR_S2 = 1.0 / torch.sqrt(torch.tensor(6.0))

    def srgb_to_linear(self, c):
        return torch.where(
            c <= self.SRGB_LINEAR_THRESH,
            c / self.SRGB_LINEAR_SCALE,
            ((c + self.SRGB_A) / (1.0 + self.SRGB_A)) ** self.SRGB_GAMMA
        )

    def rgb_to_xyz(self, rgb):  # [B,3,D,H,W] linear
        M = self.SRGB2XYZ.to(rgb.device, rgb.dtype)
        return torch.einsum("cd, b d... -> b c...", M, rgb)

    def xyz_to_lab(self, xyz, eps=1e-9):  # [B,3,D,H,W]
        Xn, Yn, Zn = self.D65_WHITE.to(xyz.device, xyz.dtype).unbind(0)
        x = xyz[:,0] / (Xn + eps); y = xyz[:,1] / (Yn + eps); z = xyz[:,2] / (Zn + eps)
        def f(t): return torch.where(t > self.LAB_EPS, t.pow(1/3), self.LAB_K * t + self.LAB_C)
        fx, fy, fz = f(x), f(y), f(z)
        L = 116.0 * fy - 16.0
        a = 500.0 * (fx - fy)
        b = 200.0 * (fy - fz)
        return torch.stack([L, a, b], dim=1)

    def rgb_to_lab_srgb(self, rgb01):        # [B,3,D,H,W] in [0,1]
        rgb_lin = self.srgb_to_linear(rgb01.clamp(0,1))
        return self.xyz_to_lab(self.rgb_to_xyz(rgb_lin))

    def rgb_to_ilr2(self, rgb01, eps=1e-6):  # [B,3,D,H,W] in [0,1]
        R,G,B = rgb01[:,0].clamp_min(0.0), rgb01[:,1].clamp_min(0.0), rgb01[:,2].clamp_min(0.0)
        S = (R+G+B).clamp_min(eps)
        r = R/S; g = G/S; b = B/S
        u1 = self.ILR_S1 * (torch.log(r+eps) - torch.log(g+eps))
        u2 = self.ILR_S2 * (torch.log(r+eps) + torch.log(g+eps) - 2.0*torch.log(b+eps))
        return torch.stack([u1, u2], dim=1)

    def whiten(self, X, eps=1e-6):           # [N,D]
        mu = X.mean(dim=0, keepdim=True)
        sd = X.std(dim=0, keepdim=True).clamp_min(eps)
        return (X - mu) / sd

    def build_xyz_grid(self, D,H,W, device, dtype):
        z = torch.linspace(-1, 1, steps=D, device=device, dtype=dtype)
        y = torch.linspace(-1, 1, steps=H, device=device, dtype=dtype)
        x = torch.linspace(-1, 1, steps=W, device=device, dtype=dtype)
        Z,Y,X = torch.meshgrid(z,y,x, indexing="ij")
        return torch.stack([X,Y,Z], dim=0)  # [3,D,H,W]

    def _batched_kmeans_pp_init(self, X, K):
        """
        Batched KMeans++ init.
        X: [B, N, D]
        Returns: C [B, K, D]
        """
        B, N, D = X.shape
        device = X.device

        # Pick first center randomly
        i0 = torch.randint(0, N, (B,), device=device)  # [B]
        C = X[torch.arange(B, device=device), i0].unsqueeze(1)  # [B, 1, D]

        min_d2 = torch.full((B, N), float('inf'), device=device)
        for j in range(1, K):
            # Distance to latest center
            d2_new = (X - C[:, j-1:j, :]).pow(2).sum(dim=-1)  # [B, N]
            min_d2 = torch.minimum(min_d2, d2_new)
            probs = (min_d2 + 1e-12) / (min_d2.sum(dim=1, keepdim=True) + 1e-12)
            idx = torch.multinomial(probs, 1)  # [B, 1]
            new_c = X[torch.arange(B, device=device).unsqueeze(1), idx]  # [B, 1, D]
            C = torch.cat([C, new_c], dim=1)
        return C  # [B, K, D]

    def _batched_kmeans(self, X, K, iters=30, tol=1e-4):
        """
        Fully batched KMeans.
        X: [B, N, D]
        Returns: C [B, K, D], A [B, N]
        """
        B, N, D = X.shape
        device = X.device

        C = self._batched_kmeans_pp_init(X, K)  # [B, K, D]

        for _ in range(iters):
            # Assign: [B, N, K] -> [B, N]
            A = torch.cdist(X, C).argmin(dim=2)  # [B, N]

            # Update centers via scatter
            A_exp = A.unsqueeze(2).expand(-1, -1, D)  # [B, N, D]
            sums = torch.zeros(B, K, D, device=device, dtype=X.dtype)
            sums.scatter_add_(1, A_exp, X)
            counts = torch.zeros(B, K, 1, device=device, dtype=X.dtype)
            counts.scatter_add_(1, A.unsqueeze(2), torch.ones(B, N, 1, device=device, dtype=X.dtype))
            empty = (counts.squeeze(2) == 0)  # [B, K]
            Cn = sums / counts.clamp_min(1)
            Cn[empty] = C[empty]

            shift = (Cn - C).norm(dim=2).mean()
            C = Cn
            if shift < tol:
                break

        A = torch.cdist(X, C).argmin(dim=2)
        return C, A

    @torch.no_grad()
    def forward(self, x, centers_init_global=None):
        """
        x:     [B,C,D,H,W] (expects RGB in x[:,:3], range [0,1] or [-1,1] OK)
        centers_init_global: [Npatch,3] in [-1,1] (optional, for compatibility)

        Returns:
          kp:   [B,K,3] in global [-1,1] (x,y,z)
          cov:  [B,K,3,3]
          meta: dict with "cluster_mass" (B,K), "cluster_eff_count" (B,K), etc.
        """
        B,C,D,H,W = x.shape
        device, dtype = x.device, x.dtype
        K = self.K

        # RGB in [0,1]
        RGB = x[:, :3]
        if RGB.min() < 0: RGB = (RGB + 1.0) * 0.5
        RGB = RGB.clamp(0,1)

        # features
        if self.feat_mode == "lab":
            feat = self.rgb_to_lab_srgb(RGB)            # [B,3,D,H,W]
        elif self.feat_mode == "ilr":
            feat = self.rgb_to_ilr2(RGB)                # [B,2,D,H,W]
        else:
            raise ValueError(f"feat_mode={self.feat_mode}")

        if self.append_xyz:
            XYZ = self.build_xyz_grid(D,H,W, device, dtype).unsqueeze(0).expand(B,-1,-1,-1,-1)
            feat = torch.cat([feat, XYZ], dim=1)    # [B,Cf(+3),D,H,W]

        Cf = feat.shape[1]

        # saliency
        if self.saliency == "L":
            if self.feat_mode == "lab":
                sal = feat[:,0]                     # [B,D,H,W]
            else:
                sal = self.rgb_to_lab_srgb(RGB)[:,0]
        else:
            sal = torch.linalg.norm(RGB, dim=1)

        # flat xyz grid
        xyz_flat = self.build_xyz_grid(D,H,W, device, dtype).view(3, -1)  # [3,DHW]

        # ---- Batched topk + feature extraction ----
        sal_flat = sal.reshape(B, -1)  # [B, DHW]
        k_keep = min(self.keep_top, sal_flat.shape[1])
        top_vals, top_idx = torch.topk(sal_flat, k=k_keep, largest=True, sorted=False)  # [B, k_keep]
        wts = top_vals.clamp_min(0)  # [B, k_keep]

        # Gather features at top indices: [B, Cf, k_keep]
        feat_flat = feat.reshape(B, Cf, -1)  # [B, Cf, DHW]
        top_idx_exp = top_idx.unsqueeze(1).expand(-1, Cf, -1)  # [B, Cf, k_keep]
        X_all = torch.gather(feat_flat, 2, top_idx_exp).permute(0, 2, 1)  # [B, k_keep, Cf]

        # Per-sample whitening (batched)
        mu_w = X_all.mean(dim=1, keepdim=True)  # [B, 1, Cf]
        sd_w = X_all.std(dim=1, keepdim=True).clamp_min(1e-6)  # [B, 1, Cf]
        X_all = (X_all - mu_w) / sd_w  # [B, k_keep, Cf]

        # Batched importance subsampling
        m = min(self.sample_m, k_keep)
        probs = (wts + 1e-9) / (wts.sum(dim=1, keepdim=True) + 1e-9)  # [B, k_keep]
        sel = torch.multinomial(probs, num_samples=m, replacement=True)  # [B, m]
        sel_exp = sel.unsqueeze(2).expand(-1, -1, Cf)  # [B, m, Cf]
        X_sub = torch.gather(X_all, 1, sel_exp)  # [B, m, Cf]

        # ---- Batched KMeans ----
        C_feat, _ = self._batched_kmeans(X_sub, K, iters=self.iters, tol=self.tol)  # [B, K, Cf]

        # Re-assign all kept voxels
        d2 = torch.cdist(X_all, C_feat).pow(2)  # [B, k_keep, K]
        assign = d2.argmin(dim=2)  # [B, k_keep]

        # Gather xyz for top indices
        pts_xyz = xyz_flat[:, top_idx.reshape(-1)].t().reshape(B, k_keep, 3)  # [B, k_keep, 3]
        # (xyz_flat is shared across batch — index with flat top_idx per sample)
        # Fix: need per-sample indexing
        pts_xyz = xyz_flat.unsqueeze(0).expand(B, -1, -1)  # [B, 3, DHW]
        top_idx_3 = top_idx.unsqueeze(1).expand(-1, 3, -1)  # [B, 3, k_keep]
        pts_xyz = torch.gather(pts_xyz, 2, top_idx_3).permute(0, 2, 1)  # [B, k_keep, 3]

        N_pts = k_keep

        # ---- Batched cluster stats ----
        # d2 to assigned center
        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, N_pts)  # [B, N_pts]
        n_idx = torch.arange(N_pts, device=device).unsqueeze(0).expand(B, -1)  # [B, N_pts]
        d2_to_center = d2[b_idx, n_idx, assign]  # [B, N_pts]

        # Per-cluster tau (median of sqrt(d2)) — batched via sort
        sqrt_d2 = d2_to_center.sqrt()  # [B, N_pts]
        sort_key = assign.float() * (sqrt_d2.max(dim=1, keepdim=True).values + 1) + sqrt_d2
        order = sort_key.argsort(dim=1)
        sorted_sqrt_d2 = torch.gather(sqrt_d2, 1, order)  # [B, N_pts]

        counts = torch.zeros(B, K, device=device, dtype=torch.long)
        counts.scatter_add_(1, assign, torch.ones(B, N_pts, device=device, dtype=torch.long))
        offsets = torch.zeros(B, K, device=device, dtype=torch.long)
        offsets[:, 1:] = counts[:, :-1].cumsum(dim=1)
        median_idx = (offsets + counts // 2).clamp(max=N_pts - 1)  # [B, K]
        tau_per_k = torch.gather(sorted_sqrt_d2, 1, median_idx).clamp_min(1e-9)  # [B, K]

        tau_pt = torch.gather(tau_per_k, 1, assign)  # [B, N_pts]
        w_soft = torch.exp(-d2_to_center / (tau_pt * tau_pt))
        w_eff = (wts ** 1.5) * w_soft  # [B, N_pts]

        # Weighted mean (centers)
        Wsum = torch.zeros(B, K, device=device, dtype=dtype)
        Wsum.scatter_add_(1, assign, w_eff)
        Wsum_safe = Wsum.clamp_min(1e-12)  # [B, K]

        w_pts = w_eff.unsqueeze(2) * pts_xyz  # [B, N_pts, 3]
        mu_sum = torch.zeros(B, K, 3, device=device, dtype=dtype)
        mu_sum.scatter_add_(1, assign.unsqueeze(2).expand(-1, -1, 3), w_pts)
        mu = mu_sum / Wsum_safe.unsqueeze(2)  # [B, K, 3]

        # Weighted covariance
        mu_gathered = torch.gather(mu, 1, assign.unsqueeze(2).expand(-1, -1, 3))  # [B, N_pts, 3]
        xc = pts_xyz - mu_gathered  # [B, N_pts, 3]
        wxc = w_eff.unsqueeze(2) * xc  # [B, N_pts, 3]
        outer = wxc.unsqueeze(3) * xc.unsqueeze(2)  # [B, N_pts, 3, 3]
        cov_sum = torch.zeros(B, K, 3, 3, device=device, dtype=dtype)
        cov_sum.scatter_add_(1, assign.view(B, -1, 1, 1).expand(-1, -1, 3, 3), outer)
        cov_mat = cov_sum / Wsum_safe.view(B, K, 1, 1)
        eye3 = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3)  # [1, 1, 3, 3]
        cov_mat = cov_mat + eye3 * (self.ridge / Wsum_safe.view(B, K, 1, 1))

        # Effective count per cluster
        wts_max = wts.max(dim=1, keepdim=True).values  # [B, 1]
        significant = (wts > wts_max * 1e-3).float()  # [B, N_pts]
        eff_count = torch.zeros(B, K, device=device, dtype=dtype)
        eff_count.scatter_add_(1, assign, significant)

        # Fix low-count clusters
        low_mask = (eff_count < 8)  # [B, K]
        if low_mask.any():
            var_iso = cov_mat.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-8)  # [B, K]
            iso_cov = eye3 * var_iso.view(B, K, 1, 1)
            cov_mat = torch.where(low_mask.view(B, K, 1, 1), iso_cov, cov_mat)

        # Fix empty clusters
        empty = (Wsum < 1e-12)  # [B, K]
        if empty.any():
            global_mean = pts_xyz.mean(dim=1, keepdim=True).expand(-1, K, -1)  # [B, K, 3]
            mu = torch.where(empty.unsqueeze(2), global_mean, mu)
            cov_mat = torch.where(empty.view(B, K, 1, 1), eye3 * self.ridge, cov_mat)

        meta = {
            "mode": f"kmeans_rgb_feat[{self.feat_mode}{'+xyz' if self.append_xyz else ''}]",
            "saliency": self.saliency,
            "K": K, "kept": self.keep_top, "sampled": self.sample_m, "iters": self.iters,
            "cluster_mass": Wsum,       # [B,K]
            "cluster_eff_count": eff_count.int(), # [B,K]
        }
        return mu, cov_mat, meta


class VoxelPatcher(nn.Module):
    """
    3D patcher in canonical PyTorch layout.
    Input:  [B, C, D, H, W]
    Output: [B, C, N, pd, ph, pw]   (non-overlapping tiles)
    """
    def __init__(self, cdim=3, volume_size=(64,64,64), patch_size=(16,16,16)):
        super().__init__()
        self.cdim = cdim
        if isinstance(volume_size, int):
            self.D = self.H = self.W = volume_size
        else:
            self.D, self.H, self.W = map(int, volume_size)
        if isinstance(patch_size, int):
            self.pd = self.ph = self.pw = patch_size
        else:
            self.pd, self.ph, self.pw = map(int, patch_size)

        # strides = patch sizes (no overlap)
        self.dd, self.dh, self.dw = self.pd, self.ph, self.pw
        self.unfold_shape = self.get_unfold_shape()
        self.patch_location_idx = self.get_patch_location_idx()  # [N, 3] with (d,h,w)

    def _num_tiles(self):
        nd = self.D // self.pd
        nh = self.H // self.ph
        nw = self.W // self.pw
        return nd, nh, nw

    def get_unfold_shape(self):
        nd, nh, nw = self._num_tiles()
        if (self.D % self.pd) or (self.H % self.ph) or (self.W % self.pw):
            raise ValueError(
                f"Volume dims (D={self.D}, H={self.H}, W={self.W}) "
                f"must be divisible by (pd={self.pd}, ph={self.ph}, pw={self.pw})."
            )
        return (self.cdim, nd, nh, nw, self.pd, self.ph, self.pw)

    def get_patch_location_idx(self):
        ds = np.arange(0, self.D, self.pd)
        hs = np.arange(0, self.H, self.ph)
        ws = np.arange(0, self.W, self.pw)
        dd, hh, ww = np.meshgrid(ds, hs, ws, indexing="ij")  # (d,h,w) order
        dhws = np.stack((dd, hh, ww), axis=-1).reshape(-1, 3)
        return torch.tensor(dhws, dtype=torch.int32)

    def get_patch_centers(self):
        mid = torch.tensor([self.pd//2, self.ph//2, self.pw//2], dtype=torch.int32)
        return self.get_patch_location_idx() + mid

    @staticmethod
    def _ensure_5d_dhw(x: torch.Tensor):
        if x.ndim != 5:
            raise ValueError(f"Expected [B,C,D,H,W], got {tuple(x.shape)}")

    def vox_to_patches(self, x: torch.Tensor) -> torch.Tensor:
        """[B,C,D,H,W] -> [B,C,N,pd,ph,pw]"""
        self._ensure_5d_dhw(x)
        B, C, D, H, W = x.shape
        nd, nh, nw = D // self.pd, H // self.ph, W // self.pw
        if (D % self.pd) or (H % self.ph) or (W % self.pw):
            raise ValueError(f"Input dims not divisible by patch size: (D={D},H={H},W={W}) vs (pd={self.pd},ph={self.ph},pw={self.pw})")

        # [B, C, nd, pd, nh, ph, nw, pw]
        x = x.view(B, C, nd, self.pd, nh, self.ph, nw, self.pw)
        # -> [B, C, nd, nh, nw, pd, ph, pw]
        x = x.permute(0, 1, 2, 4, 6, 3, 5, 7).contiguous()
        # -> [B, C, nd*nh*nw, pd, ph, pw]
        x = x.view(B, C, nd * nh * nw, self.pd, self.ph, self.pw)
        return x

    def patches_to_vox(self, patches: torch.Tensor) -> torch.Tensor:
        """[B,C,N,pd,ph,pw] -> [B,C,D,H,W]"""
        if patches.ndim != 6:
            raise ValueError(f"Expected [B,C,N,pd,ph,pw], got {tuple(patches.shape)}")
        B, C, N, pd, ph, pw = patches.shape
        nd, nh, nw = self._num_tiles()
        if N != nd * nh * nw:
            raise ValueError(f"N={N} != nd*nh*nw={nd*nh*nw}")
        if (pd, ph, pw) != (self.pd, self.ph, self.pw):
            raise ValueError(f"Patch size mismatch: {(pd,ph,pw)} vs {(self.pd,self.ph,self.pw)}")

        # [B, C, nd, nh, nw, pd, ph, pw]
        x = patches.view(B, C, nd, nh, nw, pd, ph, pw)
        # -> [B, C, nd, pd, nh, ph, nw, pw]
        x = x.permute(0, 1, 1, 5, 2, 6, 3, 7).contiguous()
        # -> [B, C, D, H, W]
        x = x.view(B, C, nd * pd, nh * ph, nw * pw)
        return x

    def forward(self, x: torch.Tensor, patches: bool = True) -> torch.Tensor:
        return self.vox_to_patches(x) if patches else self.patches_to_vox(x)

"""
Normalization
"""


class ParticleNorm(nn.Module):
    """
    experimental particle normalization module, not used in the code but left here for research
    """

    def __init__(self, particle_dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.particle_dim = particle_dim
        self.a = nn.Parameter(torch.ones(1, 1, 1, self.particle_dim))
        self.g = nn.Parameter(torch.ones(1, 1, 1, self.particle_dim))
        self.s = nn.Parameter(torch.zeros(1, 1, 1, self.particle_dim))

    def forward(self, x):
        # [bs, n_particles, T, dim]
        # if self.particle_dim > 1:
        #     dims = (1, 3)
        # else:
        dims = (1,)
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, unbiased=False, keepdim=True)
        if len(x.shape) == 3:
            d_n = (x - self.a.squeeze(2) * mean) / (var + self.eps).sqrt()
            out = d_n * self.g.squeeze(2) + self.s.squeeze(2)
        else:
            d_n = (x - self.a * mean) / (var + self.eps).sqrt()
            out = d_n * self.g + self.s
        return out


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # F.normalize: x = x / (x ** 2).sum(-1, keepdim=True).sqrt()
        return F.normalize(x, dim=-1) * self.scale * self.g


"""
Attention-based modules
"""


class SimpleRelativePositionalBias(nn.Module):
    # adapted from https://github.com/facebookresearch/mega
    def __init__(self, max_positions, num_heads=1, max_particles=None, layer_norm=False):
        super().__init__()
        self.max_positions = max_positions
        self.num_heads = num_heads
        self.max_particles = max_particles
        self.rel_pos_bias = nn.Parameter(torch.Tensor(2 * max_positions - 1, self.num_heads))
        self.ln_t = nn.LayerNorm([2 * max_positions - 1, self.num_heads]) if layer_norm else nn.Identity()

        if self.max_particles is not None:
            self.particle_rel_pos_bias = nn.Parameter(torch.Tensor(2 * max_particles - 1, self.num_heads))
            self.ln_p = nn.LayerNorm([2 * max_particles - 1, self.num_heads]) if layer_norm else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.rel_pos_bias, mean=0.0, std=std)
        if self.max_particles is not None:
            nn.init.normal_(self.particle_rel_pos_bias, mean=0.0, std=std)

    def get_particle_rel_position(self, num_particles):
        if self.max_particles is None:
            return 0.0
        if num_particles > self.max_particles:
            raise ValueError('Num particles {} going beyond max particles {}'.format(num_particles, self.max_particles))

        # seq_len * 2 -1
        in_ln = self.ln_p(self.particle_rel_pos_bias)
        b = in_ln[(self.max_particles - num_particles):(self.max_particles + num_particles - 1)]
        # seq_len * 3 - 1
        t = F.pad(b, (0, 0, 0, num_particles))
        # (seq_len * 3 - 1) * seq_len
        t = torch.tile(t, (num_particles, 1))
        t = t[:-num_particles]
        # seq_len x (3 * seq_len - 2)
        t = t.view(num_particles, 3 * num_particles - 2, b.shape[-1])
        r = (2 * num_particles - 1) // 2
        start = r
        end = t.size(1) - r
        t = t[:, start:end]  # [seq_len, seq_len, n_heads]
        t = t.permute(2, 0, 1).unsqueeze(0)  # [1, n_heads, seq_len, seq_len]
        return t

    def forward(self, seq_len, num_particles=None):
        if seq_len > self.max_positions:
            raise ValueError('Sequence length {} going beyond max length {}'.format(seq_len, self.max_positions))

        # seq_len * 2 -1
        in_ln = self.ln_t(self.rel_pos_bias)
        b = in_ln[(self.max_positions - seq_len):(self.max_positions + seq_len - 1)]
        # seq_len * 3 - 1
        t = F.pad(b, (0, 0, 0, seq_len))
        # (seq_len * 3 - 1) * seq_len
        t = torch.tile(t, (seq_len, 1))
        t = t[:-seq_len]
        # seq_len x (3 * seq_len - 2)
        t = t.view(seq_len, 3 * seq_len - 2, b.shape[-1])
        r = (2 * seq_len - 1) // 2
        start = r
        end = t.size(1) - r
        t = t[:, start:end]  # [seq_len, seq_len, n_heads]
        t = t.permute(2, 0, 1).unsqueeze(0)  # [1, n_heads, seq_len, seq_len]
        p = None
        if num_particles is not None and self.max_particles is not None:
            p = self.get_particle_rel_position(num_particles)  # [1, n_heads, n_part, n_part]
            t = t[:, :, None, :, None, :]
            p = p[:, :, :, None, :, None]
        return t, p


class CausalParticleSelfAttention(nn.Module):
    """
    A particle-based multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embed, n_head, block_size, attn_pdrop=0.1, resid_pdrop=0.1,
                 positional_bias=False, max_particles=None, linear_bias=False, torch_attn=False):
        super().__init__()
        assert n_embed % n_head == 0
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.torch_attn = torch_attn
        # if self.torch_attn:
        #     self.attn_net = nn.MultiheadAttention(embed_dim=n_embed, num_heads=n_head, dropout=attn_pdrop,
        #                                           bias=linear_bias, batch_first=True)
        #     # key, query, value projections for all heads
        #     self.key = None
        #     self.query = None
        #     self.value = None
        #     # regularization
        #     self.attn_drop = None
        #     # output projection
        #     self.proj = nn.Identity()  # already part of attn_net
        # else:
        # self.attn_net = None
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embed, n_embed, bias=linear_bias)
        self.query = nn.Linear(n_embed, n_embed, bias=linear_bias)
        self.value = nn.Linear(n_embed, n_embed, bias=linear_bias)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop) if not self.torch_attn else nn.Identity()
        # output projection
        self.proj = nn.Linear(n_embed, n_embed, bias=linear_bias)

        self.resid_drop = nn.Dropout(resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, 1, block_size, 1, block_size))
        self.n_head = n_head
        self.positional_bias = positional_bias
        self.max_particles = max_particles
        if self.positional_bias:
            self.rel_pos_bias = SimpleRelativePositionalBias(block_size, n_head, max_particles=max_particles)
        else:
            self.rel_pos_bias = nn.Identity()

    def forward(self, x):
        B, N, T, C = x.size()  # batch size, n_particles, sequence length, embedding dimensionality (n_embd)

        # if self.torch_attn:
        #     if self.positional_bias:
        #         raise NotImplementedError(f'torch_attn: {self.torch_attn}, can not use positional bias')
        #         # if self.max_particles is not None:
        #         #     bias_t, bias_p = self.rel_pos_bias(T, num_particles=N)
        #         #     bias_t = bias_t.view(1, bias_t.shape[1], 1, T, 1, T)
        #         #     bias_p = bias_p.view(1, bias_p.shape[1], N, 1, N, 1)
        #         #     mask = bias_t + bias_p
        #         # else:
        #         #     bias_t, _ = self.rel_pos_bias(T)
        #         #     bias_t = bias_t.view(1, bias_t.shape[1], 1, T, 1, T)
        #         #     mask = bias_t
        #     # For a binary mask, a True value indicates that the corresponding position is not allowed to attend.
        #     mask = self.mask[:, :, :, :T, :, :T] == 0
        #     mask = mask.repeat(B, self.n_head, N, 1, N, 1)
        #     mask = mask.view(B * self.n_head, N * T, N * T)  # (B, nh, N * T, N * T)
        #     x = x.reshape(B, N * T, C)
        #     y, _ = self.attn_net(query=x, key=x, value=x, need_weights=False, attn_mask=mask)
        # else:
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, N * T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, N * T, hs)
        q = self.query(x).view(B, N * T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, N * T, hs)
        v = self.value(x).view(B, N * T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, N * T, hs)

        if self.torch_attn:
            y = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=True,
                                               dropout_p=self.attn_pdrop if self.training else 0.0)

        else:
            # causal self-attention; Self-attend: (B, nh, N * T, hs) x (B, nh, hs, N  *T) -> (B, nh, N * T, N *T )
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, N * T, N * T)
            att = att.view(B, -1, N, T, N, T)  # (B, nh, N, T, N, T)
            if self.positional_bias:
                if self.max_particles is not None:
                    bias_t, bias_p = self.rel_pos_bias(T, num_particles=N)
                    bias_t = bias_t.view(1, bias_t.shape[1], 1, T, 1, T)
                    bias_p = bias_p.view(1, bias_p.shape[1], N, 1, N, 1)
                    att = att + bias_t + bias_p
                else:
                    bias_t, _ = self.rel_pos_bias(T)
                    bias_t = bias_t.view(1, bias_t.shape[1], 1, T, 1, T)
                    att = att + bias_t
            att = att.masked_fill(self.mask[:, :, :, :T, :, :T] == 0, float('-inf'))
            att = att.view(B, -1, N * T, N * T)  # (B, nh, N * T, N * T)
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v  # (B, nh, N*T, N*T) x (B, nh, N*T, hs) -> (B, nh, N*T, hs)

        y = y.transpose(1, 2).contiguous().view(B, N * T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        y = y.view(B, N, T, -1)
        return y


class ParticleSelfAttention(nn.Module):
    """
    A particle-based multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embed, n_head, block_size, attn_pdrop=0.1, resid_pdrop=0.1,
                 positional_bias=False, max_particles=None, linear_bias=False, torch_attn=False):
        super().__init__()
        assert n_embed % n_head == 0
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.torch_attn = torch_attn
        # if self.torch_attn:
        #     self.attn_net = nn.MultiheadAttention(embed_dim=n_embed, num_heads=n_head, dropout=attn_pdrop,
        #                                           bias=linear_bias, batch_first=True)
        #     # key, query, value projections for all heads
        #     self.key = None
        #     self.query = None
        #     self.value = None
        #     # regularization
        #     self.attn_drop = None
        #     # output projection
        #     self.proj = nn.Identity()  # already part of attn_net
        # else:
        # self.attn_net = None
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embed, n_embed, bias=linear_bias)
        self.query = nn.Linear(n_embed, n_embed, bias=linear_bias)
        self.value = nn.Linear(n_embed, n_embed, bias=linear_bias)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop) if not self.torch_attn else nn.Identity()
        # output projection
        self.proj = nn.Linear(n_embed, n_embed, bias=linear_bias)

        self.resid_drop = nn.Dropout(resid_pdrop)
        self.n_head = n_head
        self.positional_bias = positional_bias
        self.max_particles = max_particles
        if self.positional_bias:
            self.rel_pos_bias = SimpleRelativePositionalBias(block_size, n_head, max_particles=max_particles)
        else:
            self.rel_pos_bias = nn.Identity()

    def forward(self, x):
        B, N, T, C = x.size()  # batch size, n_particles, sequence length, embedding dimensionality (n_embd)
        # if self.torch_attn:
        #     if self.positional_bias:
        #         raise NotImplementedError(f'torch_attn: {self.torch_attn}, can not use positional bias')
        #         # if self.max_particles is not None:
        #         #     bias_t, bias_p = self.rel_pos_bias(T, num_particles=N)
        #         #     bias_t = bias_t.view(1, bias_t.shape[1], 1, T, 1, T)
        #         #     bias_p = bias_p.view(1, bias_p.shape[1], N, 1, N, 1)
        #         #     mask = bias_t + bias_p
        #         # else:
        #         #     bias_t, _ = self.rel_pos_bias(T)
        #         #     bias_t = bias_t.view(1, bias_t.shape[1], 1, T, 1, T)
        #         #     mask = bias_t
        #     x = x.reshape(B, N * T, C)
        #     y, _ = self.attn_net(query=x, key=x, value=x, need_weights=False)
        # else:
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, N * T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, N * T, hs)
        q = self.query(x).view(B, N * T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, N * T, hs)
        v = self.value(x).view(B, N * T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, N * T, hs)

        if self.torch_attn:
            y = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=False,
                                               dropout_p=self.attn_pdrop if self.training else 0.0)

        else:
            # causal self-attention; Self-attend: (B, nh, N * T, hs) x (B, nh, hs, N  *T) -> (B, nh, N * T, N *T )
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, N * T, N * T)
            if self.positional_bias:
                att = att.view(B, -1, N, T, N, T)  # (B, nh, N, T, N, T)
                if self.max_particles is not None:
                    bias_t, bias_p = self.rel_pos_bias(T, num_particles=N)
                    bias_t = bias_t.view(1, bias_t.shape[1], 1, T, 1, T)
                    bias_p = bias_p.view(1, bias_p.shape[1], N, 1, N, 1)
                    att = att + bias_t + bias_p
                else:
                    bias_t, _ = self.rel_pos_bias(T)
                    bias_t = bias_t.view(1, bias_t.shape[1], 1, T, 1, T)
                    att = att + bias_t
                att = att.view(B, -1, N * T, N * T)  # (B, nh, N * T, N * T)
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v  # (B, nh, N*T, N*T) x (B, nh, N*T, hs) -> (B, nh, N*T, hs)

        y = y.transpose(1, 2).contiguous().view(B, N * T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        y = y.view(B, N, T, -1)
        return y


class ParticleCrossAttention(nn.Module):
    """
    A particle-based multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embed, n_head, block_size, attn_pdrop=0.1, resid_pdrop=0.1,
                 positional_bias=False, max_particles=None, linear_bias=False, torch_attn=False, particles_first=False):
        super().__init__()
        assert n_embed % n_head == 0
        self.particles_first = particles_first
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.torch_attn = torch_attn
        # if self.torch_attn:
        #     self.attn_net = nn.MultiheadAttention(embed_dim=n_embed, num_heads=n_head, dropout=attn_pdrop,
        #                                           bias=linear_bias, batch_first=True)
        #     # key, query, value projections for all heads
        #     self.key = None
        #     self.query = None
        #     self.value = None
        #     # regularization
        #     self.attn_drop = None
        #     # output projection
        #     self.proj = nn.Identity()  # already part of attn_net
        # else:
        # self.attn_net = None
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embed, n_embed, bias=linear_bias)
        self.query = nn.Linear(n_embed, n_embed, bias=linear_bias)
        self.value = nn.Linear(n_embed, n_embed, bias=linear_bias)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop) if not self.torch_attn else nn.Identity()
        # output projection
        self.proj = nn.Linear(n_embed, n_embed, bias=linear_bias)

        self.resid_drop = nn.Dropout(resid_pdrop)
        self.n_head = n_head
        self.positional_bias = positional_bias
        self.max_particles = max_particles
        if self.positional_bias:
            self.rel_pos_bias = SimpleRelativePositionalBias(block_size, n_head, max_particles=max_particles)
        else:
            self.rel_pos_bias = nn.Identity()

    def forward(self, x_q, x_kv):
        if self.particles_first:
            B, Nq, Tq, Cq = x_q.size()  # batch size, n_particles, sequence length, embedding dimensionality (n_embd)
            _, Nkv, Tkv, Ckv = x_kv.size()  # batch size, n_particles, sequence length, embedding dimensionality (n_embd)
        else:
            B, Tq, Nq, Cq = x_q.size()  # batch size, n_particles, sequence length, embedding dimensionality (n_embd)
            _, Tkv, Nkv, Ckv = x_kv.size()  # batch size, n_particles, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x_kv).view(B, Nkv * Tkv, self.n_head, Ckv // self.n_head).transpose(1, 2)  # (B, nh, N * T, hs)
        q = self.query(x_q).view(B, Nq * Tq, self.n_head, Cq // self.n_head).transpose(1, 2)  # (B, nh, N * T, hs)
        v = self.value(x_kv).view(B, Nkv * Tkv, self.n_head, Ckv // self.n_head).transpose(1,
                                                                                           2)  # (B, nh, N * T, hs)

        if self.torch_attn:
            y = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=False,
                                               dropout_p=self.attn_pdrop if self.training else 0.0)
        else:

            # causal self-attention; Self-attend: (B, nh, N * T, hs) x (B, nh, hs, N  *T) -> (B, nh, N * T, N *T )
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, N * T, N * T)
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v  # (B, nh, N*T, N*T) x (B, nh, N*T, hs) -> (B, nh, N*T, hs)

        y = y.transpose(1, 2).contiguous().view(B, Nq * Tq, Ckv)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        if self.particles_first:
            y = y.view(B, Nq, Tq, -1)
        else:
            y = y.view(B, Tq, Nq, -1)
        return y


class MLP(nn.Module):
    def __init__(self, n_embed, resid_pdrop=0.1, hidden_dim_multiplier=4, activation='gelu'):
        super().__init__()
        self.fc_1 = nn.Linear(n_embed, hidden_dim_multiplier * n_embed)
        if activation == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU(True)
        self.proj = nn.Linear(hidden_dim_multiplier * n_embed, n_embed)
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, x):
        x = self.dropout(self.proj(self.act(self.fc_1(x))))
        return x


class FinalTransformerLayer(nn.Module):
    def __init__(self, n_embed, output_dim, bias=True, context_cond=False, residual_modulation=False, norm_type='rms'):
        super().__init__()
        if norm_type == 'rms':
            norm_layer = RMSNorm
        else:
            norm_layer = nn.LayerNorm
        self.norm = norm_layer(n_embed)
        self.head = nn.Linear(n_embed, output_dim, bias=bias)
        self.context_cond = context_cond
        self.residual_modulation = residual_modulation
        if self.context_cond:
            self.c_proj = nn.Linear(n_embed, 2 * n_embed)
            nn.init.constant_(self.c_proj.weight, 0.0)
            if self.residual_modulation:
                nn.init.constant_(self.c_proj.bias, 0.0)
            else:
                nn.init.constant_(self.c_proj.bias[:n_embed], 1.0)  # identity
                nn.init.constant_(self.c_proj.bias[n_embed:], 0.0)  # zero shift
        else:
            self.c_proj = None

    def forward(self, x, c=None):
        if self.context_cond and c is not None:
            scale, shift = self.c_proj(c).chunk(2, dim=-1)
            x = self.head(modulate(self.norm(x), scale, shift, self.residual_modulation))
        else:
            x = self.head(self.norm(x))
        return x


class MLPSwiglu(nn.Module):
    def __init__(self, n_embed, resid_pdrop=0.0, hidden_dim_multiplier=4, activation='gelu', bias=False):
        super().__init__()
        self.w1 = nn.Linear(n_embed, hidden_dim_multiplier * n_embed, bias=bias)
        self.w2 = nn.Linear(hidden_dim_multiplier * n_embed, n_embed, bias=bias)
        self.w3 = nn.Linear(n_embed, hidden_dim_multiplier * n_embed, bias=bias)

        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, x):
        x = self.dropout(self.w2(F.silu((self.w1(x))) * self.w3(x)))
        return x


class CausalBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embed, n_head, block_size, attn_pdrop=0.1, resid_pdrop=0.1, hidden_dim_multiplier=4,
                 positional_bias=False, activation='gelu', max_particles=None, norm_type='rms', context_cond=False,
                 residual_modulation=False, context_gate=False, attn_scale=1.0):
        super().__init__()
        self.max_particles = max_particles
        if norm_type == 'rms':
            norm_layer = RMSNorm
        elif norm_type == 'pn':
            norm_layer = ParticleNorm
        else:
            norm_layer = nn.LayerNorm
        self.ln1 = norm_layer(n_embed)
        self.ln2 = norm_layer(n_embed)
        self.attn = CausalParticleSelfAttention(n_embed, n_head, block_size, attn_pdrop, resid_pdrop,
                                                positional_bias=positional_bias, max_particles=max_particles)
        self.mlp = MLP(n_embed, resid_pdrop, hidden_dim_multiplier, activation=activation)
        self.attn_scale = attn_scale
        self.context_cond = context_cond
        self.residual_modulation = residual_modulation
        self.context_gate = context_gate
        self.c_multiplier = 6 if context_gate else 4
        if self.context_cond:
            self.c_proj = nn.Linear(n_embed, self.c_multiplier * n_embed)
            nn.init.constant_(self.c_proj.weight, 0.0)
            if self.residual_modulation:
                nn.init.constant_(self.c_proj.bias, 0.0)
            else:
                nn.init.constant_(self.c_proj.bias[:2 * n_embed], 1.0)  # identity
                nn.init.constant_(self.c_proj.bias[2 * n_embed: 4 * n_embed], 0.0)  # zero shift
                if self.context_gate:
                    nn.init.constant_(self.c_proj.bias[4 * n_embed:], 0.0)  # zero gate
        else:
            self.c_proj = None

    def forward(self, x, c=None):
        if self.context_cond and c is not None:
            c_proj = self.c_proj(c).chunk(self.c_multiplier, dim=-1)
            scale_a, scale_b, shift_a, shift_b = c_proj[0], c_proj[1], c_proj[2], c_proj[3]
            if self.context_gate:
                gate_a, gate_b = c_proj[4], c_proj[5]
            else:
                gate_a = gate_b = 1.0
            x = x + self.attn_scale * gate_a * self.attn(
                modulate(self.ln1(x), scale_a, shift_a, self.residual_modulation))
            x = x + gate_b * self.mlp(modulate(self.ln2(x), scale_b, shift_b, self.residual_modulation))
        else:
            x = x + self.attn_scale * self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
        return x


class SpatioTemporalBlock(nn.Module):
    """ spatio-temporal Transformer block """

    def __init__(self, n_embed, n_head, block_size, attn_pdrop=0.1, resid_pdrop=0.1, hidden_dim_multiplier=4,
                 positional_bias=False, activation='gelu', max_particles=None, norm_type='rms', causal=True,
                 context_cond=False, residual_modulation=True, context_gate=False, attn_scale=1.0,
                 cross_attn_cond=False, attention_order=('spatial', 'temporal')):
        super().__init__()
        self.context_cond = context_cond
        self.residual_modulation = residual_modulation
        self.context_gate = context_gate
        spatio_block_size = 1
        self.attn_scale = attn_scale
        self.causal = causal
        self.cross_attn_cond = cross_attn_cond
        self.attention_order = attention_order
        if self.cross_attn_cond and 'cross' not in self.attention_order:
            self.attention_order = [*attention_order, 'cross']
            # self.attention_order = ['cross', *attention_order]
            # self.attention_order = [attention_order[0], 'cross', attention_order[1]]
        self.spatio_block = SelfBlock(n_embed, n_head, spatio_block_size, attn_pdrop,
                                      resid_pdrop, hidden_dim_multiplier,
                                      positional_bias, activation=activation, max_particles=max_particles,
                                      norm_type=norm_type, context_cond=context_cond,
                                      residual_modulation=residual_modulation, context_gate=context_gate,
                                      attn_scale=attn_scale)
        temp_block_type = CausalBlock if self.causal else SelfBlock
        self.temp_block = temp_block_type(n_embed, n_head, block_size, attn_pdrop,
                                          resid_pdrop, hidden_dim_multiplier,
                                          positional_bias, activation=activation, max_particles=max_particles,
                                          norm_type=norm_type, context_cond=context_cond,
                                          residual_modulation=residual_modulation, context_gate=context_gate,
                                          attn_scale=attn_scale)
        if self.cross_attn_cond:
            self.cross_block = CrossBlock(n_embed, n_head, spatio_block_size, attn_pdrop,
                                          resid_pdrop, hidden_dim_multiplier, positional_bias, activation=activation,
                                          max_particles=max_particles,
                                          norm_type=norm_type, context_cond=context_cond,
                                          residual_modulation=residual_modulation, context_gate=context_gate,
                                          attn_scale=attn_scale, particles_first=True)
        else:
            self.cross_block = nn.Identity()

    def forward(self, x, c=None, l=None):
        # x: [b, n + 1, t, f]
        # c: context conditioning via AdaLN: [b, n+1, t, f] or None
        # l: language conditioning via cross-attention: [b, t, h, f] or None, h=N_l is the number of lang tokens
        B, N, T, F = x.shape
        for attn_type in self.attention_order:
            if attn_type == 'spatial':
                x = x.permute(0, 2, 1, 3)  # [b, t, n + 1, f]
                x = x.reshape(-1, N, 1, F)  # [b, * t, n + 1, 1, f]
                if c is not None:
                    N_c = c.shape[1]
                    c_s = c.permute(0, 2, 1, 3)  # [b, t, n + 1, f]
                    c_s = c_s.reshape(-1, N_c, 1, F)  # [b, * t, n + 1, 1, f]
                else:
                    c_s = None
                x = self.spatio_block(x, c_s)
                x = x.view(B, T, N, F)  # [b, t, n + 1, f]
                x = x.permute(0, 2, 1, 3)  # [b, n + 1, t, f]
            elif attn_type == 'temporal':
                x = x.reshape(-1, 1, T, F)  # [b * (n + 1), 1, t, f]
                if c is not None:
                    N_c = c.shape[1]
                    # c = c.reshape(B, T, N_c, F)  # [b, t, n + 1, f]
                    # c = c.permute(0, 2, 1, 3)  # [b, n + 1, t, f]
                    c_t = c.reshape(-1, 1, T, F)  # [b * (n + 1), 1, t, f]
                else:
                    c_t = None
                x = self.temp_block(x, c_t)
                x = x.view(B, N, T, F)  # [b, n + 1, t, f]
            elif attn_type == 'cross' and self.cross_attn_cond and l is not None:
                x = x.permute(0, 2, 1, 3)  # [b, t, n + 1, f]
                x = x.reshape(-1, N, 1, F)  # [b * t, n + 1, 1, f]
                N_l = l.shape[2]
                l = l.reshape(-1, N_l, 1, F)  # [b * t, N_l=h, 1, f]
                if c is not None:
                    N_c = c.shape[1]
                    c_s = c.permute(0, 2, 1, 3)  # [b, t, n + 1, f]
                    c_s = c_s.reshape(-1, N_c, 1, F)  # [b, * t, n + 1, 1, f]
                else:
                    c_s = None
                x = self.cross_block(x_q=x, x_kv=l, c=c_s)
                x = x.view(B, T, N, F)  # [b, t, n + 1, f]
                x = x.permute(0, 2, 1, 3)  # [b, n + 1, t, f]
        return x


class SelfBlock(nn.Module):
    """ self-attention Transformer block """

    def __init__(self, n_embed, n_head, block_size, attn_pdrop=0.1, resid_pdrop=0.1, hidden_dim_multiplier=4,
                 positional_bias=False, activation='gelu', max_particles=None, norm_type='ln', context_cond=False,
                 residual_modulation=False, context_gate=False, attn_scale=1.0):
        super().__init__()
        self.max_particles = max_particles
        if norm_type == 'rms':
            norm_layer = RMSNorm
        elif norm_type == 'pn':
            norm_layer = ParticleNorm
        else:
            norm_layer = nn.LayerNorm
        self.ln1 = norm_layer(n_embed)
        self.ln2 = norm_layer(n_embed)
        self.attn = ParticleSelfAttention(n_embed, n_head, block_size, attn_pdrop, resid_pdrop,
                                          positional_bias=positional_bias, max_particles=max_particles)
        self.attn_scale = attn_scale
        self.mlp = MLP(n_embed, resid_pdrop, hidden_dim_multiplier, activation=activation)
        self.context_cond = context_cond
        self.residual_modulation = residual_modulation
        self.context_gate = context_gate
        self.c_multiplier = 6 if context_gate else 4
        if self.context_cond:
            self.c_proj = nn.Linear(n_embed, self.c_multiplier * n_embed)
            nn.init.constant_(self.c_proj.weight, 0.0)
            if self.residual_modulation:
                nn.init.constant_(self.c_proj.bias, 0.0)
            else:
                nn.init.constant_(self.c_proj.bias[:2 * n_embed], 1.0)  # identity
                nn.init.constant_(self.c_proj.bias[2 * n_embed: 4 * n_embed], 0.0)  # zero shift
                if self.context_gate:
                    nn.init.constant_(self.c_proj.bias[4 * n_embed:], 0.0)  # zero gate

    def forward(self, x, c=None):
        if self.context_cond and c is not None:
            c_proj = self.c_proj(c).chunk(self.c_multiplier, dim=-1)
            scale_a, scale_b, shift_a, shift_b = c_proj[0], c_proj[1], c_proj[2], c_proj[3]
            if self.context_gate:
                gate_a, gate_b = c_proj[4], c_proj[5]
            else:
                gate_a = gate_b = 1.0
            x = x + self.attn_scale * gate_a * self.attn(
                modulate(self.ln1(x), scale_a, shift_a, self.residual_modulation))
            x = x + gate_b * self.mlp(modulate(self.ln2(x), scale_b, shift_b, self.residual_modulation))
        else:
            x = x + self.attn_scale * self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
        return x


class CrossBlock(nn.Module):
    """ cross-attention Transformer block """

    def __init__(self, n_embed, n_head, block_size, attn_pdrop=0.1, resid_pdrop=0.1, hidden_dim_multiplier=4,
                 positional_bias=False, activation='gelu', max_particles=None, norm_type='ln', particles_first=False,
                 norm_kv=False, context_cond=False,
                 residual_modulation=False, context_gate=False, attn_scale=1.0):
        super().__init__()
        self.max_particles = max_particles
        self.norm_kv = norm_kv
        if norm_type == 'rms':
            norm_layer = RMSNorm
        elif norm_type == 'pn':
            norm_layer = ParticleNorm
        else:
            norm_layer = nn.LayerNorm
        self.ln1 = norm_layer(n_embed)
        self.ln2 = norm_layer(n_embed)
        self.ln_kv = self.ln1 if self.norm_kv else nn.Identity()
        self.attn_scale = attn_scale
        self.context_cond = context_cond
        self.residual_modulation = residual_modulation
        self.context_gate = context_gate
        self.c_multiplier = 6 if context_gate else 4
        if self.context_cond:
            self.c_proj = nn.Linear(n_embed, self.c_multiplier * n_embed)
            nn.init.constant_(self.c_proj.weight, 0.0)
            if self.residual_modulation:
                nn.init.constant_(self.c_proj.bias, 0.0)
            else:
                nn.init.constant_(self.c_proj.bias[:2 * n_embed], 1.0)  # identity
                nn.init.constant_(self.c_proj.bias[2 * n_embed: 4 * n_embed], 0.0)  # zero shift
                if self.context_gate:
                    nn.init.constant_(self.c_proj.bias[4 * n_embed:], 0.0)  # zero gate
        self.attn = ParticleCrossAttention(n_embed, n_head, block_size, attn_pdrop, resid_pdrop,
                                           positional_bias=positional_bias, max_particles=max_particles,
                                           particles_first=particles_first)
        self.mlp = MLP(n_embed, resid_pdrop, hidden_dim_multiplier, activation=activation)

    def forward(self, x_q, x_kv, c=None):
        if self.context_cond and c is not None:
            c_proj = self.c_proj(c).chunk(self.c_multiplier, dim=-1)
            scale_a, scale_b, shift_a, shift_b = c_proj[0], c_proj[1], c_proj[2], c_proj[3]
            if self.context_gate:
                gate_a, gate_b = c_proj[4], c_proj[5]
            else:
                gate_a = gate_b = 1.0
            x_q = x_q + self.attn_scale * gate_a * self.attn(
                modulate(self.ln1(x_q), scale_a, shift_a, self.residual_modulation), self.ln_kv(x_kv))
            x_q = x_q + gate_b * self.mlp(modulate(self.ln2(x_q), scale_b, shift_b, self.residual_modulation))
        else:
            # x_q = x_q + self.attn(self.ln1(x_q), self.ln1(x_kv))
            x_q = x_q + self.attn_scale * self.attn(self.ln1(x_q), self.ln_kv(x_kv))
            x_q = x_q + self.mlp(self.ln2(x_q))
        return x_q


class ParticleSpatioTemporalTransformer(nn.Module):
    def __init__(self, n_embed, n_head, n_layer, block_size, output_dim, attn_pdrop=0.1, resid_pdrop=0.1,
                 hidden_dim_multiplier=4, positional_bias=False, activation='gelu', max_particles=None, norm_type='rms',
                 n_registers=0, particles_first=True, init_std=0.02,
                 causal=True, context_cond=False, residual_modulation=True, context_gate=True,
                 attention_order=('spatial', 'temporal'), cond_cross_attn=False,
                 token_pool_adaln=False,
                 pos_embed_t_adaln=False
                 ):
        super().__init__()
        self.positional_bias = positional_bias
        self.max_particles = max_particles  # for positional bias
        self.particles_first = particles_first  # expect [bs, n, t, f], else [bs, t, n, f]
        self.causal = causal
        self.n_head = n_head
        self.init_std = init_std
        self.pos_embed_t_adaln = pos_embed_t_adaln
        self.context_cond = context_cond or pos_embed_t_adaln
        self.residual_modulation = residual_modulation
        self.context_gate = context_gate
        self.attention_order = attention_order
        self.cond_cross_attn = cond_cross_attn
        self.token_pool_adaln = token_pool_adaln
        # self.attn_scale = 1 / math.sqrt(2 * 2 * n_layer)
        self.attn_scale = 1.0
        if norm_type == 'rms':
            norm_layer = RMSNorm
        elif norm_type == 'pn':
            norm_layer = ParticleNorm
        else:
            norm_layer = nn.LayerNorm
        if n_registers > 0:
            self.n_registers = n_registers
            self.registers = nn.Parameter(self.init_std * torch.randn(1, self.n_registers, 1, n_embed))
        else:
            self.n_registers = 0
            self.registers = None
        # input embedding stem
        if self.pos_embed_t_adaln:
            self.pos_embed_t_embedding = nn.Parameter(self.init_std * torch.randn(1, 1, block_size, n_embed))

        if self.positional_bias:
            self.pos_emb = nn.Identity()
        else:
            if self.pos_embed_t_adaln:
                self.pos_emb = nn.Identity()
            else:
                self.pos_emb = nn.Parameter(self.init_std * torch.randn(1, block_size, n_embed))



        attn_context_cond = context_cond or self.token_pool_adaln
        self.blocks = nn.Sequential(*[SpatioTemporalBlock(n_embed, n_head, block_size, attn_pdrop,
                                                          resid_pdrop, hidden_dim_multiplier,
                                                          positional_bias, activation=activation,
                                                          max_particles=max_particles,
                                                          norm_type=norm_type, causal=causal,
                                                          context_cond=attn_context_cond,
                                                          residual_modulation=residual_modulation,
                                                          context_gate=context_gate, attn_scale=self.attn_scale,
                                                          attention_order=self.attention_order,
                                                          cross_attn_cond=self.cond_cross_attn)
                                      for _ in range(n_layer)])

        # decoder head
        self.head = FinalTransformerLayer(n_embed, output_dim, bias=True, context_cond=self.context_cond,
                                          residual_modulation=self.residual_modulation, norm_type=norm_type)
        self.block_size = block_size
        self.n_embed = n_embed
        self.n_layer = n_layer
        # print(f"particle transformer # parameters: {sum(p.numel() for p in self.parameters())}")

    def get_block_size(self):
        return self.block_size

    def init_weights(self):
        # initialize layers
        pass
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         torch.nn.init.xavier_uniform_(m.weight)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        # if self.causal:
        #     self.apply(self._init_weights)
        # if self.positional_bias:
        #     for m in self.blocks:
        #         m.attn.rel_pos_bias.reset_parameters()

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x, c=None, l=None):
        if self.particles_first:
            b, n, t, f = x.size()
            if c is not None:
                if len(c.shape) == 3:  # [b, t, f]
                    c = c.unsqueeze(1)  # [b, t, f]
                    n_c = 1
                else:
                    n_c = c.shape[1]
        else:
            b, t, n, f = x.size()
            x = x.permute(0, 2, 1, 3)  # [bs, n, t, f]
            if c is not None:
                if len(c.shape) == 3:  # [b, t, f]
                    c = c.unsqueeze(1)  # [b, t, f]
                    n_c = 1
                else:
                    c = c.permute(0, 2, 1, 3)  # [bs, n_c, t, d]
                    n_c = c.shape[1]
        # n is the number of particles
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        assert f == self.n_embed, "invalid particle feature dim"

        # add register tokens
        if self.n_registers > 0:
            x = torch.cat([x, self.registers.repeat(b, 1, t, 1)], dim=1)
            # [bs, n + n_mem_particles, t, f]

        if not self.positional_bias and not self.pos_embed_t_adaln:
            position_embeddings = self.pos_emb[:, None, :t, :]
            x = x + position_embeddings

        # prepare condition
        if self.pos_embed_t_adaln:
            c_t = self.pos_embed_t_embedding[:, :, :t].repeat(x.shape[0], x.shape[1], 1, 1)
            if c is None:
                c = c_t
            else:
                c = c + c_t

        if self.token_pool_adaln:
            token_pool = x[:, -(self.n_registers + 1)].unsqueeze(1).repeat(1, x.shape[1], 1, 1)
            if c is None:
                c_in = token_pool
            else:
                c_in = c + token_pool
        else:
            c_in = c

        for block in self.blocks:
            # prepare condition
            # if self.token_pool_adaln:
            #     token_pool = x[:, -(self.n_registers + 1)].unsqueeze(1).repeat(1, x.shape[1], 1, 1)
            #     if c is None:
            #         c_in = token_pool
            #     else:
            #         c_in = c + token_pool
            # else:
            #     c_in = c
            # forward attention block
            # x = block(x, c, l)
            x = block(x, c_in, l)

        if self.n_registers > 0:
            x = x[:, :-self.n_registers]

        # if self.token_pool_adaln and self.context_cond:
        #     token_pool = x[:, -1].unsqueeze(1).repeat(1, x.shape[1], 1, 1)
        #     if c is None:
        #         c_in = token_pool
        #     else:
        #         c_in = c + token_pool
        # else:
        #     c_in = c

        logits = self.head(x, c)
        if not self.particles_first:
            logits = logits.permute(0, 2, 1, 3)  # [bs, t, n, f]
        return logits


class ParticleSelfAttTransformer(nn.Module):
    def __init__(self, n_embed, n_head, n_layer, block_size, output_dim, attn_pdrop=0.1, resid_pdrop=0.1,
                 hidden_dim_multiplier=4, positional_bias=False, activation='gelu', max_particles=None,
                 norm_type='rms', n_registers=0, init_std=0.02):
        super().__init__()
        self.positional_bias = positional_bias
        self.max_particles = max_particles  # for positional bias
        self.n_registers = n_registers  # "vision transformers need registers", balances the attention matrix
        # input embedding stem
        if self.positional_bias:
            self.pos_emb = nn.Identity()
        else:
            self.pos_emb = nn.Parameter(init_std * torch.randn(1, block_size, n_embed))
        if self.n_registers > 0:
            self.registers = nn.Parameter(init_std * torch.randn(1, self.n_registers, 1, n_embed))
        else:
            self.registers = None
        # transformer
        self.blocks = nn.Sequential(*[SelfBlock(n_embed, n_head, block_size, attn_pdrop,
                                                resid_pdrop, hidden_dim_multiplier,
                                                positional_bias, activation=activation, max_particles=max_particles,
                                                norm_type=norm_type)
                                      for _ in range(n_layer)])
        # decoder head
        if norm_type == 'rms':
            norm_layer = RMSNorm
        elif norm_type == 'pn':
            norm_layer = ParticleNorm
        else:
            norm_layer = nn.LayerNorm
        self.ln_f = norm_layer(n_embed)
        self.head = nn.Linear(n_embed, output_dim, bias=False)

        self.block_size = block_size
        self.n_embed = n_embed
        self.n_layer = n_layer
        # print(f"particle transformer # parameters: {sum(p.numel() for p in self.parameters())}")

    def get_block_size(self):
        return self.block_size

    def init_weights(self):
        # initialize layers
        pass
        # self.apply(self._init_weights)
        # if self.positional_bias:
        #     for m in self.blocks:
        #         m.attn.rel_pos_bias.reset_parameters()

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        # elif isinstance(module, ParticleTransformer):
        #     if not self.positional_bias:
        #         torch.nn.init.normal_(module.pos_emb, mean=0.0, std=std)

    def forward(self, x):
        # x: [b, t, n, f]
        x = x.permute(0, 2, 1, 3)  # [b, n, t, f]
        b, n, t, f = x.size()
        # n is the number of particles
        assert t <= self.block_size, f"Cannot forward, model block size is exhausted: t:{t}, block_size: {self.block_size}"
        assert f == self.n_embed, "invalid particle feature dim"

        if self.n_registers > 0 and self.registers is not None:
            registers = self.registers.repeat(b, 1, t, 1)
            x = torch.cat([x, registers], dim=1)  # [b, n+n_reg, t, f]

        if not self.positional_bias:
            position_embeddings = self.pos_emb[:, None, :t, :]
            x = x + position_embeddings
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if self.n_registers > 0:
            logits, _ = logits.split([logits.shape[1] - self.n_registers, self.n_registers], dim=1)
        logits = logits.permute(0, 2, 1, 3)  # [b, t, n, f]

        return logits


class ParticleFeatureProjection(torch.nn.Module):
    def __init__(self, in_features_dim, bg_features_dim, hidden_dim, output_dim, context_dim, add_embedding=True,
                 base_dim=32, activation='gelu', max_particles=None,
                 input_is_z=True, particle_positional_embed=True, init_std=0.02,
                 particle_score=False, ctx_cond_mode='adaln', norm_layer=True,
                 mask_inputs=True, use_z_orig=False, obj_on_film=False, mask_obj_on=False):
        super().__init__()
        assert ctx_cond_mode in ['add', 'cat', 'token', 'film', 'adaln']
        self.in_features_dim = in_features_dim
        self.bg_features_dim = bg_features_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.particle_score = particle_score
        self.add_embedding = add_embedding
        self.output_dim = output_dim
        self.base_dim = base_dim
        self.max_particles = max_particles
        self.input_is_z = input_is_z  # z or [mu, logvar]
        self.init_std = init_std

        self.mask_inputs = mask_inputs
        self.mask_obj_on = mask_obj_on
        self.use_z_orig = use_z_orig
        self.obj_on_film = obj_on_film
        self.ctx_cond_mode = ctx_cond_mode
        activation_f = nn.GELU if activation == 'gelu' else nn.ReLU
        # self.particle_dim = 2 + 2 + 1 + 1 + in_features_dim
        if self.obj_on_film:
            self.n_entities = 4
        else:
            self.n_entities = 5  # [pos, scale, obj_on, depth, features]
        if self.particle_score:
            self.n_entities += 1
        if self.use_z_orig:
            self.n_entities += 1
        if context_dim > 0 and self.ctx_cond_mode == 'cat':
            p_output_dim = 2 * output_dim
        else:
            p_output_dim = output_dim
        self.particle_dim = base_dim * self.n_entities
        # [z, z_scale, z_obj_on, z_depth, z_features]

        input_mult = 1 if self.input_is_z else 2
        self.xy_projection = nn.Sequential(nn.Linear(2 * input_mult, hidden_dim),
                                           RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                           activation_f(),
                                           nn.Linear(hidden_dim, base_dim))
        self.scale_projection = nn.Sequential(nn.Linear(2 * input_mult, hidden_dim),
                                              RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                              activation_f(),
                                              nn.Linear(hidden_dim, base_dim))
        if self.obj_on_film:
            self.obj_on_projection = nn.Sequential(nn.Linear(1 * input_mult, hidden_dim),
                                                   # RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                                   activation_f(),
                                                   nn.Linear(hidden_dim, 2 * hidden_dim))
            nn.init.constant_(self.obj_on_projection[-1].weight, 0.0)
            nn.init.constant_(self.obj_on_projection[-1].bias[:hidden_dim], 1.0)
            nn.init.constant_(self.obj_on_projection[-1].bias[hidden_dim:], 0.0)
        else:
            self.obj_on_projection = nn.Sequential(nn.Linear(1 * input_mult, hidden_dim),
                                                   RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                                   activation_f(),
                                                   nn.Linear(hidden_dim, base_dim))
        self.depth_projection = nn.Sequential(nn.Linear(1 * input_mult, hidden_dim),
                                              RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                              activation_f(),
                                              nn.Linear(hidden_dim, base_dim))
        self.features_projection = nn.Sequential(nn.Linear(in_features_dim * input_mult, hidden_dim),
                                                 RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                                 activation_f(),
                                                 nn.Linear(hidden_dim, base_dim))

        if self.mask_inputs:
            self.xy_mask = nn.Parameter(2 * torch.ones(2 * input_mult))
            self.scale_mask = nn.Parameter(0.1 * torch.ones(2 * input_mult))
            self.depth_mask = nn.Parameter(init_std * torch.randn(1 * input_mult))
            self.features_mask = nn.Parameter(init_std * torch.randn(self.in_features_dim * input_mult))
            if self.mask_obj_on:
                self.obj_on_mask = nn.Parameter(torch.zeros(1))
        if self.particle_score:
            self.score_projection = nn.Sequential(nn.Linear(1 * input_mult, hidden_dim),
                                                  RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                                  activation_f(),
                                                  nn.Linear(hidden_dim, base_dim))
        else:
            self.score_projection = nn.Identity()
        if self.use_z_orig:
            self.origin_projection = nn.Sequential(nn.Linear(4, hidden_dim),
                                                   RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                                   activation_f(),
                                                   nn.Linear(hidden_dim, base_dim))
            if self.mask_inputs:
                self.orig_mask = nn.Parameter(2 * torch.ones(4))
        else:
            self.origin_projection = nn.Identity()
        if self.obj_on_film:
            self.particle_projection_0 = nn.Sequential(nn.Linear(self.particle_dim, hidden_dim),
                                                       RMSNorm(hidden_dim))
            self.particle_projection = nn.Sequential(activation_f(),
                                                     nn.Linear(hidden_dim, output_dim))
        else:
            self.particle_projection = nn.Sequential(nn.Linear(self.particle_dim, hidden_dim),
                                                     RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                                     activation_f(),
                                                     nn.Linear(hidden_dim, output_dim))
        if bg_features_dim > 0:
            bg_output_dim = output_dim
            self.bg_features_projection = nn.Sequential(nn.Linear(bg_features_dim, hidden_dim),
                                                        RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                                        activation_f(),
                                                        nn.Linear(hidden_dim, bg_output_dim))
        else:
            self.bg_features_projection = nn.Identity()

        if self.ctx_cond_mode == 'cat' and self.context_dim > 0:
            self.p_final_projection = nn.Sequential(nn.Linear(p_output_dim, 4 * hidden_dim),
                                                    # RMSNorm(2 * hidden_dim),
                                                    activation_f(),
                                                    nn.Linear(4 * hidden_dim, output_dim))
            if bg_features_dim > 0:
                self.bg_final_projection = nn.Sequential(nn.Linear(p_output_dim, 4 * hidden_dim),
                                                         # RMSNorm(2 * hidden_dim),
                                                         activation_f(),
                                                         nn.Linear(4 * hidden_dim, output_dim))
            else:
                self.bg_final_projection = nn.Identity()
        else:
            self.p_final_projection = self.bg_final_projection = nn.Identity()
        if context_dim > 0 and self.ctx_cond_mode in ['token', 'cat']:
            self.context_projection = nn.Sequential(nn.Linear(context_dim, hidden_dim),
                                                    RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                                    activation_f(),
                                                    nn.Linear(hidden_dim, output_dim))
            if self.ctx_cond_mode == 'cat' and bg_features_dim > 0:
                self.bg_context_projection = nn.Sequential(nn.Linear(context_dim, hidden_dim),
                                                           RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                                           activation_f(),
                                                           nn.Linear(hidden_dim, output_dim))
            else:
                self.bg_context_projection = nn.Identity()
        else:
            self.context_projection = self.bg_context_projection = nn.Identity()
        if self.add_embedding:
            if particle_positional_embed:
                n_particles = 1 if self.max_particles is None else self.max_particles
            else:
                n_particles = 1  # means that all particles get the same "type" embedding
            self.particle_embedding = nn.Parameter(self.init_std * torch.randn(1, n_particles, output_dim))
            if bg_features_dim > 0:
                self.bg_embedding = nn.Parameter(self.init_std * torch.randn(1, output_dim))
            else:
                self.bg_embedding = None
            # if context_dim > 0 and self.separate_ctx_token:
            if context_dim > 0 and self.ctx_cond_mode == 'token':
                self.ctx_embedding = nn.Parameter(self.init_std * torch.randn(1, output_dim))
            else:
                self.ctx_embedding = None
        else:
            self.particle_embedding = None
            self.bg_embedding = None
            self.ctx_embedding = None

        # if self.ctx_to_act_embed and self.context_dim > 0:
        if self.ctx_cond_mode in ['add', 'film'] and self.context_dim > 0:
            ctx_out_dim = 2 * output_dim if self.ctx_cond_mode == 'film' else output_dim
            self.ctx_to_action = nn.Sequential(nn.Linear(context_dim, hidden_dim),
                                               # RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                               activation_f(),
                                               nn.Linear(hidden_dim, ctx_out_dim))
            self.ctx_to_action_bg = nn.Sequential(nn.Linear(context_dim, hidden_dim),
                                                  # RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                                  activation_f(),
                                                  nn.Linear(hidden_dim, ctx_out_dim))
            if self.ctx_cond_mode == 'film':
                self.ctx_action_norm = RMSNorm(hidden_dim)
                self.ctx_action_mlp = nn.Sequential(activation_f(),
                                                    nn.Linear(hidden_dim, hidden_dim))
                self.ctx_action_bg_norm = RMSNorm(hidden_dim)
                self.ctx_action_bg_mlp = nn.Sequential(activation_f(),
                                                       nn.Linear(hidden_dim, hidden_dim))
            else:
                self.ctx_action_norm = None
                self.ctx_action_bg_norm = None
                self.ctx_action_mlp = None
                self.ctx_action_bg_mlp = None
        else:
            self.ctx_to_action = None
            self.ctx_to_action_bg = None
            self.ctx_action_norm = None
            self.ctx_action_bg_norm = None
            self.ctx_action_mlp = None
            self.ctx_action_bg_mlp = None

        self.init_weights()

    def init_weights(self):
        # pass
        if self.ctx_to_action is not None:
            nn.init.constant_(self.ctx_to_action[-1].weight, 0.0)
            if self.ctx_cond_mode == 'film':
                nn.init.constant_(self.ctx_to_action[-1].bias, 0.0)
            else:
                nn.init.constant_(self.ctx_to_action[-1].bias, 0.0)
        if self.ctx_to_action_bg is not None:
            nn.init.constant_(self.ctx_to_action_bg[-1].weight, 0.0)
            if self.ctx_cond_mode == 'film':
                nn.init.constant_(self.ctx_to_action_bg[-1].bias, 0.0)
            else:
                nn.init.constant_(self.ctx_to_action_bg[-1].bias, 0.0)

    def forward(self, z, z_scale, z_obj_on, z_depth, z_features, z_bg_features=None, z_context=None,
                z_score=None, z_orig=None):
        # z, z_scale, z_velocity: [bs, n_particles, 2]
        # z_depth, z_obj_on: [bs, n_particles, 1]
        # z_features: [bs, n_particles, in_features_dim]
        # z_bg_features: [bs, bg_features_dim]
        # z_context: [bs, context_dim]

        n_particles = z_features.shape[-2]

        # add origin and offset
        if self.use_z_orig and z_orig is not None:
            z_offset = z - z_orig
            z_orig_tot = torch.cat([z_orig, z_offset], dim=-1)
        else:
            z_orig_tot = z_orig
        # apply masks
        if self.mask_inputs:
            z_gate = torch.where(z_obj_on > 0.2, 1.0, 0.0)
            z = z_gate * z + (1 - z_gate) * self.xy_mask
            z_scale = z_gate * z_scale + (1 - z_gate) * self.scale_mask
            z_depth = z_gate * z_depth + (1 - z_gate) * self.depth_mask
            z_features = z_gate * z_features + (1 - z_gate) * self.features_mask
            if self.use_z_orig and z_orig is not None:
                z_orig_mask = self.orig_mask
                z_orig_tot = z_gate * z_orig_tot + (1 - z_gate) * z_orig_mask
            if self.mask_obj_on:
                z_obj_on = z_gate * z_obj_on + (1 - z_gate) * self.obj_on_mask

        z_proj = self.xy_projection(z)
        z_scale_proj = self.scale_projection(z_scale)
        if len(z_obj_on.shape) == 2:
            z_obj_on = z_obj_on.unsqueeze(-1)
        z_obj_on_proj = self.obj_on_projection(z_obj_on)
        z_depth_proj = self.depth_projection(z_depth)
        z_features_proj = self.features_projection(z_features)

        if self.obj_on_film:
            z_all = torch.cat([z_proj, z_scale_proj, z_depth_proj, z_features_proj], dim=-1)
        else:
            z_all = torch.cat([z_proj, z_scale_proj, z_obj_on_proj, z_depth_proj, z_features_proj], dim=-1)
        if self.particle_score and z_score is not None:
            # apply masks
            z_score_proj = self.score_projection(z_score)
            z_all = torch.cat([z_all, z_score_proj], dim=-1)
        if self.use_z_orig and z_orig is not None:
            z_orig_proj = self.origin_projection(z_orig_tot)
            z_all = torch.cat([z_all, z_orig_proj], dim=-1)
        # z_all: [bs, n_particles, 2 + 2 + 1 + 1 + in_features_dim]
        if self.obj_on_film:
            oscale, oshift = z_obj_on_proj.chunk(2, dim=-1)
            z_all_proj = self.particle_projection(oscale * self.particle_projection_0(z_all) + oshift)
        else:
            z_all_proj = self.particle_projection(z_all)
        # [bs, n_particles, output_dim]  or [bs, n_particle, hidden_dim]
        if z_bg_features is not None:
            z_bg_features_proj = self.bg_features_projection(z_bg_features)  # [bs, output_dim] or [bs, hidden_dim]
        else:
            z_bg_features_proj = None

        if z_context is not None and self.ctx_cond_mode in ['cat', 'token']:
            if self.ctx_cond_mode == 'token':
                z_context_proj = self.context_projection(z_context)  # [bs, output_dim] or [bs, hidden_dim]
            elif self.ctx_cond_mode == 'cat':
                if len(z_context.shape) != len(z_all_proj.shape):
                    z_context_fg = z_context.unsqueeze(1)
                    z_context_proj = self.context_projection(z_context_fg)  # [bs, 1, output_dim] or [bs, 1, hidden_dim]
                    z_all_proj = torch.cat([z_all_proj, z_context_proj.repeat(1, n_particles, 1)], dim=-1)
                else:
                    z_context_fg = z_context[:, :-1]
                    z_context_proj = self.context_projection(z_context_fg)
                    # [bs, n_part, output_dim] or [bs, n_part hidden_dim]
                    z_all_proj = torch.cat([z_all_proj, z_context_proj], dim=-1)
                if z_bg_features is not None:
                    if len(z_context.shape) != len(z_all_proj.shape):
                        z_context_bg = z_context
                    else:
                        z_context_bg = z_context[:, -1]
                    z_context_proj_bg = self.bg_context_projection(z_context_bg)
                    z_bg_features_proj = torch.cat([z_bg_features_proj, z_context_proj_bg], dim=-1)
        else:
            z_context_proj = None
        z_all_proj = self.p_final_projection(z_all_proj)
        if z_bg_features is not None:
            z_bg_features_proj = self.bg_final_projection(z_bg_features_proj)
        if self.ctx_cond_mode in ['add', 'film'] and self.context_dim > 0:
            if len(z_context.shape) != len(z_all_proj.shape):
                z_context_fg = z_context.unsqueeze(1)
            else:
                z_context_fg = z_context[:, :-1]
            ctx_act = self.ctx_to_action(z_context_fg)
            if self.ctx_cond_mode == 'film':
                ctx_scale, ctx_shift = ctx_act.chunk(2, dim=-1)
                z_all_proj = (ctx_scale + 1.0) * self.ctx_action_norm(z_all_proj) + ctx_shift
                z_all_proj = self.ctx_action_mlp(z_all_proj)
            else:
                z_all_proj = z_all_proj + ctx_act
            if z_bg_features is not None:
                if len(z_context.shape) != len(z_all_proj.shape):
                    z_context_bg = z_context
                else:
                    z_context_bg = z_context[:, -1]
                ctx_act_bg = self.ctx_to_action_bg(z_context_bg)
                if self.ctx_cond_mode == 'film':
                    ctx_bg_scale, ctx_bg_shift = ctx_act_bg.chunk(2, dim=-1)
                    z_bg_features_proj = (ctx_bg_scale + 1.0) * self.ctx_action_bg_norm(
                        z_bg_features_proj) + ctx_bg_shift
                    z_bg_features_proj = self.ctx_action_bg_mlp(z_bg_features_proj)
                else:
                    z_bg_features_proj = z_bg_features_proj + ctx_act_bg
        if self.add_embedding:
            z_all_proj = z_all_proj + self.particle_embedding
            if z_bg_features is not None:
                z_bg_features_proj = z_bg_features_proj + self.bg_embedding
            # if z_context is not None and self.separate_ctx_token:
            if z_context is not None and self.ctx_cond_mode == 'token':
                z_context_proj = z_context_proj + self.ctx_embedding
        if z_bg_features is not None and z_context is not None and self.ctx_cond_mode == 'token':
            z_context_proj_p = z_context_proj.unsqueeze(1) if len(z_context_proj.shape) == 2 else z_context_proj
            z_processed = torch.cat([z_all_proj, z_bg_features_proj.unsqueeze(1), z_context_proj_p], dim=1)
            # [bs, n_particles + 2, output_dim]
        elif z_bg_features is not None:
            z_processed = torch.cat([z_all_proj, z_bg_features_proj.unsqueeze(1)], dim=1)
        # elif z_context is not None and self.separate_ctx_token:
        elif z_context is not None and self.ctx_cond_mode == 'token':
            z_processed = torch.cat([z_all_proj, z_context_proj.unsqueeze(1)], dim=1)
        else:
            z_processed = z_all_proj
        return z_processed


class ParticleAttributesProjection(nn.Module):
    def __init__(self, n_particles, in_features_dim, hidden_dim, output_dim, bg_features_dim, add_ctx_token=False,
                 base_dim=32, depth=True, obj_on=True, base_var=False, bg=True, activation='gelu', init_std=0.2,
                 cat_particle_num=False, norm_layer=True, particle_score=False,
                 mask_inputs=True, use_z_orig=False, obj_on_film=False, mask_obj_on=False):
        super().__init__()
        self.n_particles = n_particles
        self.in_features_dim = in_features_dim
        self.bg_features_dim = bg_features_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.with_depth = depth
        self.with_obj_on = obj_on
        self.with_var = base_var
        self.with_bg = bg
        self.with_score = particle_score
        self.add_ctx_token = add_ctx_token
        self.cat_particle_num = cat_particle_num
        self.norm_layer = norm_layer
        self.mask_inputs = mask_inputs
        self.mask_obj_on = mask_obj_on
        self.use_z_orig = use_z_orig
        self.obj_on_film = obj_on_film

        activation_f = nn.GELU if activation == 'gelu' else nn.ReLU

        # ==== 3D CHANGES: positions & scales are length-3 ====
        pos_dim   = 3
        scale_dim = 3
        origin_dim = 2 * pos_dim  # (orig_xyz, offset_xyz) when use_z_orig

        self.base_dim = base_dim
        self.n_entities = 3  # z, z_scale, z_features
        if self.with_depth:      self.n_entities += 1
        if self.with_obj_on and not self.obj_on_film: self.n_entities += 1
        if self.with_var:        self.n_entities += 1
        if self.with_score:      self.n_entities += 1
        if self.use_z_orig:      self.n_entities += 1
        if self.cat_particle_num:
            self.n_entities += 1
            self.particle_num_embed = nn.Parameter(0.02 * torch.randn(1, self.n_particles, self.base_dim))
        self.particle_dim = self.base_dim * self.n_entities

        # --- projections ---
        Norm = (lambda d: RMSNorm(d)) if norm_layer else (lambda d: nn.Identity())

        self.xy_projection = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim), Norm(hidden_dim), activation_f(), nn.Linear(hidden_dim, self.base_dim)
        )
        self.scale_projection = nn.Sequential(
            nn.Linear(scale_dim, hidden_dim), Norm(hidden_dim), activation_f(), nn.Linear(hidden_dim, self.base_dim)
        )

        # NOTE: keep var projection shape as in your code (you control z_base_var):
        if self.with_var:
            self.var_projection = nn.Sequential(
                nn.Linear(5, hidden_dim), Norm(hidden_dim), activation_f(), nn.Linear(hidden_dim, self.base_dim)
            )

        if self.with_obj_on:
            if self.obj_on_film:
                self.obj_on_projection = nn.Sequential(
                    nn.Linear(1, hidden_dim),
                    activation_f(),
                    nn.Linear(hidden_dim, 2 * hidden_dim)
                )
                nn.init.constant_(self.obj_on_projection[-1].weight, 0.0)
                nn.init.constant_(self.obj_on_projection[-1].bias[:hidden_dim], 1.0)
                nn.init.constant_(self.obj_on_projection[-1].bias[hidden_dim:], 0.0)
            else:
                self.obj_on_projection = nn.Sequential(
                    nn.Linear(1, hidden_dim), Norm(hidden_dim), activation_f(), nn.Linear(hidden_dim, self.base_dim)
                )

            if self.mask_inputs:
                # ==== 3D CHANGES: mask lengths ====
                self.xy_mask       = nn.Parameter(2 * torch.ones(pos_dim))
                self.scale_mask    = nn.Parameter(0.1 * torch.ones(scale_dim))
                self.features_mask = nn.Parameter(init_std * torch.randn(in_features_dim))
                if self.mask_obj_on:
                    self.obj_on_mask = nn.Parameter(torch.zeros(1))

        if self.with_depth:
            self.depth_projection = nn.Sequential(
                nn.Linear(1, hidden_dim), Norm(hidden_dim), activation_f(), nn.Linear(hidden_dim, self.base_dim)
            )
            if self.with_obj_on and self.mask_inputs:
                self.depth_mask = nn.Parameter(init_std * torch.randn(1))

        self.features_projection = nn.Sequential(
            nn.Linear(in_features_dim, hidden_dim), Norm(hidden_dim), activation_f(), nn.Linear(hidden_dim, self.base_dim)
        )

        if self.with_score:
            self.score_projection = nn.Sequential(
                nn.Linear(1, hidden_dim), Norm(hidden_dim), activation_f(), nn.Linear(hidden_dim, self.base_dim)
            )

        if self.with_bg:
            self.bg_projection = nn.Sequential(
                nn.Linear(bg_features_dim, hidden_dim), Norm(hidden_dim), activation_f(), nn.Linear(hidden_dim, output_dim)
            )

        if self.use_z_orig:
            # ==== 3D CHANGES: 6 inputs (orig_xyz + offset_xyz) ====
            self.origin_projection = nn.Sequential(
                nn.Linear(origin_dim, hidden_dim), Norm(hidden_dim), activation_f(), nn.Linear(hidden_dim, base_dim)
            )
            if self.mask_inputs:
                self.orig_mask = nn.Parameter(2 * torch.ones(origin_dim))

        if self.obj_on_film:
            self.particle_projection_0 = nn.Sequential(
                nn.Linear(self.particle_dim, hidden_dim), RMSNorm(hidden_dim)
            )
            self.particle_projection = nn.Sequential(
                activation_f(), nn.Linear(hidden_dim, output_dim)
            )
        else:
            self.particle_projection = nn.Sequential(
                nn.Linear(self.particle_dim, hidden_dim), Norm(hidden_dim), activation_f(), nn.Linear(hidden_dim, output_dim)
            )

        if self.add_ctx_token:
            self.ctx_embedding = nn.Parameter(init_std * torch.randn(1, 1, 1, output_dim))

    def init_weights(self):
        pass

    def forward(self, z, z_scale, z_obj_on, z_depth, z_features, z_bg_features=None, z_base_var=None, z_score=None,
                z_orig=None):
        """
        Shapes (3D):
          z:         [B, T, N, 3] or [B, N, 3]
          z_scale:   [B, T, N, 3] or [B, N, 3]
          z_obj_on:  [B, T, N, 1] or [B, N, 1]
          z_depth:   [B, T, N, 1] or [B, N, 1]
          z_features:[B, T, N, F] or [B, N, F]
          z_bg_features: [B, Fbg]
          z_orig:    [B, T, N, 3] (orig) if use_z_orig, else None
        Returns:
          [B, T, N(+bg)(+ctx), output_dim]
        """
        # add origin and offset if requested
        if self.use_z_orig and z_orig is not None:
            z_offset   = z - z_orig                       # [.., 3]
            z_orig_tot = torch.cat([z_orig, z_offset], dim=-1)  # [.., 6]
        else:
            z_orig_tot = z_orig

        # optional gating based on objectness
        if self.with_obj_on and self.mask_inputs:
            z_gate = torch.where(z_obj_on > 0.2, 1.0, 0.0)
            z         = z_gate * z         + (1 - z_gate) * self.xy_mask
            z_scale   = z_gate * z_scale   + (1 - z_gate) * self.scale_mask
            z_features= z_gate * z_features+ (1 - z_gate) * self.features_mask
            if self.use_z_orig and z_orig is not None:
                z_orig_tot = z_gate * z_orig_tot + (1 - z_gate) * self.orig_mask
            if self.mask_obj_on:
                z_obj_on = z_gate * z_obj_on + (1 - z_gate) * self.obj_on_mask

        # project parts
        z_proj         = self.xy_projection(z)
        z_scale_proj   = self.scale_projection(z_scale)
        z_features_proj= self.features_projection(z_features)

        z_all = torch.cat([z_proj, z_scale_proj, z_features_proj], dim=-1)

        if self.with_obj_on:
            z_obj_on_proj = self.obj_on_projection(z_obj_on)
            if not self.obj_on_film:
                z_all = torch.cat([z_all, z_obj_on_proj], dim=-1)

        if self.with_depth:
            if self.with_obj_on and self.mask_inputs:
                z_depth = z_gate * z_depth + (1 - z_gate) * self.depth_mask
            z_depth_proj = self.depth_projection(z_depth)
            z_all = torch.cat([z_all, z_depth_proj], dim=-1)

        if self.with_var and z_base_var is not None:
            z_var_proj = self.var_projection(z_base_var)   # you control z_base_var's last dim
            z_all = torch.cat([z_all, z_var_proj], dim=-1)

        if self.with_score and z_score is not None:
            z_score_proj = self.score_projection(z_score)
            z_all = torch.cat([z_all, z_score_proj], dim=-1)

        if self.use_z_orig and z_orig is not None:
            z_orig_proj = self.origin_projection(z_orig_tot)  # [.., base_dim]
            z_all = torch.cat([z_all, z_orig_proj], dim=-1)

        if self.cat_particle_num:
            if len(z.shape) == 4:  # [B,T,N,3]
                p_embed = self.particle_num_embed.unsqueeze(1).repeat(z.shape[0], z.shape[1], 1, 1)
            else:                  # [B,N,3]
                p_embed = self.particle_num_embed.repeat(z.shape[0], 1, 1)
            z_all = torch.cat([z_all, p_embed], dim=-1)

        # final projection (with optional FiLM by obj_on)
        if self.with_obj_on and self.obj_on_film:
            oscale, oshift = z_obj_on_proj.chunk(2, dim=-1)
            z_all_proj = self.particle_projection(oscale * self.particle_projection_0(z_all) + oshift)
        else:
            z_all_proj = self.particle_projection(z_all)  # [..., output_dim]

        # append bg token
        # ----- append bg token (robust to extra dims) -----
        if self.with_bg:
            assert z_bg_features is not None, "z_bg_features required when bg=True"

            def _normalize_bg_feats(x):
                # Accepts [B,F], [B,1,F], [B,T,F], [B,T,1,F], [B,T,N,F], [B,1,1,F], [B,1,1,1,F]
                if x.dim() == 2:             # [B,F]
                    return x
                if x.dim() == 3:             # [B,?,F]  (T or 1)
                    return x                 # keep [B,?,F]
                if x.dim() == 4:             # [B, T, N, F] or [B,1,1,F]
                    return x.mean(dim=-2)    # -> [B, T, F]  (or [B,1,F])
                if x.dim() == 5:             # [B, T, 1, 1, F] etc.
                    return x.squeeze(-2).squeeze(-2)  # -> [B, T, F]
                raise RuntimeError(f"Unexpected z_bg_features shape: {tuple(x.shape)}")

            bgf = _normalize_bg_feats(z_bg_features)                # [B,F] or [B,T,F]
            bgp = self.bg_projection(bgf)                           # [B,D] or [B,T,D]

            if z_all_proj.dim() == 4:                               # [B,T,N,D]
                B, T, _, D = z_all_proj.shape
                if bgp.dim() == 2:                                  # [B,D] -> tile over T
                    bgp = bgp[:, None, :].expand(B, T, D)           # [B,T,D]
                # now [B,T,D] -> [B,T,1,D]
                bgp = bgp.unsqueeze(2)
                z_all_proj = torch.cat([z_all_proj, bgp], dim=2)    # append 1 BG token per timestep
            elif z_all_proj.dim() == 3:                             # [B,N,D]
                B, _, D = z_all_proj.shape
                if bgp.dim() == 3:                                  # [B,T,D] -> pick T=1 or mean over T
                    bgp = bgp.mean(dim=1)                           # [B,D]
                # [B,D] -> [B,1,D]
                z_all_proj = torch.cat([z_all_proj, bgp[:, None, :]], dim=1)
            else:
                raise RuntimeError(f"Unexpected z_all_proj shape: {tuple(z_all_proj.shape)}")


        # optional ctx token
        if self.add_ctx_token:
            if len(z_all_proj.shape) == 4:  # [B,T,N,D]
                B, T = z_all_proj.shape[:2]
                z_all_proj = torch.cat(
                    [z_all_proj, self.ctx_embedding.expand(B, T, 1, -1)], dim=2
                )
            else:                            # [B,N,D]
                B = z_all_proj.shape[0]
                z_all_proj = torch.cat(
                    [z_all_proj, self.ctx_embedding.expand(B, 1, -1)], dim=1
                )

        return z_all_proj



class ParticlePool(nn.Module):
    def __init__(self, pool_mode='mean', pool_dim=-2, keepdim=True):
        super().__init__()
        assert pool_mode in ['mean', 'max', 'sum', 'none', 'last', 'token', 'mlp']
        self.pool_mode = pool_mode
        self.pool_dim = pool_dim
        self.keepdim = keepdim

    def forward(self, x):
        if self.pool_mode == 'mean':
            return x.mean(self.pool_dim, keepdim=self.keepdim)
        elif self.pool_mode == 'sum':
            return x.sum(self.pool_dim, keepdim=self.keepdim)
        elif self.pool_mode == 'max':
            return x.max(self.pool_dim, keepdim=self.keepdim)[0]
        else:
            return x


class ParticleAttributeDecoder(nn.Module):
    def __init__(self, n_particles, input_dim, hidden_dim, features_dim, bg_features_dim=None,
                 depth=False, obj_on=False, features=False, bg_features=False,
                 offset_logvar=False,
                 activation='gelu', dropout=0.0, shared_logvar=False,
                 output_ctx_logvar=True, features_dist='gauss'):
        super().__init__()
        # decoder to map back from PTE's inner dim to the particle's original dimension
        self.n_particles = n_particles
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.features_dist = features_dist
        self.features_dim = features_dim
        self.bg_features_dim = bg_features_dim
        self.offset_logvar = offset_logvar
        self.with_depth = depth
        self.with_obj_on = obj_on
        self.with_features = features
        self.with_bg_features = bg_features
        self.use_fg_backbone = (self.with_obj_on or self.with_depth or self.with_features)
        self.shared_logvar = shared_logvar
        self.output_ctx_logvar = output_ctx_logvar
        activation_f = nn.GELU if activation == 'gelu' else nn.ReLU
        if self.use_fg_backbone:
            self.fg_backbone = nn.Identity()
            if self.with_obj_on:
                self.obj_on_head = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                                 activation_f(),
                                                 nn.Linear(hidden_dim, 1)
                                                 )  # log_a, log_b
            if self.with_depth:
                self.depth_head = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                                activation_f(),
                                                nn.Linear(hidden_dim, 2)
                                                )  # mu_z, logvar_z
            if self.with_features:
                output_feat_dim = 2 * features_dim if (features_dist != 'categorical') else features_dim
                self.features_head = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                                   activation_f(),
                                                   nn.Linear(hidden_dim, output_feat_dim)
                                                   )  # mu_features, logvar_features
        if self.with_bg_features:
            output_bg_feat_dim = 2 * bg_features_dim if (features_dist != 'categorical') else bg_features_dim
            self.bg_backbone = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                             activation_f(),
                                             )
            self.bg_features_head = nn.Linear(hidden_dim, output_bg_feat_dim)  # mu_features, logvar_features

        self.init_weights()

    def init_weights(self):
        if self.with_features and self.features_dist != 'categorical':
            nn.init.constant_(self.features_head[-1].weight[:self.features_dim], 0.0)
            nn.init.constant_(self.features_head[-1].bias[:self.features_dim], 0.0)
            nn.init.constant_(self.features_head[-1].weight[self.features_dim:], 0.0)
            nn.init.constant_(self.features_head[-1].bias[self.features_dim:], math.log(0.001 ** 2))
        if self.with_bg_features and self.features_dist != 'categorical':
            nn.init.constant_(self.bg_features_head.weight[:self.bg_features_dim], 0.0)
            nn.init.constant_(self.bg_features_head.bias[:self.bg_features_dim], 0.0)
            nn.init.constant_(self.bg_features_head.weight[self.bg_features_dim:], 0.0)
            nn.init.constant_(self.bg_features_head.bias[self.bg_features_dim:], math.log(0.001 ** 2))

    def forward(self, x):
        # x: [bs, n_particles, input_dim]
        # bs, n_particles, in_dim = x.shape
        bs, ts, n_particles = x.shape[0], x.shape[1], x.shape[2]
        # the following assumes fg_particles + bg_particle + context particle
        fg_particles = n_particles - 2 if self.with_bg_features else n_particles - 1
        if self.use_fg_backbone:
            x_fg = x[:, :, :fg_particles]
            fg_features = self.fg_backbone(x_fg)
            if self.with_depth:
                depth = self.depth_head(fg_features)
                mu_depth, logvar_depth = torch.chunk(depth, 2, dim=-1)
            else:
                mu_depth = logvar_depth = None
            if self.with_obj_on:
                obj_on = self.obj_on_head(fg_features)
                lobj_on_a = lobj_on_b = obj_on
            else:
                lobj_on_a = lobj_on_b = None
            if self.with_features:
                features = self.features_head(fg_features)
                if self.features_dist != 'categorical':
                    mu_features, logvar_features = torch.chunk(features, 2, dim=-1)
                else:
                    mu_features = logvar_features = features
            else:
                mu_features = logvar_features = None
        else:
            mu_depth = logvar_depth = None
            lobj_on_a = lobj_on_b = None
            mu_features = logvar_features = None

        if self.with_bg_features:
            x_bg = x[:, :, fg_particles]
            bg_features = self.bg_backbone(x_bg)
            bg_features = self.bg_features_head(bg_features)
            if self.features_dist != 'categorical':
                mu_bg_features, logvar_bg_features = torch.chunk(bg_features, 2, dim=-1)
            else:
                mu_bg_features = logvar_bg_features = bg_features
        else:
            mu_bg_features = logvar_bg_features = None

        decoder_out = {'mu_depth': mu_depth, 'logvar_depth': logvar_depth,
                       'lobj_on_a': lobj_on_a, 'lobj_on_b': lobj_on_b,
                       'mu_features': mu_features, 'logvar_features': logvar_features,
                       'mu_bg_features': mu_bg_features, 'logvar_bg_features': logvar_bg_features}

        return decoder_out


class ParticleFeatureDecoderDyn(nn.Module):
    def __init__(self, input_dim, features_dim, bg_features_dim, hidden_dim, kp_activation='tanh', max_delta=1.0,
                 context_dim=7, activation='gelu', shared_logvar=False, logvar_min=-10.0,
                 logvar_max=10.0, ctx_as_token=False, dec_ctx=False,
                 norm_type='rms', dropout=0.0, particle_score=False,
                 features_dist='gauss', n_fg_categories=8, n_fg_classes=4, n_bg_categories=4, n_bg_classes=4,
                 scale_init=None):
        super().__init__()
        # decoder to map back from PTE's inner dim to the particle's original dimension
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.features_dist = features_dist
        self.n_fg_categories = n_fg_categories
        self.n_fg_classes = n_fg_classes
        self.n_bg_categories = n_bg_categories
        self.n_bg_classes = n_bg_classes
        self.features_dim = features_dim
        self.bg_features_dim = bg_features_dim
        self.ctx_dim = context_dim
        self.particle_score = particle_score
        self.kp_activation = kp_activation
        self.max_delta = max_delta
        self.shared_logvar = shared_logvar
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.ctx_as_token = ctx_as_token
        self.dec_ctx = dec_ctx
        self.n_attributes = 5  # [xy, scale, depth, transp, features]
        self.scale_init = scale_init
        activation_f = nn.GELU if activation == 'gelu' else nn.ReLU

        xy_output_dim = 2 if self.shared_logvar else 4
        scale_output_dim = 2 if self.shared_logvar else 4
        depth_output_dim = 1 if self.shared_logvar else 2
        output_features_logvar = (not self.shared_logvar and self.features_dist != 'categorical')
        feature_output_dim = 2 * features_dim if output_features_logvar else features_dim
        if bg_features_dim > 0:
            bg_output_dim = 2 * bg_features_dim if output_features_logvar else bg_features_dim
        else:
            bg_output_dim = 0
        if self.shared_logvar:
            self.offset_xy_logvar = nn.Parameter(torch.zeros(1, 1, 1))
            self.scale_xy_logvar = nn.Parameter(torch.zeros(1, 1, 1))
            self.depth_logvar = nn.Parameter(torch.zeros(1, 1, 1))
            self.features_logvar = nn.Parameter(torch.zeros(1, 1, 1))
            if bg_features_dim > 0:
                self.bg_features_logvar = nn.Parameter(torch.zeros(1, 1))

        self.offset_xy_head = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                            activation_f(),
                                            nn.Linear(hidden_dim, xy_output_dim)
                                            )  # mu_ox, logvar_ox, mu_oy, logvar_oy

        self.scale_xy_head = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                           activation_f(),
                                           nn.Linear(hidden_dim, scale_output_dim)
                                           )  # mu_sx, logvar_sx, mu_sy, logvar_sy
        self.obj_on_head = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                         activation_f(),
                                         nn.Linear(hidden_dim, 1)
                                         )  # log_a, log_b
        self.depth_head = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        activation_f(),
                                        nn.Linear(hidden_dim, depth_output_dim)
                                        )  # mu_z, logvar_z
        self.features_head = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                           activation_f(),
                                           nn.Linear(hidden_dim, feature_output_dim)
                                           )  # mu_features, logvar_features
        if self.particle_score:
            self.score_head = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                            activation_f(),
                                            nn.Linear(hidden_dim, 2)
                                            )  # mu_score, logvar_score
            self.n_attributes += 1
        else:
            self.score_head = nn.Identity()

        if self.bg_features_dim > 0:
            self.bg_backbone = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                             # RMSNorm(hidden_dim),
                                             activation_f(),
                                             )
            self.bg_features_head = self.get_mlp_head(bg_output_dim)  # mu_features, logvar_features
        if self.ctx_dim > 0 and self.dec_ctx and self.ctx_as_token:
            self.backbone = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                          activation_f(),
                                          )
            self.context_head = self.get_mlp_head(2 * self.ctx_dim)  # mu_features, logvar_features
        else:
            self.backbone = nn.Identity()
            self.context_head = nn.Identity()
        self.init_weights()

    def init_weights(self):
        # pass

        torch.nn.init.constant_(self.offset_xy_head[-1].weight[:2], 0.0)
        torch.nn.init.constant_(self.offset_xy_head[-1].bias[:2], 0.0)

        if not self.shared_logvar:
            torch.nn.init.constant_(self.offset_xy_head[-1].weight[2:], 0.0)
            torch.nn.init.constant_(self.offset_xy_head[-1].bias[2:], math.log(0.1 ** 2))

        if self.scale_init is not None:
            scale_init = 0.75 * self.scale_init + 1e-5
            torch.nn.init.constant_(self.scale_xy_head[-1].weight[:2], 0.0)
            torch.nn.init.constant_(self.scale_xy_head[-1].bias[:2], np.log(scale_init / (1 - scale_init)))

            torch.nn.init.constant_(self.scale_xy_head[-1].weight[2:], 0.0)
            torch.nn.init.constant_(self.scale_xy_head[-1].bias[2:], math.log(0.2 ** 2))  # 0.1

        # torch.nn.init.constant_(self.obj_on_head[-1].weight, 0.0)
        # torch.nn.init.constant_(self.obj_on_head[-1].bias, -0.3)

        if self.particle_score:
            torch.nn.init.constant_(self.score_head[-1].weight, 0.0)
            torch.nn.init.constant_(self.score_head[-1].bias, 0.0)

    def get_mlp_head(self, output_dim):
        return nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x):
        # x: [bs, n_particles + 2, input_dim]
        bs, n_particles, in_dim = x.shape

        if self.ctx_dim > 0 and self.ctx_as_token:
            fg_features, bg_features, ctx_features = x.split([n_particles - 2, 1, 1], dim=1)
            bg_features = self.bg_backbone(bg_features)
            ctx_features = self.backbone(ctx_features)
        elif self.bg_features_dim > 0:
            fg_features, bg_features = x.split([n_particles - 1, 1], dim=1)
            bg_features = self.bg_backbone(bg_features)
            ctx_features = None
        else:
            fg_features = x
            bg_features = None
            ctx_features = None
        xy = scale = obj_on = depth = features = scores = fg_features

        n, f = xy.shape[1], xy.shape[-1]
        offset_features = xy
        offset_xy = self.offset_xy_head(offset_features)
        if self.shared_logvar:
            mu_offset = offset_xy
            logvar_offset = self.offset_xy_logvar.repeat(mu_offset.shape[0], mu_offset.shape[1], mu_offset.shape[-1])
        else:
            offset_xy = offset_xy.view(bs, -1, offset_xy.shape[-1])
            mu_offset, logvar_offset = torch.chunk(offset_xy, chunks=2, dim=-1)

        if self.kp_activation == "tanh":
            mu_offset = torch.tanh(mu_offset)
        elif self.kp_activation == "sigmoid":
            mu_offset = torch.sigmoid(mu_offset)

        # apply max delta
        mu_offset = self.max_delta * mu_offset

        scale_features = scale
        scale_xy = self.scale_xy_head(scale_features)
        if self.shared_logvar:
            mu_scale = scale_xy
            logvar_scale = self.scale_xy_logvar.repeat(mu_scale.shape[0], mu_scale.shape[1], mu_scale.shape[-1])
        else:
            scale_xy = scale_xy.view(bs, -1, scale_xy.shape[-1])
            mu_scale, logvar_scale = torch.chunk(scale_xy, chunks=2, dim=-1)

        obj_on_1 = self.obj_on_head(obj_on)
        obj_on_1 = obj_on_1.view(bs, -1, 1)
        lobj_on_a = lobj_on_b = obj_on_1
        depth = self.depth_head(depth)
        if self.shared_logvar:
            mu_depth = depth
            logvar_depth = self.depth_logvar.repeat(mu_depth.shape[0], mu_depth.shape[1], 1)
        else:
            depth = depth.view(bs, -1, 2)
            mu_depth, logvar_depth = torch.chunk(depth, 2, dim=-1)

        feat_features = features
        features = self.features_head(feat_features)
        if self.features_dist == 'categorical':
            mu_features = logvar_features = features
        else:
            if self.shared_logvar:
                mu_features = features
                logvar_features = self.features_logvar.repeat(mu_features.shape[0], mu_features.shape[1],
                                                              mu_features.shape[-1])
            else:
                features = features.view(bs, -1, 2 * self.features_dim)
                mu_features, logvar_features = torch.chunk(features, 2, dim=-1)

        if self.particle_score:
            score = self.score_head(scores)
            score = score.view(bs, -1, 2)
            mu_score, logvar_score = torch.chunk(score, chunks=2, dim=-1)
        else:
            mu_score = logvar_score = None

        if self.bg_features_dim > 0:
            f_bg = bg_features.shape[-1]
            bg_features = self.bg_features_head(bg_features.squeeze(1))
            if self.features_dist == 'categorical':
                mu_bg_features = logvar_bg_features = bg_features
            else:
                if self.shared_logvar:
                    mu_bg_features = bg_features
                    logvar_bg_features = self.bg_features_logvar.repeat(mu_bg_features.shape[0],
                                                                        mu_bg_features.shape[-1])
                else:
                    mu_bg_features, logvar_bg_features = torch.chunk(bg_features, 2, dim=-1)
        else:
            mu_bg_features = logvar_bg_features = None

        if self.ctx_dim > 0 and self.ctx_as_token and self.dec_ctx:
            context_features = self.context_head(ctx_features.squeeze(1))
            mu_context, logvar_context = torch.chunk(context_features, 2, dim=-1)
        else:
            mu_context = logvar_context = None

        decoder_out = {'mu_offset': mu_offset,
                       'logvar_offset': logvar_offset, 'lobj_on_a': lobj_on_a, 'lobj_on_b': lobj_on_b,
                       'obj_on': obj_on, 'mu_depth': mu_depth, 'logvar_depth': logvar_depth,
                       'mu_scale': mu_scale, 'logvar_scale': logvar_scale, 'mu_features': mu_features,
                       'logvar_features': logvar_features, 'mu_bg_features': mu_bg_features,
                       'logvar_bg_features': logvar_bg_features, 'mu_context': mu_context,
                       'logvar_context': logvar_context, 'mu_score': mu_score, 'logvar_score': logvar_score}
        return decoder_out


"""
CNN-based modules
"""

class ObjectDecoderCNN(nn.Module):
    def __init__(self, patch_size, num_chans=4, bottleneck_size=128, pad_mode='replicate',
                 embed_position=False, use_resblock=False, context_dim=0, normalize_rgb=False,
                 res_from_fc=8, activation='gelu', ch_mult=(1, 2, 3), base_ch=32, final_cnn_ch=32,
                 num_res_blocks=2, cnn_mid_blocks=False, mlp_hidden_dim=256,
                 init_zero_bias=True, init_conv_layers=True, init_conv_fg_std=0.02):
        super().__init__()

        # ---- config / flags ----
        self.init_zero_bias = init_zero_bias
        self.init_conv_layers = init_conv_layers
        self.init_conv_fg_std = init_conv_fg_std
        self.patch_size = patch_size
        self.num_chans = num_chans
        self.embed_position = embed_position
        self.use_resblock = use_resblock
        self.features_dim = bottleneck_size
        self.activation = activation
        self.cnn_mid_blocks = cnn_mid_blocks
        self.mlp_hidden_dim = mlp_hidden_dim
        self.context_dim = context_dim
        self.normalize_rgb = normalize_rgb

        # ---- 3D seed (fc_res³) ----
        self.in_ch = final_cnn_ch
        self.fc_res = res_from_fc                           # e.g., 6 or 8
        feature_map_size = self.fc_res ** 3                 # *** 3D ***
        if self.features_dim % feature_map_size == 0:
            self.ch_feature_dim = max(1, math.ceil(self.features_dim / feature_map_size))
            self.from_latent_lin = nn.Identity()
        else:
            self.ch_feature_dim = final_cnn_ch
            flattened = self.ch_feature_dim * (self.fc_res ** 3)
            self.from_latent_lin = self.get_mlp(self.features_dim, flattened)

        # optional extra processing on the seed block (keep as identity by default)
        self.from_latent = nn.Identity()

        # ---- 3D decoder ----
        self.num_upsample = max(int(np.log2(patch_size)) - int(np.log2(self.fc_res)), 0)
        attn_res = [max(self.patch_size // 16, 1)]
        ch_mult = ch_mult[:self.num_upsample + 1]
        z_channels = self.ch_feature_dim
        self.cnn = Decoder(  # your Decoder must be the 3D version
            ch=base_ch, out_ch=self.num_chans, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
            attn_resolutions=attn_res, dropout=0.0, resamp_with_conv=True,
            resolution=self.patch_size, z_channels=z_channels, give_pre_end=False,
            padding_mode=pad_mode, residual=self.use_resblock, upsample_method='nearest',
            mid_blocks=cnn_mid_blocks
        )

        # final channel projector (lazy-init because C' depends on runtime)
        self._cnn_out_proj_inited = False
        self.cnn_out_proj = None

        # init
        self.init_weights()

    def forward(self, x, context=None):
        """
        x: [B, N, feat]
        returns: [B*N, num_chans, patch_size, patch_size, patch_size]
        """
        import torch.nn.functional as F
        B, N = x.shape[0], x.shape[1]
        D = H = W = self.patch_size

        # latent -> 3D seed
        x = self.from_latent_lin(x)  # [B, N, ch_feature_dim * (fc_res^3)]
        x = x.view(-1, self.ch_feature_dim, self.fc_res, self.fc_res, self.fc_res)  # [B*N,C,fr,fr,fr]

        z = self.from_latent(x)      # identity by default
        y = self.cnn(z)              # [B*N, C', D', H', W']  (3D decoder)

        # force channels to num_chans
        Cprime = y.shape[1]
        if not self._cnn_out_proj_inited:
            self.cnn_out_proj = nn.Conv3d(
                Cprime, self.num_chans, kernel_size=1, bias=True
            ).to(y.device)
            self._cnn_out_proj_inited = True

            # ---- NEW: break symmetry between R/G/B output channels ----
            with torch.no_grad():
                # keep alpha (ch0) as-in, nudge RGB filters slightly differently
                if self.num_chans >= 4:
                    # small multiplicative factors for channels 1,2,3
                    scales = torch.tensor([1.0, 0.97, 1.03, 1.06],
                                          dtype=self.cnn_out_proj.weight.dtype,
                                          device=self.cnn_out_proj.weight.device)
                    for c in range(min(self.num_chans, scales.numel())):
                        self.cnn_out_proj.weight[c].mul_(scales[c])

                    # tiny different biases so gradients diverge early
                    if self.cnn_out_proj.bias is not None:
                        # alpha bias stays 0; tiny offsets for RGB
                        self.cnn_out_proj.bias[1].fill_(+1e-3)
                        self.cnn_out_proj.bias[2].fill_(-1e-3)
                        self.cnn_out_proj.bias[3].fill_(+2e-3)
            # ---- END NEW ----

        if Cprime != self.num_chans:
            y = self.cnn_out_proj(y)


        # force spatial size to (patch_size)^3
        if y.shape[2:] != (D, H, W):
            y = F.interpolate(y, size=(D, H, W), mode="trilinear", align_corners=False)

        return y


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if self.init_conv_layers:
                    nn.init.normal_(m.weight, 0.0, self.init_conv_fg_std)
                if self.init_zero_bias and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_mlp(self, in_dim, out_dim, linear=False):
        if linear:
            return nn.Linear(in_dim, out_dim)
        else:
            activation_f = nn.GELU if self.activation == 'gelu' else nn.ReLU
            hidden_dim = self.mlp_hidden_dim
            mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                activation_f(),
                                nn.Linear(hidden_dim, out_dim))

            return mlp




class ObjectDecoderCNNFILM(nn.Module):
    def __init__(self, patch_size, num_chans=4, bottleneck_size=128, pad_mode='replicate', embed_position=False,
                 use_resblock=False, context_dim=0, normalize_rgb=False, res_from_fc=8, activation='gelu',
                 ch_mult=(1, 2, 3), base_ch=32, final_cnn_ch=32, num_res_blocks=2, cnn_mid_blocks=False,
                 mlp_hidden_dim=256,
                 # initialization
                 init_zero_bias=True,  # zero bias for conv and linear layers
                 init_conv_layers=True,  # initialize conv layers with normal dist
                 init_conv_fg_std=0.02,  # std for conv fg normal dist
                 ):
        super().__init__()

        # initialization
        self.init_zero_bias = init_zero_bias  # zero bias for conv and linear layers
        self.init_conv_layers = init_conv_layers  # initialize conv layers with normal dist
        self.init_conv_fg_std = init_conv_fg_std  # std for conv fg normal dist

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.num_chans = num_chans
        self.embed_position = embed_position
        self.use_resblock = use_resblock
        self.features_dim = bottleneck_size
        self.activation = activation
        self.cnn_mid_blocks = cnn_mid_blocks
        self.mlp_hidden_dim = mlp_hidden_dim

        self.context_dim = context_dim
        self.normalize_rgb = normalize_rgb

        self.in_ch = final_cnn_ch
        self.fc_res = res_from_fc
        fc_out_dim = self.in_ch * (self.fc_res ** 2)
        fc_in_dim = bottleneck_size if not self.embed_position else 2 * bottleneck_size

        feature_map_size = self.fc_res ** 2

        if self.features_dim % feature_map_size == 0:
            self.ch_feature_dim = math.ceil(max(self.features_dim / (res_from_fc ** 2), 1))
            output_z_cnn = (self.ch_feature_dim, self.fc_res, self.fc_res)
            flattened_z_cnn = np.prod(output_z_cnn)

            self.projection_mode = 'fcn'
            self.from_latent_lin = nn.Identity()
            self.from_latent = nn.Identity()
        else:
            self.ch_feature_dim = final_cnn_ch
            output_z_cnn = (self.ch_feature_dim, self.fc_res, self.fc_res)
            flattened_z_cnn = np.prod(output_z_cnn)

            self.projection_mode = 'fc'
            self.from_latent_lin = self.get_mlp(self.features_dim, flattened_z_cnn)
            self.from_latent = nn.Identity()

        self.info = (f'ObjectDecoderCNN: requested latent size: {self.features_dim}, '
                     f'cnn input (h*w): {feature_map_size}, (latent_size / h*w)={self.features_dim / feature_map_size} ->'
                     f' latent projection mode: {self.projection_mode},'
                     f' project {self.features_dim} -> {output_z_cnn} ({flattened_z_cnn})')

        n_film_layers = 1
        self.film_layer = self.get_mlp(in_dim=self.context_dim, out_dim=n_film_layers * 2 * self.in_ch)

        self.num_upsample = max(int(np.log2(patch_size[0])) - int(np.log2(self.fc_res)), 0)
        # print(f'ObjDecCNN: fc to cnn num upsample: {num_upsample}')
        attn_res = [max(self.patch_size[0] // 16, 1)]
        ch_mult = ch_mult[:self.num_upsample + 1]
        z_channels = self.ch_feature_dim
        self.cnn = Decoder(ch=base_ch, out_ch=self.num_chans, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                           attn_resolutions=attn_res, dropout=0.0, resamp_with_conv=True,
                           resolution=self.patch_size[0], z_channels=z_channels, give_pre_end=False,
                           padding_mode=pad_mode, residual=self.use_resblock, upsample_method='nearest',
                           mid_blocks=cnn_mid_blocks)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if self.init_conv_layers:
                    nn.init.normal_(m.weight, 0.0, self.init_conv_fg_std)
                if self.init_zero_bias and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_mlp(self, in_dim, out_dim, linear=False):
        if linear:
            return nn.Linear(in_dim, out_dim)
        else:
            activation_f = nn.GELU if self.activation == 'gelu' else nn.ReLU
            hidden_dim = self.mlp_hidden_dim
            mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                activation_f(),
                                nn.Linear(hidden_dim, out_dim))

            return mlp

    def forward(self, x, context=None):
        # x: [bs, n_kp, feat] ; context: [bs, ctx_dim]
        bs, n_kp = x.shape[0], x.shape[1]
        D, H, W = (self.patch_size if isinstance(self.patch_size, tuple)
                else (self.patch_size, self.patch_size, self.patch_size))

        # FiLM params -> broadcast over 3D
        ctx_param = self.film_layer(context)                  # [bs, 2*in_ch]
        ctx_param = ctx_param.view(bs, 1, 2, self.in_ch)      # [bs,1,2,C]
        ctx_param = ctx_param.repeat(1, n_kp, 1, 1)           # [bs,n_kp,2,C]
        ctx_param = ctx_param.view(-1, 2, self.in_ch, 1, 1, 1)  # [bs*N,2,C,1,1,1]
        ctx_gammas, ctx_betas = ctx_param[:, 0], ctx_param[:, 1]  # [bs*N,C,1,1,1]

        x = self.from_latent_lin(x)
        x = x.view(-1, self.ch_feature_dim, self.fc_res, self.fc_res, self.fc_res)  # [B*N,C,fd,fh,fw]
        z = self.from_latent(x)

        conv_in = ctx_gammas * z + ctx_betas
        out = self.cnn(conv_in).view(-1, self.num_chans, D, H, W)  # [B*N,C,D,H,W]

        if self.num_chans == 1:
            # Occupancy logits
            return out

        out_a  = torch.sigmoid(out[:, :1])
        out_rem = out[:, 1:]

        if self.num_chans == 4:
            rgb_act = torch.tanh if self.normalize_rgb else torch.sigmoid
            out_rgb = rgb_act(out_rem)
            return torch.cat([out_a, out_rgb], dim=1)

        if self.num_chans == 5:
            rgb_act = torch.tanh if self.normalize_rgb else torch.sigmoid
            out_rgb, out_d = out_rem[:, :3], out_rem[:, 3:]
            out_rgb = rgb_act(out_rgb)
            out_d   = torch.sigmoid(out_d)
            return torch.cat([out_a, out_rgb, out_d], dim=1)

        return torch.cat([out_a, torch.sigmoid(out_rem)], dim=1)



class ObjectDecoderCNNConcat(nn.Module):
    def __init__(self, patch_size, num_chans=4, bottleneck_size=128, pad_mode='replicate', embed_position=False,
                 use_resblock=False, context_dim=7, normalize_rgb=False, res_from_fc=8,
                 ch_mult=(1, 2, 3), base_ch=32, final_cnn_ch=32, num_res_blocks=2, cnn_mid_blocks=False,
                 mlp_hidden_dim=256):
        super().__init__()
        assert context_dim > 0, f'ObjectDecoderCNNFILM: context dim - {context_dim} must be > 0'
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.num_chans = num_chans
        self.embed_position = embed_position
        self.use_resblock = use_resblock
        self.features_dim = bottleneck_size
        self.context_dim = context_dim
        self.normalize_rgb = normalize_rgb
        self.cnn_mid_blocks = cnn_mid_blocks
        self.mlp_hidden_dim = mlp_hidden_dim

        self.in_ch = final_cnn_ch
        self.fc_res = res_from_fc
        fc_out_dim = self.in_ch * (self.fc_res ** 2)

        feature_map_size = self.fc_res ** 2

        if self.features_dim % feature_map_size == 0:
            self.ch_feature_dim = math.ceil(max(self.features_dim / (res_from_fc ** 2), 1))
            output_z_cnn = (self.ch_feature_dim, self.fc_res, self.fc_res)
            flattened_z_cnn = np.prod(output_z_cnn)

            self.projection_mode = 'fcn'
            self.from_latent_lin = nn.Identity()
        else:
            self.ch_feature_dim = final_cnn_ch
            output_z_cnn = (self.ch_feature_dim, self.fc_res, self.fc_res)
            flattened_z_cnn = np.prod(output_z_cnn)

        self.info = (f'ObjectDecoderCNNConcat: requested latent size: {self.features_dim}, '
                     f'cnn input (h*w): {feature_map_size}, (latent_size / h*w)={self.features_dim / feature_map_size} ->'
                     f' latent projection mode: {self.projection_mode},'
                     f' project {self.features_dim} -> {output_z_cnn} ({flattened_z_cnn})')

        self.from_latent = nn.Conv3d(in_channels=self.ch_feature_dim, out_channels=self.in_ch, kernel_size=1)
        self.context_projection = nn.Sequential(nn.Linear(self.context_dim, mlp_hidden_dim),
                                                nn.GELU(),
                                                nn.Linear(mlp_hidden_dim, self.in_ch))

        self.num_upsample = max(int(np.log2(patch_size[0])) - int(np.log2(self.fc_res)), 0)
        # print(f'ObjDecCNN: fc to cnn num upsample: {num_upsample}')
        attn_res = [max(self.patch_size[0] // 16, 1)]
        ch_mult = ch_mult[:self.num_upsample + 1]
        self.cnn = Decoder(ch=base_ch, out_ch=self.num_chans, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                           attn_resolutions=attn_res, dropout=0.0, resamp_with_conv=True,
                           resolution=self.patch_size[0], z_channels=2 * final_cnn_ch, give_pre_end=False,
                           padding_mode=pad_mode, mid_blocks=cnn_mid_blocks)

    def forward(self, x, context):
        # x: [bs, n_kp, feat]
        # context: [bs, feat]
        bs, n_kp = x.shape[0], x.shape[1]

        ctx_param = self.context_projection(context)  # [bs, hidden_size]
        ctx_param = ctx_param.view(ctx_param.shape[0], 1, self.in_ch)
        ctx_param = ctx_param[:, :, :, None, None].repeat(1, n_kp, 1, self.fc_res, self.fc_res)
        ctx_param = ctx_param.view(-1, *ctx_param.shape[2:])

        x = self.from_latent_lin(x)
        x = x.view(-1, self.ch_feature_dim, self.fc_res, self.fc_res)
        z = self.from_latent(x)
        conv_in = torch.cat([z, ctx_param], dim=1)
        out = self.cnn(conv_in).view(-1, self.num_chans, *self.patch_size)
        out_a, out_rgb = torch.split(out, [1, out.shape[1] - 1], dim=1)
        rgb_func = torch.tanh if self.normalize_rgb else torch.sigmoid
        out = torch.cat([torch.sigmoid(out_a), rgb_func(out_rgb)], dim=1)
        return out

# -------- 3D latent -> seed feature-map projectors --------

class FCToCNN3D(nn.Module):
    def __init__(self, target_hw=16, n_ch=8, pad_mode='replicate',
                 features_dim=2, use_resblock=False, context_dim=0,
                 res_from_fc=8, activation='gelu', mlp_hidden_dim=256):
        super().__init__()
        self.features_dim = features_dim
        self.n_ch = n_ch
        self.fmap_size = res_from_fc         # fd = fh = fw
        self.use_resblock = use_resblock
        self.context_dim = context_dim
        self.activation = activation
        self.mlp_hidden_dim = mlp_hidden_dim

        feature_map_size = self.fmap_size ** 3

        if self.features_dim % feature_map_size == 0:
            self.ch_features_dim = max(self.features_dim // feature_map_size, 1)
            self.projection_mode = 'fcn'
            self.from_latent_lin = nn.Identity()
        else:
            self.ch_features_dim = self.n_ch
            self.projection_mode = 'fc'
            flattened = self.ch_features_dim * feature_map_size
            self.from_latent_lin = self._mlp(self.features_dim, flattened)

        self.info = (f'FCToCNN3D: requested latent {self.features_dim}, seed edge={self.fmap_size}, '
                     f'mode={self.projection_mode}, outC={self.ch_features_dim}')

    def _mlp(self, in_dim, out_dim, linear=False):
        if linear:
            return nn.Linear(in_dim, out_dim)
        act = nn.GELU if self.activation == 'gelu' else nn.ReLU
        return nn.Sequential(nn.Linear(in_dim, self.mlp_hidden_dim), act(), nn.Linear(self.mlp_hidden_dim, out_dim))

    def forward(self, features, context=None):
        B = features.shape[0]
        x = self.from_latent_lin(features)
        x = x.view(B, self.ch_features_dim, self.fmap_size, self.fmap_size, self.fmap_size)
        return x  # [B, C0, D0, H0, W0]


class FCToCNNFILM3D(nn.Module):
    def __init__(self, target_hw=16, n_ch=8, pad_mode='replicate',
                 features_dim=2, use_resblock=False, context_dim=0,
                 res_from_fc=8, activation='gelu', mlp_hidden_dim=256):
        super().__init__()
        self.features_dim = features_dim
        self.n_ch = n_ch
        self.fmap_size = res_from_fc
        self.use_resblock = use_resblock
        self.context_dim = context_dim
        self.activation = activation
        self.mlp_hidden_dim = mlp_hidden_dim

        feature_map_size = self.fmap_size ** 3

        if self.features_dim % feature_map_size == 0:
            self.ch_features_dim = max(self.features_dim // feature_map_size, 1)
            self.from_latent_lin = nn.Identity()
        else:
            self.ch_features_dim = self.n_ch
            flattened = self.ch_features_dim * feature_map_size
            self.from_latent_lin = self._mlp(self.features_dim, flattened)

        # one FiLM layer on the seed channels
        self.film = self._mlp(self.context_dim, 2 * self.ch_features_dim, linear=True)

        self.info = (f'FCToCNNFILM3D: latent {self.features_dim}, seed edge={self.fmap_size}, outC={self.ch_features_dim}')

    def _mlp(self, in_dim, out_dim, linear=False):
        if linear:
            return nn.Linear(in_dim, out_dim)
        act = nn.GELU if self.activation == 'gelu' else nn.ReLU
        return nn.Sequential(nn.Linear(in_dim, self.mlp_hidden_dim), act(), nn.Linear(self.mlp_hidden_dim, out_dim))

    def forward(self, features, context):
        B = features.shape[0]
        x = self.from_latent_lin(features)
        x = x.view(B, self.ch_features_dim, self.fmap_size, self.fmap_size, self.fmap_size)  # [B,C,D,H,W]

        gam_beta = self.film(context)                             # [B, 2*C]
        gamma, beta = gam_beta.chunk(2, dim=-1)                  # [B, C], [B, C]
        gamma = gamma.view(B, -1, 1, 1, 1)
        beta  = beta.view(B, -1, 1, 1, 1)
        return gamma * x + beta                                   # [B,C,D,H,W]


class FCToCNNConcat3D(nn.Module):
    def __init__(self, target_hw=16, n_ch=8, pad_mode='replicate',
                 features_dim=2, use_resblock=False, context_dim=0,
                 res_from_fc=8, mlp_hidden_dim=256, activation='gelu'):
        super().__init__()
        assert context_dim > 0
        self.features_dim = features_dim
        self.n_ch = n_ch
        self.fmap_size = res_from_fc
        self.context_dim = context_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.activation = activation

        feature_map_size = self.fmap_size ** 3

        if self.features_dim % feature_map_size == 0:
            self.ch_features_dim = max(self.features_dim // feature_map_size, 1)
            self.from_latent_lin = nn.Identity()
        else:
            self.ch_features_dim = self.n_ch
            flattened = self.ch_features_dim * feature_map_size
            self.from_latent_lin = self._mlp(self.features_dim, flattened)

        # 1×1×1 conv to map seed channels -> n_ch
        self.from_latent = nn.Conv3d(self.ch_features_dim, self.n_ch, kernel_size=1)
        # project context to a spatial volume with n_ch channels
        self.context_projection = nn.Sequential(
            nn.Linear(self.context_dim, mlp_hidden_dim),
            nn.ReLU(True),
            nn.Linear(mlp_hidden_dim, self.n_ch)
        )

        self.info = (f'FCToCNNConcat3D: latent {self.features_dim}, seed edge={self.fmap_size}, outC={self.n_ch}')

    def _mlp(self, in_dim, out_dim, linear=False):
        if linear:
            return nn.Linear(in_dim, out_dim)
        act = nn.GELU if self.activation == 'gelu' else nn.ReLU
        return nn.Sequential(nn.Linear(in_dim, self.mlp_hidden_dim), act(), nn.Linear(self.mlp_hidden_dim, out_dim))

    def forward(self, features, context):
        B = features.shape[0]
        x = self.from_latent_lin(features)
        x = x.view(B, -1, self.fmap_size, self.fmap_size, self.fmap_size)   # [B,Cs,D,H,W]
        z = self.from_latent(x)                                             # [B,n_ch,D,H,W]

        ctx = self.context_projection(context).view(B, self.n_ch, 1, 1, 1)
        ctx = ctx.repeat(1, 1, self.fmap_size, self.fmap_size, self.fmap_size)  # [B,n_ch,D,H,W]
        return torch.cat([z, ctx], dim=1)                                   # [B, 2*n_ch, D,H,W]

class BgDecoder(nn.Module):
    def __init__(self, cdim=3, image_size=64,
                 pad_mode='replicate', dropout=0.0, learned_bg_feature_dim=16,
                 use_resblock=False, context_dim=0, film=False, timestep_horizon=1,
                 bg_res_from_fc=8, bg_ch_mult=(1, 2, 3), bg_base_ch=32, bg_final_cnn_ch=32, num_res_blocks=2,
                 decode_with_ctx=False, normalize_rgb=False, cnn_mid_blocks=False, mlp_hidden_dim=256,
                 init_zero_bias=True, init_conv_layers=True, init_conv_bg_std=0.005):
        super().__init__()

        self.image_size = image_size
        self.feature_map_edge = int(image_size // (2 ** (len(bg_ch_mult) - 1)))  # seed edge
        self.dropout = dropout
        self.learned_bg_feature_dim = learned_bg_feature_dim
        self.context_dim = context_dim
        self.cdim = cdim
        self.use_resblock = use_resblock
        self.film = film
        self.decode_with_ctx = decode_with_ctx
        self.normalize_rgb = normalize_rgb
        self.timestep_horizon = (timestep_horizon + 1) if timestep_horizon > 1 else 1
        self.cnn_mid_blocks = cnn_mid_blocks
        self.mlp_hidden_dim = mlp_hidden_dim

        self.init_zero_bias = init_zero_bias
        self.init_conv_layers = init_conv_layers
        self.init_conv_bg_std = init_conv_bg_std

        # ------- 3D latent -> seed -------
        if self.context_dim > 0 and self.decode_with_ctx:
            latent_proj_net = FCToCNNFILM3D if self.film else FCToCNNConcat3D
        else:
            latent_proj_net = FCToCNN3D

        decoder_base_ch = bg_final_cnn_ch  # just a label for Decoder "ch" param

        self.latent_to_feat_map = latent_proj_net(
            target_hw=self.feature_map_edge,
            n_ch=decoder_base_ch,
            features_dim=self.learned_bg_feature_dim,
            pad_mode=pad_mode,
            use_resblock=self.use_resblock,
            context_dim=self.context_dim,
            res_from_fc=bg_res_from_fc,
            mlp_hidden_dim=mlp_hidden_dim
        )

        # ------- upsampling depth & channel multipliers -------
        self.num_bg_upsample = max(int(np.log2(self.image_size)) - int(np.log2(self.feature_map_edge)), 0)
        attn_res = [max(self.image_size // 16, 1)]
        bg_ch_mult = bg_ch_mult[:self.num_bg_upsample + 1]

        # in_z_ch depends on projector (Concat doubles it)
        if self.decode_with_ctx and self.context_dim > 0 and not self.film:
            in_z_ch = 2 * self.latent_to_feat_map.n_ch
        else:
            # FILM3D and plain 3D keep channels = ch_features_dim (FILM uses same C)
            in_z_ch = getattr(self.latent_to_feat_map, 'ch_features_dim', decoder_base_ch)

            # For Concat3D we returned 2*n_ch; for FILM3D and FCToCNN3D we used ch_features_dim
            if isinstance(self.latent_to_feat_map, FCToCNN3D):
                in_z_ch = self.latent_to_feat_map.ch_features_dim
            elif isinstance(self.latent_to_feat_map, FCToCNNFILM3D):
                in_z_ch = self.latent_to_feat_map.ch_features_dim
            elif isinstance(self.latent_to_feat_map, FCToCNNConcat3D):
                in_z_ch = 2 * self.latent_to_feat_map.n_ch

        # ------- 3D decoder -------
        self.cnn = Decoder(
            ch=decoder_base_ch, out_ch=self.cdim, ch_mult=bg_ch_mult, num_res_blocks=num_res_blocks,
            attn_resolutions=attn_res, dropout=0.0, resamp_with_conv=True,
            resolution=self.image_size, z_channels=in_z_ch, give_pre_end=False,
            padding_mode=pad_mode, residual=self.use_resblock, upsample_method='nearest',
            mid_blocks=cnn_mid_blocks
        )

        self.info = self.latent_to_feat_map.info
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if self.init_conv_layers:
                    nn.init.normal_(m.weight, 0, self.init_conv_bg_std)
                if self.init_zero_bias and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def decode_all(self, z_bg_features, z_ctx=None, warmup=False):
        feature_vol = self.latent_to_feat_map(z_bg_features, z_ctx)   # [B, C0, D0, H0, W0]
        bg_rec = self.cnn(feature_vol)                                # [B, cdim, D, H, W]
        if self.cdim == 1:
            return bg_rec  # logits
        act = torch.tanh if self.normalize_rgb else torch.sigmoid
        return act(bg_rec)

    def forward(self, z_bg_features, z_ctx=None, warmup=False):
        return self.decode_all(z_bg_features, z_ctx, warmup)


class BgEncoder(nn.Module):
    def __init__(self, cdim=3, image_size=64, pad_mode='replicate', dropout=0.0,
                 learned_feature_dim=16, use_resblock=False, activation='gelu', cnn_mid_blocks=False,
                 ch_mult=(1, 2, 3), base_ch=32, final_cnn_ch=32, num_res_blocks=2, interaction_features=False,
                 mlp_hidden_dim=256, timestep_horizon=1, add_particle_temp_embed=False, init_std=0.2,
                 features_dist='gauss', n_bg_categories=4, n_bg_classes=4,
                 # init
                 init_zero_bias=True, init_conv_layers=True, init_conv_bg_std=0.005):
        super().__init__()

        # ----- core config -----
        self.image_size = image_size              # assume cubic volume: H=W=L=image_size
        self.dropout = dropout
        self.features_dim = learned_feature_dim
        self.features_dist = features_dist
        self.n_bg_categories = n_bg_categories
        self.n_bg_classes = n_bg_classes
        assert learned_feature_dim > 0
        self.cdim = cdim
        self.n_kp_enc = final_cnn_ch
        self.interaction_features = interaction_features
        self.use_resblock = use_resblock
        self.activation = activation
        self.cnn_mid_blocks = cnn_mid_blocks
        self.mlp_hidden_dim = mlp_hidden_dim
        self.timestep_horizon = (timestep_horizon + 1) if timestep_horizon > 1 else 1
        self.add_particle_temp_embed = add_particle_temp_embed

        # init flags
        self.init_zero_bias = init_zero_bias
        self.init_conv_layers = init_conv_layers
        self.init_conv_bg_std = init_conv_bg_std

        # output fmap size per axis after the 3D encoder pyramid
        self.out_axis = int(image_size // (2 ** (len(ch_mult) - 1)))
        feature_map_size = self.out_axis * self.out_axis * self.out_axis

        # ----- 3D CNN encoder -----
        attn_res = [max(self.image_size // 16, 1)]
        self.bg_cnn_enc = Encoder(
            ch=base_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
            attn_resolutions=attn_res, dropout=0.0, resamp_with_conv=True,
            in_channels=self.cdim, resolution=self.image_size, z_channels=final_cnn_ch, double_z=False,
            padding_mode=pad_mode, residual=self.use_resblock, in_conv_kernel_size=3, mid_blocks=cnn_mid_blocks
        )
        self.cnn_out_shape = self.get_cnn_shape_3d()  # (C_out, D_out, H_out, W_out)

        # whether we output logvar
        self.output_logvar = (not self.interaction_features and self.features_dist != 'categorical')

        # ----- projection head (FCN vs FC) -----
        if self.features_dim % feature_map_size == 0:
            # FCN mode: 1x1x1 conv to reduce channels, keep spatial
            self.ch_learned_feature_dim = math.ceil(max(self.features_dim / feature_map_size, 1))
            out_ch = 2 * self.ch_learned_feature_dim if self.output_logvar else self.ch_learned_feature_dim
            self.to_latent = nn.Conv3d(in_channels=final_cnn_ch, out_channels=out_ch, kernel_size=1)
            output_z_cnn = (self.ch_learned_feature_dim, *self.cnn_out_shape[-3:])
            flattened_z_cnn = int(np.prod(output_z_cnn))

            if self.timestep_horizon > 1 and self.add_particle_temp_embed:
                # [1, T, C, D, H, W] to broadcast over batch
                self.temp_embed = nn.Parameter(
                    init_std * torch.randn(1, self.timestep_horizon, final_cnn_ch,
                                           self.cnn_out_shape[-3], self.cnn_out_shape[-2], self.cnn_out_shape[-1])
                )
            else:
                self.temp_embed = None

            self.projection_mode = 'fcn'
            self.to_mu = nn.Identity()
            self.to_logvar = nn.Identity()
        else:
            # FC mode: flatten and MLP
            self.ch_learned_feature_dim = final_cnn_ch
            self.to_latent = nn.Identity()
            output_z_cnn = (self.ch_learned_feature_dim, *self.cnn_out_shape[-3:])
            flattened_z_cnn = int(np.prod(output_z_cnn))

            if self.timestep_horizon > 1 and self.add_particle_temp_embed:
                self.temp_embed = nn.Parameter(init_std * torch.randn(1, self.timestep_horizon, flattened_z_cnn))
            else:
                self.temp_embed = None

            self.projection_mode = 'fc'
            self.to_mu = self.get_mlp(flattened_z_cnn, self.features_dim)
            self.to_logvar = self.get_mlp(flattened_z_cnn, self.features_dim) if self.output_logvar else nn.Identity()

        self.info = (f'BgEncoder3D: requested latent={self.features_dim}, '
                     f'cnn (D*H*W)={feature_map_size}, mode={self.projection_mode}, '
                     f'project {output_z_cnn} ({flattened_z_cnn}) -> {self.features_dim}')

        self.init_weights()

    # ---------- inits ----------
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if self.init_conv_layers:
                    nn.init.normal_(m.weight, 0, self.init_conv_bg_std)
                if self.init_zero_bias and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # leave defaults
                pass

    def get_mlp(self, in_dim, out_dim, linear=False):
        if linear:
            return nn.Linear(in_dim, out_dim)
        activation_f = nn.GELU if self.activation == 'gelu' else nn.ReLU
        hidden_dim = self.mlp_hidden_dim
        return nn.Sequential(nn.Linear(in_dim, hidden_dim), activation_f(), nn.Linear(hidden_dim, out_dim))

    def get_cnn_shape_3d(self):
        dummy = torch.rand(1, self.cdim, self.image_size, self.image_size, self.image_size)
        out = self.bg_cnn_enc(dummy)
        if isinstance(out, tuple):
            out = out[1]
        return out.shape[1:]  # (C_out, D_out, H_out, W_out)

    # ---------- forward pieces ----------
    def encode_bg_features(self, x, masks=None, timesteps=None):
        """
        x:      [B, C, H, W, L]
        masks:  [B, 1, H, W, L]  (optional, background mask)
        """
        if masks is not None:
            x_in = x * masks
        else:
            x_in = x

        enc_out = self.bg_cnn_enc(x_in)
        cnn_features = enc_out[1] if isinstance(enc_out, tuple) else enc_out  # [B, C', D', H', W']

        if self.projection_mode == 'fcn' and self.temp_embed is not None:
            # add temporal embedding across timesteps if provided
            # reshape to [B, T, C', D', H', W'], add, then back
            B = cnn_features.shape[0]
            if timesteps is None:
                timesteps = 1
            feat = cnn_features.view(-1, timesteps, *cnn_features.shape[1:]) + self.temp_embed[:, :timesteps]
            cnn_features = feat.view(B, *cnn_features.shape[1:])

        z_feat = self.to_latent(cnn_features)  # [B, out_ch(or C'), D', H', W']

        if self.projection_mode == 'fc':
            flat = z_feat.view(z_feat.shape[0], -1)  # [B, C'*D'*H'*W']
            if self.temp_embed is not None and timesteps is not None and timesteps > 1:
                B = flat.shape[0]
                feat = flat.view(-1, timesteps, *flat.shape[1:]) + self.temp_embed[:, :timesteps]
                flat = feat.view(B, -1)
            if self.interaction_features:
                mu_bg = self.to_mu(flat); logvar_bg = None
            else:
                mu_bg = self.to_mu(flat); logvar_bg = self.to_logvar(flat)
        else:
            # FCN mode: collapse spatial to vector at the end
            flat = z_feat.view(z_feat.shape[0], -1)
            if self.interaction_features:
                mu_bg = self.to_mu(flat); logvar_bg = None
            else:
                mu_bg = self.to_mu(flat); logvar_bg = self.to_logvar(flat)

        return mu_bg, logvar_bg

    def encode_all(self, x, masks=None, deterministic=False, timesteps=None):
        """
        x: [B, C, H, W, L]
        """
        mu_bg, logvar_bg = self.encode_bg_features(x, masks, timesteps)
        if self.interaction_features:
            z_bg = mu_bg
        else:
            z_bg = mu_bg if deterministic else self.reparameterize(mu_bg, logvar_bg)
        # 3D: z_kp placeholder is 3D coords
        z_kp = torch.zeros(mu_bg.shape[0], 1, 3, device=x.device, dtype=torch.float)
        return {'mu_bg': mu_bg, 'logvar_bg': logvar_bg, 'z_bg': z_bg, 'z_kp': z_kp}

    def forward(self, x, masks=None, deterministic=False, timesteps=None):
        out = self.encode_all(x, masks, deterministic, timesteps)
        return {'mu_bg': out['mu_bg'], 'logvar_bg': out['logvar_bg'],
                'z_bg': out['z_bg'], 'z_kp': out['z_kp']}


class ParticleAttributeEncoder(nn.Module):
    """
    Glimpse-encoder: encodes patches visual features in a variational fashion (mu, log-variance).
    Useful for object-based scenes.
    """

    def __init__(self, anchor_size, image_size, n_particles, cnn_channels=(16, 16, 32), margin=0, ch=3, max_offset=1.0,
                 kp_activation='tanh', use_resblock=False, hidden_dim=512, pad_mode='replicate', depth=False,
                 obj_on=True, scale=True, activation='gelu',
                 ch_mult=(1, 2, 3), base_ch=32, final_cnn_ch=32, num_res_blocks=2, cnn_mid_blocks=False,
                 timestep_horizon=1, add_particle_temp_embed=False, init_std=0.2,
                 obj_on_min=1e-4, obj_on_max=100.0,
                 init_zero_bias=True,  # zero bias for conv and linear layers
                 init_conv_layers=True,  # initialize conv layers with normal dist
                 init_conv_fg_std=0.02,  # std for conv fg normal dist
                 ):
        super().__init__()
        self.anchor_size = anchor_size
        self.channels = cnn_channels
        self.image_size = image_size
        self.n_particles = n_particles
        self.patch_size = np.round(anchor_size * (image_size - 1)).astype(int)
        self.margin = margin
        self.crop_size = self.patch_size + 2 * margin
        self.ch = ch
        self.use_resblock = use_resblock
        self.kp_activation = kp_activation
        self.max_offset = max_offset  # max offset of x-y, [-max_offset, +max_offset]
        self.hidden_dim = hidden_dim
        self.with_depth = depth
        self.with_obj_on = obj_on
        self.with_scale = scale
        self.cnn_mid_blocks = cnn_mid_blocks
        self.timestep_horizon = timestep_horizon
        self.add_particle_temp_embed = add_particle_temp_embed
        self.obj_on_min = obj_on_min
        self.obj_on_max = obj_on_max
        self.init_std = init_std
        # initialization
        self.init_zero_bias = init_zero_bias  # zero bias for conv and linear layers
        self.init_conv_layers = init_conv_layers  # initialize conv layers with normal dist
        self.init_conv_fg_std = init_conv_fg_std  # std for conv fg normal dist

        attn_res = [max(self.crop_size // 16, 1)]
        self.cnn = Encoder(ch=base_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                           attn_resolutions=attn_res, dropout=0.0, resamp_with_conv=True, in_channels=self.ch,
                           resolution=self.crop_size, z_channels=final_cnn_ch, double_z=False, padding_mode=pad_mode,
                           residual=self.use_resblock, mid_blocks=cnn_mid_blocks)

        c_out, fD, fH, fW = self.cnn.conv_output_size  # from your Encoder3D.calc_conv_output_size()
        fc_in_dim = c_out * fD * fH * fW
        if self.add_particle_temp_embed and self.timestep_horizon > 1:
            self.temp_embed = nn.Parameter(init_std * torch.randn(1, self.timestep_horizon, 1, fc_in_dim))
        else:
            self.temp_embed = None
        activation_f = nn.GELU if activation == 'gelu' else nn.ReLU

        self.backbone = nn.Identity()
        self.xy_head = nn.Sequential(
            nn.Linear(fc_in_dim, self.hidden_dim),
            (nn.GELU() if activation == 'gelu' else nn.ReLU()),
            nn.Linear(self.hidden_dim, 6)  # mu_x, mu_y, mu_z, logvar_x, logvar_y, logvar_z
        )
        scale_output = 6 if self.with_scale else 3
        self.scale_xy_head = nn.Sequential(
            nn.Linear(fc_in_dim, self.hidden_dim),
            (nn.GELU() if activation == 'gelu' else nn.ReLU()),
            nn.Linear(self.hidden_dim, scale_output)  # mu_sx, mu_sy, mu_sz, logvar_sx, logvar_sy, logvar_sz
        )
        if self.with_obj_on:
            self.obj_on_head = nn.Sequential(nn.Linear(fc_in_dim, self.hidden_dim),
                                             activation_f(),
                                             nn.Linear(self.hidden_dim, 1, bias=False))  # [log_obj_on_a, log_obj_on_b]

        else:
            self.obj_on_head = None
        if self.with_depth:
            self.depth_head = nn.Sequential(nn.Linear(fc_in_dim, self.hidden_dim),
                                            activation_f(),
                                            nn.Linear(self.hidden_dim, 2))  # mu_depth, logvar_depth
        else:
            self.depth_head = None
        self.init_weights()

    def init_weights(self):
        # ---- generic inits ----
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if self.init_conv_layers:
                    nn.init.normal_(m.weight, 0.0, self.init_conv_fg_std)
                if self.init_zero_bias and (m.bias is not None):
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                if self.init_zero_bias and (m.bias is not None):
                    nn.init.constant_(m.bias, 0.0)

        # ---- xy_head: [mu_x, mu_y, mu_z, logvar_x, logvar_y, logvar_z] ----
        W = self.xy_head[-1].weight
        b = self.xy_head[-1].bias
        nn.init.constant_(W, 0.0)
        if b is not None:
            nn.init.constant_(b, 0.0)
            # set only logvar biases
            b.data[3:6] = math.log(0.01 ** 2)

        # ---- scale_xy_head: [mu_sx, mu_sy, mu_sz, logvar_sx, logvar_sy, logvar_sz] ----
        if self.with_scale:
            Ws = self.scale_xy_head[-1].weight
            bs = self.scale_xy_head[-1].bias
            nn.init.constant_(Ws, 0.0)
            if bs is not None:
                nn.init.constant_(bs, 0.0)
                bs.data[3:6] = math.log(0.1 ** 2)  # your chosen init variance for scale

        # ---- obj_on_head (Beta params are produced elsewhere from this logit) ----
        if self.with_obj_on:
            Wo = self.obj_on_head[-1].weight
            bo = self.obj_on_head[-1].bias
            nn.init.constant_(Wo, 0.0)
            if bo is not None:
                nn.init.constant_(bo, 0.0)

        # ---- depth_head: [mu_depth, logvar_depth] (optional) ----
        if self.with_depth and (self.depth_head is not None):
            Wd = self.depth_head[-1].weight
            bd = self.depth_head[-1].bias
            nn.init.constant_(Wd, 0.0)
            if bd is not None:
                nn.init.constant_(bd, 0.0)
                bd.data[1] = math.log(0.01 ** 2)

    def forward(self, x, kp, z_scale=None, timesteps=None, deterministic=False):
        # x: [batch_size, ch, H, W, L]
        # kp: [batch_size, n_kp, 3] in [-1, 1]
        batch_size, C, D, H, W = x.shape
        _, n_kp, _ = kp.shape

        x_repeated = x.unsqueeze(1).repeat(1, n_kp, 1, 1, 1, 1)   # [B, n_kp, C, D, H, W]
        x_repeated = x_repeated.view(-1, C, D, H, W)    

        if z_scale is None:
            frac = torch.tensor(
                [self.patch_size / W,   # sx (x-axis)
                self.patch_size / H,   # sy (y-axis)
                self.patch_size / D],  # sz (z-axis)
                device=x.device, dtype=kp.dtype
            ).view(1, 1, 3)
            z_scale = frac.expand_as(kp)     # [B, n_kp, 3] (sx,sy,sz)
        else:
            z_scale = torch.sigmoid(z_scale)


        z_pos   = kp.reshape(-1, kp.shape[-1])                        # [B*n_kp, 3]
        z_scale = z_scale.view(-1, z_scale.shape[-1])                 # [B*n_kp, 3]

        out_dims = (batch_size * n_kp, C, self.patch_size, self.patch_size, self.patch_size)  # (N,C,D,H,W)
        cropped_objects = spatial_transform(
            x_repeated, kp.reshape(-1,3), z_scale.view(-1,3),
            out_dims, inverse=False, padding_mode='border'
        )
        # cropped_objects: [B*n_kp, C, pd, ph, pw]

        # [batch_size * n_kp, ch, patch_size, patch_size]

        # encode objects - fc
        enc_out = self.cnn(cropped_objects)
        if isinstance(enc_out, tuple):
            cropped_objects_cnn = enc_out[1]
        else:
            cropped_objects_cnn = enc_out

        cropped_objects_flat = cropped_objects_cnn.view(batch_size, n_kp, -1)
        backbone_features = self.backbone(cropped_objects_flat)

        # projection
        backbone_features = self.backbone(backbone_features)
        if timesteps is not None and self.temp_embed is not None:
            orig_shape = backbone_features.shape
            new_feat = backbone_features.view(-1, timesteps, *backbone_features.shape[1:]) + self.temp_embed[:,
            :timesteps]
            backbone_features = new_feat.view(orig_shape)

        if self.with_obj_on:
            obj_on_feat = backbone_features
            obj_on = self.obj_on_head(obj_on_feat)

            obj_on = obj_on.view(batch_size, n_kp, 1)
            lobj_on_a = lobj_on_b = obj_on
            obj_on_a_gate = lobj_on_a.sigmoid()
            obj_on_a = ((1 - obj_on_a_gate) * self.obj_on_min + obj_on_a_gate * self.obj_on_max).exp()
            obj_on_b_gate = 1 - (lobj_on_b * 0 + lobj_on_a).sigmoid()
            obj_on_b = ((1 - obj_on_b_gate) * self.obj_on_min + obj_on_b_gate * self.obj_on_max).exp()
            obj_on_beta_dist = torch.distributions.Beta(obj_on_a, obj_on_b)
            mu_obj_on = obj_on_beta_dist.mean
            if deterministic:
                z_obj_on = obj_on_beta_dist.mean
            else:
                z_obj_on = obj_on_beta_dist.rsample()
        else:
            lobj_on_a = lobj_on_b = obj_on = None
            obj_on_a = obj_on_b = z_obj_on = mu_obj_on = None

        xyz = self.xy_head(backbone_features).view(batch_size, n_kp, -1)
        mu, logvar = torch.chunk(xyz, chunks=2, dim=-1)  # each [..., 3]

        scale_vec = self.scale_xy_head(backbone_features).view(batch_size, n_kp, -1)
        if self.with_scale:
            mu_scale, logvar_scale = torch.chunk(scale_vec, chunks=2, dim=-1)  # each [..., 3]
        else:
            mu_scale, logvar_scale = scale_vec, None


        if self.kp_activation == "tanh":
            mu = self.max_offset * torch.tanh(mu)
        elif self.kp_activation == "sigmoid":
            mu = self.max_offset * torch.sigmoid(mu)

        if self.with_depth:
            depth = self.depth_head(backbone_features)
            depth = depth.view(batch_size, n_kp, 2)
            mu_depth, logvar_depth = torch.chunk(depth, 2, dim=-1)
        else:
            mu_depth = logvar_depth = None

        # print("MU 0: ", mu[0,0])

        spatial_out = {'mu': mu, 'logvar': logvar, 'mu_scale': mu_scale, 'logvar_scale': logvar_scale,
                       'lobj_on_a': lobj_on_a, 'lobj_on_b': lobj_on_b, 'obj_on': obj_on,
                       'mu_depth': mu_depth, 'logvar_depth': logvar_depth, 'obj_on_a': obj_on_a, 'obj_on_b': obj_on_b,
                       'z_obj_on': z_obj_on, 'mu_obj_on': mu_obj_on}
        return spatial_out

class ParticleFeaturesEncoder(nn.Module):
    """
    Glimpse-encoder for voxel patches: encodes visual features (mu, logvar) from 3D crops
    centered at 3D keypoints.

    x:  [B, C, D, H, W]   (D=z, H=y, W=x)
    kp: [B, K, 3]         (x, y, z) in [-1, 1]
    """

    def __init__(self, anchor_size, features_dim, image_size, margin=0, ch=3,
                 use_resblock=False, hidden_dim=256, pad_mode='replicate', activation='gelu',
                 ch_mult=(1, 2, 3), base_ch=32, final_cnn_ch=32, num_res_blocks=2, output_logvar=True,
                 cnn_mid_blocks=False, timestep_horizon=1, add_particle_temp_embed=False, init_std=0.2,
                 init_zero_bias=True, init_conv_layers=True, init_conv_fg_std=0.02):
        super().__init__()
        self.anchor_size = anchor_size
        self.image_size  = image_size              # assume cubic D=H=W=image_size
        self.patch_size  = int(np.round(anchor_size * (image_size - 1)))
        self.margin      = margin
        self.crop_size   = self.patch_size + 2 * margin
        self.ch          = ch
        self.use_resblock = use_resblock
        self.features_dim = features_dim
        self.output_logvar = output_logvar
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.cnn_mid_blocks = cnn_mid_blocks
        self.timestep_horizon = timestep_horizon
        self.add_particle_temp_embed = add_particle_temp_embed

        self.init_zero_bias = init_zero_bias
        self.init_conv_layers = init_conv_layers
        self.init_conv_fg_std = init_conv_fg_std

        attn_res = [max(self.crop_size // 16, 1)]
        self.cnn = Encoder(
            ch=base_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
            attn_resolutions=attn_res, dropout=0.0, resamp_with_conv=True, in_channels=self.ch,
            resolution=self.crop_size, z_channels=final_cnn_ch, double_z=False,
            padding_mode=pad_mode, residual=self.use_resblock, mid_blocks=cnn_mid_blocks
        )

        # ---- Use actual CNN output shape on correct device ----
        self.cnn_out_shape = self.get_cnn_shape()                 # [C_out, fD, fH, fW]
        C_out, fD, fH, fW = self.cnn_out_shape
        fmap_elems = fD * fH * fW                                  # <-- use this, not a heuristic

        # ---- Decide projection path using true fmap size ----
        if self.features_dim % fmap_elems == 0:
            self.ch_feature_dim = max(self.features_dim // fmap_elems, 1)
            z_out_channels = 2 * self.ch_feature_dim if self.output_logvar else self.ch_feature_dim
            self.to_latent = nn.Conv3d(in_channels=final_cnn_ch, out_channels=z_out_channels, kernel_size=1)
            self.projection_mode = 'fcn'
            self.to_mu = nn.Identity()
            self.to_logvar = nn.Identity()
            if self.timestep_horizon > 1 and self.add_particle_temp_embed:
                self.temp_embed = nn.Parameter(
                    init_std * torch.randn(1, self.timestep_horizon, 1, final_cnn_ch, fD, fH, fW)
                )
            else:
                self.temp_embed = None
        else:
            self.ch_feature_dim = final_cnn_ch
            self.to_latent = nn.Identity()
            flattened = self.ch_feature_dim * fmap_elems
            self.projection_mode = 'fc'
            self.to_mu = self.get_mlp(flattened, self.features_dim)
            self.to_logvar = self.get_mlp(flattened, self.features_dim) if self.output_logvar else nn.Identity()
            if self.timestep_horizon > 1 and self.add_particle_temp_embed:
                self.temp_embed = nn.Parameter(init_std * torch.randn(1, self.timestep_horizon, 1, flattened))
            else:
                self.temp_embed = None

        self.init_weights()

        self.info = (f'ParticleFeaturesEncoder3D: requested latent size: {self.features_dim}, '
                     f'cnn output (C,fD,fH,fW)=({C_out},{fD},{fH},{fW}), '
                     f'(latent / fmap_elems)={self.features_dim / max(fmap_elems,1):.3f} -> '
                     f'projection: {self.projection_mode}')

    def get_cnn_shape(self):
        # build dummy on same device/dtype as module
        p = next(self.parameters(), None)
        dev = p.device if p is not None else torch.device('cpu')
        dt  = p.dtype  if p is not None else torch.float32
        dummy = torch.zeros(1, self.ch, self.crop_size, self.crop_size, self.crop_size, device=dev, dtype=dt)
        out = self.cnn(dummy)
        if isinstance(out, tuple): out = out[1]
        return out.shape[1:]  # [C, fD, fH, fW]
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if self.init_conv_layers:
                    nn.init.normal_(m.weight, 0.0, self.init_conv_fg_std)
                if self.init_zero_bias and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_mlp(self, in_dim, out_dim, linear=False):
            if linear:
                return nn.Linear(in_dim, out_dim)
            activation_f = nn.GELU if self.activation == 'gelu' else nn.ReLU
            hidden_dim = self.hidden_dim
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                activation_f(),
                nn.Linear(hidden_dim, out_dim)
            )
    def forward(self, x, kp, z_scale=None, timesteps=None, obj_on=None):
        """
        x:   [B, C, D, H, W]     (D=z, H=y, W=x)
        kp:  [B, K, 3]           (x,y,z) in [-1,1]
        """
        B, C, D, H, W = x.shape

     
        assert x.shape[1] == self.ch, f"Expected {self.ch} channels, got {x.shape[1]}"
        K = kp.shape[1]

        # repeat per keypoint
        x_rep = x.unsqueeze(1).repeat(1, K, 1, 1, 1, 1)     # [B, K, C, D, H, W]
        x_rep = x_rep.view(-1, C, D, H, W)                 # [B*K, C, D, H, W]

        # per-axis scales (sx, sy, sz)
        if z_scale is None:
            frac = torch.tensor(
                [self.patch_size / W,   # sx  (x-axis)
                 self.patch_size / H,   # sy  (y-axis)
                 self.patch_size / D],  # sz  (z-axis)
                device=x.device, dtype=kp.dtype
            ).view(1, 1, 3)
            z_scale = frac.expand_as(kp)                   # [B, K, 3]
        else:
            z_scale = torch.sigmoid(z_scale)

        z_pos   = kp.reshape(-1, 3)                        # [B*K, 3]
        z_scale = z_scale.view(-1, 3)                      # [B*K, 3]

        out_dims = (B * K, C, self.patch_size, self.patch_size, self.patch_size)  # (N,C,D,H,W)
        crops = spatial_transform(x_rep, z_pos, z_scale, out_dims, inverse=False, padding_mode='border')
        # [B*K, C, ps, ps, ps]

        # encode crops
        enc = self.cnn(crops)
        crops_cnn = enc[1] if isinstance(enc, tuple) else enc   # [B*K, C', fD, fH, fW]

        if obj_on is not None:
            obj_on = obj_on.view(-1)                            # [B*K]
            crops_cnn = crops_cnn * obj_on[:, None, None, None, None]

        # temporal embed (FCN path)
        if self.projection_mode == 'fcn' and self.temp_embed is not None:
            orig = crops_cnn.shape
            crops_cnn = crops_cnn.view(B, K, *orig[1:])         # [B, K, C', fD, fH, fW]
            crops_cnn = crops_cnn.view(B // max(timesteps or 1), timesteps or 1, K, *orig[1:])
            crops_cnn = crops_cnn + self.temp_embed[:, : (timesteps or 1)]
            crops_cnn = crops_cnn.view(orig)

        z_feat = self.to_latent(crops_cnn)                      # [B*K, Cz, fD, fH, fW] or Identity
        z_flat = z_feat.view(B, K, -1)                          # [B, K, *]

        # temporal embed (FC path)
        if self.projection_mode == 'fc' and self.temp_embed is not None:
            orig = z_flat.shape                                  # [B, K, F]
            z_flat = z_flat.view(B // max(timesteps or 1), timesteps or 1, K, *orig[2:])
            z_flat = z_flat + self.temp_embed[:, : (timesteps or 1)]
            z_flat = z_flat.view(orig)

        if self.output_logvar:
            mu_features    = self.to_mu(z_flat)
            logvar_features = self.to_logvar(z_flat)
        else:
            mu_features    = self.to_mu(z_flat)
            logvar_features = None

        crops_batched = crops.view(B, K, *crops.shape[1:])      # [B, K, C, ps, ps, ps]

        return {
            'mu_features': mu_features,
            'logvar_features': logvar_features,
            'cropped_objects': crops_batched
        }



"""
DLP components
"""

# expects you already have:
#   - VoxelPatcher(cdim, volume_size, patch_size) with attributes ph,pw,pl
#   - Encoder(...)
#   - AlternativeSpatialSoftmaxKP3D(...)

class DLPPrior(nn.Module):
    def __init__(self, cdim=3, volume_size=64, n_kp=1,
                 pad_mode='replicate',
                 patch_size=16, n_kp_prior=64,
                 kp_range=(-1, 1),
                 use_resblock=False,
                 filtering_heuristic='none',
                 ch_mult=(1, 2, 3), base_ch=32, num_res_blocks=2, cnn_mid_blocks=False,
                 init_zero_bias=True,
                 init_ssm_last_layer=True,
                 init_conv_layers=True,
                 init_conv_fg_std=0.02,
                 use_kmeans_prior=True, kmeans_iters=5, kmeans_tau=1.0,):
        super().__init__()

        # ---- fixed grid shape (no per-batch dependence) ----
        if isinstance(volume_size, int):
            D = H = W = int(volume_size)
        else:
            assert len(volume_size) == 3, "volume_size must be int or (D,H,W)"
            D, H, W = map(int, volume_size)
        self.D, self.H, self.W = D, H, W
        self.volume_size = (D, H, W)

        self.kp_range = kp_range  # typically (-1,1)
        self.n_kp = int(n_kp)
        self.n_kp_prior = int(n_kp_prior)
        self.patch_size = int(patch_size)
        self.cdim = cdim
        self.use_resblock = use_resblock
        self.cnn_mid_blocks = cnn_mid_blocks
        assert filtering_heuristic in ['distance', 'variance', 'random', 'none']
        self.filtering_heuristic = filtering_heuristic

        # inits
        self.init_zero_bias = init_zero_bias
        self.init_ssm_last_layer = init_ssm_last_layer
        self.init_conv_layers = init_conv_layers
        self.init_conv_fg_std = init_conv_fg_std

        # 3D patcher (fixed grid)
        self.patcher = VoxelPatcher(cdim=cdim, volume_size=(D, H, W), patch_size=self.patch_size)
        pd, ph, pw = self.patcher.pd, self.patcher.ph, self.patcher.pw
        self.nz, self.ny, self.nx = D // pd, H // ph, W // pw   # counts along (z,y,x)
        self.num_patches = self.nx * self.ny * self.nz
        self.n_kp_total = self.n_kp * self.num_patches
        self.n_kp_prior = min(self.n_kp_total, self.n_kp_prior)

        # CNN encoder over patches
        attn_res = [max(self.patch_size // 16, 1)]
        self.enc = Encoder(ch=base_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                           attn_resolutions=attn_res, dropout=0.0, resamp_with_conv=True, in_channels=cdim,
                           resolution=self.patch_size, z_channels=self.n_kp, double_z=False, padding_mode='replicate',
                           residual=self.use_resblock, mid_blocks=cnn_mid_blocks)

        # 3D spatial softmax -> per-patch KPs (local coords) + covariance
        self.ssm = AlternativeSpatialSoftmaxKP3D(kp_range=kp_range)


        # -------- precompute tile origins/centers in voxel-index space; keep as buffers --------
        origins = self._precompute_patch_origins_xyz(
            W, H, D, pw, ph, pd
        ) 
        centers = origins + torch.tensor([(pw - 1) / 2.0, (ph - 1) / 2.0, (pd - 1) / 2.0], dtype=torch.float32).view(1, 3)
        self.register_buffer("patch_origins_xyz_idx", origins)
        self.register_buffer("patch_centers_xyz_idx", centers)
        self.register_buffer("size_minus1", torch.tensor([W - 1, H - 1, D - 1], dtype=torch.float32))  # (x,y,z)
        self.register_buffer("patch_size_vec", torch.tensor([pw, ph, pd], dtype=torch.float32))
        
        self.use_kmeans_prior = use_kmeans_prior
        self.kmeans_iters = int(kmeans_iters)
        self.kmeans_tau = float(kmeans_tau)

        # TODO: Make this configurable values
        rgbk_feat_mode="lab"     # {"lab","ilr"}
        rgbk_append_xyz=False
        rgbk_saliency="L"        # {"L","alpha","rgbnorm"}
        rgbk_keep_top=80_000
        rgbk_sample_m=50_000
        rgbk_iters=30
        rgbk_tol=1e-4
        rgbk_ridge=1e-4
        self.rgb_km = FeatureKMeansRGB(
                K=self.n_kp_prior,
                feat_mode=rgbk_feat_mode,
                append_xyz=rgbk_append_xyz,
                saliency=rgbk_saliency,
                keep_top=rgbk_keep_top,
                sample_m=rgbk_sample_m,
                iters=rgbk_iters,
                tol=rgbk_tol,
                ridge=rgbk_ridge,
            )
        self.init_weights()

    # ---------- init helpers ----------
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if self.init_conv_layers:
                    nn.init.normal_(m.weight, 0.0, self.init_conv_fg_std)
                if self.init_zero_bias and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if self.init_ssm_last_layer:
            # bias the final conv slightly negative like your original
            m = self.enc.conv_out
            nn.init.constant_(m.weight, -1.0 * self.init_conv_fg_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _precompute_patch_origins_xyz(W, H, L, pw, ph, pl):
        xs = torch.arange(0, W, step=pw, dtype=torch.float32)
        ys = torch.arange(0, H, step=ph, dtype=torch.float32)
        zs = torch.arange(0, L, step=pl, dtype=torch.float32)
        X, Y, Z = torch.meshgrid(xs, ys, zs, indexing='ij')  # [nx,ny,nz]
        return torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=-1)  # [N,3]

    # ---------- mapping utilities (no last_shape) ----------
    def get_global_kp(self, local_kp):
        """
        local_kp: [B, N, K, 3] in kp_range (e.g., (-1,1)).
        Returns global coords in kp_range, using fixed grid shape and precomputed patch origins.
        """
        xmin, xmax = self.kp_range
        px, py, pz = self.patch_size_vec.unbind(-1)
        # map local (-1,1) -> [0, ps-1]
        scale = torch.stack([px - 1.0, py - 1.0, pz - 1.0]).to(local_kp).view(1,1,1,3)
        lk01   = (local_kp - xmin) / (xmax - xmin)         # (x,y,z) in [0,1]
        lk_idx = lk01 * scale                               # voxel index units (x,y,z)

        origins_xyz = self.patch_origins_xyz_idx.to(local_kp).view(1, -1, 1, 3)  # (x,y,z)
        idx_xyz = lk_idx + origins_xyz

        size_m1_xyz = self.size_minus1.to(local_kp).view(1,1,1,3)                # [W-1,H-1,D-1] (x,y,z)
        g_xyz01 = idx_xyz / size_m1_xyz
        g_xyz   = g_xyz01 * (xmax - xmin) + xmin     
        return g_xyz

    def get_patch_centers(self):
        """
        Returns patch centers in global kp_range coords, shape [N,3].
        """
        xmin, xmax = self.kp_range
        centers = self.patch_centers_xyz_idx
        g01 = centers / self.size_minus1  # [N,3] / [3]
        return g01 * (xmax - xmin) + xmin

    def get_distance_from_patch_centers(self, kp_global):
        """
        kp_global: [B,N,K,3] already in global kp_range.
        Returns squared L2 distance to each patch center: [B,N,K]
        """
        centers = self.get_patch_centers().to(kp_global.device, kp_global.dtype)  # [N,3]
        centers_b = centers.view(1, -1, 1, 3)
        return ((kp_global - centers_b) ** 2).sum(dim=-1)

    def weighted_cov_proximity(self,pts, weights, center,
                                alpha=1.5,           # ↑ emphasize high-mass voxels
                                tau_mode="median",   # proximity length scale
                                tau_mult=1.0,        # ↓ smaller → more shrink
                                lam0=1e-4,           # base ridge
                                mass_norm=True):
        """
        pts: [M,3] in global [-1,1]
        weights: [M] >=0 (voxel mass)
        center: [3] cluster center (x,y,z)

        Effective weights = mass^alpha * exp(-||x-c||^2 / tau^2)
        tau picked per-cluster from distances (robust).
        """
        w = weights.clamp_min(0)

        # distances to center
        d2 = ((pts - center[None])**2).sum(dim=-1)     # [M]

        # pick tau from distances
        if tau_mode == "median":
            tau = d2.median().sqrt() * tau_mult + 1e-9
        elif tau_mode == "p75":
            tau = d2.kthvalue(int(0.75*max(1, d2.numel()))).values.sqrt() * tau_mult + 1e-9
        else:
            tau = d2.mean().sqrt() * tau_mult + 1e-9

        w_eff = (w**alpha) * torch.exp(-d2 / (tau*tau))  # [M]
        W = w_eff.sum() + 1e-12

        # weighted mean and covariance
        mu = (w_eff[:,None] * pts).sum(dim=0) / W
        xc = pts - mu[None]
        cov = (w_eff[:,None,None] * (xc[:,:,None] * xc[:,None,:])).sum(dim=0) / W

        # ridge scaled inversely with mass → more mass = tighter cov
        lam = lam0 / (W.item() if mass_norm else 1.0)
        cov = cov + torch.eye(3, device=cov.device, dtype=cov.dtype) * lam
        return mu, cov, W

    def kmeans_hard(self,x, K, init_centers=None, iters=50, tol=1e-4):
        """
        x: [N,3] points in (-1,1) global coords
        K: number of clusters
        init_centers: [M,3] optional; if provided but M!=K, reduce/expand via KMeans++
        """
        device = x.device
        x = x.float()
        if init_centers is None:
            # KMeans++ on data
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
            # down/up select init to exactly K
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

    # ---------- main API ----------
    def vox_to_patches(self, x):
        return self.patcher.vox_to_patches(x)

    def patches_to_vox(self, x):
        return self.patcher.patches_to_vox(x)

    def zyx_to_xyz(self, v):  # [...,3]
        return torch.stack([v[..., 2], v[..., 1], v[..., 0]], dim=-1)

    def encode_prior(self, x, filtering_heuristic='none', k=None,
                     precomputed_prior=None):
        """
        x: [B, C, D, H, W]  (D=z, H=y, W=x)
        precomputed_prior: optional (kp, cov) tuple from offline kmeans cache
        """
        if precomputed_prior is not None:
            kp, cov = precomputed_prior
            return kp.to(x.device, x.dtype), cov.to(x.device, x.dtype)

        B, C, D, H, W = x.shape
        assert (D, H, W) == self.volume_size, f"got {(D,H,W)}, expected {self.volume_size}"

        # patchify (DHW) -> [B, C, N, pd, ph, pw]
        patches = self.vox_to_patches(x)                          # DHW patcher
        patches = patches.permute(0, 2, 1, 3, 4, 5).contiguous()  # [B, N, C, pd, ph, pw]
        N = patches.shape[1]
        pd, ph, pw = self.patcher.pd, self.patcher.ph, self.patcher.pw

        # if self.kp_mode == "kmeans":
        if True:
            kp, cov, meta = self.rgb_km(x, centers_init_global=self.get_patch_centers().to(x))
            return kp, cov


        # encode
        patches_bn = patches.view(-1, C, pd, ph, pw)              # [B*N, C, pd, ph, pw]
        enc_out = self.enc(patches_bn)                            # -> [B*N, K, pd, ph, pw]
        z = enc_out[1] if isinstance(enc_out, tuple) else enc_out
        assert z.dim() == 5 and z.shape[1] == self.n_kp, f"expected [B*, {self.n_kp}, Dp, Hp, Wp], got {z.shape}"

        kp_local, cov_local = self.ssm(z, probs=False, variance=True)  # [B*N,K,3], [B*N,K,3,3]
        kp_local  = kp_local.view(B, N, self.n_kp, 3)                  # (x,y,z) in patch-normalized coords
        cov_local = cov_local.view(B, N, self.n_kp, 3, 3)

        kp_global = self.get_global_kp(kp_local)                       # [B,N,K,3]  (x,y,z) in global kp_range
        cov_global = cov_local                                         # (optionally rescale to global units)
        kp_global_xyz = self.zyx_to_xyz(kp_global)                     # convert to (z,y,x) for output
        # ---- filtering ----
        if filtering_heuristic == 'distance':
            scores = self.get_distance_from_patch_centers(kp_global)       # [B,N,K]
            scores = scores.view(B, -1)
            M = scores.shape[1]
            k_keep = min(k if k is not None else self.n_kp_prior, M)
            _, idx = torch.topk(scores, k=k_keep, dim=-1, largest=True)    # farthest
        elif filtering_heuristic == 'variance':
            tr = cov_global[..., 0, 0] + cov_global[..., 1, 1] + cov_global[..., 2, 2]
            scores = tr.view(B, -1)
            M = scores.shape[1]
            k_keep = min(k if k is not None else self.n_kp_prior, M)
            _, idx = torch.topk(scores, k=k_keep, dim=-1, largest=False)   # smallest var
        elif filtering_heuristic == 'random':
            kp_flat = kp_global.view(B, -1, 3)
            M = kp_flat.shape[1]
            k_keep = min(k if k is not None else self.n_kp_prior, M)
            idx = torch.rand(B, M, device=x.device).argsort(dim=-1)[:, :k_keep]
            b = torch.arange(B, device=x.device)[:, None]
            return kp_flat[b, idx], cov_global.view(B, -1, 3, 3)[b, idx]
        else:
            # none: return all
            kp_global_xyz = self.zyx_to_xyz(kp_global) 
            return kp_global_xyz.view(B, -1, 3), cov_global.view(B, -1, 3, 3)

        # gather filtered
        b = torch.arange(B, device=x.device)[:, None]
        return kp_global_xyz.view(B, -1, 3)[b, idx], cov_global.view(B, -1, 3, 3)[b, idx]

    def forward(self, x, precomputed_prior=None):
        return self.encode_prior(x, filtering_heuristic=self.filtering_heuristic,
                                 precomputed_prior=precomputed_prior)



class ParticleInteractionEncoder(nn.Module):
    def __init__(self, n_kp_enc, dropout=0.0, learned_feature_dim=16, learned_bg_feature_dim=16, embed_init_std=0.2,
                 projection_dim=128, timestep_horizon=1, pte_layers=1, pte_heads=1,
                 attn_norm_type='rms', hidden_dim=256, use_resblock=True, pad_mode='replicate',
                 temporal_interaction=True, interaction_depth=False, interaction_obj_on=False, activation='gelu',
                 scale_anchor=None,
                 interaction_features=False, ch_mult=(1, 2, 3), base_ch=32, final_cnn_ch=32, num_res_blocks=2, cdim=3,
                 image_size=64, n_views=1, bg=True, use_img_input=True, cnn_mid_blocks=False,
                 particle_positional_embed=True,
                 particle_score=False, norm_layer=True, add_particle_temp_embed=False,
                 features_dist='gauss', n_fg_categories=8, n_fg_classes=4, n_bg_categories=4, n_bg_classes=4,
                 obj_on_min=1e-4, obj_on_max=100.0,
                 particle_anchors=None, use_z_orig=False,
                 init_zero_bias=True,  # zero bias for conv and linear layers
                 init_conv_layers=True,  # initialize conv layers with normal dist
                 init_conv_fg_std=0.02,  # std for conv fg normal dist
                 ):
        super(ParticleInteractionEncoder, self).__init__()
        """
        DLP Foreground Module -- extract objects from an image

        """
        self.n_kp_enc = n_kp_enc
        self.dropout = dropout
        self.learned_feature_dim = learned_feature_dim
        self.learned_bg_feature_dim = learned_bg_feature_dim
        self.features_dist = features_dist
        self.n_fg_categories = n_fg_categories
        self.n_fg_classes = n_fg_classes
        self.n_bg_categories = n_bg_categories
        self.n_bg_classes = n_bg_classes
        assert learned_feature_dim > 0, "learned_feature_dim must be greater than 0"
        self.embed_init_std = embed_init_std
        self.projection_dim = projection_dim
        self.timestep_horizon = (timestep_horizon + 1) if timestep_horizon > 1 else 1
        self.attn_norm_type = attn_norm_type
        self.hidden_dim = hidden_dim
        self.temporal_interaction = temporal_interaction
        self.interaction_depth = interaction_depth
        self.interaction_obj_on = interaction_obj_on
        self.interaction_features = interaction_features
        self.with_bg = bg
        self.use_img_input = use_img_input
        self.activation = activation
        self.cnn_mid_blocks = cnn_mid_blocks
        self.particle_score = particle_score
        self.obj_on_min = obj_on_min
        self.obj_on_max = obj_on_max
        self.add_particle_temp_embed = add_particle_temp_embed
        self.scale_anchor = scale_anchor
        self.use_z_orig = use_z_orig
        self.n_views = n_views

        # initialization
        self.init_zero_bias = init_zero_bias  # zero bias for conv and linear layers
        self.init_conv_layers = init_conv_layers  # initialize conv layers with normal dist
        self.init_conv_fg_std = init_conv_fg_std  # std for conv fg normal dist

        if particle_anchors is None:
            self.register_buffer('particles_anchor', torch.zeros(1, self.n_kp_enc, 3))
            print(" SETTING ")
            self.use_z_orig = False
        else:
            self.register_buffer('particles_anchor', particle_anchors)

        n_particles = self.n_kp_enc  # [n_kp_enc]

        if self.use_img_input:
            # cnn stuff
            self.ctx_pre_pte_latent_dim = projection_dim  # can also be ctx dim
            self.image_size = image_size
            self.output_feat_map_size = int(image_size // (2 ** (len(ch_mult) - 1)))
            self.cdim = cdim

            attn_res = [max(self.image_size // 16, 1)]
            self.ctx_cnn_enc = Encoder(ch=base_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                                       attn_resolutions=attn_res, dropout=0.0, resamp_with_conv=True,
                                       in_channels=self.cdim,
                                       resolution=self.image_size, z_channels=final_cnn_ch, double_z=False,
                                       padding_mode=pad_mode, residual=use_resblock, in_conv_kernel_size=3,
                                       mid_blocks=cnn_mid_blocks)
            self.cnn_out_shape = self.get_cnn_shape()


            # Detect 3D context encoder from output shape (C,D,H,W) vs (C,H,W)
            self.is_3d_ctx = (len(self.cnn_out_shape) == 4)

            if self.is_3d_ctx:
                C_out, D_out, H_out, W_out = self.cnn_out_shape
                vox_elems = D_out * H_out * W_out

                # FCN-style projection (1x1x1 conv) if divisible, else flatten+MLP
                if self.ctx_pre_pte_latent_dim % vox_elems == 0:
                    self.ch_learned_feature_dim = self.ctx_pre_pte_latent_dim // vox_elems
                    # 3D 1x1x1 conv to reach desired channels before flatten
                    self.to_latent = nn.Conv3d(in_channels=final_cnn_ch,
                                            out_channels=self.ch_learned_feature_dim,
                                            kernel_size=1)
                    flattened_z_cnn = self.ch_learned_feature_dim * vox_elems
                    self.projection_mode = 'fcn3d'
                    self.to_latent_lin = nn.Identity()
                else:
                    self.ch_learned_feature_dim = final_cnn_ch
                    self.to_latent = nn.Identity()
                    flattened_z_cnn = final_cnn_ch * vox_elems
                    self.projection_mode = 'fc3d'
                    self.to_latent_lin = self.get_mlp(flattened_z_cnn, self.ctx_pre_pte_latent_dim)

            self.info = (f'ParticleInteractionEncoder: requested latent size: {self.ctx_pre_pte_latent_dim}, '
                         f' latent projection mode: {self.projection_mode},')

            # end cnn stuff
            n_particles += 1  # [ctx + n_kp_enc]
            self.ctx_embeddings = nn.Parameter(
                self.embed_init_std * torch.randn(1, 1, 1, projection_dim))
        else:
            self.info = f'ParticleInteractionEncoder: not using image as input context'
        if self.with_bg:
            n_particles += 1
            self.bg_embeddings = nn.Parameter(self.embed_init_std * torch.randn(1, 1, 1, projection_dim))

        # entities positional embeddings
        if particle_positional_embed:
            self.particle_embeddings = nn.Parameter(
                self.embed_init_std * torch.randn(1, 1, self.n_kp_enc, projection_dim))
        else:
            self.particle_embeddings = nn.Parameter(self.embed_init_std * torch.randn(1, 1, 1, projection_dim))

        # interaction encoder
        self.basic_particle_proj = ParticleAttributesProjection(n_particles=self.n_kp_enc,
                                                                in_features_dim=self.learned_feature_dim,
                                                                hidden_dim=self.hidden_dim,
                                                                output_dim=projection_dim,
                                                                bg_features_dim=self.learned_bg_feature_dim,
                                                                add_ctx_token=False,
                                                                depth=not self.interaction_depth,
                                                                obj_on=not self.interaction_obj_on,
                                                                base_var=False, bg=self.with_bg,
                                                                particle_score=self.particle_score,
                                                                norm_layer=norm_layer,
                                                                use_z_orig=self.use_z_orig)
        if self.add_particle_temp_embed and not self.temporal_interaction:
            self.temp_embed = nn.Parameter(
                self.embed_init_std * torch.randn(1, self.timestep_horizon, 1, projection_dim))
        else:
            self.temp_embed = None

        if self.n_views > 1:
            self.view_embeddings = nn.Parameter(
                self.embed_init_std * torch.randn(1, 1, self.n_views, 1, projection_dim))
        else:
            self.view_embeddings = None

        block_size = self.timestep_horizon if self.temporal_interaction else 1
        self.pte = ParticleSelfAttTransformer(n_embed=self.projection_dim, n_head=pte_heads,
                                              n_layer=pte_layers,
                                              block_size=block_size,
                                              output_dim=self.projection_dim, attn_pdrop=dropout,
                                              resid_pdrop=dropout,
                                              hidden_dim_multiplier=4, positional_bias=False,
                                              activation=activation,
                                              max_particles=None, norm_type=attn_norm_type,
                                              init_std=embed_init_std)

        self.particle_decoder = ParticleAttributeDecoder(n_particles=self.n_kp_enc, input_dim=projection_dim,
                                                         hidden_dim=self.hidden_dim,
                                                         features_dim=learned_feature_dim,
                                                         bg_features_dim=learned_bg_feature_dim,
                                                         depth=self.interaction_depth,
                                                         obj_on=self.interaction_obj_on,
                                                         features=self.interaction_features,
                                                         bg_features=(self.interaction_features and self.with_bg),
                                                         features_dist=self.features_dist)
        self.init_weights()

    def init_weights(self):
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if self.init_conv_layers:
                    nn.init.normal_(m.weight, 0, self.init_conv_fg_std)
                if self.init_zero_bias and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if self.init_zero_bias and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.particle_decoder.init_weights()
        self.pte.init_weights()

    def get_mlp(self, in_dim, out_dim, linear=False):
        if linear:
            return nn.Linear(in_dim, out_dim)
        else:
            activation_f = nn.GELU if self.activation == 'gelu' else nn.ReLU
            hidden_dim = self.hidden_dim
            mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                activation_f(),
                                nn.Linear(hidden_dim, out_dim))
            return mlp

    def get_cnn_shape(self):

        # Heuristic: check whether the ctx encoder is 3D
        is_3d = any(isinstance(m, nn.Conv3d) for m in self.ctx_cnn_enc.modules())

        if is_3d:
            # 3D conv path: B,C,D,H,W
            depth = getattr(self, "image_depth", self.image_size)  # fallback if you don't set image_depth
            dummy_input = torch.rand(1, self.cdim, depth, self.image_size, self.image_size)
        else:
            # 2D conv path: B,C,H,W
            dummy_input = torch.rand(1, self.cdim, self.image_size, self.image_size)

        out = self.ctx_cnn_enc(dummy_input)
        if isinstance(out, tuple):
            out = out[1]
        return out.shape[1:]


    def encode_ctx_features(self, x, masks=None):
        """
        x: [B,C,D,H,W] (3D) or [B,C,H,W] (2D)
        returns: [B, projection_dim]
        """
        if masks is not None:
            x_in = x * masks
        else:
            x_in = x

        enc_out = self.ctx_cnn_enc(x_in)
        cnn_features = enc_out[1] if isinstance(enc_out, tuple) else enc_out  # 5D for 3D, 4D for 2D

        # 3D path
        if cnn_features.dim() == 5:
            # to_latent is Conv3d or Identity sized for (B,C,D,H,W)
            feat = self.to_latent(cnn_features)                 # [B,C',D,H,W]
            feat = feat.reshape(feat.shape[0], -1)              # [B, C'*D*H*W]
            feat = self.to_latent_lin(feat)                     # [B, proj_dim]
            return feat

        # 2D path (unchanged)
        feat = self.to_latent(cnn_features)                     # [B,C',H,W]
        feat = feat.view(feat.shape[0], -1)                     # [B, C'*H*W]
        feat = self.to_latent_lin(feat)                         # [B, proj_dim]
        return feat


    def encode_all(self, x, z, z_scale, z_obj_on, z_depth, z_features, z_bg_features=None, z_base_var=None,
                   z_score=None, patch_id_embed=None, deterministic=False, warmup=False,
                   detach_before_proj=False):
        """
        output order:
        if with_bg and ctx_pool_mode='token': [n_particles, bg, ctx, ctx_token*]
        else: [n_particles, ctx, ctx_token*]
        """
        # x: [bs * n_views, t, ch, h, w]
        bs, timestep_horizon = z.shape[0], z.shape[1]
        z_v = z.detach() if detach_before_proj else z
        z_scale_v = z_scale.detach() if detach_before_proj else z_scale
        z_obj_on_v = z_obj_on.detach() if (z_obj_on is not None and detach_before_proj) else z_obj_on
        z_depth_v = z_depth.detach() if (z_depth is not None and detach_before_proj) else z_depth
        z_features_v = z_features.detach() if detach_before_proj else z_features
        if not self.with_bg:
            z_bg_features = None
        z_bg_features_v = z_bg_features.detach() if (
                z_bg_features is not None and detach_before_proj) else z_bg_features
        z_base_var_v = z_base_var.detach() if z_base_var is not None else z_base_var
        z_score_v = z_score.detach() if z_score is not None else z_score
        if self.use_z_orig:
            print("USE Z ORIGIN")
            z_orig_v = self.particles_anchor.unsqueeze(0).repeat(z_v.shape[0], z_v.shape[1], 1, 1)
        else:
            z_orig_v = None

        particle_projection = self.basic_particle_proj(z=z_v,
                                                       z_scale=z_scale_v,
                                                       z_obj_on=z_obj_on_v,
                                                       z_depth=z_depth_v,
                                                       z_features=z_features_v,
                                                       z_bg_features=z_bg_features_v,
                                                       z_base_var=z_base_var_v,
                                                       z_score=z_score_v,
                                                       z_orig=z_orig_v)
        # add entity pos embeddings
        if self.particle_embeddings.shape[2] == 1:
            p_embeddings = self.particle_embeddings.repeat(bs, timestep_horizon, z.shape[2], 1)
        else:
            p_embeddings = self.particle_embeddings.repeat(bs, timestep_horizon, 1, 1)
        if patch_id_embed is not None:
            p_embeddings = p_embeddings + patch_id_embed
        if self.with_bg:
            bg_embeddings = self.bg_embeddings.repeat(bs, timestep_horizon, 1, 1)
            p_embeddings = torch.cat([p_embeddings, bg_embeddings], dim=2)
        particle_projection = particle_projection + p_embeddings

        if self.use_img_input:
            # Keep shape as-is; encode_ctx_features will handle 2D vs 3D
            x_in = x
            ctx_features = self.encode_ctx_features(x_in)      # [B, proj_dim] or [B,T,proj_dim] if you ever add time here
            # Match the temporal dims you expect downstream:
            # Your code expects [B, T, 1, proj_dim]. If you don't have time, synthesize T=1.
            if ctx_features.dim() == 2:                         # [B, D]
                ctx_features = ctx_features[:, None, :]         # [B, 1, D]
            ctx_features = ctx_features[:, :, None, :]          # [B, T, 1, D]
            ctx_features = ctx_features + self.ctx_embeddings.repeat(ctx_features.shape[0],
                                                                    ctx_features.shape[1], 1, 1)
            particle_projection = torch.cat([particle_projection, ctx_features], dim=2)

            # [bs, t, n_p + 2, proj_dim] if with_bg else [bs, t, n_p + 1, proj_dim]
        #     # [bs, t, n_p + 2, proj_dim]

        if self.n_views > 1:
            # [bs * n_views, t, n, d] -> [bs, t, n_views, n, d] -> [bs, t, n_views * n, d]
            particle_projection = particle_projection.view(-1, self.n_views, *particle_projection.shape[1:])
            particle_projection = particle_projection.permute(0, 2, 1, 3, 4)  # [bs, t, n_views, n, d]
            particle_projection = particle_projection + self.view_embeddings
            particle_projection = particle_projection.reshape(particle_projection.shape[0],
                                                              particle_projection.shape[1],
                                                              -1,
                                                              particle_projection.shape[-1])  # [bs, t, n_views * n, d]

        if timestep_horizon > 1 and not self.temporal_interaction:
            if self.add_particle_temp_embed:
                particle_projection = particle_projection + self.temp_embed[:, :timestep_horizon]
            particle_projection = particle_projection.view(-1, 1, *particle_projection.shape[2:])
            # [bs * ts, 1, n, f]
        particles_out = self.pte(particle_projection)
        particles_out = particles_out.view(-1, timestep_horizon, *particles_out.shape[2:])
        # [bs, ts, n, f]
        if self.n_views > 1:
            # [bs, t, n_views * n, d] -> [bs * n_views, t, n, d]
            particles_out = particles_out.view(particles_out.shape[0], timestep_horizon, self.n_views, -1,
                                               particles_out.shape[-1])
            particles_out = particles_out.permute(0, 2, 1, 3, 4)
            particles_out = particles_out.reshape(-1, *particles_out.shape[2:])
        particle_decoder_out = self.particle_decoder(particles_out)  # [bs * n_views, t, n, d]
        # unpack
        mu_depth = particle_decoder_out['mu_depth']
        logvar_depth = particle_decoder_out['logvar_depth']
        if self.interaction_depth:
            z_depth = reparameterize(mu_depth, logvar_depth) if not deterministic else mu_depth
        else:
            z_depth = None
        mu_features = particle_decoder_out['mu_features']
        logvar_features = particle_decoder_out['logvar_features']
        mu_bg_features = particle_decoder_out['mu_bg_features']
        logvar_bg_features = particle_decoder_out['logvar_bg_features']
        if self.interaction_features:
            mu_features = z_features + mu_features
            if self.features_dist == 'categorical':
                logits = mu_features.view(*mu_features.shape[:-1], self.n_fg_categories, self.n_fg_classes)
                # [bs, T, n_p, n_categories, n_classes]
                probs = logits.softmax(dim=-1)  # [bs, T, n_p, n_categories, n_classes]
                if deterministic:
                    samples = torch.argmax(probs.view(-1, probs.shape[-1]), dim=-1, keepdim=True)
                    samples = F.one_hot(samples.squeeze(-1), num_classes=self.n_fg_classes)
                    samples = samples.view(probs.shape)
                    # straight-through
                    z_features = samples.detach() + (probs - probs.detach())
                    z_features = z_features.view(*mu_features.shape)  # [bs, T, n_p, n_categories * n_classes]
                else:
                    samples = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1)
                    samples = F.one_hot(samples.squeeze(-1), num_classes=self.n_fg_classes)
                    samples = samples.view(probs.shape)
                    # straight-through
                    z_features = samples.detach() + (probs - probs.detach())
                    z_features = z_features.view(*mu_features.shape)  # [bs, T, n_p, n_categories * n_classes]
            else:
                # logvar_features = logvar_features.clamp_max(math.log(0.2 ** 2))
                z_features = reparameterize(mu_features, logvar_features) if not deterministic else mu_features
            if self.with_bg:
                mu_bg_features = z_bg_features + mu_bg_features
                if self.features_dist == 'categorical':
                    logits_bg = mu_bg_features.view(*mu_bg_features.shape[:-1], self.n_bg_categories, self.n_bg_classes)
                    # [bs, T, n_p, n_categories, n_classes]
                    probs_bg = logits_bg.softmax(dim=-1)  # [bs, T, n_p, n_categories, n_classes]
                    if deterministic:
                        samples_bg = torch.argmax(probs_bg.view(-1, probs_bg.shape[-1]), dim=-1, keepdim=True)
                        samples_bg = F.one_hot(samples_bg.squeeze(-1), num_classes=self.n_bg_classes)
                        samples_bg = samples_bg.view(probs_bg.shape)
                        # straight-through
                        z_bg_features = samples_bg.detach() + (probs_bg - probs_bg.detach())
                        z_bg_features = z_bg_features.view(
                            *mu_bg_features.shape)  # [bs, T, n_p, n_categories * n_classes]
                    else:
                        samples_bg = torch.multinomial(probs_bg.view(-1, probs_bg.shape[-1]), num_samples=1)
                        samples_bg = F.one_hot(samples_bg.squeeze(-1), num_classes=self.n_bg_classes)
                        samples_bg = samples_bg.view(probs_bg.shape)
                        # straight-through
                        z_bg_features = samples_bg.detach() + (probs_bg - probs_bg.detach())
                        z_bg_features = z_bg_features.view(
                            *mu_bg_features.shape)  # [bs, T, n_p, n_categories * n_classes]
                else:
                    # logvar_bg_features = logvar_bg_features.clamp_max(math.log(0.2 ** 2))
                    z_bg_features = reparameterize(mu_bg_features,
                                                   logvar_bg_features) if not deterministic else mu_bg_features
        else:
            z_features = z_bg_features = None
        lobj_on_a = particle_decoder_out['lobj_on_a']
        lobj_on_b = particle_decoder_out['lobj_on_b']
        if self.interaction_obj_on:
            obj_on_a_gate = (lobj_on_a).sigmoid()
            obj_on_a = ((1 - obj_on_a_gate) * self.obj_on_min + obj_on_a_gate * self.obj_on_max).exp()
            obj_on_b_gate = 1 - (lobj_on_b * 0 + lobj_on_a).sigmoid()
            obj_on_b = ((1 - obj_on_b_gate) * self.obj_on_min + obj_on_b_gate * self.obj_on_max).exp()
            obj_on_beta_dist = torch.distributions.Beta(obj_on_a, obj_on_b)
            mu_obj_on = obj_on_beta_dist.mean
            z_obj_on = obj_on_beta_dist.rsample() if not deterministic else obj_on_beta_dist.mean
        else:
            obj_on_a = obj_on_b = z_obj_on = mu_obj_on = None

        encode_dict = {'mu_depth': mu_depth, 'logvar_depth': logvar_depth, 'z_depth': z_depth,
                       'obj_on_a': obj_on_a, 'obj_on_b': obj_on_b, 'z_obj_on': z_obj_on, 'mu_obj_on': mu_obj_on,
                       'mu_features': mu_features, 'logvar_features': logvar_features, 'z_features': z_features,
                       'mu_bg_features': mu_bg_features, 'logvar_bg_features': logvar_bg_features,
                       'z_bg_features': z_bg_features, 'z_scale': z_scale, 'z': z}
        return encode_dict

    def forward(self, x, z, z_scale, z_obj_on, z_depth, z_features, z_bg_features=None, z_base_var=None, z_score=None,
                patch_id_embed=None, deterministic=False, warmup=False):
        output_dict = self.encode_all(x, z, z_scale, z_obj_on, z_depth, z_features, z_bg_features, z_base_var, z_score,
                                      patch_id_embed, deterministic=deterministic, warmup=warmup)
        return output_dict


class ParticleContextEncoder(nn.Module):
    def __init__(self, n_kp_enc, dropout=0.1, learned_feature_dim=16, learned_bg_feature_dim=16, embed_init_std=0.02,
                 projection_dim=128, timestep_horizon=1, pte_layers=1, pte_heads=1,
                 attn_norm_type='rms', context_dim=7, hidden_dim=256,
                 activation='gelu',
                 ctx_pool_mode='none', bg=True, causal=True, particle_positional_embed=True,
                 particle_score=False, norm_layer=True,
                 shared_logvar=False, ctx_dist='gauss', n_ctx_categories=4, n_ctx_classes=4,
                 particle_anchors=None, use_z_orig=False,
                 ctx_pool_dim=256, n_pool_ctx_categories=8, n_pool_ctx_classes=8, global_ctx_pool=False):
        super(ParticleContextEncoder, self).__init__()
        """
        This module takes in temporal sequence of particles and outputs latent context,
        which can be per-particle, or global, depending on the pooling type.

        """
        assert ctx_pool_mode in ['none', 'mean', 'max', 'token', 'last', 'mlp']
        self.ctx_pool_mode = ctx_pool_mode
        self.n_kp_enc = n_kp_enc
        self.dropout = dropout
        self.learned_feature_dim = learned_feature_dim
        self.learned_bg_feature_dim = learned_bg_feature_dim
        assert learned_feature_dim > 0, "learned_feature_dim must be greater than 0"
        self.embed_init_std = embed_init_std
        self.projection_dim = projection_dim
        self.timestep_horizon = (timestep_horizon + 1) if timestep_horizon > 1 else 1
        self.attn_norm_type = attn_norm_type
        self.context_dist = ctx_dist
        self.n_ctx_categories = n_ctx_categories
        self.n_ctx_classes = n_ctx_classes
        self.context_dim = context_dim
        self.learned_ctx_token = (ctx_pool_mode == 'token')
        self.n_pool_ctx_categories = n_pool_ctx_categories
        self.n_pool_ctx_classes = n_pool_ctx_classes
        self.ctx_pool_dim = ctx_pool_dim
        if self.context_dist == 'categorical':
            self.ctx_pool_dim = int(self.n_pool_ctx_categories * self.n_pool_ctx_classes)
        self.global_ctx_pool = global_ctx_pool
        self.hidden_dim = hidden_dim
        self.with_bg = bg
        self.activation = activation
        self.is_causal = causal
        # assert not (ctx_pool_mode == 'none' and not self.use_img_input), \
        #     f'context pooling mode can not be "{ctx_pool_mode}" without using image encoder!'
        self.particle_score = particle_score
        self.shared_logvar = shared_logvar
        self.use_z_orig = use_z_orig
        if particle_anchors is None:
            self.register_buffer('particles_anchor', torch.zeros(1, 1, self.n_kp_enc))
            self.use_z_orig = False
        else:
            self.register_buffer('particles_anchor', particle_anchors)

        n_particles = self.n_kp_enc  # [n_kp_enc]
        # entities in attn: [bg*, n_particles, ctx, ctx_tokens*]
        if self.learned_ctx_token:
            n_particles += 1
            self.ctx_token_embeddings = nn.Parameter(
                self.embed_init_std * torch.randn(1, 1, 1, projection_dim))
        if self.learned_ctx_token or self.ctx_pool_mode == 'last':
            block_size = 1  # this means token pooling does not depend on the temporal horizon
            self.cross_attn_block = CrossBlock(n_embed=self.projection_dim, n_head=pte_heads,
                                               block_size=block_size,
                                               attn_pdrop=dropout,
                                               resid_pdrop=dropout,
                                               hidden_dim_multiplier=4, positional_bias=False,
                                               activation='gelu',
                                               max_particles=None, norm_type=attn_norm_type)
        else:
            self.cross_attn_block = None
        if self.with_bg:
            n_particles += 1
            self.bg_embeddings = nn.Parameter(self.embed_init_std * torch.randn(1, 1, 1, projection_dim))

        # entities positional embeddings
        if particle_positional_embed:
            self.particle_embeddings = nn.Parameter(
                self.embed_init_std * torch.randn(1, 1, self.n_kp_enc, projection_dim))
        else:
            self.particle_embeddings = nn.Parameter(self.embed_init_std * torch.randn(1, 1, 1, projection_dim))

        # interaction encoder
        proj_out_dim = projection_dim
        self.basic_particle_proj = ParticleAttributesProjection(n_particles=self.n_kp_enc,
                                                                in_features_dim=self.learned_feature_dim,
                                                                hidden_dim=self.hidden_dim,
                                                                output_dim=proj_out_dim,
                                                                bg_features_dim=self.learned_bg_feature_dim,
                                                                add_ctx_token=False,
                                                                depth=True,
                                                                obj_on=True,
                                                                base_var=False, bg=self.with_bg,
                                                                norm_layer=norm_layer,
                                                                particle_score=self.particle_score,
                                                                use_z_orig=self.use_z_orig)

        block_size = self.timestep_horizon
        self.pte = ParticleSpatioTemporalTransformer(n_embed=self.projection_dim, n_head=pte_heads,
                                                     n_layer=pte_layers,
                                                     block_size=block_size,
                                                     output_dim=self.projection_dim, attn_pdrop=dropout,
                                                     resid_pdrop=dropout,
                                                     hidden_dim_multiplier=4, positional_bias=False,
                                                     activation='gelu',
                                                     max_particles=None, norm_type=attn_norm_type,
                                                     particles_first=False, init_std=embed_init_std,
                                                     causal=self.is_causal)

        self.particle_decoder = ParticleContextDecoder(n_particles=self.n_kp_enc, input_dim=projection_dim,
                                                       hidden_dim=self.hidden_dim,
                                                       context_dim=self.context_dim,
                                                       context_dist=self.context_dist,
                                                       n_ctx_categories=self.n_ctx_categories,
                                                       n_ctx_classes=self.n_ctx_classes,
                                                       learned_ctx_token=self.learned_ctx_token,
                                                       ctx_pool_mode=self.ctx_pool_mode,
                                                       shared_logvar=self.shared_logvar,
                                                       output_ctx_logvar=(ctx_dist != 'categorical'))
        self.init_weights()

    def init_weights(self):
        self.particle_decoder.init_weights()
        self.pte.init_weights()

    def encode_all(self, z, z_scale, z_obj_on, z_depth, z_features, z_bg_features=None, z_base_var=None,
                   z_score=None, patch_id_embed=None, deterministic=False, warmup=False,
                   detach_before_proj=False):
        """
        output order:
        if with_bg and ctx_pool_mode='token': [n_particles, bg, ctx, ctx_token*]
        else: [n_particles, ctx, ctx_token*]
        """
        bs, timestep_horizon = z.shape[0], z.shape[1]
        z_v = z.detach() if detach_before_proj else z
        z_scale_v = z_scale.detach() if detach_before_proj else z_scale
        z_obj_on_v = z_obj_on.detach() if (z_obj_on is not None and detach_before_proj) else z_obj_on
        z_depth_v = z_depth.detach() if (z_depth is not None and detach_before_proj) else z_depth
        z_features_v = z_features.detach() if detach_before_proj else z_features
        if not self.with_bg:
            z_bg_features = None
        z_bg_features_v = z_bg_features.detach() if (
                z_bg_features is not None and detach_before_proj) else z_bg_features
        z_base_var_v = z_base_var.detach() if z_base_var is not None else z_base_var
        z_score_v = z_score.detach() if z_score is not None else z_score
        if self.use_z_orig:
            z_orig_v = self.particles_anchor.unsqueeze(0).repeat(z_v.shape[0], z_v.shape[1], 1, 1)
        else:
            z_orig_v = None

        particle_projection = self.basic_particle_proj(z=z_v,
                                                       z_scale=z_scale_v,
                                                       z_obj_on=z_obj_on_v,
                                                       z_depth=z_depth_v,
                                                       z_features=z_features_v,
                                                       z_bg_features=z_bg_features_v,
                                                       z_base_var=z_base_var_v,
                                                       z_score=z_score_v,
                                                       z_orig=z_orig_v)
        # [bs, T, n_kp + 1, projection_dim or 2 * pctx_dim]

        # add entity pos embeddings
        if self.particle_embeddings.shape[2] == 1:
            p_embeddings = self.particle_embeddings.repeat(bs, timestep_horizon, self.n_kp_enc, 1)
        else:
            p_embeddings = self.particle_embeddings.repeat(bs, timestep_horizon, 1, 1)
        if patch_id_embed is not None:
            p_embeddings = p_embeddings + patch_id_embed
        if self.with_bg:
            bg_embeddings = self.bg_embeddings.repeat(bs, timestep_horizon, 1, 1)
            p_embeddings = torch.cat([p_embeddings, bg_embeddings], dim=2)
        particle_projection = particle_projection + p_embeddings

        particles_out = self.pte(particle_projection)
        particles_out = particles_out.view(bs, timestep_horizon, *particles_out.shape[2:])
        # [bs, ts, n, f]

        if self.learned_ctx_token or self.ctx_pool_mode == 'last':
            if self.learned_ctx_token:
                q_particles = self.ctx_token_embeddings.repeat(bs, timestep_horizon, 1, 1)
                q_particles = q_particles.view(bs * timestep_horizon, 1, *q_particles.shape[2:])
                # [bs * t, 1, 1, embed_dim]
                kv_particles = particles_out[:, :, :self.n_kp_enc + 1]  # only fg + bg particles
                kv_particles = kv_particles.reshape(bs * timestep_horizon, 1, *kv_particles.shape[2:])
                # [bs * t, 1, n_particles + 1, embed_dim]
            else:
                # 'last' pooling
                kv_particles, q_particles = particles_out.split([particles_out.shape[2] - 1, 1], dim=2)
            ctx_ca = self.cross_attn_block(q_particles, kv_particles)
            # [bs * t, 1, 1, embed_dim]
            particles_out = torch.cat([kv_particles, ctx_ca], dim=2)
            particles_out = particles_out.view(bs, timestep_horizon, *particles_out.shape[2:])

        particle_decoder_out = self.particle_decoder(particles_out, deterministic=deterministic)
        # unpack
        mu_context = particle_decoder_out['mu_context']
        logvar_context = particle_decoder_out['logvar_context']
        z_context = particle_decoder_out['z_context']

        encode_dict = {'mu_context': mu_context, 'logvar_context': logvar_context, 'z_context': z_context}
        return encode_dict

    def forward(self, z, z_scale, z_obj_on, z_depth, z_features, z_bg_features=None, z_base_var=None,
                z_score=None, patch_id_embed=None, deterministic=False, warmup=False):
        output_dict = self.encode_all(z, z_scale, z_obj_on, z_depth, z_features, z_bg_features, z_base_var, z_score,
                                      patch_id_embed, deterministic=deterministic, warmup=warmup)
        return output_dict


class ParticleContextDecoder(nn.Module):
    def __init__(self, n_particles, input_dim, hidden_dim,
                 context_dist='gauss',
                 context_dim=7,
                 n_ctx_categories=4,
                 n_ctx_classes=4,
                 learned_ctx_token=False,
                 ctx_pool_mode='none',
                 activation='gelu',
                 shared_logvar=False,
                 output_ctx_logvar=True,
                 projection_base_dim=32,
                 conditional=False,
                 cond_dim=512):
        super().__init__()
        # decoder to map back from PTE's inner dim to the particle's original dimension
        self.n_particles = n_particles
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.context_dist = context_dist
        self.n_ctx_categories = n_ctx_categories
        self.n_ctx_classes = n_ctx_classes
        self.ctx_dim = context_dim
        self.learned_ctx_token = learned_ctx_token
        self.ctx_pool_mode = ctx_pool_mode
        self.shared_logvar = shared_logvar
        self.output_ctx_logvar = output_ctx_logvar
        self.projection_base_dim = projection_base_dim
        self.conditional = conditional
        self.cond_dim = cond_dim
        activation_f = nn.GELU if activation == 'gelu' else nn.ReLU
        base_dim = self.projection_base_dim
        ctx_output_dim = self.ctx_dim if (self.shared_logvar or not output_ctx_logvar) else 2 * self.ctx_dim
        if self.shared_logvar and self.output_ctx_logvar:
            self.ctx_logvar = nn.Parameter(torch.zeros(1, 1, self.ctx_dim))

        if self.conditional:
            # cond projection to FiLM parameters
            self.cond_projection = nn.Sequential(nn.Linear(input_dim, self.hidden_dim),
                                                 activation_f(),
                                                 nn.Linear(self.hidden_dim, 2 * self.hidden_dim))
            # init to zeros (=identity)
            nn.init.constant_(self.cond_projection[-1].weight, 0.0)
            nn.init.constant_(self.cond_projection[-1].bias, 0.0)
            self.ctx_ln = RMSNorm(self.hidden_dim)

            # ctx projection
            if self.ctx_pool_mode == 'mlp':
                self.context_projection = nn.Sequential(nn.Linear(self.cond_dim, base_dim),
                                                        activation_f(),
                                                        nn.Flatten(start_dim=-2, end_dim=-1),
                                                        nn.Linear((self.n_particles + 1) * base_dim, self.hidden_dim),
                                                        activation_f())
            else:
                self.context_projection = nn.Sequential(nn.Linear(self.cond_dim, hidden_dim),
                                                        activation_f(),
                                                        ParticlePool(pool_mode=self.ctx_pool_mode, pool_dim=-2),
                                                        nn.Linear(self.hidden_dim, self.hidden_dim))

            self.context_head = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                              activation_f(),
                                              nn.Linear(self.hidden_dim, ctx_output_dim))

        else:
            self.cond_projection = self.ctx_ln = nn.Identity()
            if self.ctx_pool_mode == 'mlp':
                ctx_head_in_dim = self.hidden_dim
                self.context_projection = nn.Sequential(nn.Linear(input_dim, base_dim),
                                                        activation_f(),
                                                        nn.Flatten(start_dim=-2, end_dim=-1),
                                                        nn.Linear((self.n_particles + 1) * base_dim, ctx_head_in_dim),
                                                        activation_f())
            else:
                self.context_projection = nn.Identity()
                ctx_head_in_dim = input_dim

            self.context_head = nn.Sequential(ParticlePool(pool_mode=self.ctx_pool_mode, pool_dim=-2),
                                              nn.Linear(ctx_head_in_dim, ctx_output_dim))

        self.init_weights()

    def init_weights(self):
        pass

    def reparameterize(self, mu_context, logvar_context, deterministic=False):
        if self.context_dist == 'beta':
            mu_context = torch.exp(mu_context)
            logvar_context = torch.exp(logvar_context)
            beta_context = Beta(mu_context, logvar_context)
            z_context = beta_context.rsample() if not deterministic else beta_context.mean
        elif self.context_dist == 'categorical':
            # raise NotImplementedError(f'context dist: {self.context_dist}')
            logits = mu_context.view(*mu_context.shape[:-1], self.n_ctx_categories, self.n_ctx_classes)
            # [bs, T, n_p, n_categories, n_classes]
            probs = logits.softmax(dim=-1)  # [bs, T, n_p, n_categories, n_classes]
            if deterministic:
                samples = torch.argmax(probs.view(-1, probs.shape[-1]), dim=-1, keepdim=True)
                samples = F.one_hot(samples.squeeze(-1), num_classes=self.n_ctx_classes)
                samples = samples.view(probs.shape)
                # straight-through
                z_context = samples.detach() + (probs - probs.detach())
                z_context = z_context.view(*mu_context.shape)  # [bs, T, n_p, n_categories * n_classes]
            else:
                samples = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1)
                samples = F.one_hot(samples.squeeze(-1), num_classes=self.n_ctx_classes)
                samples = samples.view(probs.shape)
                # straight-through
                z_context = samples.detach() + (probs - probs.detach())
                z_context = z_context.view(*mu_context.shape)  # [bs, T, n_p, n_categories * n_classes]
        else:
            z_context = reparameterize(mu_context, logvar_context) if not deterministic else mu_context

        return z_context

    def forward(self, x, c=None, deterministic=False):
        # x: [bs, n_particles, input_dim]
        # bs, n_particles, in_dim = x.shape
        bs, ts, n_particles = x.shape[0], x.shape[1], x.shape[2]
        if self.ctx_pool_mode == 'last' or self.learned_ctx_token:
            # original
            # in_x = x[:, :, -1]  # only one kl-term for the token
            # same kl-weight as per-particles ctx
            if self.ctx_pool_mode == 'token':
                in_x = x[:, :, -1:].repeat(1, 1, n_particles - 1, 1)
            else:
                in_x = x[:, :, -1:].repeat(1, 1, n_particles, 1)
            if self.conditional and c is not None:
                # cond_proj = self.cond_projection(c)
                # scale, shift = cond_proj.chunk(2, dim=-1)
                #
                # ctx_proj = self.context_projection(in_x)
                # ctx_feat = self.context_head(modulate(self.ctx_ln(ctx_proj), scale, shift, residual=True))

                # --- other direction --- #
                cond_proj = self.cond_projection(x[:, :, -1])
                scale, shift = cond_proj.chunk(2, dim=-1)

                ctx_proj = self.context_projection(c)
                ctx_feat = self.context_head(modulate(self.ctx_ln(ctx_proj), scale, shift, residual=True))
            else:
                ctx_feat = self.context_head(in_x)  # [bs, T, dim]
        else:
            # consider only fg + bg particles for pooling
            ctx_feat = x[:, :, :self.n_particles + 1]
            if self.conditional and c is not None:
                # cond_proj = self.cond_projection(c)
                # if len(cond_proj.shape) == 3:
                #     # [bs, t, d] -> [bs, t, 1, d]
                #     cond_proj = cond_proj.unsqueeze(-2)
                # scale, shift = cond_proj.chunk(2, dim=-1)
                #
                # ctx_proj = self.context_projection(ctx_feat)
                # ctx_feat = self.context_head(modulate(self.ctx_ln(ctx_proj), scale, shift, residual=True))

                # --- other direction --- #
                cond_proj = self.cond_projection(ctx_feat)
                scale, shift = cond_proj.chunk(2, dim=-1)

                ctx_proj = self.context_projection(c)
                if len(ctx_proj.shape) == 3:
                    # [bs, t, d] -> [bs, t, 1, d]
                    ctx_proj = ctx_proj.unsqueeze(-2)
                ctx_feat = self.context_head(modulate(self.ctx_ln(ctx_proj), scale, shift, residual=True))
            else:
                # [bs, ts, hidden_dim]
                ctx_feat = self.context_head(self.context_projection(ctx_feat))

        context_features = ctx_feat
        if self.shared_logvar and self.output_ctx_logvar:
            mu_context = context_features
            if len(mu_context.shape) == 3:
                # [bs, t, dim]
                logvar_context = self.ctx_logvar.repeat(mu_context.shape[0], mu_context.shape[1], 1)
            else:
                logvar_context = self.ctx_logvar.unsqueeze(1).repeat(mu_context.shape[0],
                                                                     mu_context.shape[1],
                                                                     mu_context.shape[2],
                                                                     1)
        elif not self.output_ctx_logvar:
            mu_context = context_features
            logvar_context = None
        else:
            mu_context, logvar_context = torch.chunk(context_features, 2, dim=-1)

        z_context = self.reparameterize(mu_context, logvar_context, deterministic)
        decoder_out = {'mu_context': mu_context, 'logvar_context': logvar_context, 'z_context': z_context}

        return decoder_out


class ParticleEncoder(nn.Module):
    def __init__(self, cdim=3, image_size=64,
                 pad_mode='replicate', dropout=0.0, n_kp_per_patch=1, n_kp_prior=20,
                 patch_size=16, n_kp_enc=20, n_kp_dec=None, learned_feature_dim=16,
                 kp_range=(-1, 1), kp_activation="tanh", anchor_s=0.25,
                 use_resblock=True, embed_init_std=0.2, projection_dim=128, timestep_horizon=1,
                 filtering_heuristic='none', obj_ch_mult_prior=(1, 2),
                 obj_ch_mult=(1, 2, 3), obj_base_ch=32, obj_final_cnn_ch=32, num_res_blocks=2,
                 interaction_features=False, interaction_obj_on=False, interaction_depth=True,
                 temporal_interaction=True, cnn_mid_blocks=False, mlp_hidden_dim=256,
                 embed_prior_patch_pos=False, add_particle_temp_embed=False,
                 features_dist='gauss', n_fg_categories=8, n_fg_classes=4,
                 use_null_features_embed=True, obj_on_min=1e-4, obj_on_max=100.0, warmup_n_kp_ratio=0.35,
                 # initialization
                 init_zero_bias=True,  # zero bias for conv and linear layers
                 init_ssm_last_layer=True,  # spatial softmax initialization
                 init_conv_layers=True,  # initialize conv layers with normal dist
                 init_conv_fg_std=0.02,  # std for conv fg normal dist
                 separate_depth_features=False,
                 depth_feature_dim=16,):
        super(ParticleEncoder, self).__init__()
        """
        DLP Foreground Module – Extracts objects from an image using keypoints and learned features. 
        Combines posterior CNN for full image processing and prior CNN for patch-based keypoint proposals.
        
        Args:
        cdim (int, default=3): Number of channels in the input image.
        image_size (int, default=64): Resolution of the input image (assumes square images).
        pad_mode (str, default='replicate'): Padding mode for CNNs, options are 'zeros' or 'replicate'.
        dropout (float, default=0.0): Dropout rate for CNNs (not used in practice).
        n_kp_per_patch (int, default=1): Number of keypoints proposed per patch.
        n_kp_prior (int, default=20): Number of keypoints filtered from prior proposals.
        patch_size (int, default=16): Size of patches for the prior keypoint proposal network.
        n_kp_enc (int, default=20): Number of posterior keypoints to learn.
        n_kp_dec (int, optional): Number of keypoints for decoder (if different from encoder).
        learned_feature_dim (int, default=16): Dimensionality of latent visual features for glimpses.
        kp_range (tuple, default=(-1, 1)): Range for keypoints; options are (-1, 1) or (0, 1).
        kp_activation (str, default='tanh'): Activation function for keypoints; 'tanh' for range (-1, 1), 'sigmoid' for range (0, 1).
        anchor_s (float, default=0.25): Glimpse size as a ratio of image size (e.g., 0.25 → glimpse size is 0.25 * image_size).
        use_resblock (bool, default=True): Whether to use residual blocks in CNNs.
        embed_init_std (float, default=0.2): Standard deviation for initializing learned tokens.
        projection_dim (int, default=128): Dimensionality of embeddings for transformer input.
        timestep_horizon (int, default=1): Maximum timesteps the model processes at once.
        filtering_heuristic (str, default='none'): Method for filtering prior keypoints. Options: 'distance', 'variance', 'random', 'none'.
        obj_ch_mult (tuple, default=(1, 2, 3)): Multiplicative factors for object feature channels at each CNN stage.
        obj_base_ch (int, default=32): Base number of channels in object feature extractor.
        obj_final_cnn_ch (int, default=32): Number of channels in the final object CNN layer.
        num_res_blocks (int, default=2): Number of residual blocks in object feature extractor.
        interaction_features (bool, default=False): Whether to compute interaction-based features.
        interaction_obj_on (bool, default=False): Whether to include "object-on" features for interactions.
        interaction_depth (bool, default=True): Whether to compute depth information for interactions.
        temporal_interaction (bool, default=True): Whether to model temporal interactions between features.
        cnn_mid_blocks (bool, default=False): Whether to include intermediate blocks in the CNN.
        mlp_hidden_dim (int, default=256): Hidden dimensionality for MLP layers.
        embed_prior_patch_pos (bool, default=False): Whether to embed positional information for prior patches.
        add_particle_temp_embed (bool, default=False): Whether to add temporal embeddings to particles.
        features_dist (str, default='gauss'): Distribution type for keypoint features. Options: 'gauss'.
        n_fg_categories (int, default=8): Number of foreground categories for classification.
        n_fg_classes (int, default=4): Number of foreground classes for classification.
        use_null_features_embed (bool, default=True): Whether to use a learned embedding for filtered-out particles.
        obj_on_min (float, default=1e-4): Minimum concentration value in Beta dist for transparency" probabilities.
        obj_on_max (float, default=100.0): Maximum concentration value in Beta dist for transparency" probabilities.
        """
        self.image_size = image_size
        self.dropout = dropout
        self.kp_range = kp_range
        self.n_kp_per_patch = n_kp_per_patch
        self.n_kp_enc = n_kp_enc
        self.n_kp_dec = self.n_kp_enc if n_kp_dec is None else n_kp_dec
        self.n_kp_prior = n_kp_prior
        self.kp_activation = kp_activation
        self.patch_size = patch_size
        self.anchor_patch_s = patch_size / image_size
        self.features_dim = int(image_size // (2 ** (len(obj_ch_mult) - 1)))
        self.learned_feature_dim = learned_feature_dim
        self.features_dist = features_dist
        self.n_fg_categories = n_fg_categories
        self.n_fg_classes = n_fg_classes
        assert learned_feature_dim > 0, "learned_feature_dim must be greater than 0"
        self.anchor_s = anchor_s
        self.obj_patch_size = np.round(anchor_s * (image_size - 1)).astype(int)
        self.cdim = cdim
        self.use_resblock = use_resblock
        self.embed_init_std = embed_init_std
        self.projection_dim = projection_dim
        self.timestep_horizon = (timestep_horizon + 1) if timestep_horizon > 1 else 1
        self.num_patches = int((image_size // self.patch_size) ** 3)
        self.interaction_features = interaction_features
        self.interaction_depth = interaction_depth
        self.interaction_obj_on = interaction_obj_on
        self.temporal_interaction = temporal_interaction
        self.add_particle_temp_embed = add_particle_temp_embed
        self.cnn_mid_blocks = cnn_mid_blocks
        self.mlp_hidden_dim = mlp_hidden_dim
        self.embed_prior_patch_pos = embed_prior_patch_pos
        self.obj_on_min = obj_on_min
        self.obj_on_max = obj_on_max
        self.use_null_features_embed = use_null_features_embed
        self.warmup_n_kp_ratio = warmup_n_kp_ratio
        # initialization
        self.init_zero_bias = init_zero_bias  # zero bias for conv and linear layers
        self.init_ssm_last_layer = init_ssm_last_layer  # spatial softmax initialization
        self.init_conv_layers = init_conv_layers  # initialize conv layers with normal dist
        self.init_conv_fg_std = init_conv_fg_std  # std for conv fg normal dist
        
        self.separate_depth_features = separate_depth_features
        self.depth_feature_dim = depth_feature_dim

        self.prior_encoder = DLPPrior(cdim=cdim, volume_size=image_size, n_kp=self.n_kp_per_patch,
                                      patch_size=patch_size, kp_range=kp_range, pad_mode=pad_mode,
                                      n_kp_prior=n_kp_prior,
                                      filtering_heuristic=filtering_heuristic,
                                      ch_mult=obj_ch_mult_prior, base_ch=obj_base_ch, num_res_blocks=num_res_blocks,
                                      use_resblock=use_resblock, cnn_mid_blocks=cnn_mid_blocks,
                                      init_ssm_last_layer=init_ssm_last_layer, init_conv_layers=init_conv_layers,
                                      init_conv_fg_std=init_conv_fg_std)

        # attribute encoder - anchor (z_a), offset (z_o), scale (z_s)
        anchor_s_att = patch_size / image_size
        self.particle_attribute_enc = ParticleAttributeEncoder(anchor_size=anchor_s, image_size=image_size,
                                                               n_particles=self.n_kp_prior,
                                                               margin=0, ch=cdim,
                                                               kp_activation=kp_activation,
                                                               use_resblock=use_resblock,
                                                               max_offset=1.0,
                                                               pad_mode=pad_mode, depth=not self.interaction_depth,
                                                               obj_on=not self.interaction_obj_on,
                                                               ch_mult=obj_ch_mult, base_ch=obj_base_ch,
                                                               final_cnn_ch=obj_final_cnn_ch,
                                                               num_res_blocks=num_res_blocks,
                                                               cnn_mid_blocks=cnn_mid_blocks,
                                                               hidden_dim=mlp_hidden_dim,
                                                               timestep_horizon=self.timestep_horizon,
                                                               add_particle_temp_embed=add_particle_temp_embed,
                                                               init_std=embed_init_std,
                                                               obj_on_min=self.obj_on_min,
                                                               obj_on_max=self.obj_on_max,
                                                               init_zero_bias=init_zero_bias,
                                                               init_conv_layers=init_conv_layers,
                                                               init_conv_fg_std=init_conv_fg_std)
        # appearance encoder - visual features encoder (z_f)
        output_logvar = (not self.interaction_features and self.features_dist != 'categorical')
        
        # Split RGB and Depth Feature Encoder when RGBD

        self.particle_features_enc = ParticleFeaturesEncoder(anchor_s, learned_feature_dim,
                                                        image_size,
                                                        margin=0, ch=cdim, pad_mode=pad_mode,
                                                        ch_mult=obj_ch_mult, base_ch=obj_base_ch,
                                                        final_cnn_ch=obj_final_cnn_ch,
                                                        num_res_blocks=num_res_blocks,
                                                        output_logvar=output_logvar,
                                                        use_resblock=use_resblock, cnn_mid_blocks=cnn_mid_blocks,
                                                        hidden_dim=mlp_hidden_dim,
                                                        timestep_horizon=self.timestep_horizon,
                                                        add_particle_temp_embed=add_particle_temp_embed,
                                                        init_zero_bias=init_zero_bias,
                                                        init_conv_layers=init_conv_layers,
                                                        init_conv_fg_std=init_conv_fg_std
                                                        )
        num_patches = self.prior_encoder.num_patches  # nx * ny * nz

        if self.embed_prior_patch_pos:
            # one embed per patch (+1 for the null slot if you want a null-embed too)
            self.patch_id_embed = nn.Parameter(
                self.embed_init_std * torch.randn(1, num_patches + 1, mlp_hidden_dim)
            )
        else:
            self.patch_id_embed = None

        # centers already in kp_range, shape [N, 3]
        pc = self.prior_encoder.get_patch_centers().unsqueeze(0)     # [1, N, 3]
        patch_centers = pc                                            # no extra scaling

        # append 3D null center
        null_center = torch.zeros(1, 1, 3, device=pc.device, dtype=pc.dtype)
        patch_centers = torch.cat([patch_centers, null_center], dim=1)  # [1, N+1, 3]

        self.register_buffer('patch_centers', patch_centers)

        # scale prior stays the same
        self.register_buffer(
            'mu_scale_prior',
            torch.tensor(np.log(self.anchor_s / (1 - self.anchor_s + 1e-5)), dtype=torch.float32)
        )
        self.init_weights()

    def init_weights(self):
        self.prior_encoder.init_weights()
        # initialize ssm and other conv ins the same
        # conv1 = self.prior_encoder.enc.conv_in
        # conv2 = self.particle_attribute_enc.cnn.conv_in
        # conv3 = self.particle_features_enc.cnn.conv_in
        # if conv1.weight.shape == conv2.weight.shape:
        #     with torch.no_grad():
        #         conv2.weight.copy_(conv1.weight)
        #         if conv1.bias is not None and conv2.bias is not None:
        #             conv2.bias.copy_(conv1.bias)
        # if conv1.weight.shape == conv3.weight.shape:
        #     with torch.no_grad():
        #         conv3.weight.copy_(conv1.weight)
        #         if conv1.bias is not None and conv3.bias is not None:
        #             conv3.bias.copy_(conv1.bias)

    def encode_prior(self, x, precomputed_prior=None):
        return self.prior_encoder(x, precomputed_prior=precomputed_prior)

    def encode_pos_scale_with_prior(self, x, deterministic=False, warmup=False, timesteps=None,
                                    precomputed_prior=None):
        batch_size, ch, d, h, w = x.shape

        # prior now returns (x,y,z) and full covariance
        kp_p, cov_kp = self.encode_prior(x, precomputed_prior=precomputed_prior)  # kp_p: [B, n_kp_prior, 3], cov_kp: [B, n_kp_prior, 3, 3]
        
        # kp_init: [B, n_kp_prior, 3] in [-1, 1]
        kp_init = kp_p

        # 0) create or filter anchors
        if kp_init is None:
            # randomly sample n_kp_prior kp in 3D
            mu = torch.rand(batch_size, self.n_kp_prior, 3, device=x.device) * 2 - 1
        else:
            mu = kp_init  # [B, n_kp_prior, 3]

        # keep logvar simple (zero) unless you want to use diag(cov_kp)
        logvar = torch.zeros_like(mu)  # [B, n_kp_prior, 3]

        # deterministic base for Chamfer-KL (unchanged behavior)
        z_base = mu + 0.0 * logvar  # [B, n_kp_prior, 3]

        # 1) posterior offsets and scale
        particle_stats_dict = self.particle_attribute_enc(
            x, z_base, timesteps=timesteps, deterministic=deterministic
        )

        mu_offset     = particle_stats_dict['mu']           # [..., 3]
        logvar_offset = particle_stats_dict['logvar']       # [..., 3]
        mu_scale      = particle_stats_dict['mu_scale']     # [..., 3]
        logvar_scale  = particle_stats_dict['logvar_scale'] # [..., 3] or None

        if not self.interaction_obj_on:
            lobj_on_a   = particle_stats_dict['lobj_on_a']
            lobj_on_b   = particle_stats_dict['lobj_on_b']
            obj_on_a    = particle_stats_dict['obj_on_a']
            obj_on_b    = particle_stats_dict['obj_on_b']
            mu_obj_on   = particle_stats_dict['mu_obj_on']
            z_obj_on    = particle_stats_dict['z_obj_on']
        else:
            obj_on_a = obj_on_b = z_obj_on = mu_obj_on = None

        if not self.interaction_depth:
            mu_depth     = particle_stats_dict['mu_depth']
            logvar_depth = particle_stats_dict['logvar_depth']
            z_depth      = mu_depth if deterministic else reparameterize(mu_depth, logvar_depth)
        else:
            mu_depth = logvar_depth = z_depth = None

        # final position
        mu_tot    = z_base + mu_offset
        logvar_tot = logvar_offset
        mu_scale   = self.mu_scale_prior + mu_scale

        # reparameterize
        if deterministic:
            z_offset = mu_offset
            z_scale  = mu_scale
        else:
            z_offset = reparameterize(mu_offset, logvar_offset)
            z_scale  = reparameterize(mu_scale, logvar_scale) if logvar_scale is not None else mu_scale

        z = z_base + z_offset  # [B, n_kp_prior, 3]
        # z = z_base

        # --- NEW: use covariance properly ---
        # per-axis variance from the prior covariance (diag)
        var_kp = torch.diagonal(cov_kp, dim1=-2, dim2=-1)  # [B, n_kp_prior, 3]
        z_base_var = var_kp.detach()
        z_base_cov = cov_kp.detach()


        # optional confidence feature (same shape as logvar_offset)
        confidence_score = particle_stats_dict['logvar'].detach()  # [B, n_kp_prior, 3]
        # concat for a small feature vector per kp (length 6): [prior_var_xyz | posterior_logvar_xyz]
        z_base_var = torch.cat([z_base_var, confidence_score], dim=-1)  # [B, n_kp_prior, 6]

        # simple integer id for each kp
        z_base_id = torch.arange(z_base.shape[-2], device=z_base.device)[None, :, None]  # [1, n_kp_prior, 1]
        z_base_id = z_base_id.repeat(z_base.shape[0], 1, 1)  # [B, n_kp_prior, 1]

        patch_id_embed = self.patch_id_embed.repeat(mu_tot.shape[0], 1, 1) if self.embed_prior_patch_pos else None

        # normalize to [-1,1] using feature length instead of magic number
        feat_len = z_base_var.shape[-1]  # 6
        mu_score = (z_base_var.sum(-1, keepdim=True) / float(feat_len)) * 2 - 1  # [B, n_kp_prior, 1]
        logvar_score = math.log(0.2 ** 2) * torch.ones_like(mu_score)
        z_score = mu_score

        # variance filtering (use summed prior variance; small is better)
        total_var = var_kp.sum(-1)  # [B, n_kp_prior]

        if self.n_kp_enc < self.n_kp_prior:
            n_filter = self.n_kp_enc if not warmup else min(self.n_kp_enc, int(self.warmup_n_kp_ratio * self.n_kp_prior))
            _, embed_ind = torch.topk(total_var, k=n_filter, dim=-1, largest=False)
            batch_ind = torch.arange(batch_size, device=x.device)[:, None]

            mu_tot       = mu_tot[batch_ind, embed_ind]        # [B, n_kp_enc, 3]
            z_base       = z_base[batch_ind, embed_ind]        # [B, n_kp_enc, 3]
            z_base_var   = z_base_var[batch_ind, embed_ind]    # [B, n_kp_enc, 6]
            z_base_id    = z_base_id[batch_ind, embed_ind]     # [B, n_kp_enc, 1]
            mu_offset    = mu_offset[batch_ind, embed_ind]     # [B, n_kp_enc, 3]
            logvar_offset= logvar_offset[batch_ind, embed_ind] # [B, n_kp_enc, 3]
            z            = z[batch_ind, embed_ind]             # [B, n_kp_enc, 3]
            z_offset     = z_offset[batch_ind, embed_ind]      # [B, n_kp_enc, 3]
            z_scale      = z_scale[batch_ind, embed_ind]       # [B, n_kp_enc, 3]
            mu_scale     = mu_scale[batch_ind, embed_ind]      # [B, n_kp_enc, 3]
            mu_score     = mu_score[batch_ind, embed_ind]      # [B, n_kp_enc, 1]
            logvar_score = logvar_score[batch_ind, embed_ind]  # [B, n_kp_enc, 1]
            z_score      = z_score[batch_ind, embed_ind]       # [B, n_kp_enc, 1]
            z_base_cov  = z_base_cov[batch_ind, embed_ind]   # [B, n_kp_enc, 3,3]

            if logvar_scale is not None:
                logvar_scale = logvar_scale[batch_ind, embed_ind]  # [B, n_kp_enc, 3]

            if not self.interaction_obj_on:
                obj_on_a   = obj_on_a[batch_ind, embed_ind]
                obj_on_b   = obj_on_b[batch_ind, embed_ind]
                mu_obj_on  = mu_obj_on[batch_ind, embed_ind]
                z_obj_on   = z_obj_on[batch_ind, embed_ind]

            if not self.interaction_depth:
                z_depth      = z_depth[batch_ind, embed_ind]
                mu_depth     = mu_depth[batch_ind, embed_ind]
                logvar_depth = logvar_depth[batch_ind, embed_ind]

            if self.embed_prior_patch_pos:
                patch_id_embed = patch_id_embed[batch_ind, embed_ind]

        out_dict = {
            'mu': mu, 'logvar': logvar, 'z_base': z_base, 'z': z, 'mu_tot': mu_tot,
            'patch_id_embed': patch_id_embed,
            'mu_scale': mu_scale, 'logvar_scale': logvar_scale, 'z_scale': z_scale,
            'mu_depth': mu_depth, 'logvar_depth': logvar_depth, 'z_depth': z_depth,
            'mu_offset': mu_offset, 'logvar_offset': logvar_offset, 'z_offset': z_offset,
            'kp_p': kp_p,
            # keep both for convenience: full covariance + per-axis variance
            'cov_kp': cov_kp, 'var_kp': var_kp,
            'z_base_var': z_base_var, 'total_var': total_var, 'z_base_cov': z_base_cov,
            'obj_on_a': obj_on_a, 'obj_on_b': obj_on_b, 'z_obj_on': z_obj_on, 'mu_obj_on': mu_obj_on,
            'z_base_id': z_base_id, 'mu_score': mu_score, 'logvar_score': logvar_score, 'z_score': z_score
        }
        return out_dict


    def sample_gauss(self, mu, logvar, deterministic, interaction_features):
        if deterministic or interaction_features or (logvar is None):
            return mu
        return reparameterize(mu, logvar)

    def sample_categorical_logits(self, logits, deterministic, n_cat, n_cls):
        # logits: [..., n_cat*n_cls]
        shape = logits.shape
        probs = logits.view(*shape[:-1], n_cat, n_cls).softmax(dim=-1)
        if deterministic:
            idx = torch.argmax(probs.view(-1, n_cls), dim=-1, keepdim=True)    # [M,1]
            onehot = F.one_hot(idx.squeeze(-1), num_classes=n_cls)
            onehot = onehot.view(*probs.shape).to(logits.dtype)
        else:
            # multinomial over the last dim (classes) per category
            draws = torch.multinomial(
                probs.view(-1, n_cls), num_samples=1, replacement=True
            )  # [M,1]
            onehot = F.one_hot(draws.squeeze(-1), num_classes=n_cls)
            onehot = onehot.view(*probs.shape).to(logits.dtype)
        # straight-through
        st = onehot.detach() + (probs - probs.detach())
        return st.view(*shape)  # back to [..., n_cat*n_cls]

    def gate_with_null(self, mu, gate, null_embed):
        # gate: broadcastable zeros/ones
        if null_embed is None:
            return mu  # no-op if you didn't set one
        if null_embed.shape[-1] != mu.shape[-1]:
            pad = mu.new_zeros(*null_embed.shape[:-1], mu.shape[-1] - null_embed.shape[-1])
            null_embed = torch.cat([null_embed, pad], dim=-1)
        return gate * mu + (1.0 - gate) * null_embed
    def encode_appearance(self, x, z, z_scale, deterministic=False, timesteps=None, obj_on=None):
        """
        Unified (original) behavior:
        - one encoder self.particle_features_enc over x
        - supports features_dist={'gauss','categorical'}
        - gating by obj_on -> null_feature_embed
        - interaction_features toggles sampling vs passthrough

        Split behavior (if self.separate_depth_features=True):
        - runs self.particle_features_enc_rgb on x[..., :3, :, :]
        - runs self.particle_features_enc_depth on x[..., 3:4, :, :]
        - returns separate depth features under '*_depth' keys
        - keeps original keys pointing to RGB branch for back-compat
        - also returns concatenated '*_total' (rgb||depth) when both exist & gaussian
        """
        obj_enc_out = self.particle_features_enc(x, z, z_scale=z_scale, timesteps=timesteps)
        mu_features     = obj_enc_out['mu_features']
        logvar_features = obj_enc_out['logvar_features']
        cropped_objects = obj_enc_out['cropped_objects']

        # obj_on gating
        if obj_on is not None:
            gate = (obj_on > 0.2).to(mu_features.dtype)
            null = self.null_feature_embed
            mu_features = self.gate_with_null(mu_features, gate, null)

        # sampling
        if not self.interaction_features:
            if self.features_dist == 'categorical':
                z_features = self.sample_categorical_logits(
                    mu_features, deterministic, self.n_fg_categories, self.n_fg_classes
                )
            else:  # 'gauss'
                z_features = self.sample_gauss(mu_features, logvar_features)
        else:
            z_features = mu_features



        return {
            'mu_features':           mu_features,
            'logvar_features':       logvar_features,
            'z_features':            z_features,
            'cropped_objects':       cropped_objects,
            # depth-specific keys None for caller uniformity
            'mu_depth_features':     None,
            'logvar_depth_features': None,
            'z_depth_features':      None,
            'cropped_objects_rgb':   cropped_objects,
            'cropped_objects_d':     None,
            'cropped_objects_4ch':   cropped_objects if cropped_objects.shape[2] == 4 else None,
            # totals equal unified values
            'mu_features_total':     mu_features,
            'logvar_features_total': logvar_features,
            'z_features_total':      z_features,
        }


    def encode_all(self, x, deterministic=False, warmup=False, precomputed_prior=None):
        # make sure x is [bs, T, ch, h, w, l]
        if x.dim() == 5:
            # x: [B, C, D, H, W]  -> add T=1
            x = x.unsqueeze(1)  # -> [B, 1, C, D, H, W]

        bs, timestep_horizon, ch, D, H, W = x.shape  # DHW (z,y,x)
        x = x.view(bs * timestep_horizon, ch, D, H, W)  # [B*T, C, D, H, W]

        # repeat precomputed_prior along T if needed
        if precomputed_prior is not None and timestep_horizon > 1:
            kp_pre, cov_pre = precomputed_prior
            kp_pre = kp_pre.unsqueeze(1).expand(-1, timestep_horizon, -1, -1).reshape(bs * timestep_horizon, *kp_pre.shape[1:])
            cov_pre = cov_pre.unsqueeze(1).expand(-1, timestep_horizon, -1, -1, -1).reshape(bs * timestep_horizon, *cov_pre.shape[1:])
            precomputed_prior = (kp_pre, cov_pre)

        # ---- stage 1: positions & scales (DHW throughout) ----
        stage1_dict = self.encode_pos_scale_with_prior(
            x, deterministic=deterministic, warmup=warmup, timesteps=timestep_horizon,
            precomputed_prior=precomputed_prior
        )

        # --- unpack ---
        kp_p         = stage1_dict['kp_p']
        cov_kp       = stage1_dict['cov_kp']          # <-- CHANGED: use covariance, not var
        var_kp      = stage1_dict['var_kp']
        z_base_var   = stage1_dict['z_base_var']
        z_base_cov   = stage1_dict['z_base_cov']
        total_var    = stage1_dict['total_var']
        patch_id_embed = stage1_dict['patch_id_embed']

        z_base       = stage1_dict['z_base']
        mu_offset    = stage1_dict['mu_offset']
        logvar_offset= stage1_dict['logvar_offset']
        z_offset     = stage1_dict['z_offset']
        mu_tot       = stage1_dict['mu_tot']
        z            = stage1_dict['z']
        mu_scale     = stage1_dict['mu_scale']
        logvar_scale = stage1_dict['logvar_scale']
        z_scale      = stage1_dict['z_scale']

        # may be None
        mu_depth     = stage1_dict['mu_depth']
        logvar_depth = stage1_dict['logvar_depth']
        z_depth      = stage1_dict['z_depth']
        obj_on_a     = stage1_dict['obj_on_a']
        obj_on_b     = stage1_dict['obj_on_b']
        mu_obj_on    = stage1_dict['mu_obj_on']
        z_obj_on     = stage1_dict['z_obj_on']

        mu_score     = stage1_dict['mu_score']
        logvar_score = stage1_dict['logvar_score']
        z_score      = stage1_dict['z_score']

        if self.n_kp_enc != self.n_kp_dec and self.interaction_features and self.use_null_features_embed:
            # rank by summed prior covariance trace across x,y,z (smaller = sharper/more confident)
            total_var = cov_kp.diagonal(dim1=-2, dim2=-1).sum(-1)     # <-- CHANGED: trace(cov) per keypoint, shape [B, n_kp_enc]
            n_filter = self.n_kp_dec if not warmup else min(self.n_kp_dec, int(self.warmup_n_kp_ratio * self.n_kp_enc))
            _, embed_ind = torch.topk(total_var, k=n_filter, dim=-1, largest=False)

            batch_ind   = torch.arange(z.shape[0], device=z.device)[:, None]
            z_app       = z[batch_ind, embed_ind].contiguous()          # [B, n_kp_dec, 3]
            z_scale_app = z_scale[batch_ind, embed_ind].contiguous()    # [B, n_kp_dec, 3]

            stage2_dict = self.encode_appearance(
                x, z_app, z_scale_app, deterministic=deterministic,
                timesteps=timestep_horizon, obj_on=None
            )

            # unpack
            cropped_objects     = stage2_dict['cropped_objects']
            mu_features_app     = stage2_dict['mu_features']
            mu_depth_features   = stage2_dict['mu_depth_features']         # None
            logvar_features     = stage2_dict['logvar_features']           # None
            logvar_depth_features = stage2_dict['logvar_depth_features']   # None
            z_features_app      = stage2_dict['z_features']
            z_depth_features    = stage2_dict['z_depth_features']

            mu_features = self.null_feature_embed.repeat(z.shape[0], self.n_kp_enc, 1)
            mu_features[batch_ind, embed_ind] = mu_features_app

            mu_features_depth = self.null_feature_depth_embed.repeat(z.shape[0], self.n_kp_enc, 1)
            mu_features_depth[batch_ind, embed_ind] = mu_depth_features

            z_features = mu_features
            z_depth_features = mu_features_depth

        else:
            stage2_dict = self.encode_appearance(
                x, z, z_scale, deterministic=deterministic, timesteps=timestep_horizon, obj_on=None
            )
            # unpack
            cropped_objects       = stage2_dict['cropped_objects']
            mu_features           = stage2_dict['mu_features']
            logvar_features       = stage2_dict['logvar_features']
            z_features            = stage2_dict['z_features']
            z_depth_features      = stage2_dict['z_depth_features']

        # reshape to [bs, T, ...]
        z_base       = z_base.view(bs, timestep_horizon, *z_base.shape[1:])
        z_base_var   = z_base_var.view(bs, timestep_horizon, *z_base_var.shape[1:])
        z_base_cov   = z_base_cov.view(bs, timestep_horizon, *z_base_cov.shape[1:])
        if patch_id_embed is not None:
            patch_id_embed = patch_id_embed.view(bs, timestep_horizon, *patch_id_embed.shape[1:])
        mu_offset    = mu_offset.view(bs, timestep_horizon, *mu_offset.shape[1:])
        logvar_offset= logvar_offset.view(bs, timestep_horizon, *logvar_offset.shape[1:])
        z_offset     = z_offset.view(bs, timestep_horizon, *z_offset.shape[1:])
        mu_tot       = mu_tot.view(bs, timestep_horizon, *mu_tot.shape[1:])
        z            = z.view(bs, timestep_horizon, *z.shape[1:])
        mu_scale     = mu_scale.view(bs, timestep_horizon, *mu_scale.shape[1:])
        if logvar_scale is not None:
            logvar_scale = logvar_scale.view(bs, timestep_horizon, *logvar_scale.shape[1:])
        z_scale      = z_scale.view(bs, timestep_horizon, *z_scale.shape[1:])
        if not self.interaction_features:
            mu_features     = mu_features.view(bs, timestep_horizon, *mu_features.shape[1:])
            logvar_features = logvar_features.view(bs, timestep_horizon, *logvar_features.shape[1:])
        z_features    = z_features.view(bs, timestep_horizon, *z_features.shape[1:])
        if z_depth_features is not None:
            z_depth_features = z_depth_features.view(bs, timestep_horizon, *z_depth_features.shape[1:])
        cropped_objects = cropped_objects.view(-1, *cropped_objects.shape[2:])
        if not self.interaction_depth:
            mu_depth     = mu_depth.view(bs, timestep_horizon, *mu_depth.shape[1:])
            logvar_depth = logvar_depth.view(bs, timestep_horizon, *logvar_depth.shape[1:])
            z_depth      = z_depth.view(bs, timestep_horizon, *z_depth.shape[1:])
        if not self.interaction_obj_on:
            obj_on_a   = obj_on_a.view(bs, timestep_horizon, *obj_on_a.shape[1:])
            obj_on_b   = obj_on_b.view(bs, timestep_horizon, *obj_on_b.shape[1:])
            mu_obj_on  = mu_obj_on.view(bs, timestep_horizon, *mu_obj_on.shape[1:])
            z_obj_on   = z_obj_on.view(bs, timestep_horizon, *z_obj_on.shape[1:])
        mu_score     = mu_score.view(bs, timestep_horizon, *mu_score.shape[1:])
        logvar_score = logvar_score.view(bs, timestep_horizon, *logvar_score.shape[1:])
        z_score      = z_score.view(bs, timestep_horizon, *z_score.shape[1:])

        encode_dict = {
            'mu_anchor': z_base, 'logvar_anchor': torch.zeros_like(z_base),
            'z_base': z_base, 'z': z,
            'mu_offset': mu_offset, 'logvar_offset': logvar_offset, 'z_offset': z_offset, 'mu_tot': mu_tot,
            'mu_features': mu_features, 'logvar_features': logvar_features, 'z_features': z_features,
            'z_depth_features': z_depth_features,
            'cropped_objects': cropped_objects.detach(), 'patch_id_embed': patch_id_embed,
            'obj_on_a': obj_on_a, 'obj_on_b': obj_on_b, 'z_obj_on': z_obj_on, 'mu_obj_on': mu_obj_on,
            'mu_depth': mu_depth, 'logvar_depth': logvar_depth, 'z_depth': z_depth,
            'mu_scale': mu_scale, 'logvar_scale': logvar_scale, 'z_scale': z_scale,
            'kp_p': kp_p, 'cov_kp': cov_kp, 'var_kp': var_kp,             
            'z_base_var': z_base_var, 'mu_score': mu_score, 'z_base_cov': z_base_cov,
            'logvar_score': logvar_score, 'z_score': z_score
        }
        return encode_dict


    def forward(self, x, deterministic=False, warmup=False, precomputed_prior=None):
        output_dict = self.encode_all(x, deterministic, warmup, precomputed_prior=precomputed_prior)
        return output_dict


class DLPEncoder(nn.Module):
    def __init__(self,
                 # Input configuration
                 cdim=3,  # Number of input image channels
                 image_size=64,  # Input image size (assumed square)
                 n_views=1,  # number of input views (e.g., multiple cameras)
                 pad_mode='replicate',  # Padding mode for CNNs
                 dropout=0.0,  # Dropout rate (not typically used)

                 # Keypoint and patch configuration
                 n_kp_per_patch=1,  # Number of keypoints per patch
                 n_kp_prior=20,  # Number of keypoints to filter from proposals
                 patch_size=16,  # Patch size for keypoint proposal network
                 n_kp_enc=20,  # Number of posterior keypoints to learn
                 n_kp_dec=None,  # Number of keypoints for decoder (if different from encoder)
                 warmup_n_kp_ratio=0.35,
                 mask_bg_in_enc=True,  # before encoding the bg, mask with the particles' obj_on

                 # Feature dimensions
                 learned_feature_dim=16,  # Dimension of learned visual features
                 learned_bg_feature_dim=16,  # Dimension of background features
                 kp_range=(-1, 1),  # Range for keypoint coordinates
                 kp_activation="tanh",  # Activation for keypoint coordinates
                 anchor_s=0.25,  # Glimpse size ratio

                 # Network architecture
                 use_resblock=True,  # Use residual blocks
                 embed_init_std=0.02,  # Standard deviation for embedding initialization
                 projection_dim=128,  # Embedding dimension for transformer

                 # Transformer configuration
                 timestep_horizon=1,  # Maximum timesteps to process at once
                 pte_layers=1,  # Number of particle transformer encoder layers
                 pte_heads=1,  # Number of particle transformer encoder heads
                 context_dim=16,  # Context latent dimension
                 filtering_heuristic='none',  # Method to filter prior keypoints
                 attn_norm_type='rms',  # Normalization type for attention

                 # Object encoder configuration
                 obj_ch_mult_prior=(1, 2,),  # Channel multipliers for prior patch encoder (kp proposals)
                 obj_ch_mult=(1, 2, 3),  # Channel multipliers for object encoder
                 obj_base_ch=32,  # Base channels for object encoder
                 obj_final_cnn_ch=32,  # Final CNN channels for object encoder
                 cnn_mid_blocks=False,  # Use middle blocks in CNN
                 mlp_hidden_dim=256,  # Hidden dimension for MLPs
                 pte_inner_dim=256,  # Inner dimension for particle transformer

                 # Background decoder configuration
                 bg_ch_mult=(1, 2, 3),  # Channel multipliers for background encoder
                 bg_base_ch=32,  # Base channels for background encoder
                 bg_final_cnn_ch=32,  # Final CNN channels for background encoder
                 num_res_blocks=2,  # Number of residual blocks

                 # Interaction configuration
                 ctx_pool_mode='none',  # Mode for pooling context features
                 interaction_depth=True,  # Enable depth interaction between particles
                 interaction_obj_on=False,  # Enable transparency interaction
                 interaction_features=True,  # Enable feature interaction
                 particle_score=False,  # Use particle confidence scores

                 # Embedding options
                 add_particle_temp_embed=False,  # Add temporal embeddings to particles
                 particle_positional_embed=True,  # Add positional embeddings to particles

                 # Context modeling
                 ctx_enc=None,
                 causal_ctx=True,  # Use causal attention for context
                 pte_ctx_layers=1,  # Number of context transformer layers
                 pte_ctx_heads=1,  # Number of context transformer heads
                 ctx_dist='gauss',  # Distribution type for context
                 n_ctx_categories=4,  # Number of context categories
                 n_ctx_classes=4,  # Number of context classes per category
                 global_ctx_pool=False,  # learn global latent context in addition to per-particle context
                 pool_ctx_dim=256,  # pool dimension for the global ctx latent
                 n_pool_ctx_categories=8,  # Number of global context categories (if categorical)
                 n_pool_ctx_classes=4,  # Number of global context classes per category
                 global_local_fuse_mode='none',  # concatenate/add global and local z_ctx to condition the dynamics
                 condition_local_on_global=True,  # condition z_context on z_context_global

                 # Distribution configuration
                 features_dist='gauss',  # Distribution type for features
                 n_fg_categories=8,  # Number of foreground categories, 'categorical' dist
                 n_fg_classes=4,  # Number of foreground classes per category, 'categorical' dist
                 n_bg_categories=4,  # Number of background categories, 'categorical' dist
                 n_bg_classes=4,  # Number of background classes per category, 'categorical' dist
                 obj_on_min=1e-4,  # Minimum concentration in Beta dist transparency value
                 obj_on_max=100,  # Maximum concentration in Beta dist transparency value
                 use_z_orig=True,  # Use original patch center coordinates as features

                 # initialization
                 init_zero_bias=True,  # zero bias for conv and linear layers
                 init_ssm_last_layer=True,  # spatial softmax initialization
                 init_conv_layers=True,  # initialize conv layers with normal dist
                 init_conv_fg_std=0.02,  # std for conv fg normal dist
                 init_conv_bg_std=0.005,  # std for conv bg normal dist (<fg -> prioritize fg in learning)
                 #RGBD 
                 separate_depth_features=False, # use separate feature encoder for RGB and Depth channels
                 depth_feature_dim=16, # feature dimension for depth channel
                 ):
        """
        DLP Encoder Module

        A neural network module that extracts object-centric representations from images using
        the Deep Latent Particles (DLP) approach. This encoder processes images to identify
        and represent objects as particles with learned attributes.

        Args:
            cdim (int): Number of input image channels. Defaults to 3.
            image_size (int): Size of input images (assumed square). Defaults to 64.
            pad_mode (str): Padding mode for CNNs ('zeros' or 'replicate'). Defaults to 'replicate'.
            dropout (float): Dropout rate for CNNs (typically unused). Defaults to 0.0.
            n_kp_per_patch (int): Number of keypoints to extract per patch. Defaults to 1.
            n_kp_prior (int): Number of keypoints to filter from proposals. Defaults to 20.
            patch_size (int): Size of patches for keypoint proposal network. Defaults to 16.
            n_kp_enc (int): Number of posterior keypoints to learn. Defaults to 20.
            n_kp_dec (Optional[int]): Number of keypoints for decoder. If None, equals n_kp_enc. Defaults to None.
            learned_feature_dim (int): Dimension of learned visual features. Defaults to 16.
            learned_bg_feature_dim (int): Dimension of background features. Defaults to 16.
            kp_range (tuple): Range for keypoint coordinates, either (-1, 1) or (0, 1). Defaults to (-1, 1).
            kp_activation (str): Activation for keypoint coordinates ('tanh' or 'sigmoid'). Defaults to 'tanh'.
            anchor_s (float): Glimpse size as ratio of image_size. Defaults to 0.25.
            use_resblock (bool): Use residual blocks in network. Defaults to True.
            embed_init_std (float): Standard deviation for embedding initialization. Defaults to 0.02.
            projection_dim (int): Embedding dimension for transformer. Defaults to 128.
            timestep_horizon (int): Maximum number of timesteps to process at once. Defaults to 1.
            pte_layers (int): Number of particle transformer encoder layers. Defaults to 1.
            pte_heads (int): Number of particle transformer encoder heads. Defaults to 1.
            context_dim (int): Dimension of context latent space. Defaults to 16.
            filtering_heuristic (str): Method to filter prior keypoints. Defaults to 'none'.
            attn_norm_type (str): Normalization type for attention blocks. Defaults to 'rms'.
            obj_ch_mult_prior (tuple): Channel multipliers for prior patch encoder. Defaults to (1, 2, 3).
            obj_ch_mult (tuple): Channel multipliers for object encoder. Defaults to (1, 2, 3).
            obj_base_ch (int): Base channels for object encoder. Defaults to 32.
            obj_final_cnn_ch (int): Final CNN channels for object encoder. Defaults to 32.
            cnn_mid_blocks (bool): Use middle blocks in CNN. Defaults to False.
            mlp_hidden_dim (int): Hidden dimension for MLPs. Defaults to 256.
            pte_inner_dim (int): Inner dimension for particle transformer. Defaults to 256.
            bg_ch_mult (tuple): Channel multipliers for background encoder. Defaults to (1, 2, 3).
            bg_base_ch (int): Base channels for background encoder. Defaults to 32.
            bg_final_cnn_ch (int): Final CNN channels for background encoder. Defaults to 32.
            num_res_blocks (int): Number of residual blocks. Defaults to 2.
            ctx_pool_mode (str): Mode for pooling context features. Defaults to 'none'.
            interaction_depth (bool): Enable modeling depth by interaction between particles. Defaults to True.
            interaction_obj_on (bool): Enable modeling transparency by interaction. Defaults to False.
            interaction_features (bool): Enable modeling features by interaction. Defaults to True.
            particle_score (bool): Use particle confidence scores. Defaults to False.
            add_particle_temp_embed (bool): Add temporal embeddings to particles. Defaults to False.
            particle_positional_embed (bool): Add positional embeddings to particles. Defaults to True.
            causal_ctx (bool): Use causal attention for context. Defaults to True.
            pte_ctx_layers (int): Number of context transformer layers. Defaults to 1.
            pte_ctx_heads (int): Number of context transformer heads. Defaults to 1.
            ctx_dist (str): Distribution type for context ('gauss' or 'categorical'). Defaults to 'gauss'.
            n_ctx_categories (int): Number of context categories if categorical. Defaults to 4.
            n_ctx_classes (int): Number of context classes per category. Defaults to 4.
            features_dist (str): Distribution type for features ('gauss' or 'categorical'). Defaults to 'gauss'.
            n_fg_categories (int): Number of foreground categories if categorical. Defaults to 8.
            n_fg_classes (int): Number of foreground classes per category. Defaults to 4.
            n_bg_categories (int): Number of background categories if categorical. Defaults to 4.
            n_bg_classes (int): Number of background classes per category. Defaults to 4.
            obj_on_min (float): Minimum concentration value in Beta dist for transparency value. Defaults to 1e-4.
            obj_on_max (float): Maximum concentration value in Beta dist transparency value. Defaults to 100.
            use_z_orig (bool): Use original patch center coordinates. Defaults to True.

        Notes:
            The encoder operates in several stages:
            1. Patch Processing: Divides input image into patches and processes each
            2. Keypoint Proposal: Generates candidate keypoints using spatial softmax
            3. Feature Extraction: Learns visual features around each keypoint
            4. Particle Interaction: Models relationships between particles
            5. Context Modeling: Captures dynamics for the latent context (if enabled)

            The module supports both Gaussian and categorical distributions for
            features and context variables.

        The architecture uses a combination of CNNs and transformers:
            - CNNs for initial feature extraction from patches
            - Transformer encoders for modeling particle interactions
            - Separate pathways for foreground and background processing
            - Optional causal attention for temporal modeling
        """
        super(DLPEncoder, self).__init__()
        self.cdim = cdim
        self.image_size = image_size
        self.n_views = n_views
        self.dropout = dropout
        self.kp_range = kp_range
        self.n_kp_per_patch = n_kp_per_patch
        self.n_kp_enc = n_kp_enc
        self.n_kp_prior = n_kp_prior
        self.n_kp_dec = self.n_kp_enc if n_kp_dec is None else n_kp_dec
        self.warmup_n_kp_ratio = warmup_n_kp_ratio
        self.kp_activation = kp_activation
        self.patch_size = patch_size
        self.anchor_patch_s = patch_size / image_size
        self.features_dim = int(image_size // (2 ** (len(bg_ch_mult) - 1)))
        self.learned_feature_dim = learned_feature_dim
        self.learned_bg_feature_dim = learned_bg_feature_dim
        assert learned_feature_dim > 0, "learned_feature_dim must be greater than 0"
        self.features_dist = features_dist
        self.n_fg_categories = n_fg_categories
        self.n_fg_classes = n_fg_classes
        self.n_bg_categories = n_bg_categories
        self.n_bg_classes = n_bg_classes

        # self.context_dist = ctx_dist
        # self.n_ctx_categories = n_ctx_categories
        # self.n_ctx_classes = n_ctx_classes
        self.context_dim = context_dim
        # # global latent context
        # self.global_ctx_pool = global_ctx_pool
        # self.pool_ctx_dim = pool_ctx_dim
        # self.n_pool_ctx_categories = n_pool_ctx_categories
        # self.n_pool_ctx_classes = n_pool_ctx_classes
        # if self.context_dist == 'categorical':
        #     self.pool_ctx_dim = int(self.n_pool_ctx_categories * self.n_pool_ctx_classes)
        # self.global_local_fuse_mode = global_local_fuse_mode
        # self.condition_local_on_global = condition_local_on_global
        self.mask_bg_in_enc = mask_bg_in_enc  # before encoding the bg, mask with the particles' obj_on
        self.anchor_s = anchor_s
        self.obj_patch_size = np.round(anchor_s * (image_size - 1)).astype(int)
        self.obj_on_min = obj_on_min
        self.obj_on_max = obj_on_max
        self.use_resblock = use_resblock
        self.embed_init_std = embed_init_std
        self.projection_dim = projection_dim
        self.timestep_horizon = (timestep_horizon + 1) if timestep_horizon > 1 else 1
        self.num_patches = int((image_size // self.patch_size) ** 3)
        self.attn_norm_type = attn_norm_type
        self.use_z_orig = use_z_orig
        self.interaction_depth = interaction_depth
        self.interaction_obj_on = interaction_obj_on
        self.interaction_features = interaction_features
        self.use_particle_inter_enc = (self.interaction_features or self.interaction_depth or self.interaction_obj_on)
        self.add_particle_temp_embed = add_particle_temp_embed
        self.temporal_interaction = False  # True=allow to attend over timesteps

        self.use_ctx_enc = (self.context_dim > 0)
        # self.ctx_pool_mode = ctx_pool_mode
        # self.causal_ctx = causal_ctx
        self.particle_score = particle_score
        self.cnn_mid_blocks = cnn_mid_blocks
        self.mlp_hidden_dim = mlp_hidden_dim

        # initialization
        self.init_zero_bias = init_zero_bias  # zero bias for conv and linear layers
        self.init_ssm_last_layer = init_ssm_last_layer  # spatial softmax initialization
        self.init_conv_layers = init_conv_layers  # initialize conv layers with normal dist
        self.init_conv_fg_std = init_conv_fg_std  # std for conv fg normal dist
        self.init_conv_bg_std = init_conv_bg_std  # std for conv bg normal dist

        #RGBD
        self.separate_depth_features = separate_depth_features
        self.depth_feature_dim = depth_feature_dim

        self.register_buffer('scale_anchor', torch.tensor(np.log(anchor_s / (1 - anchor_s + 1e-5))))
        use_norm_layer = True  # norm layer in the pre-attention projections modules
        self.particle_enc = ParticleEncoder(cdim=cdim,
                                            image_size=image_size,
                                            pad_mode=pad_mode,
                                            n_kp_per_patch=self.n_kp_per_patch,
                                            n_kp_prior=self.n_kp_prior,
                                            patch_size=self.patch_size, n_kp_enc=self.n_kp_enc, n_kp_dec=self.n_kp_dec,
                                            learned_feature_dim=learned_feature_dim,
                                            kp_range=kp_range, kp_activation=kp_activation, anchor_s=anchor_s,
                                            use_resblock=use_resblock, embed_init_std=embed_init_std,
                                            projection_dim=projection_dim, timestep_horizon=timestep_horizon,
                                            filtering_heuristic=filtering_heuristic,
                                            obj_ch_mult_prior=obj_ch_mult_prior,
                                            obj_ch_mult=obj_ch_mult,
                                            obj_base_ch=obj_base_ch,
                                            obj_final_cnn_ch=obj_final_cnn_ch, num_res_blocks=num_res_blocks,
                                            interaction_features=interaction_features,
                                            interaction_obj_on=interaction_obj_on,
                                            interaction_depth=interaction_depth,
                                            temporal_interaction=self.temporal_interaction,
                                            cnn_mid_blocks=cnn_mid_blocks,
                                            mlp_hidden_dim=mlp_hidden_dim, embed_prior_patch_pos=False,
                                            add_particle_temp_embed=self.add_particle_temp_embed,
                                            features_dist=self.features_dist, n_fg_categories=n_fg_categories,
                                            n_fg_classes=n_fg_classes, obj_on_min=self.obj_on_min,
                                            obj_on_max=self.obj_on_max, warmup_n_kp_ratio=self.warmup_n_kp_ratio,
                                            init_zero_bias=init_zero_bias,
                                            init_ssm_last_layer=init_ssm_last_layer,
                                            init_conv_layers=init_conv_layers,
                                            init_conv_fg_std=init_conv_fg_std,
                                            separate_depth_features=separate_depth_features,
                                            depth_feature_dim=depth_feature_dim)

        self.prior_encoder = self.particle_enc.prior_encoder
        self.bg_encoder = BgEncoder(cdim=cdim, image_size=image_size, pad_mode=pad_mode,
                                    learned_feature_dim=learned_bg_feature_dim, use_resblock=use_resblock,
                                    ch_mult=bg_ch_mult, base_ch=bg_base_ch, final_cnn_ch=bg_final_cnn_ch,
                                    num_res_blocks=num_res_blocks, interaction_features=interaction_features,
                                    cnn_mid_blocks=cnn_mid_blocks, mlp_hidden_dim=mlp_hidden_dim,
                                    timestep_horizon=timestep_horizon,
                                    add_particle_temp_embed=self.add_particle_temp_embed,
                                    features_dist=self.features_dist, n_bg_categories=n_bg_categories,
                                    n_bg_classes=n_bg_classes,
                                    init_zero_bias=init_zero_bias,
                                    init_conv_layers=init_conv_layers,
                                    init_conv_bg_std=init_conv_bg_std)

        # centers already in kp_range, shape [N,3]
        patch_centers = self.prior_encoder.get_patch_centers().unsqueeze(0)  # [1, N, 3]

        # append 3D null particle (same device/dtype)
        null_center = torch.zeros(1, 1, 3, device=patch_centers.device, dtype=patch_centers.dtype)
        patch_centers = torch.cat([patch_centers, null_center], dim=1)       # [1, N+1, 3]

        # keep as buffer
        self.register_buffer('patch_centers', patch_centers)

        # anchors: duplicate each patch center n_kp_per_patch times, exclude the null at the end
        particle_anchors = (
            patch_centers[:, :-1, :]                          # [1, N, 3]
            .unsqueeze(2)                                     # [1, N, 1, 3]
            .expand(-1, -1, self.n_kp_per_patch, -1)         # [1, N, n_kp_per_patch, 3]
            .reshape(1, -1, 3)                                # [1, N*n_kp_per_patch, 3]
        )

        print("z origin: ", use_z_orig)
        print("particle anchors: ", particle_anchors.shape)

        if self.use_particle_inter_enc:
            print("USING INTERACTION ENCODER")
            self.particle_inter_enc = ParticleInteractionEncoder(n_kp_enc=n_kp_enc, dropout=0.0,
                                                                 learned_feature_dim=learned_feature_dim,
                                                                 learned_bg_feature_dim=learned_bg_feature_dim,
                                                                 embed_init_std=embed_init_std,
                                                                 projection_dim=projection_dim,
                                                                 timestep_horizon=timestep_horizon,
                                                                 pte_layers=pte_layers,
                                                                 pte_heads=pte_heads,
                                                                 attn_norm_type=attn_norm_type, pad_mode=pad_mode,
                                                                 use_resblock=use_resblock,
                                                                 hidden_dim=mlp_hidden_dim,
                                                                 temporal_interaction=self.temporal_interaction,
                                                                 interaction_features=interaction_features,
                                                                 interaction_depth=interaction_depth,
                                                                 interaction_obj_on=interaction_obj_on,
                                                                 cdim=cdim, image_size=image_size, n_views=self.n_views,
                                                                 ch_mult=bg_ch_mult, base_ch=bg_base_ch,
                                                                 final_cnn_ch=bg_final_cnn_ch,
                                                                 num_res_blocks=num_res_blocks,
                                                                 bg=True, use_img_input=True,
                                                                 cnn_mid_blocks=cnn_mid_blocks,
                                                                 particle_score=True,
                                                                 particle_positional_embed=particle_positional_embed,
                                                                 norm_layer=use_norm_layer,
                                                                 add_particle_temp_embed=self.add_particle_temp_embed,
                                                                 features_dist=self.features_dist,
                                                                 n_fg_categories=n_fg_categories,
                                                                 n_fg_classes=n_fg_classes,
                                                                 n_bg_categories=n_bg_categories,
                                                                 n_bg_classes=n_bg_classes,
                                                                 scale_anchor=self.scale_anchor,
                                                                 obj_on_min=self.obj_on_min,
                                                                 obj_on_max=self.obj_on_max,
                                                                 particle_anchors=particle_anchors,
                                                                 use_z_orig=self.use_z_orig,
                                                                 init_zero_bias=init_zero_bias,
                                                                 init_conv_layers=init_conv_layers,
                                                                 init_conv_fg_std=init_conv_fg_std
                                                                 )
        else:
            self.particle_inter_enc = None

        self.ctx_enc = ctx_enc


        self.init_weights()

    def init_weights(self):
        self.particle_enc.init_weights()
        self.bg_encoder.init_weights()
        self.prior_encoder.init_weights()
        # if self.with_ctx:
        #     self.ctx_enc.init_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                pass
                # nn.init.normal_(m.weight, 0, 0.01)
                # if m.bias is not None:
                #    nn.init.constant_(m.bias, 0)
            #         # print(m.__repr__())
            #     elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            #         nn.init.constant_(m.weight, 1)
            #         nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5), mode='fan_in')
                # nn.init.normal_(m.weight, 0, 0.02)
                # if m.bias is not None:
                #    nn.init.constant_(m.bias, 0)
                # use pytorch's default
                pass
    def get_bg_mask_from_particle_glimpses(self, z, z_obj_on, mask_size, z_scale=None, detach_grad=True):
        """
        3D version.
        z:         [B, K, 3] in [-1, 1] (x, y, z)
        z_obj_on:  [B, K] or [B, K, 1]
        mask_size: int or (D, H, W) of the feature/grid you want the mask for
        z_scale:   None or [B, K, 3] (unnormalized -> sigmoid)
        Returns:
        bg_mask: [B, 1, D, H, W]  (1 where background, 0 in object cubes)
        """
        if detach_grad:
            with torch.no_grad():
                if z_scale is None:
                    obj_fmap_masks = create_masks_fast(z.detach(), anchor_s=self.anchor_s, feature_dim=mask_size)
                else:
                    obj_fmap_masks = create_masks_with_scale(
                        z.detach(), anchor_s=self.anchor_s, image_size=mask_size, scale=z_scale.detach()
                    )
                z_gate = (z_obj_on.detach() > 0.2).to(obj_fmap_masks.dtype)
                if z_gate.dim() == 2:
                    z_gate = z_gate[:, :, None, None, None, None]
                else:
                    z_gate = z_gate[:, :, None, None, None, None]  # ensure broadcast to [B,K,1,D,H,W]
                obj_fmap_masks = obj_fmap_masks.clamp(0, 1) * z_gate
                bg_mask = 1 - obj_fmap_masks.squeeze(2).sum(1, keepdim=True).clamp(0, 1)
        else:
            # only z_obj_on gating is detached in original; keep the same spirit
            if z_scale is None:
                obj_fmap_masks = create_masks_fast(z, anchor_s=self.anchor_s, feature_dim=mask_size)
            else:
                obj_fmap_masks = create_masks_with_scale(
                    z, anchor_s=self.anchor_s, image_size=mask_size, scale=z_scale
                )
            if z_obj_on.dim() == 2:
                z_gate = (z_obj_on > 0.2).to(obj_fmap_masks.dtype)[:, :, None, None, None, None]
            else:
                z_gate = (z_obj_on > 0.2).to(obj_fmap_masks.dtype)[:, :, None, None, None, None]
            obj_fmap_masks = obj_fmap_masks.clamp(0, 1) * z_gate
            bg_mask = 1 - obj_fmap_masks.squeeze(2).sum(1, keepdim=True).clamp(0, 1)

        return bg_mask
    def encode_all(self, x, deterministic=False, warmup=False, actions=None, actions_mask=None, lang_embed=None,
                x_goal=None, deterministic_goal=True, precomputed_prior=None):
        """
        encoding steps:
        1. encode bg: x -> bg_enc -> [bs * T, projection_dim]
        2. encode patches: x -> patch_enc -> [bs * T, n_patches, projection_dim ]
        3. encode particles: [patches, bg, particle_tokens, bg_token, ctx_token] -> pte -> [bs, T, n_particles + 2, dim]
        """
        # make sure x is [bs, T, ch, h, w, l]
        if len(x.shape) == 5:
            # that means x: [bs, ch, h, w, l]
            x = x.unsqueeze(1)  # -> [bs, T=1, ch, h, w, l]
        bs, timestep_horizon, ch, D, H, W = x.shape

        if x_goal is not None:
            if len(x_goal.shape) == 5:
                x_goal = x_goal.unsqueeze(1)  # -> [bs, T=1, ch, h, w, l]
            x = torch.cat([x, x_goal], dim=1)  # [bs, T+1, ...]

        # encode particles +++++++
        particle_dict = self.particle_enc(x, deterministic, warmup, precomputed_prior=precomputed_prior)

        kp_p              = particle_dict['kp_p']
        cov_kp            = particle_dict['cov_kp']        # full [bs,T,n_kp,3,3]
        var_kp            = particle_dict['var_kp']        # diag only [bs,T,n_kp,3]
        patch_id_embed    = particle_dict['patch_id_embed']
        z_base            = particle_dict['z_base']
        z                 = particle_dict['z']
        mu_offset         = particle_dict['mu_offset']
        logvar_offset     = particle_dict['logvar_offset']
        z_offset          = particle_dict['z_offset']
        mu_tot            = particle_dict['mu_tot']
        z_base_var        = particle_dict['z_base_var']
        z_base_cov        = particle_dict['z_base_cov']
        mu_scale          = particle_dict['mu_scale']
        logvar_scale      = particle_dict['logvar_scale']
        z_scale           = particle_dict['z_scale']
        mu_depth          = particle_dict['mu_depth']
        logvar_depth      = particle_dict['logvar_depth']
        z_depth           = particle_dict['z_depth']
        obj_on_a          = particle_dict['obj_on_a']
        obj_on_b          = particle_dict['obj_on_b']
        mu_obj_on         = particle_dict['mu_obj_on']
        z_obj_on          = particle_dict['z_obj_on']
        mu_features       = particle_dict['mu_features']
        logvar_features   = particle_dict['logvar_features']
        z_features        = particle_dict['z_features']
        cropped_objects   = particle_dict['cropped_objects']

        z_score           = particle_dict['z_score']
        mu_score          = particle_dict['mu_score']
        logvar_score      = particle_dict['logvar_score']
        # -------------------------------------------

        if x_goal is not None and deterministic_goal:
            z = torch.cat([z[:, :-1], mu_tot[:, -1:]], dim=1)
            if z_obj_on is not None:
                z_obj_on = torch.cat([z_obj_on[:, :-1], Beta(obj_on_a[:, -1:], obj_on_b[:, -1:]).mean], dim=1)
            z_scale = torch.cat([z_scale[:, :-1], mu_scale[:, -1:]], dim=1)
            if z_depth is not None:
                z_depth = torch.cat([z_depth[:, :-1], mu_depth[:, -1:]], dim=1)
            if not self.interaction_features:
                z_features = torch.cat([z_features[:, :-1], mu_features[:, -1:]], dim=1)

        # +++++++ encode bg +++++++
        # x: [bs, T, ch, h, w, l]  ->  [bs*T, ch, h, w, l]
        x  = x.view(-1, *x.shape[2:])
        z_v = z.view(-1, *z.shape[2:])  # [bs*T, n_kp_enc, 3]

        if self.n_kp_dec != self.n_kp_enc:
            # variance filtering (use summed prior variance / score; last dim can be 3 or 4)
            total_var = z_base_var.view(-1, *z_base_var.shape[2:]).sum(-1)  # [bs*T, n_kp_enc]
            n_filter = self.n_kp_dec if not warmup else min(self.n_kp_dec, int(self.warmup_n_kp_ratio * self.n_kp_enc))
            _, embed_ind = torch.topk(total_var, k=n_filter, dim=-1, largest=False)
            # make selection
            batch_ind = torch.arange(z_v.shape[0], device=z_v.device)[:, None]
            z_v = z_v[batch_ind, embed_ind]  # [bs*T, n_kp_dec, 3]

        if self.interaction_obj_on:
            z_obj_on_v = torch.ones(z_v.shape[0], z_v.shape[1], device=x.device, dtype=torch.float)
        else:
            # z_obj_on: [bs, T, n_kp, 1] -> [bs*T, n_kp]
            z_obj_on_v = z_obj_on.view(-1, *z_obj_on.shape[2:]).squeeze(-1)
            if self.n_kp_dec != self.n_kp_enc:
                z_obj_on_v = z_obj_on_v[batch_ind, embed_ind]  # [bs*T, n_kp_dec]

        if self.mask_bg_in_enc:
            bg_enc_mask = self.get_bg_mask_from_particle_glimpses(
                    z_v, z_obj_on_v, mask_size=(x.shape[-3], x.shape[-2], x.shape[-1])
                )
            bg_dict = self.bg_encoder(x, bg_enc_mask, deterministic, timestep_horizon)
        else:
            bg_enc_mask = None
            bg_dict = self.bg_encoder(x, None, deterministic, timestep_horizon)  # unmasked bg
        mu_bg_features = bg_dict['mu_bg']
        mu_bg_features = mu_bg_features.view(bs, -1, mu_bg_features.shape[-1])
        logvar_bg_features = bg_dict['logvar_bg']
        if logvar_bg_features is not None:
            logvar_bg_features = logvar_bg_features.view(bs, -1, logvar_bg_features.shape[-1])
        z_bg_features = bg_dict['z_bg']
        z_bg_features = z_bg_features.view(bs, -1, z_bg_features.shape[-1])
        if x_goal is not None and deterministic_goal and not self.interaction_features:
            z_bg_features = torch.cat([z_bg_features[:, :-1], mu_bg_features[:, -1:]], dim=1)

        if self.use_particle_inter_enc:
            z_in_inter = z_base + z_offset  # so we can detach z_base (ssm) if more stable
            inter_dict = self.particle_inter_enc(x, z_in_inter, z_scale, z_obj_on, z_depth, z_features, z_bg_features,
                                                 z_base_var, z_score, patch_id_embed,
                                                 deterministic=deterministic, warmup=warmup)
            if self.interaction_features:
                mu_features = inter_dict['mu_features']
                logvar_features = inter_dict['logvar_features']
                z_features = inter_dict['z_features']

                if x_goal is not None and deterministic_goal:
                    z_features = torch.cat([z_features[:, :-1], mu_features[:, -1:]], dim=1)

                if inter_dict.get('mu_bg_features') is not None:
                    mu_bg_features = inter_dict['mu_bg_features']
                    logvar_bg_features = inter_dict['logvar_bg_features']
                    z_bg_features = inter_dict['z_bg_features']

                    if x_goal is not None and deterministic_goal:
                        z_bg_features = torch.cat([z_bg_features[:, :-1], mu_bg_features[:, -1:]], dim=1)
            if self.interaction_obj_on:
                obj_on_a = inter_dict['obj_on_a']
                obj_on_b = inter_dict['obj_on_b']
                mu_obj_on = inter_dict['mu_obj_on']
                z_obj_on = inter_dict['z_obj_on']
                if x_goal is not None and deterministic_goal:
                    z_obj_on = torch.cat([z_obj_on[:, :-1], Beta(obj_on_a[:, -1:], obj_on_b[:, -1:]).mean], dim=1)
            if self.interaction_depth:
                mu_depth = inter_dict['mu_depth']
                logvar_depth = inter_dict['logvar_depth']
                z_depth = inter_dict['z_depth']

                if x_goal is not None and deterministic_goal:
                    z_depth = torch.cat([z_depth[:, :-1], mu_depth[:, -1:]], dim=1)

        if self.use_ctx_enc:
            z_in_ctx = z_base + z_offset  # so we can detach z_base (ssm) if more stable
            if x_goal is not None and deterministic_goal:
                z_in_ctx = torch.cat([z_in_ctx[:, :-1], mu_tot[:, -1:]], dim=1)
            z_scale_in_ctx = z_scale
            z_obj_on_in_ctx = z_obj_on
            z_depth_in_ctx = z_depth
            z_features_in_ctx = z_features
            z_bg_features_in_ctx = z_bg_features

            ctx_dict = self.ctx_enc(z_in_ctx, z_scale_in_ctx, z_obj_on_in_ctx, z_depth_in_ctx,
                                    z_features_in_ctx, z_bg_features_in_ctx, z_base_var,
                                    z_score, patch_id_embed, deterministic=deterministic, warmup=warmup,
                                    actions=actions, actions_mask=actions_mask, lang_embed=lang_embed)
            z_goal_proj = ctx_dict['z_goal_proj']
            # global context
            mu_context_global = ctx_dict['mu_context_global']
            logvar_context_global = ctx_dict['logvar_context_global']
            z_context_global = ctx_dict['z_context_global']

            mu_context_global_dyn = ctx_dict['mu_context_global_dyn']
            logvar_context_global_dyn = ctx_dict['logvar_context_global_dyn']
            z_context_global_dyn = ctx_dict['z_context_global_dyn']

            # local context
            mu_context = ctx_dict['mu_context']
            logvar_context = ctx_dict['logvar_context']
            z_context = ctx_dict['z_context']

            mu_context_dyn = ctx_dict['mu_context_dyn']
            logvar_context_dyn = ctx_dict['logvar_context_dyn']
            z_context_dyn = ctx_dict['z_context_dyn']
        else:
            mu_context_global = logvar_context_global = z_context_global = None
            mu_context_global_dyn = logvar_context_global_dyn = z_context_global_dyn = None
            mu_context = logvar_context = z_context = None
            mu_context_dyn = logvar_context_dyn = z_context_dyn = None
            z_goal_proj = None

        if x_goal is not None:
            # remove last timestep
            z_base = z_base[:, :-1].contiguous()
            z = z[:, :-1].contiguous()
            mu_offset = mu_offset[:, :-1].contiguous()
            logvar_offset = logvar_offset[:, :-1].contiguous()
            z_offset = z_offset[:, :-1].contiguous()
            mu_tot = mu_tot[:, :-1].contiguous()
            mu_features = mu_features[:, :-1].contiguous()
            logvar_features = logvar_features[:, :-1].contiguous()
            z_features = z_features[:, :-1].contiguous()
            mu_bg_features = mu_bg_features[:, :-1].contiguous()
            logvar_bg_features = logvar_bg_features[:, :-1].contiguous()
            z_bg_features = z_bg_features[:, :-1].contiguous()
            obj_on_a = obj_on_a[:, :-1].contiguous()
            obj_on_b = obj_on_b[:, :-1].contiguous()
            z_obj_on = z_obj_on[:, :-1].contiguous()
            if mu_obj_on is not None:
                mu_obj_on = mu_obj_on[:, :-1].contiguous()
            z_base_var = z_base_var[:, :-1].contiguous()
            z_base_cov = z_base_cov[:, :-1].contiguous()
            mu_depth = mu_depth[:, :-1].contiguous()
            logvar_depth = logvar_depth[:, :-1].contiguous()
            z_depth = z_depth[:, :-1].contiguous()
            mu_scale = mu_scale[:, :-1].contiguous()
            logvar_scale = logvar_scale[:, :-1].contiguous()
            z_scale = z_scale[:, :-1].contiguous()
            kp_p = kp_p.view(bs, -1, *kp_p.shape[1:])[:, :-1].reshape(-1, *kp_p.shape[1:])  # orig: [bs * T, N, 2]
            var_kp = var_kp.view(bs, -1, *var_kp.shape[1:])[:, :-1].reshape(-1,
                                                                            *var_kp.shape[1:])  # orig: [bs * T, N, 2]
            bg_enc_mask = bg_enc_mask.view(bs, -1, *bg_enc_mask.shape[1:])[:, :-1].reshape(-1, *bg_enc_mask.shape[
                1:])  # orig: [bs * t, 1, im_size, im_size]
            mu_score = mu_score[:, :-1].contiguous()
            logvar_score = logvar_score[:, :-1].contiguous()
            z_score = z_score[:, :-1].contiguous()
        # TODO: Make sure all the necessary information from the depth encoding is returned and USED
        encode_dict = {'mu_anchor': z_base, 'logvar_anchor': torch.zeros_like(z_base), 'z_base': z_base, 'z': z,
                       'mu_offset': mu_offset, 'logvar_offset': logvar_offset, 'z_offset': z_offset, 'mu_tot': mu_tot,
                       'mu_features': mu_features, 'logvar_features': logvar_features, 'z_features': z_features,
                       'mu_bg_features': mu_bg_features, 'logvar_bg_features': logvar_bg_features,
                       'z_bg_features': z_bg_features, 'mu_context': mu_context, 'logvar_context': logvar_context,
                       'z_context': z_context,
                       'mu_context_global': mu_context_global, 'logvar_context_global': logvar_context_global,
                       'z_context_global': z_context_global,
                       'cropped_objects': cropped_objects.detach(), 'patch_id_embed': patch_id_embed,
                       'obj_on_a': obj_on_a, 'obj_on_b': obj_on_b, 'obj_on': z_obj_on, 'mu_obj_on': mu_obj_on,
                       'z_base_var': z_base_var, 'cov_kp': cov_kp, 'z_base_cov': z_base_cov,
                       'mu_depth': mu_depth, 'logvar_depth': logvar_depth, 'z_depth': z_depth,
                       'mu_scale': mu_scale, 'logvar_scale': logvar_scale, 'z_scale': z_scale,
                       'kp_p': kp_p, 'var_kp': var_kp, 'bg_enc_mask': bg_enc_mask,
                       'mu_score': mu_score, 'logvar_score': logvar_score, 'z_score': z_score,
                       'mu_context_dyn': mu_context_dyn, 'logvar_context_dyn': logvar_context_dyn,
                       'z_context_dyn': z_context_dyn,
                       'mu_context_global_dyn': mu_context_global_dyn,
                       'logvar_context_global_dyn': logvar_context_global_dyn,
                       'z_context_global_dyn': z_context_global_dyn,
                       'z_goal_proj': z_goal_proj
                       }
        return encode_dict

    def forward(self, x, deterministic=False, warmup=False, actions=None, actions_mask=None, lang_embed=None,
                x_goal=None, precomputed_prior=None):
        output_dict = self.encode_all(x, deterministic, warmup, actions=actions, actions_mask=actions_mask,
                                      lang_embed=lang_embed, x_goal=x_goal,
                                      precomputed_prior=precomputed_prior)
        return output_dict


class DLPDecoder(nn.Module):
    def __init__(self,
                 # Input configuration
                 cdim=3,  # Number of input image channels
                 image_size=64,  # Input image size (assumed square)
                 pad_mode='replicate',  # Padding mode for CNNs
                 dropout=0.0,  # Dropout rate
                 normalize_rgb=False,  # Normalize RGB output to [-1, 1]

                 # Feature dimensions
                 learned_feature_dim=16,  # Dimension of learned visual features
                 learned_bg_feature_dim=16,  # Dimension of background features
                 anchor_s=0.25,  # Glimpse size ratio
                 n_kp_enc=16,  # Number of keypoints to decode
                 context_dim=0,  # Dimension of context features

                 # Network architecture
                 use_resblock=True,  # Use residual blocks
                 timestep_horizon=1,  # Maximum timesteps to process
                 decode_with_ctx=False,  # Use context in decoding
                 cnn_mid_blocks=False,  # Use middle blocks in CNN
                 mlp_hidden_dim=256,  # Hidden dimension for MLPs

                 # Object decoder configuration
                 obj_res_from_fc=8,  # Initial resolution for object decoder
                 obj_ch_mult=(1, 2, 3),  # Channel multipliers for object decoder
                 obj_base_ch=32,  # Base channels for object decoder
                 obj_final_cnn_ch=32,  # Final CNN channels for object decoder

                 # Background decoder configuration
                 bg_res_from_fc=8,  # Initial resolution for background decoder
                 bg_ch_mult=(1, 2, 3),  # Channel multipliers for background decoder
                 bg_base_ch=32,  # Base channels for background decoder
                 bg_final_cnn_ch=32,  # Final CNN channels for background decoder
                 num_res_blocks=2,  # Number of residual blocks

                 # initialization
                 init_zero_bias=True,  # zero bias for conv and linear layers
                 init_conv_layers=True,  # initialize conv layers with normal dist
                 init_conv_fg_std=0.02,  # std for conv fg normal dist
                 init_conv_bg_std=0.005,  # std for conv bg normal dist (<fg -> prioritize fg in learning)
                 
                 separate_depth_features=False,  # separate depth features when using RGBD input
                 depth_feature_dim=4,  # depth feature dimension (if separate_depth_features)
                 ):
        """
        DLP Decoder Module

        A neural network module that reconstructs images from object-centric representations using
        the Deep Latent Particles (DLP) approach. This decoder transforms particle representations
        back into image space, handling both foreground objects and background separately.

        Args:
            cdim (int): Number of input image channels. Defaults to 3.
            image_size (int): Size of input images (assumed square). Defaults to 64.
            pad_mode (str): Padding mode for CNNs ('zeros' or 'replicate'). Defaults to 'replicate'.
            dropout (float): Dropout rate for networks. Defaults to 0.0.
            normalize_rgb (bool): Normalize RGB output to [-1, 1] range. Defaults to False.
            learned_feature_dim (int): Dimension of learned visual features. Defaults to 16.
            learned_bg_feature_dim (int): Dimension of background features. Defaults to 16.
            anchor_s (float): Glimpse size as ratio of image_size (e.g., 0.25 for 32px glimpse on 128px image).
                            Defaults to 0.25.
            n_kp_enc (int): Number of keypoints to decode. Defaults to 16.
            context_dim (int): Dimension of context features. Set to 0 to disable context. Defaults to 0.
            use_resblock (bool): Use residual blocks in decoders. Defaults to True.
            timestep_horizon (int): Maximum number of timesteps to process at once. Defaults to 1.
            decode_with_ctx (bool): Use context information during decoding. Defaults to False.
            cnn_mid_blocks (bool): Use middle blocks in CNN decoders. Defaults to False.
            mlp_hidden_dim (int): Hidden dimension for MLPs. Defaults to 256.
            obj_res_from_fc (int): Initial resolution for object decoder from fully connected layer.
                                 Defaults to 8.
            obj_ch_mult (tuple): Channel multipliers for progressive object decoder stages.
                               Defaults to (1, 2, 3).
            obj_base_ch (int): Base number of channels for object decoder. Defaults to 32.
            obj_final_cnn_ch (int): Number of channels in final object CNN layer. Defaults to 32.
            bg_res_from_fc (int): Initial resolution for background decoder from fully connected layer.
                                Defaults to 8.
            bg_ch_mult (tuple): Channel multipliers for progressive background decoder stages.
                              Defaults to (1, 2, 3).
            bg_base_ch (int): Base number of channels for background decoder. Defaults to 32.
            bg_final_cnn_ch (int): Number of channels in final background CNN layer. Defaults to 32.
            num_res_blocks (int): Number of residual blocks per resolution level. Defaults to 2.

        Architecture Details:
            The decoder consists of two main pathways:
            1. Object Decoder:
               - Processes each particle independently
               - Progressively upsamples from initial resolution (obj_res_from_fc)
               - Optionally incorporates context information

            2. Background Decoder:
               - Processes background features
               - Similar progressive upsampling architecture

        Notes:
            - The decoder uses spatial transformer networks (STN) for differentiable
              rendering of particles
        """
        super(DLPDecoder, self).__init__()
        self.occupancy_mode  = (cdim ==1)
        self.occupancy_prior = 0.05
        self.alpha_prior = 0.05  
        self.image_size = image_size
        self.feature_map_size = image_size
        self.n_kp_enc = n_kp_enc
        self.dropout = dropout
        self.learned_feature_dim = learned_feature_dim
        self.learned_bg_feature_dim = learned_bg_feature_dim
        assert learned_feature_dim > 0, "learned_feature_dim must be greater than 0"
        self.anchor_s = anchor_s
        self.context_dim = context_dim
        self.obj_patch_size = np.round(anchor_s * (image_size - 1)).astype(int)
        self.cdim = cdim
        self.use_resblock = use_resblock
        self.decode_with_ctx = decode_with_ctx
        self.normalize_rgb = normalize_rgb
        self.timestep_horizon = (timestep_horizon + 1) if timestep_horizon > 1 else 1
        self.cnn_mid_blocks = cnn_mid_blocks
        self.mlp_hidden_dim = mlp_hidden_dim
        self.context_dim = context_dim

        # initialization
        self.init_zero_bias = init_zero_bias  # zero bias for conv and linear layers
        self.init_conv_layers = init_conv_layers  # initialize conv layers with normal dist
        self.init_conv_fg_std = init_conv_fg_std  # std for conv fg normal dist
        self.init_conv_bg_std = init_conv_bg_std  # std for conv bg normal dist

        self.separate_depth_features = separate_depth_features
        self.depth_feature_dim = depth_feature_dim
        # object decoder
        if self.context_dim > 0 and self.decode_with_ctx:
            particle_dec_net = ObjectDecoderCNNFILM
        else:
            particle_dec_net = ObjectDecoderCNN
        
        particle_out_ch = 1 if self.occupancy_mode else (cdim + 1)
        self.particle_dec = particle_dec_net(patch_size=self.obj_patch_size, num_chans=particle_out_ch,
                                            bottleneck_size=learned_feature_dim,
                                            use_resblock=self.use_resblock,
                                            pad_mode='replicate', context_dim=context_dim, normalize_rgb=normalize_rgb,
                                            res_from_fc=obj_res_from_fc,
                                            ch_mult=obj_ch_mult, base_ch=obj_base_ch, final_cnn_ch=obj_final_cnn_ch,
                                            num_res_blocks=num_res_blocks, cnn_mid_blocks=cnn_mid_blocks,
                                            mlp_hidden_dim=mlp_hidden_dim,
                                            init_zero_bias=init_zero_bias,
                                            init_conv_layers=init_conv_layers,
                                            init_conv_fg_std=init_conv_fg_std
                                            )

        self.num_obj_upsample = self.particle_dec.num_upsample # TODO:This never gets used
        # bg decoder
        self.bg_dec = BgDecoder(cdim=cdim, image_size=image_size,
                                pad_mode='replicate', learned_bg_feature_dim=learned_bg_feature_dim,
                                use_resblock=use_resblock, context_dim=context_dim, film=decode_with_ctx,
                                timestep_horizon=timestep_horizon,
                                bg_res_from_fc=bg_res_from_fc, bg_ch_mult=bg_ch_mult, bg_base_ch=bg_base_ch,
                                bg_final_cnn_ch=bg_final_cnn_ch, num_res_blocks=num_res_blocks,
                                decode_with_ctx=decode_with_ctx, normalize_rgb=normalize_rgb,
                                cnn_mid_blocks=cnn_mid_blocks, mlp_hidden_dim=mlp_hidden_dim,
                                init_zero_bias=init_zero_bias, init_conv_layers=init_conv_layers,
                                init_conv_bg_std=init_conv_bg_std
                                )
        self.num_bg_upsample = self.bg_dec.num_bg_upsample
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                pass
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if self.init_zero_bias and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        ### NEW: bias-init the occupancy head to logit(prior)
        if self.occupancy_mode:
            p0 = min(max(self.occupancy_prior, 1e-4), 1 - 1e-4)
            logit_p0 = math.log(p0 / (1 - p0))
            # We try a few common module names to locate the final 3D conv
            occ_bias = None
            for attr_name in ["final_conv", "out_conv", "to_rgb", "to_logits"]:
                mod = getattr(self.particle_dec, attr_name, None)
                if isinstance(mod, nn.Conv3d) and mod.bias is not None:
                    try:
                        with torch.no_grad():
                            mod.bias[self.occ_channel_index].fill_(logit_p0)
                            occ_bias = mod.bias
                    except Exception:
                        pass
                    break
        else:
            p0 = min(max(self.alpha_prior, 1e-4), 1 - 1e-4)
            logit_p0 = math.log(p0 / (1 - p0))
            for attr_name in ["final_conv", "out_conv", "to_rgb", "to_logits"]:
                mod = getattr(self.particle_dec, attr_name, None)
                if isinstance(mod, nn.Conv3d) and mod.bias is not None:
                    with torch.no_grad():
                        mod.bias[0].fill_(logit_p0)
                    break

    def translate_patches(self, kp_batch, patches_batch, scale=None, translation=None, scale_normalized=False):
        """
        3D version.
        kp_batch:      [B, N, 3] in [-1, 1]
        patches_batch: [B, N, Cpatch, ps, ps, ps]
        scale:         None or [B, N, 3] or [B, N, 1]
        returns:       [B, N, Cpatch, H, W, L]
        """
        B, N, Cpatch, ps, _, _ = patches_batch.shape
        H = W = self.feature_map_size
        L = self.feature_map_size  # use one cube size for now (keeps parity with encoder)

        if scale is None:
            z_scale = torch.tensor([ps / H, ps / W, ps / L], device=kp_batch.device, dtype=kp_batch.dtype)
            z_scale = z_scale.view(1, 1, 3).expand_as(kp_batch)      # [B, N, 3]
        else:
            z_scale = scale if scale_normalized else torch.sigmoid(scale)

        z_pos   = kp_batch.reshape(-1, 3)                             # [B*N, 3]
        z_scale = z_scale.view(-1, z_scale.shape[-1])                 # [B*N, 3]

        patches_batch = patches_batch.reshape(-1, Cpatch, ps, ps, ps) # [B*N, Cpatch, ps, ps, ps]
        out_dims = (B * N, Cpatch, H, W, L)

        # uses your 3D spatial_transform (already upgraded earlier)
        trans = spatial_transform(
            patches_batch, z_pos, z_scale, out_dims, inverse=True, padding_mode='border'
        )
        return trans.view(B, N, Cpatch, H, W, L)


    def get_objects_occupancy(self, z_kp, z_features, z_scale=None, z_ctx=None, translation=None):
        """
        Decode per-object occupancy logits and place them in the volume.
        Returns:
        occ_logits_per_obj: [B,N,1,D,H,W]
        occ_prob_per_obj:   [B,N,1,D,H,W]
        occ_prob_composite: [B,1,D,H,W]   via probabilistic OR over objects
        """
        patches = self.particle_dec(z_features, context=z_ctx)        # [B*N, 1, ps, ps, ps] (logits)
        B, N = z_kp.shape[:2]

        patches = patches.view(B, N, *patches.shape[1:])              # [B,N,1,ps,ps,ps]
        patches_t = self.translate_patches(z_kp, patches, z_scale, translation)  # [B,N,1,D,H,W]

        occ_logits = patches_t                                        # logits per object
        occ_prob   = torch.sigmoid(occ_logits)                        # probs per object

        # p_total_objects = 1 - Π_k (1 - p_k * on_k)
        obj_on = self.obj_on if hasattr(self, "obj_on") else None  # not used here; pass in decode_objects
        return occ_logits, occ_prob
    def composite_occupancy_no_alpha(self,occ_prob, obj_on, eps=1e-8):
        """
        occ_prob: [B,N,1,D,H,W] in [0,1]
        obj_on:   [B,N] or [B,N,1]
        -> p_obj: [B,1,D,H,W]   = 1 - Π_k (1 - p_k)
        """
        if obj_on.dim() == 3: obj_on = obj_on.squeeze(-1)
        gate = obj_on[:, :, None, None, None, None]  # [B,N,1,1,1,1]
        p_k = torch.clamp(gate * occ_prob, 0.0, 1.0)
        log_1m = torch.log(torch.clamp(1.0 - p_k, min=eps))
        return 1.0 - torch.exp(log_1m.sum(dim=1, keepdim=False))  # [B,1,D,H,W]


    def decode_rgb_unified(self, z_kp, z_features, z_scale=None, z_ctx=None, translation=None):
        """
        Unified decoder path (3D) that supports arbitrary content channels.
        Expect at least 2 channels: [alpha, content...]
        - particle_dec outputs C over a cube [ps, ps, ps]
        - translate to [B, N, C, H, W, L]
        Returns:
        dec_patches: [B,N,C,ps,ps,ps]
        a_obj:       [B,N,1,H,W,L]
        content_obj: [B,N,Cc,H,W,L]  (Cc can be 1 for occupancy, 3 for RGB, etc.)
        d_obj:       [B,N,1,H,W,L] or None
        """
        patches = self.particle_dec(z_features, context=z_ctx)        # [B*N, C, ps, ps, ps]
        patches = patches.view(-1, z_kp.shape[1], *patches.shape[1:]) # [B, N, C, ps, ps, ps]
        patches_t = self.translate_patches(z_kp, patches, z_scale, translation)  # [B,N,C,D,H,W]


        C = patches_t.shape[2]
        assert C >= 1 + self.cdim, f"need [alpha + {self.cdim} rgb], got C={C}"

        a_logits = patches_t[:, :, :1]                         # [B,N,1,D,H,W]
        a_obj    = torch.sigmoid(a_logits)                     # α in [0,1]
        rgb_raw  = patches_t[:, :, 1:1+self.cdim]              # [B,N,3,D,H,W]
        if self.normalize_rgb:
            rgb_obj = torch.tanh(rgb_raw)                      # [-1,1]
        else:
            rgb_obj = torch.sigmoid(rgb_raw)                   # [0,1]

        
        return patches, a_obj, rgb_obj, None                    # d_obj=None

    def composite_rgb(self, a_obj, rgb_obj, obj_on, eps=1e-6):
        # a_obj: [B,N,1,D,H,W] in [0,1], rgb_obj: [B,N,3,D,H,W]
        if obj_on.dim() == 3: obj_on = obj_on.squeeze(-1)      # [B,N]
        gate = obj_on[:, :, None, None, None, None]            # [B,N,1,1,1,1]
        a = torch.clamp(a_obj * gate, 0.0, 1.0)                # gate off disabled objects
        # per-voxel mixture weights from alpha only (no ordering)
        w = a / (a.sum(dim=1, keepdim=True) + eps)             # [B,N,1,D,H,W]
        rgb_comp = (w * rgb_obj).sum(dim=1)                    # [B,3,D,H,W]
        alpha_sum = a.sum(dim=1, keepdim=False).clamp(max=1.0)   # [B,1,D,H,W]
        bg_mask = 1.0 - alpha_sum
        # return per-object α for optional aux losses/visualization
        return a, bg_mask, rgb_comp, None


    def decode_objects(
        self, z_kp, z_features, obj_on, z_scale=None, translation=None, z_depth=None,
        z_ctx=None,
    ):
        """
        If self.separate_depth_features:
            - use particle_dec_rgb on z_features (RGB features)
            - use particle_dec_depth on z_depth_features (Depth features)
        Else:
            - use particle_dec on z_features (unified: α+RGB or α+RGB+D)
        """

        if getattr(self, "occupancy_mode", False):
            occ_logits, occ_prob = self.get_objects_occupancy(z_kp, z_features, z_scale=z_scale, z_ctx=z_ctx, translation=translation)
            p_obj = self.composite_occupancy_no_alpha(occ_prob, obj_on)  # [B,1,D,H,W]
            return {
                "occ_logits_per_obj": occ_logits,
                "occ_prob_per_obj":   occ_prob,
                "occ_prob_composite": p_obj,
            }
        dec_rgb_patches, a_obj, rgb_obj, d_obj = self.decode_rgb_unified(
            z_kp, z_features, z_scale=z_scale, z_ctx=z_ctx, translation=translation
        )

        dec_depth_patches = None  # unified depth is not per-patch separate object unless you want to expose it

        # Composite (always uses z_depth for ordering)
        alpha_masks, bg_mask, dec_rgb_comp, dec_depth_comp = self.composite_rgb(a_obj, rgb_obj, obj_on)

        return dec_rgb_patches, dec_rgb_comp, alpha_masks, bg_mask, dec_depth_comp, dec_depth_patches, rgb_obj

    def decode_all(
        self, z, z_scale, z_features, obj_on, z_depth,
        z_bg_features, z_ctx=None, warmup=False
    ):
        # ---- flatten time exactly like your original ----
        if len(z.shape) == 4:
            B, T = z.shape[0], z.shape[1]
            z              = z.view(-1, *z.shape[2:])
            z_scale        = z_scale.view(-1, *z_scale.shape[2:])
            obj_on         = obj_on.view(-1, *obj_on.shape[2:])
            z_depth        = z_depth.view(-1, *z_depth.shape[2:])
            z_features     = z_features.view(-1, *z_features.shape[2:])
            z_bg_features  = z_bg_features.view(-1, *z_bg_features.shape[2:])
            if z_ctx is not None:
                z_ctx = z_ctx.view(-1, *z_ctx.shape[2:])
        else:
            T = 1

        # ensure obj_on is [B*, N]
        if obj_on.dim() == 3:
            obj_on = obj_on.squeeze(-1)

        if getattr(self, "occupancy_mode", False):
            occ_out = self.decode_objects(z, z_features, obj_on, z_scale=z_scale, z_ctx=z_ctx)
            p_obj   = occ_out["occ_prob_composite"]                 # [B*,1,D,H,W]

            # BG prior as occupancy logits in channel 0
            bg_raw    = self.bg_dec(z_bg_features, z_ctx)           # [B*, C_bg, D,H,W]
            bg_logits = bg_raw[:, :1, ...]                          # [B*,1,D,H,W]
            p_bg      = torch.sigmoid(bg_logits)


            # Ensure same spatial shape/order; resample bg if needed
            if p_bg.shape[-3:] != p_obj.shape[-3:]:
                p_bg = F.interpolate(p_bg, size=p_obj.shape[-3:], mode="trilinear", align_corners=False)

            p_total = 1.0 - (1.0 - p_bg) * (1.0 - p_obj)            # union BG + objects

            return {
                "rec":                p_total,      
                "occ_logits_per_obj": occ_out["occ_logits_per_obj"],
                "occ_prob_per_obj":   occ_out["occ_prob_per_obj"],
                "occ_prob_composite": p_obj,
                "bg_rec":             bg_raw,
                "bg_logits":          bg_logits,
                "bg_prob":            p_bg,
                "bg_mask":            1.0 - p_total,

                # RGB/Depth unused here
                "dec_objects":        None,
                "dec_objects_trans":  None,
                "alpha_masks":        None,
                "rec_rgb":            None,
                "rec_depth":          None,
                "dec_depth_trans":    None,
                "dec_depth_patches":  None,
            }
        # =========================
        # RGB / RGBD PATH (unchanged)
        # =========================
        (dec_objects, dec_objects_trans, alpha_masks, bg_mask,
        dec_depth_trans, dec_depth_patches, rgb_obj) = self.decode_objects(
            z, z_features, obj_on, z_depth=z_depth, z_scale=z_scale, z_ctx=z_ctx
        )

        bg_rec = self.bg_dec(z_bg_features, z_ctx)   # [B*, C_bg, *spatial*]

        # Ensure bg_rec matches object spatial dims
        if bg_rec.shape[-3:] != dec_objects_trans.shape[-3:]:
            bg_rec = F.interpolate(bg_rec, size=dec_objects_trans.shape[-3:], mode="trilinear", align_corners=False)

        C_bg   = bg_rec.shape[1]


        if C_bg >= 3:
            rec_rgb = bg_mask * bg_rec[:, :3, ...] + dec_objects_trans
        else:
            rec_rgb = dec_objects_trans
        rec_depth = None
        have_obj_depth = (dec_depth_trans is not None)
        have_bg_depth  = (C_bg > 3)
        if have_obj_depth or have_bg_depth:
            bg_depth = bg_rec[:, 3:4, ...] if have_bg_depth else (
                torch.zeros_like(dec_depth_trans) if dec_depth_trans is not None else None
            )
            if bg_depth is None:
                rec_depth = dec_depth_trans
            elif dec_depth_trans is None:
                rec_depth = bg_depth
            else:
                rec_depth = bg_mask * bg_depth + dec_depth_trans
        
        rec = torch.cat([rec_rgb, rec_depth], dim=1) if rec_depth is not None else rec_rgb

        return {
            'rec': rec,
            'dec_objects': dec_objects,
            'dec_objects_trans': dec_objects_trans,
            'alpha_masks': alpha_masks,
            'bg_mask': bg_mask,
            'bg_rec': bg_rec,
            'rec_rgb': rec_rgb,
            'rec_depth': rec_depth,
            'dec_depth_trans': dec_depth_trans,
            'dec_depth_patches': dec_depth_patches,
            'rgb_obj': rgb_obj,  
        }



    def forward(self, z, z_scale, z_features, obj_on_sample, z_depth, z_bg_features, z_ctx=None,
                warmup=False):
        return self.decode_all(z, z_scale, z_features, obj_on_sample, z_depth, z_bg_features, z_ctx, warmup)


class DLPContext(nn.Module):
    def __init__(self, n_kp_enc, dropout=0.1, learned_feature_dim=16, learned_bg_feature_dim=16, embed_init_std=0.02,
                 projection_dim=128, timestep_horizon=1, pte_layers=1, pte_heads=1,
                 attn_norm_type='rms', context_dim=7, hidden_dim=256,
                 activation='gelu',
                 ctx_pool_mode='none', bg=True, n_views=1, causal=True, particle_positional_embed=True,
                 particle_score=False, norm_layer=True,
                 shared_logvar=False, ctx_dist='gauss', n_ctx_categories=4, n_ctx_classes=4,
                 particle_anchors=None, use_z_orig=False,
                 ctx_pool_dim=256, n_pool_ctx_categories=8, n_pool_ctx_classes=8, global_ctx_pool=False,
                 token_pool_cross_attn=False, global_local_fuse_mode='none', condition_local_on_global=True,
                 # external conditioning
                 action_condition=False,  # condition on actions
                 action_dim=0,  # dimension of input actions
                 random_action_condition=False,  # condition on random actions
                 random_action_dim=0,  # dimension of sampled random actions
                 null_action_embed=False,  # learn a "no-input-action" embedding, to learn on action-free videos as well
                 action_as_particle=False,  # if False, use AdaLN conditioning
                 language_condition=False,  # condition on language embedding
                 language_embed_dim=0,  # embedding dimension for each token
                 language_max_len=64,  # maximum tokens per prompt
                 language_condition_type='self',  # cross-attention ('cross') or self-attention ('self')
                 img_goal_condition=False,
                 img_goal_condition_type='adaln',
                 # cross-attention ('cross'), adaptive LN ('adaln') or self-attention ('self')
                 pos_embed_t_adaln=True,  # pos embeddings for timesteps using adaln
                 pos_embed_p_adaln=True,  # pos embeddings for particles using adaln
                 pos_embed_objon_adaln=False,  # pos embeddings for particles transparency using adaln
                 particle_pool_adaln=False
                 ):
        super(DLPContext, self).__init__()
        """
        This module takes in temporal sequence of particles and outputs latent context,
        which can be per-particle, or global, depending on the pooling type.
        This module shares attention layers and has different head for posterior latent context (inverse model)
        and prior latent context (policy).
        """
        assert ctx_pool_mode in ['none', 'mean', 'max', 'token', 'last', 'mlp']
        self.ctx_pool_mode = ctx_pool_mode
        self.n_kp_enc = n_kp_enc
        self.dropout = dropout
        self.learned_feature_dim = learned_feature_dim
        self.learned_bg_feature_dim = learned_bg_feature_dim
        assert learned_feature_dim > 0, "learned_feature_dim must be greater than 0"
        self.embed_init_std = embed_init_std
        self.projection_dim = projection_dim
        self.timestep_horizon = (timestep_horizon + 1) if timestep_horizon > 1 else 1
        self.attn_norm_type = attn_norm_type
        self.context_dist = ctx_dist
        self.n_ctx_categories = n_ctx_categories
        self.n_ctx_classes = n_ctx_classes
        self.context_dim = context_dim
        self.learned_ctx_token = (ctx_pool_mode == 'token' and self.context_dim > 0)
        self.n_pool_ctx_categories = n_pool_ctx_categories
        self.n_pool_ctx_classes = n_pool_ctx_classes
        self.ctx_pool_dim = ctx_pool_dim
        if self.context_dist == 'categorical':
            self.ctx_pool_dim = int(self.n_pool_ctx_categories * self.n_pool_ctx_classes)
        self.global_ctx_pool = global_ctx_pool
        self.global_local_fuse_mode = global_local_fuse_mode
        self.condition_local_on_global = condition_local_on_global
        self.token_pool_cross_attn = token_pool_cross_attn
        self.hidden_dim = hidden_dim
        self.with_bg = bg
        self.n_views = n_views
        self.activation = activation
        self.is_causal = causal
        self.particle_score = particle_score
        self.shared_logvar = shared_logvar
        self.use_z_orig = use_z_orig
        self.pos_embed_t_adaln = pos_embed_t_adaln
        self.pos_embed_p_adaln = pos_embed_p_adaln
        self.pos_embed_objon_adaln = pos_embed_objon_adaln
        self.particle_pool_adaln = particle_pool_adaln

        # actions
        self.action_condition = action_condition
        self.action_dim = action_dim
        self.random_action_condition = random_action_condition
        self.random_action_dim = random_action_dim
        self.learn_null_action_embed = null_action_embed
        self.action_as_particle = action_as_particle
        # language
        self.language_condition = language_condition
        self.language_embed_dim = language_embed_dim
        self.language_max_len = language_max_len
        self.language_condition_type = language_condition_type
        assert self.language_condition_type in ['cross', 'self'], \
            f'lang condition type {self.language_condition_type} is not supported'
        # image goal
        self.img_goal_condition = img_goal_condition
        self.img_goal_condition_type = img_goal_condition_type
        assert self.img_goal_condition_type in ['cross', 'self', 'adaln'], \
            f'img goal condition type {self.img_goal_condition_type} is not supported'

        if self.learn_null_action_embed:
            self.null_action_embeddings = nn.Parameter(
                self.embed_init_std * torch.randn(1, 1, self.action_dim))
        else:
            self.null_action_embeddings = None

        if self.action_condition and self.action_dim > 0:
            if self.action_as_particle:
                self.action_proj = nn.Sequential(nn.Linear(self.action_dim, hidden_dim),
                                                 RMSNorm(hidden_dim))
            else:
                self.action_proj = nn.Sequential(nn.Linear(self.action_dim, hidden_dim),
                                                 RMSNorm(hidden_dim),
                                                 nn.GELU())
        else:
            self.action_proj = None

        if self.random_action_condition and self.random_action_dim > 0:
            if self.action_as_particle:
                self.random_action_proj = nn.Sequential(nn.Linear(self.random_action_dim, hidden_dim),
                                                        RMSNorm(hidden_dim))
            else:
                self.random_action_proj = nn.Sequential(nn.Linear(self.random_action_dim, hidden_dim),
                                                        RMSNorm(hidden_dim),
                                                        nn.GELU())
        else:
            self.random_action_proj = None

        if self.language_condition and self.language_embed_dim > 0:
            # self.lang_proj = nn.Sequential(nn.Linear(self.language_embed_dim, hidden_dim),
            #                                RMSNorm(hidden_dim),
            #                                nn.GELU(),
            #                                nn.Linear(hidden_dim, hidden_dim))
            self.lang_proj = nn.Linear(self.language_embed_dim, hidden_dim)
            if self.language_condition_type == 'self':
                self.lang_pos_embed = nn.Parameter(
                    self.embed_init_std * torch.randn(1, 1, 1, hidden_dim))
            else:
                # self.lang_pos_embed = nn.Parameter(
                #     self.embed_init_std * torch.randn(1, 1, self.language_max_len, hidden_dim))
                self.lang_pos_embed = None
        else:
            self.lang_proj = None
            self.lang_pos_embed = None

        if self.img_goal_condition:
            if self.img_goal_condition_type == 'adaln':
                self.goal_proj = nn.Sequential(nn.Linear(projection_dim, hidden_dim),
                                               RMSNorm(hidden_dim),
                                               nn.GELU())
            else:
                self.goal_proj = nn.Sequential(nn.Linear(projection_dim, hidden_dim),
                                               RMSNorm(hidden_dim),
                                               nn.GELU(),
                                               nn.Linear(hidden_dim, hidden_dim))
            # self.goal_proj = nn.Linear(projection_dim, hidden_dim)
            # self.goal_proj = nn.Identity()
        else:
            self.goal_proj = None

        if self.particle_pool_adaln:
            self.particle_pool_proj = nn.Sequential(nn.Linear(projection_dim, self.hidden_dim),
                                                    nn.GELU(),
                                                    ParticlePool(pool_mode='mean', pool_dim=-2),
                                                    nn.Linear(self.hidden_dim, self.hidden_dim),
                                                    RMSNorm(hidden_dim),
                                                    nn.GELU())
        else:
            self.particle_pool_proj = None

        if particle_anchors is None:
            self.register_buffer('particles_anchor', torch.zeros(1, 1, self.n_kp_enc))
            self.use_z_orig = False
        else:
            self.register_buffer('particles_anchor', particle_anchors)

        n_particles = self.n_kp_enc  # [n_kp_enc]

        # entities in attn: [bg*, n_particles, ctx, ctx_tokens*]
        if (self.learned_ctx_token or self.ctx_pool_mode == 'last') and self.token_pool_cross_attn:
            if self.learned_ctx_token:
                n_particles += 1
                self.ctx_token_embeddings = nn.Parameter(
                    self.embed_init_std * torch.randn(1, 1, 1, projection_dim))
            block_size = 1  # this means token pooling does not depend on the temporal horizon
            self.cross_attn_block = CrossBlock(n_embed=self.projection_dim, n_head=pte_heads,
                                               block_size=block_size,
                                               attn_pdrop=dropout,
                                               resid_pdrop=dropout,
                                               hidden_dim_multiplier=4, positional_bias=False,
                                               activation='gelu',
                                               max_particles=None, norm_type=attn_norm_type)
        elif self.learned_ctx_token:
            n_particles += 1
            self.ctx_token_embeddings = nn.Parameter(
                self.embed_init_std * torch.randn(1, 1, 1, projection_dim))
            self.cross_attn_block = None
        else:
            self.ctx_token_embeddings = None
            self.cross_attn_block = None
        if self.with_bg:
            n_particles += 1
            self.bg_embeddings = nn.Parameter(self.embed_init_std * torch.randn(1, 1, 1, projection_dim))
        if self.img_goal_condition and self.img_goal_condition_type == 'self':
            n_particles += self.n_kp_enc
            if self.with_bg:
                n_particles += 1
            self.goal_embeddings = nn.Parameter(self.embed_init_std * torch.randn(1, 1, 1, projection_dim))
        if self.action_condition and self.action_as_particle:
            self.action_embeddings = nn.Parameter(self.embed_init_std * torch.randn(1, 1, 1, projection_dim))
            n_particles += 1
        if self.random_action_condition and self.action_as_particle:
            self.random_action_embeddings = nn.Parameter(self.embed_init_std * torch.randn(1, 1, 1, projection_dim))
            n_particles += 1
        if self.language_condition and self.language_condition_type == 'self':
            n_particles += self.language_max_len
        # if self.pos_embed_t_adaln:
        # self.pos_embed_t_embeddings = nn.Parameter(self.embed_init_std * torch.randn(1, 1, 1, projection_dim))
        # self.pos_embed_t_embeddings = nn.Parameter(
        #     self.embed_init_std * torch.randn(1, self.timestep_horizon, 1, self.hidden_dim))
        # self.learned_token_projection = nn.Sequential(nn.Linear(projection_dim, hidden_dim),
        #                                               RMSNorm(hidden_dim),
        #                                               nn.GELU())
        # if self.img_goal_condition and self.img_goal_condition_type == 'adaln':
        #     self.token_pool_embeddings_goal = nn.Parameter(
        #         self.embed_init_std * torch.randn(1, 1, 1, projection_dim))
        # n_particles += 1

        # entities positional embeddings
        self.particle_pos_embed = particle_positional_embed and not self.pos_embed_p_adaln
        if self.particle_pos_embed:
            self.particle_embeddings = nn.Parameter(
                self.embed_init_std * torch.randn(1, 1, self.n_kp_enc, projection_dim))
        else:
            self.particle_embeddings = nn.Parameter(self.embed_init_std * torch.randn(1, 1, 1, projection_dim))

        if self.n_views > 1:
            self.view_embeddings = nn.Parameter(
                self.embed_init_std * torch.randn(1, 1, self.n_views, 1, projection_dim))
        else:
            self.view_embeddings = None

        if self.pos_embed_p_adaln:
            n_particles += (self.n_views - 1) * (self.n_kp_enc + 1)
            self.pos_p_embeddings = nn.Parameter(
                self.embed_init_std * torch.randn(1, 1, n_particles, self.hidden_dim))

        if self.pos_embed_objon_adaln:
            self.objon_embeddings = nn.Sequential(nn.Linear(1, self.hidden_dim),
                                                  RMSNorm(hidden_dim),
                                                  nn.GELU())

        # interaction encoder
        proj_out_dim = projection_dim
        self.basic_particle_proj = ParticleAttributesProjection(n_particles=self.n_kp_enc,
                                                                in_features_dim=self.learned_feature_dim,
                                                                hidden_dim=self.hidden_dim,
                                                                output_dim=proj_out_dim,
                                                                bg_features_dim=self.learned_bg_feature_dim,
                                                                add_ctx_token=False,
                                                                depth=True,
                                                                obj_on=True,
                                                                base_var=False, bg=self.with_bg,
                                                                norm_layer=norm_layer,
                                                                particle_score=self.particle_score,
                                                                use_z_orig=self.use_z_orig)

        block_size = self.timestep_horizon
        pte_action_cond = (self.action_condition or self.random_action_condition) and not self.action_as_particle
        pte_goal_cond = (self.img_goal_condition and self.img_goal_condition_type == 'adaln')
        pte_context_cond = pte_action_cond or pte_goal_cond or self.particle_pool_adaln or self.pos_embed_p_adaln
        lang_cross_attn = (self.language_condition and self.language_condition_type == 'cross')
        img_goal_cross_attn = (self.img_goal_condition and self.img_goal_condition_type == 'cross')
        cross_attn = lang_cross_attn or img_goal_cross_attn
        self.pte = ParticleSpatioTemporalTransformer(n_embed=self.projection_dim, n_head=pte_heads,
                                                     n_layer=pte_layers,
                                                     block_size=block_size,
                                                     output_dim=self.projection_dim, attn_pdrop=dropout,
                                                     resid_pdrop=dropout,
                                                     hidden_dim_multiplier=4, positional_bias=False,
                                                     activation='gelu',
                                                     max_particles=None, norm_type=attn_norm_type,
                                                     particles_first=False, init_std=embed_init_std,
                                                     causal=self.is_causal,
                                                     context_cond=pte_context_cond,
                                                     residual_modulation=pte_context_cond,
                                                     context_gate=pte_context_cond,
                                                     cond_cross_attn=cross_attn,
                                                     pos_embed_t_adaln=self.pos_embed_t_adaln)

        if self.global_ctx_pool:
            # global
            global_ctx_pool = 'token' if self.ctx_pool_mode == 'none' else self.ctx_pool_mode
            self.global_posterior_decoder = ParticleContextDecoder(n_particles=self.n_kp_enc, input_dim=projection_dim,
                                                                   hidden_dim=self.hidden_dim,
                                                                   context_dim=self.ctx_pool_dim,
                                                                   context_dist=self.context_dist,
                                                                   n_ctx_categories=self.n_pool_ctx_categories,
                                                                   n_ctx_classes=self.n_pool_ctx_classes,
                                                                   learned_ctx_token=self.learned_ctx_token,
                                                                   ctx_pool_mode=global_ctx_pool,
                                                                   shared_logvar=self.shared_logvar,
                                                                   output_ctx_logvar=(ctx_dist != 'categorical'),
                                                                   conditional=False, cond_dim=0)
            self.global_prior_decoder = ParticleContextDecoder(n_particles=self.n_kp_enc, input_dim=projection_dim,
                                                               hidden_dim=self.hidden_dim,
                                                               context_dim=self.ctx_pool_dim,
                                                               context_dist=self.context_dist,
                                                               n_ctx_categories=self.n_pool_ctx_categories,
                                                               n_ctx_classes=self.n_pool_ctx_classes,
                                                               learned_ctx_token=self.learned_ctx_token,
                                                               ctx_pool_mode=global_ctx_pool,
                                                               shared_logvar=self.shared_logvar,
                                                               output_ctx_logvar=(ctx_dist != 'categorical'),
                                                               conditional=False, cond_dim=0)

            # local
            self.posterior_decoder = ParticleContextDecoder(n_particles=self.n_kp_enc, input_dim=projection_dim,
                                                            hidden_dim=self.hidden_dim,
                                                            context_dim=self.context_dim,
                                                            context_dist=self.context_dist,
                                                            n_ctx_categories=self.n_ctx_categories,
                                                            n_ctx_classes=self.n_ctx_classes,
                                                            learned_ctx_token=False,
                                                            ctx_pool_mode="none",
                                                            shared_logvar=self.shared_logvar,
                                                            output_ctx_logvar=(ctx_dist != 'categorical'),
                                                            conditional=self.condition_local_on_global,
                                                            cond_dim=self.ctx_pool_dim)
            self.prior_decoder = ParticleContextDecoder(n_particles=self.n_kp_enc, input_dim=projection_dim,
                                                        hidden_dim=self.hidden_dim,
                                                        context_dim=self.context_dim,
                                                        context_dist=self.context_dist,
                                                        n_ctx_categories=self.n_ctx_categories,
                                                        n_ctx_classes=self.n_ctx_classes,
                                                        learned_ctx_token=False,
                                                        ctx_pool_mode="none",
                                                        shared_logvar=self.shared_logvar,
                                                        output_ctx_logvar=(ctx_dist != 'categorical'),
                                                        conditional=self.condition_local_on_global,
                                                        cond_dim=self.ctx_pool_dim)
        else:
            self.global_posterior_decoder = self.global_prior_decoder = nn.Identity()
            self.posterior_decoder = ParticleContextDecoder(n_particles=self.n_kp_enc, input_dim=projection_dim,
                                                            hidden_dim=self.hidden_dim,
                                                            context_dim=self.context_dim,
                                                            context_dist=self.context_dist,
                                                            n_ctx_categories=self.n_ctx_categories,
                                                            n_ctx_classes=self.n_ctx_classes,
                                                            learned_ctx_token=self.learned_ctx_token,
                                                            ctx_pool_mode=self.ctx_pool_mode,
                                                            shared_logvar=self.shared_logvar,
                                                            output_ctx_logvar=(ctx_dist != 'categorical'),
                                                            conditional=False, cond_dim=0)
            self.prior_decoder = ParticleContextDecoder(n_particles=self.n_kp_enc, input_dim=projection_dim,
                                                        hidden_dim=self.hidden_dim,
                                                        context_dim=self.context_dim,
                                                        context_dist=self.context_dist,
                                                        n_ctx_categories=self.n_ctx_categories,
                                                        n_ctx_classes=self.n_ctx_classes,
                                                        learned_ctx_token=self.learned_ctx_token,
                                                        ctx_pool_mode=self.ctx_pool_mode,
                                                        shared_logvar=self.shared_logvar,
                                                        output_ctx_logvar=(ctx_dist != 'categorical'),
                                                        conditional=False, cond_dim=0)
        self.init_weights()

    def init_weights(self):
        self.posterior_decoder.init_weights()
        self.prior_decoder.init_weights()
        self.pte.init_weights()

    def encode_all(self, z, z_scale, z_obj_on, z_depth, z_features, z_bg_features=None, z_base_var=None,
                   z_score=None, patch_id_embed=None, deterministic=False, warmup=False,
                   detach_before_proj=False, encode_posterior=True, encode_prior=True, actions=None, actions_mask=None,
                   lang_embed=None, z_goal=None, detach_z_goal=False):
        """
        output order:
        if with_bg and ctx_pool_mode='token': [n_particles, bg, ctx, ctx_token*]
        else: [n_particles, ctx, ctx_token*]
        """
        # x: [bs, t, ch, h, w]
        bs, timestep_horizon = z.shape[0], z.shape[1]
        z_v = z.detach() if detach_before_proj else z
        z_scale_v = z_scale.detach() if detach_before_proj else z_scale
        z_obj_on_v = z_obj_on.detach() if (z_obj_on is not None and detach_before_proj) else z_obj_on
        z_depth_v = z_depth.detach() if (z_depth is not None and detach_before_proj) else z_depth
        z_features_v = z_features.detach() if detach_before_proj else z_features
        if not self.with_bg:
            z_bg_features = None
        z_bg_features_v = z_bg_features.detach() if (
                z_bg_features is not None and detach_before_proj) else z_bg_features
        z_base_var_v = z_base_var.detach() if z_base_var is not None else z_base_var
        z_score_v = z_score.detach() if z_score is not None else z_score
        if self.use_z_orig:
            z_orig_v = self.particles_anchor.unsqueeze(0).repeat(z_v.shape[0], z_v.shape[1], 1, 1)
        else:
            z_orig_v = None

        particle_projection = self.basic_particle_proj(z=z_v,
                                                       z_scale=z_scale_v,
                                                       z_obj_on=z_obj_on_v,
                                                       z_depth=z_depth_v,
                                                       z_features=z_features_v,
                                                       z_bg_features=z_bg_features_v,
                                                       z_base_var=z_base_var_v,
                                                       z_score=z_score_v,
                                                       z_orig=z_orig_v)
        # [bs, T, n_kp + 1, projection_dim or 2 * pctx_dim]

        # add entity pos embeddings
        if self.particle_embeddings.shape[2] == 1:
            p_embeddings = self.particle_embeddings.repeat(bs, timestep_horizon, self.n_kp_enc, 1)
        else:
            p_embeddings = self.particle_embeddings.repeat(bs, timestep_horizon, 1, 1)
        if patch_id_embed is not None:
            p_embeddings = p_embeddings + patch_id_embed
        if self.with_bg:
            bg_embeddings = self.bg_embeddings.repeat(bs, timestep_horizon, 1, 1)
            p_embeddings = torch.cat([p_embeddings, bg_embeddings], dim=2)
        particle_projection = particle_projection + p_embeddings

        c = c_a = c_r = c_g = l = None  # conditions
        if self.img_goal_condition and z_goal is None:
            particle_projection, goal_projection = particle_projection.split([particle_projection.shape[1] - 1, 1],
                                                                             dim=1)
            # goal_projection: [bs, 1, N, d]
            if detach_z_goal:
                goal_projection = goal_projection.detach()
            z_goal = self.goal_proj(goal_projection)
            l_or_cg = z_goal.repeat(1, particle_projection.shape[1], 1, 1)
            timestep_horizon = timestep_horizon - 1
            if self.img_goal_condition_type == 'cross':
                l = l_or_cg
            elif self.img_goal_condition_type == 'self':
                p_g = l_or_cg + self.goal_embeddings
                particle_projection = torch.cat([particle_projection, p_g], dim=2)  # [bs, T, n_p * 2]
            elif self.img_goal_condition_type == 'adaln':
                c_g = l_or_cg
                c = c_g
                # if self.token_pool_adaln:
                #     token_pool_goal = self.token_pool_embeddings_goal.repeat(c.shape[0], c.shape[1], 1, 1)
                #     c = torch.cat([c, token_pool_goal], dim=2)
                # if c is None:
                #     c = cg
                # else:
                #     c = c + cg
        elif self.img_goal_condition and z_goal is not None:
            l_or_cg = z_goal.repeat(1, particle_projection.shape[1], 1, 1)
            # if self.img_goal_condition_type == 'cross' and l is None:
            if self.img_goal_condition_type == 'cross':
                l = l_or_cg
            elif self.img_goal_condition_type == 'self':
                p_g = l_or_cg + self.goal_embeddings
                particle_projection = torch.cat([particle_projection, p_g], dim=2)  # [bs, T, n_p * 2]
            elif self.img_goal_condition_type == 'adaln':
                c_g = l_or_cg
                c = c_g
                # if self.token_pool_adaln:
                #     token_pool_goal = self.token_pool_embeddings_goal.repeat(c.shape[0], c.shape[1], 1, 1)
                #     c = torch.cat([c, token_pool_goal], dim=2)
                # if c is None:
                #     c = cg
                # else:
                #     c = c + cg

        if self.learned_ctx_token and not self.token_pool_cross_attn:
            if self.img_goal_condition and self.img_goal_condition_type == 'self':
                n_goal_particles = self.n_kp_enc
                if self.with_bg:
                    n_goal_particles += 1
                pp, pg = particle_projection.split([particle_projection.shape[2] - n_goal_particles, n_goal_particles],
                                                   dim=2)
                pc = self.ctx_token_embeddings.repeat(bs, timestep_horizon, 1, 1)
                particle_projection = torch.cat([pp, pc, pg], dim=1)
            else:
                particle_projection = torch.cat([particle_projection,
                                                 self.ctx_token_embeddings.repeat(bs, timestep_horizon, 1, 1)], dim=2)

        if self.random_action_condition:
            # if self.img_goal_condition and z_goal is None:
            #     rand_action_horizon = timestep_horizon - 1
            # else:
            #     rand_action_horizon = timestep_horizon
            rand_action_horizon = timestep_horizon
            random_actions = torch.rand(bs, rand_action_horizon, self.random_action_dim,
                                        device=particle_projection.device)
            c_r = self.random_action_proj(random_actions)
            if self.action_as_particle:
                if len(c_r.shape) == 3:
                    c_r = c_r.unsqueeze(2)  # [bs, t, 1, f]
                    action_embeddings = self.random_action_embeddings.repeat(bs, timestep_horizon, 1, 1)
                    c_r = c_r + action_embeddings
                    particle_projection = torch.cat([particle_projection, c_r], dim=2)
            else:
                if len(c_r.shape) == 3:
                    # n_particles = particle_projection.shape[2] if not self.token_pool_adaln \
                    #     else particle_projection.shape[2] + 1
                    n_particles = particle_projection.shape[2]
                    c_r = c_r.unsqueeze(2).repeat(1, 1, n_particles, 1)  # [bs, t, n, f]
                if c is None:
                    c = c_r
                else:
                    c = c + c_r

        if self.action_condition and actions is not None:
            if self.learn_null_action_embed and actions_mask is not None:
                # action_mask: [batch_size, T] or [batch_size, T, 1], 1 where use action, 0 replace action
                # Expand mask
                if len(actions_mask.shape) == 2:
                    actions_mask = actions_mask.bool().unsqueeze(-1)  # (batch_size, seq_len, 1)
                # Expand null embedding to match
                null_action_embeds = self.null_action_embeddings.expand(actions.size(0), actions.size(1), -1)

                # Blend
                actions = actions * actions_mask + null_action_embeds * (~actions_mask)

            c_a = self.action_proj(actions)
            if self.action_as_particle:
                if len(c_a.shape) == 3:
                    c_a = c_a.unsqueeze(2)  # [bs, t, 1, f]
                    action_embeddings = self.action_embeddings.repeat(bs, timestep_horizon, 1, 1)
                    c_a = c_a + action_embeddings
                    particle_projection = torch.cat([particle_projection, c_a], dim=2)
            else:
                if len(c_a.shape) == 3:
                    # n_particles = particle_projection.shape[2] if not self.token_pool_adaln \
                    #     else particle_projection.shape[2] + 1
                    n_particles = particle_projection.shape[2]
                    c_a = c_a.unsqueeze(2).repeat(1, 1, n_particles, 1)  # [bs, t, n, f]
                if c is None:
                    c = c_a
                else:
                    c = c + c_a

        # views
        if self.n_views > 1:
            # [bs * n_views, t, n, d] -> [bs, n_views, t, n, d] -> [bs, t, n_views * n, d]
            particle_projection = particle_projection.view(-1, self.n_views, *particle_projection.shape[1:])
            # [bs, n_views, t, n, d]
            particle_projection = particle_projection.permute(0, 2, 1, 3, 4)  # [bs, t, n_views, n, d]
            # add view embeddings
            particle_projection = particle_projection + self.view_embeddings
            particle_projection = particle_projection.reshape(particle_projection.shape[0],
                                                              particle_projection.shape[1],
                                                              -1,
                                                              particle_projection.shape[-1])  # [bs, t, n_views * n, d]
            if c is not None:
                c = c.view(-1, self.n_views, *c.shape[1:])
                c = c.permute(0, 2, 1, 3, 4)  # [bs, t, n_views, n, d]
                c = c.reshape(c.shape[0], c.shape[1], -1, c.shape[-1])  # [bs, t, n_views * n, d]
                # if self.img_goal_condition and self.img_goal_condition_type == 'adaln' and self.token_pool_adaln:
                #     token_pool_goal = self.token_pool_embeddings_goal.repeat(c.shape[0], c.shape[1], 1, 1)
                #     c = torch.cat([c, token_pool_goal], dim=2)
        # else:
        #     if c is not None and self.img_goal_condition and self.img_goal_condition_type == 'adaln' and self.token_pool_adaln:
        #         token_pool_goal = self.token_pool_embeddings_goal.repeat(c.shape[0], c.shape[1], 1, 1)
        #         c = torch.cat([c, token_pool_goal], dim=2)

        if self.language_condition and lang_embed is not None and l is None:
            l = self.lang_proj(lang_embed)
            if len(l.shape) == 3:
                # [bs, h=N_l, f]
                l = l.unsqueeze(1).repeat(1, timestep_horizon, 1, 1)  # [bs, t, h=N_l, f]
            elif l.shape[1] != timestep_horizon:
                # [bs, 1, h=N_l, f]
                l = l.repeat(1, timestep_horizon, 1, 1)  # [bs, t, h=N_l, f]
            # clip max tokens
            l = l[:, :, :self.language_max_len].contiguous()
            # add positional embeddings
            if self.lang_pos_embed is not None:
                l = l + self.lang_pos_embed[:, :, :self.language_max_len]
            if self.language_condition_type == 'self':
                particle_projection = torch.cat([particle_projection, l[:particle_projection.shape[0]]], dim=2)
                # [bs, T, n_p + n_l, dim], if n_views > 1, we make sure the effective batch size is the same
                # note that we assume the language instruction is the same for all views here
                l = None
        # if self.token_pool_adaln:
        #     pool_tokens = self.token_pool_embeddings.repeat(particle_projection.shape[0],
        #                                                     particle_projection.shape[1], 1, 1)
        #     particle_projection = torch.cat([particle_projection, pool_tokens], dim=2)

        # if self.pos_embed_t_adaln:
        #     c_t = self.pos_embed_t_embeddings[:, :particle_projection.shape[1]]  # match the timesteps
        #     # c_t = self.learned_token_projection(c_t)
        #     c_t = c_t.repeat(particle_projection.shape[0], 1, particle_projection.shape[2], 1)
        #     if c is not None:
        #         c = c + c_t
        #     else:
        #         c = c_t

        if self.particle_pool_adaln:
            c_p = self.particle_pool_proj(particle_projection)  # [bs, T, 1, dim]
            c_p = c_p.repeat(1, 1, particle_projection.shape[2], 1)
            if c is not None:
                c = c + c_p
            else:
                c = c_p

        if self.pos_embed_p_adaln:
            c_pe = self.pos_p_embeddings[:, :, :particle_projection.shape[2]].repeat(particle_projection.shape[0],
                                                                                     particle_projection.shape[1],
                                                                                     1, 1)
            if c is not None:
                c = c + c_pe
            else:
                c = c_pe

        if self.pos_embed_objon_adaln:
            z_obj_on_proj = z_obj_on_v[:, :timestep_horizon]
            c_objon = self.objon_embeddings(z_obj_on_proj)  # [bs, t, n, dim]
            # add zeros for the bg particle
            c_objon_bg = torch.zeros([c_objon.shape[0], c_objon.shape[1], 1, c_objon.shape[-1]], device=c_objon.device)
            c_objon = torch.cat([c_objon, c_objon_bg], dim=2)  # [bs, t, n + 1, dim]
            if self.n_views > 1:
                c_objon = c_objon.view(-1, self.n_views, *c_objon.shape[1:])
                c_objon = c_objon.permute(0, 2, 1, 3, 4)  # [bs, t, n_views, n, d]
                c_objon = c_objon.reshape(c_objon.shape[0], c_objon.shape[1], -1,
                                          c_objon.shape[-1])  # [bs, t, n_views * n, d]
            total_particles = particle_projection.shape[2]
            c_objon_other = torch.zeros(c_objon.shape[0], c_objon.shape[1], total_particles - c_objon.shape[2],
                                        c_objon.shape[3], device=c_objon.device)
            c_objon = torch.cat([c_objon, c_objon_other], dim=2)
            if c is not None:
                c = c + c_objon
            else:
                c = c_objon

        # if self.img_goal_condition and l is None and z_goal is None:
        #     particle_projection, goal_projection = particle_projection.split([particle_projection.shape[1] - 1, 1],
        #                                                                      dim=1)
        #     # goal_projection: [bs, 1, N, d]
        #     z_goal = self.goal_proj(goal_projection)
        #     l = goal_projection.repeat(1, particle_projection.shape[1], 1, 1)
        #     timestep_horizon = timestep_horizon - 1
        #     if detach_z_goal:
        #         l = l.detach()
        # elif self.img_goal_condition and l is None and z_goal is not None:
        #     l = z_goal.repeat(1, particle_projection.shape[1], 1, 1)
        #     if detach_z_goal:
        #         l = l.detach()

        particles_out = self.pte(particle_projection, c=c, l=l)
        # particles_out = particles_out.view(bs, timestep_horizon, *particles_out.shape[2:])
        # particles_out = particles_out.view(bs, -1, *particles_out.shape[2:])
        # [bs, ts, n, f]
        # if self.token_pool_adaln:
        #     particles_out = particles_out[:, :, :-1].contiguous()
        if self.language_condition and self.language_condition_type == 'self':
            particles_out = particles_out[:, :, :-self.language_max_len].contiguous()
        if self.n_views > 1:
            # [bs, t, n_views * n, d] -> [bs, n_views, t, n, d] -> [bs * n_views, t, n, d]
            particles_out = particles_out.view(particles_out.shape[0], particles_out.shape[1],
                                               self.n_views, -1, particles_out.shape[-1])
            particles_out = particles_out.permute(0, 2, 1, 3, 4)  # [bs, n_views, t, n, d]
            particles_out = particles_out.reshape(-1, *particles_out.shape[2:])  # [bs * n_views, t, n, d]
        if self.action_condition and self.action_as_particle:
            particles_out = particles_out[:, :, :-1].contiguous()
        if self.random_action_condition and self.action_as_particle:
            particles_out = particles_out[:, :, :-1].contiguous()
        if self.img_goal_condition and self.img_goal_condition_type == 'self':
            n_goal_particles = self.n_kp_enc
            if self.with_bg:
                n_goal_particles += 1
            particles_out = particles_out[:, :, :-n_goal_particles].contiguous()

        if (self.learned_ctx_token or self.ctx_pool_mode == 'last') and self.token_pool_cross_attn:
            if self.learned_ctx_token:
                q_particles = self.ctx_token_embeddings.repeat(bs, timestep_horizon, 1, 1)
                q_particles = q_particles.view(bs * timestep_horizon, 1, *q_particles.shape[2:])
                # [bs * t, 1, 1, embed_dim]
                kv_particles = particles_out[:, :, :self.n_kp_enc + 1]  # only fg + bg particles
                kv_particles = kv_particles.reshape(bs * timestep_horizon, 1, *kv_particles.shape[2:])
                # [bs * t, 1, n_particles + 1, embed_dim]
            else:
                # 'last' pooling
                kv_particles, q_particles = particles_out.split([particles_out.shape[2] - 1, 1], dim=2)
            ctx_ca = self.cross_attn_block(q_particles, kv_particles)
            # [bs * t, 1, 1, embed_dim]
            particles_out = torch.cat([kv_particles, ctx_ca], dim=2)
            particles_out = particles_out.view(bs, timestep_horizon, *particles_out.shape[2:])

        if encode_posterior:
            if self.global_ctx_pool:
                # global
                particle_decoder_out_global = self.global_posterior_decoder(particles_out, deterministic=deterministic)
                # unpack
                mu_context_global = particle_decoder_out_global['mu_context']
                logvar_context_global = particle_decoder_out_global['logvar_context']
                z_context_global = particle_decoder_out_global['z_context']
                # local
                if self.condition_local_on_global:
                    c = z_context_global
                else:
                    c = None
                particle_decoder_out = self.posterior_decoder(particles_out, c=c, deterministic=deterministic)
                # unpack
                mu_context = particle_decoder_out['mu_context']
                logvar_context = particle_decoder_out['logvar_context']
                z_context = particle_decoder_out['z_context']

                if self.global_local_fuse_mode != 'none':
                    if len(z_context_global.shape) != len(z_context.shape):
                        z_context_global = z_context_global.unsqueeze(2).repeat(1, 1, z_context.shape[2], 1)
                    elif z_context_global.shape[2] != z_context.shape[2]:
                        z_context_global = z_context_global.repeat(1, 1, z_context.shape[2], 1)
                    if self.global_local_fuse_mode == 'concat':
                        z_context = torch.cat([z_context, z_context_global], dim=-1)
                    else:
                        # add
                        z_context = z_context + z_context_global
            else:
                mu_context_global = logvar_context_global = z_context_global = None
                particle_decoder_out = self.posterior_decoder(particles_out, deterministic=deterministic)
                # unpack
                mu_context = particle_decoder_out['mu_context']
                logvar_context = particle_decoder_out['logvar_context']
                z_context = particle_decoder_out['z_context']
        else:
            mu_context = logvar_context = z_context = None
            mu_context_global = logvar_context_global = z_context_global = None

        if encode_prior:
            if self.global_ctx_pool:
                # global
                prior_decoder_out_global = self.global_prior_decoder(particles_out, deterministic=deterministic)
                # unpack
                mu_context_global_dyn = prior_decoder_out_global['mu_context']
                logvar_context_global_dyn = prior_decoder_out_global['logvar_context']
                z_context_global_dyn = prior_decoder_out_global['z_context']

                # local
                if self.condition_local_on_global:
                    if z_context_global is None:
                        # sampling
                        c = z_context_global_dyn
                    else:
                        # teacher-forcing: shift inverse-model output by one timestep
                        c = torch.cat([z_context_global[:, 1:], z_context_global[:, -1:]], dim=1)
                else:
                    c = None
                prior_decoder_out = self.prior_decoder(particles_out, c=c, deterministic=deterministic)
                # unpack
                mu_context_dyn = prior_decoder_out['mu_context']
                logvar_context_dyn = prior_decoder_out['logvar_context']
                z_context_dyn = prior_decoder_out['z_context']

                if self.global_local_fuse_mode != 'none':
                    if len(z_context_global_dyn.shape) != len(z_context_dyn.shape):
                        z_context_global_dyn = z_context_global_dyn.unsqueeze(2).repeat(1, 1, z_context_dyn.shape[2], 1)
                    elif z_context_global_dyn.shape[2] != z_context_dyn.shape[2]:
                        z_context_global_dyn = z_context_global_dyn.repeat(1, 1, z_context_dyn.shape[2], 1)
                    if self.global_local_fuse_mode == 'concat':
                        z_context_dyn = torch.cat([z_context_dyn, z_context_global_dyn], dim=-1)
                    else:
                        # add
                        z_context_dyn = z_context_dyn + z_context_global_dyn
            else:
                mu_context_global_dyn = logvar_context_global_dyn = z_context_global_dyn = None
                prior_decoder_out = self.prior_decoder(particles_out, deterministic=deterministic)
                # unpack
                mu_context_dyn = prior_decoder_out['mu_context']
                logvar_context_dyn = prior_decoder_out['logvar_context']
                z_context_dyn = prior_decoder_out['z_context']
        else:
            mu_context_dyn = logvar_context_dyn = z_context_dyn = None
            mu_context_global_dyn = logvar_context_global_dyn = z_context_global_dyn = None

        encode_dict = {'mu_context': mu_context, 'logvar_context': logvar_context, 'z_context': z_context,
                       'mu_context_dyn': mu_context_dyn, 'logvar_context_dyn': logvar_context_dyn,
                       'z_context_dyn': z_context_dyn,
                       'mu_context_global': mu_context_global, 'logvar_context_global': logvar_context_global,
                       'z_context_global': z_context_global,
                       'mu_context_global_dyn': mu_context_global_dyn,
                       'logvar_context_global_dyn': logvar_context_global_dyn,
                       'z_context_global_dyn': z_context_global_dyn,
                       'z_goal_proj': z_goal,
                       }
        return encode_dict

    def forward(self, z, z_scale, z_obj_on, z_depth, z_features, z_bg_features=None, z_base_var=None,
                z_score=None, patch_id_embed=None, deterministic=False, warmup=False,
                encode_posterior=True, encode_prior=True, actions=None, actions_mask=None, lang_embed=None,
                z_goal=None):
        output_dict = self.encode_all(z, z_scale, z_obj_on, z_depth, z_features, z_bg_features, z_base_var, z_score,
                                      patch_id_embed, deterministic=deterministic, warmup=warmup,
                                      encode_posterior=encode_posterior, encode_prior=encode_prior, actions=actions,
                                      actions_mask=actions_mask, lang_embed=lang_embed, z_goal=z_goal)
        return output_dict


class DLPDynamics(nn.Module):
    def __init__(self,
                 features_dim,
                 bg_features_dim,
                 hidden_dim,
                 projection_dim,
                 n_head=8,  # Number of attention heads
                 n_layer=2,  # Number of attention layers
                 block_size=12,  # Timestep horizon
                 dropout=0.1,
                 kp_activation='tanh',  # Keypoint activation function
                 predict_delta=False,  # Predict position deltas instead of absolute positions
                 max_delta=1.5,  # Maximum delta value for predictions
                 positional_bias=False,  # Use positional bias in dynamics
                 max_particles=None,  # Maximum particles for positional bias
                 context_dim=7,  # Context latent dimension
                 attn_norm_type='rms',  # Normalization type for attention
                 n_fg_particles=None,  # Number of foreground particles
                 ctx_pool_mode='none',  # Context pooling mode
                 ctx_mode='adaln',  # Conditioning type for latent context
                 particle_score=False,  # Include particle confidence scores
                 particle_positional_embed=True,  # Use positional embeddings for particles
                 scale_anchor=None,  # Anchor scale for particle dynamics
                 init_std=0.02,  # Standard deviation for initialization
                 pint_ctx_layers=6,  # Number of PINT context transformer layers
                 pint_ctx_heads=8,  # Number of PINT context transformer heads
                 ctx_dist='gauss',  # Context distribution type ('gauss' or 'categorical')
                 n_ctx_categories=4,  # Number of context categories
                 n_ctx_classes=4,  # Number of context classes per category
                 residual_modulation=True,  # Use residual modulation for dynamics
                 context_gate=True,  # Use gating for context features
                 context_decoder=None,  # Decoder configuration for context
                 features_dist='gauss',  # Distribution type for features
                 n_fg_categories=8,  # Number of foreground feature categories
                 n_fg_classes=4,  # Number of foreground feature classes per category
                 n_bg_categories=4,  # Number of background feature categories
                 n_bg_classes=4,  # Number of background feature classes per category
                 particle_anchors=None,  # Anchors for particles
                 scale_init=None,  # Initial scale for particles
                 obj_on_min=1e-4,  # Minimum transparency concentration value
                 obj_on_max=100,  # Maximum transparency concentration value
                 use_z_orig=True,  # Include original patch coordinates in features
                 n_views=1,  # number of input views (e.g., multiple cameras)
                 # external conditioning
                 action_condition=False,  # condition on actions
                 action_dim=0,  # dimension of input actions
                 random_action_condition=False,  # condition on random actions
                 random_action_dim=0,  # dimension of sampled random actions
                 null_action_embed=False,  # learn a "no-input-action" embedding, to learn on action-free videos as well
                 pos_embed_t_adaln=True,  # pos embeddings for timesteps using adaln
                 pos_embed_p_adaln=True,  # pos embeddings for particles using adaln
                 pos_embed_objon_adaln=False,  # pos embeddings for particles transparency using adaln
                 # language_condition=False,  # condition on language embedding
                 # language_embed_dim=0,  # embedding dimension for each token
                 # language_max_len=64,  # maximum tokens per prompt
                 ):
        super(DLPDynamics, self).__init__()
        """
        Args:
        features_dim (int): Dimension of visual features.
        bg_features_dim (int): Dimension of background features.
        hidden_dim (int): Hidden dimension for dynamics layers.
        projection_dim (int): Projection dimension for dynamics.
        n_head (int): Number of attention heads. Defaults to 8.
        n_layer (int): Number of attention layers. Defaults to 2.
        block_size (int): Timestep horizon for dynamics. Defaults to 12.
        dropout (float): Dropout rate for transformer layers. Defaults to 0.1.
        kp_activation (str): Activation function for keypoints ('tanh' or 'relu'). Defaults to 'tanh'.
        predict_delta (bool): Predict position deltas instead of absolute positions. Defaults to False.
        max_delta (float): Maximum value for delta predictions. Defaults to 1.5.
        positional_bias (bool): Use positional bias in dynamics computations. Defaults to False.
        max_particles (Optional[int]): Maximum number of particles for positional bias. Defaults to None.
        context_dim (int): Dimension of the latent context. Defaults to 7.
        attn_norm_type (str): Normalization type for attention ('rms' or 'layer'). Defaults to 'rms'.
        n_fg_particles (Optional[int]): Number of foreground particles. Defaults to None.
        ctx_pool_mode (str): Pooling mode for context ('none', 'mean', etc.). Defaults to 'none'.
        ctx_mode (str): Conditioning mode for latent context ('adaln', etc.). Defaults to 'adaln'.
        particle_score (bool): Include particle confidence scores as features. Defaults to False.
        particle_positional_embed (bool): Use positional embeddings for particles. Defaults to True.
        scale_anchor (Optional[float]): Anchor scale for dynamics. Defaults to None.
        init_std (float): Standard deviation for parameter initialization. Defaults to 0.02.
        pint_ctx_layers (int): Number of PINT context transformer layers. Defaults to 6.
        pint_ctx_heads (int): Number of PINT context transformer heads. Defaults to 8.
        ctx_dist (str): Distribution type for context ('gauss' or 'categorical'). Defaults to 'gauss'.
        n_ctx_categories (int): Number of context categories if categorical. Defaults to 4.
        n_ctx_classes (int): Number of context classes per category. Defaults to 4.
        residual_modulation (bool): Apply residual modulation to dynamics features. Defaults to True.
        context_gate (bool): Use gating mechanisms for context integration. Defaults to True.
        context_decoder (Optional[str]): Configuration of context decoder. Defaults to None.
        features_dist (str): Distribution type for features ('gauss' or 'categorical'). Defaults to 'gauss'.
        n_fg_categories (int): Number of foreground feature categories. Defaults to 8.
        n_fg_classes (int): Number of foreground feature classes per category. Defaults to 4.
        n_bg_categories (int): Number of background feature categories. Defaults to 4.
        n_bg_classes (int): Number of background feature classes per category. Defaults to 4.
        particle_anchors (Optional[Any]): Anchors for particle initialization. Defaults to None.
        scale_init (Optional[float]): Initial scale value for particles. Defaults to None.
        obj_on_min (float): Minimum concentration for Beta distribution in transparency. Defaults to 1e-4.
        obj_on_max (float): Maximum concentration for Beta distribution in transparency. Defaults to 100.
        use_z_orig (bool): Include original patch coordinates in particle features. Defaults to True.

        DLP Dynamics with Context:
        This module predicts particle dynamics across timesteps using a PINT-based transformer. Each particle's attributes,
        such as position, scale, and transparency, evolve over time, guided by latent context variables.

        """

        self.predict_delta = predict_delta
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.max_delta = max_delta
        self.max_particles = max_particles  # for positional bias
        self.n_fg_particles = n_fg_particles
        self.learned_feature_dim = features_dim
        self.learned_bg_feature_dim = bg_features_dim
        self.features_dist = features_dist
        self.n_fg_categories = n_fg_categories
        self.n_fg_classes = n_fg_classes
        self.n_bg_categories = n_bg_categories
        self.n_bg_classes = n_bg_classes
        self.context_dist = ctx_dist
        self.n_ctx_categories = n_ctx_categories
        self.n_ctx_classes = n_ctx_classes
        self.context_dim = context_dim
        self.particle_score = particle_score
        self.attn_norm_type = attn_norm_type
        assert ctx_mode in ['add', 'cat', 'token', 'film', 'adaln']
        self.ctx_mode = ctx_mode
        self.ctx_pool_mode = ctx_pool_mode
        # ['last'-last token is ctx, otherwise, use pool op over the particles to generate context]
        self.init_std = init_std
        self.obj_on_min = obj_on_min
        self.obj_on_max = obj_on_max
        self.use_z_orig = use_z_orig  # use the origin of the particles (the center of the source patch) as attribute
        self.n_views = n_views  # number of input views (e.g., multiple cameras)
        use_norm_layer = True  # norm layer in the projections modules

        # actions
        self.action_condition = action_condition
        self.action_dim = action_dim
        self.random_action_condition = random_action_condition
        self.random_action_dim = random_action_dim
        self.learn_null_action_embed = null_action_embed
        # language
        # self.language_condition = language_condition
        # self.language_embed_dim = language_embed_dim
        # self.language_max_len = language_max_len

        # token adaln
        self.pos_embed_t_adaln = pos_embed_t_adaln
        self.pos_embed_p_adaln = pos_embed_p_adaln
        self.pos_embed_objon_adaln = pos_embed_objon_adaln

        if self.learn_null_action_embed and self.action_condition:
            self.null_action_embeddings = nn.Parameter(
                self.init_std * torch.randn(1, 1, self.action_dim))
        else:
            self.null_action_embeddings = None

        if scale_anchor is None:
            self.register_buffer('scale_anchor', torch.tensor(0.0))
        else:
            self.register_buffer('scale_anchor',
                                 torch.tensor(np.log(0.75 * scale_anchor / (1 - 0.75 * scale_anchor + 1e-5))))
        if particle_anchors is None:
            self.register_buffer('particles_anchor', torch.zeros(1, 1, self.n_fg_particles))
            self.use_z_orig = False
        else:
            self.register_buffer('particles_anchor', particle_anchors)

        self.particle_pos_embed = particle_positional_embed and not self.pos_embed_p_adaln

        proj_max_particles = self.n_fg_particles
        self.particle_projection = ParticleFeatureProjection(features_dim, bg_features_dim,
                                                             hidden_dim, self.projection_dim, context_dim=context_dim,
                                                             max_particles=proj_max_particles, add_embedding=True,
                                                             ctx_cond_mode=self.ctx_mode,
                                                             particle_positional_embed=self.particle_pos_embed,
                                                             init_std=self.init_std, particle_score=self.particle_score,
                                                             norm_layer=use_norm_layer,
                                                             use_z_orig=self.use_z_orig)
        if self.ctx_mode == 'adaln' and self.context_dim > 0:
            self.context_proj = nn.Linear(self.context_dim, hidden_dim)
            # self.context_proj = nn.Sequential(nn.Linear(self.context_dim, hidden_dim),
            #                                   RMSNorm(hidden_dim),
            #                                   nn.GELU())
            if self.action_condition and self.action_dim > 0:
                self.action_proj = nn.Linear(self.action_dim, hidden_dim)
            else:
                self.action_proj = None
            if self.random_action_condition and self.random_action_dim > 0:
                self.random_action_proj = nn.Linear(self.random_action_dim, hidden_dim)
            else:
                self.random_action_proj = None
            self.cond_activation = nn.GELU()
        else:
            self.context_proj = None
            self.action_proj = None
            self.cond_activation = None

        if self.n_views > 1:
            self.view_embeddings = nn.Parameter(self.init_std * torch.randn(1, self.n_views, 1, 1, self.projection_dim))
        else:
            self.view_embeddings = None

        if self.pos_embed_p_adaln and (self.ctx_mode == 'adaln'):
            n_particles = self.n_views * (self.n_fg_particles + 1)
            self.pos_p_embeddings = nn.Parameter(
                self.init_std * torch.randn(1, n_particles, 1, hidden_dim))
        if self.pos_embed_objon_adaln:
            self.objon_embeddings = nn.Sequential(nn.Linear(1, hidden_dim),
                                                  RMSNorm(hidden_dim),
                                                  nn.GELU())

        self.particle_transformer = ParticleSpatioTemporalTransformer(self.projection_dim, n_head, n_layer,
                                                                      block_size, self.projection_dim,
                                                                      attn_pdrop=dropout, resid_pdrop=dropout,
                                                                      hidden_dim_multiplier=4,
                                                                      positional_bias=positional_bias,
                                                                      activation='gelu',
                                                                      max_particles=max_particles,
                                                                      norm_type=attn_norm_type,
                                                                      init_std=self.init_std, causal=True,
                                                                      context_cond=(self.ctx_mode == 'adaln'),
                                                                      residual_modulation=residual_modulation,
                                                                      context_gate=context_gate,
                                                                      pos_embed_t_adaln=self.pos_embed_t_adaln)

        self.particle_decoder = ParticleFeatureDecoderDyn(self.projection_dim, features_dim, bg_features_dim,
                                                          hidden_dim, kp_activation=kp_activation, max_delta=max_delta,
                                                          context_dim=context_dim,
                                                          ctx_as_token=(self.ctx_mode == 'token'),
                                                          dec_ctx=False, norm_type=attn_norm_type, dropout=dropout,
                                                          particle_score=self.particle_score,
                                                          features_dist=self.features_dist,
                                                          n_fg_categories=n_fg_categories,
                                                          n_fg_classes=n_fg_classes, n_bg_categories=n_bg_categories,
                                                          n_bg_classes=n_bg_classes, scale_init=scale_init)
        self.context_decoder = context_decoder

    def init_weights(self):
        self.particle_projection.init_weights()
        self.particle_transformer.init_weights()
        self.particle_decoder.init_weights()

    def sample(self, z, z_scale, z_obj_on, z_depth, z_features, z_bg_features, z_context=None,
               z_score=None, steps=10, deterministic=False, deterministic_particles=True, actions=None,
               actions_mask=None, lang_embed=None, z_goal=None, return_context_posterior=False):
        """
        Samples a sequence of particle states based on the given conditioning inputs and internal model dynamics.

        Args:
            z (torch.Tensor): Initial particle positions, shape `(batch_size, timesteps, n_particles, 2)`.
            z_scale (torch.Tensor): Scale of particles, shape `(batch_size, timesteps, n_particles, 2)`.
            z_obj_on (torch.Tensor): transparency probabilities, shape `(batch_size, timesteps, n_particles, 1)`.
            z_depth (torch.Tensor): Depth of particles, shape `(batch_size, timesteps, n_particles, 1)`.
            z_features (torch.Tensor): Particle features, shape `(batch_size, timesteps, n_particles, in_features_dim)`.
            z_bg_features (torch.Tensor): Background features, shape `(batch_size, timesteps, bg_features_dim)`.
            z_context (torch.Tensor, optional): Dynamic context encoding, shape `(batch_size, timesteps, context_dim)`.
            z_score (torch.Tensor, optional): Particle scores, shape `(batch_size, timesteps, n_particles, 1)`.
                If not provided, it defaults to zeros.
            steps (int): Number of forward sampling steps. Defaults to 10.
            deterministic (bool): If True, the sampling is deterministic. Defaults to False.
            deterministic_particles (bool): If True, uses deterministic particles during sampling. Defaults to True.

        Returns:
            dict: A dictionary containing sampled outputs:
                - `z` (torch.Tensor): Updated particle positions.
                - `z_scale` (torch.Tensor): Updated particle scales.
                - `z_obj_on` (torch.Tensor): Updated transparency probabilities.
                - `z_depth` (torch.Tensor): Updated particle depths.
                - `z_features` (torch.Tensor): Updated particle features.
                - `z_bg_features` (torch.Tensor): Updated background features.
                - `z_context` (torch.Tensor): Generated or updated context.
                - `z_score` (torch.Tensor): Updated particle scores.

        Notes:
            - The function iteratively generates future particle states using a transformer-based architecture.
            - Reparameterization techniques are employed for stochastic sampling when `deterministic=False`.
            - Quadratic complexity is involved in the sampling process due to the block size and transformer operations.
        """
        block_size = self.particle_transformer.get_block_size()
        # z, z_scale: [bs, T, n_particles, 2]
        # z_depth, z_obj_on: [bs, T, n_particles, 1]
        # z_features: [bs, T, n_particles, in_features_dim]
        # z_bg_features: [bs, T, bg_features_dim]
        # z_context: [bs, T, context_dim]
        if z_score is None:
            z_score = torch.zeros(z.shape[0], z.shape[1], z.shape[2], 1, dtype=torch.float, device=z.device)

        mu_context_posterior = z_context_posterior = z_context  # initialize in case they are needed
        bs, timestep_horizon, n_particles, _ = z.shape
        for k in range(steps):
            # first generate context, then use the context with the current particles
            if self.context_dim > 0:
                start_step = max(z.shape[1] - block_size, 0)
                end_step = min(start_step + block_size, z.shape[1])
                # check if context was provided
                if z_context is None or z_context.shape[1] < z.shape[1]:
                    # generate context
                    if actions is not None:
                        actions_in = actions[:, start_step:end_step]
                    else:
                        actions_in = None
                    if actions_mask is not None:
                        actions_mask_in = actions_mask[:, start_step:end_step]
                    else:
                        actions_mask_in = None
                    ctx_dec_out = self.context_decoder(z=z[:, -block_size:],
                                                       z_scale=z_scale[:, -block_size:],
                                                       z_obj_on=z_obj_on[:, -block_size:],
                                                       z_depth=z_depth[:, -block_size:],
                                                       z_features=z_features[:, -block_size:],
                                                       z_bg_features=z_bg_features[:, -block_size:],
                                                       z_score=z_score[:, -block_size:],
                                                       deterministic=deterministic,
                                                       encode_posterior=return_context_posterior,
                                                       encode_prior=True,
                                                       actions=actions_in,
                                                       actions_mask=actions_mask_in,
                                                       lang_embed=lang_embed,
                                                       z_goal=z_goal)
                    z_context_last = ctx_dec_out['z_context_dyn'][:, -1:]

                    new_z_context = z_context_last
                    if z_context is None:
                        # that means that it the very first step
                        z_context = new_z_context
                    else:
                        z_context = torch.cat([z_context, new_z_context], dim=1)
                        if return_context_posterior:
                            mu_context_posterior_last = ctx_dec_out['mu_context']
                            z_context_posterior_last = ctx_dec_out['z_context']
                            if z_context_posterior_last.shape[1] > 1:
                                new_mu_context_posterior = mu_context_posterior_last[:, -1:]
                                new_z_context_posterior = z_context_posterior_last[:, -1:]
                                if z_context_posterior is None:
                                    mu_context_posterior = new_mu_context_posterior
                                    z_context_posterior = new_z_context_posterior
                                else:
                                    mu_context_posterior = torch.cat([mu_context_posterior,
                                                                      new_mu_context_posterior], dim=1)
                                    z_context_posterior = torch.cat([z_context_posterior,
                                                                     new_z_context_posterior], dim=1)

                # prepare input to dyn module
                # start_step = max(z.shape[1] - block_size, 0)
                # end_step = min(start_step + block_size, z.shape[1])
                z_context_v = z_context[:, start_step:end_step].reshape(-1, *z_context.shape[2:])
            else:
                z_context_v = None

            # project particles
            z_v = z[:, -block_size:].reshape(-1, *z.shape[2:])
            z_scale_v = z_scale[:, -block_size:].reshape(-1, *z_scale.shape[2:])
            z_obj_on_v = z_obj_on[:, -block_size:].reshape(-1, *z_obj_on.shape[2:])
            z_depth_v = z_depth[:, -block_size:].reshape(-1, *z_depth.shape[2:])
            z_features_v = z_features[:, -block_size:].reshape(-1, *z_features.shape[2:])
            z_bg_features_v = z_bg_features[:, -block_size:].reshape(-1, *z_bg_features.shape[2:])
            z_score_v = z_score[:, -block_size:].reshape(-1, *z_score.shape[2:])
            if self.use_z_orig:
                z_orig_v = self.particles_anchor.repeat(z_v.shape[0], 1, 1)
            else:
                z_orig_v = None

            particle_projection = self.particle_projection(z_v, z_scale_v, z_obj_on_v, z_depth_v, z_features_v,
                                                           z_bg_features_v, z_context_v, z_score_v, z_orig_v)
            # [bs * T, n_particles + 1, projection_dim]
            particle_proj_int = particle_projection
            # unroll forward
            particle_proj_int = particle_proj_int.view(bs, -1, *particle_proj_int.shape[1:])
            # [bs, T, n_particles + 2, projection_dim]
            particle_proj_int = particle_proj_int.permute(0, 2, 1, 3)
            # [bs, n_particles + 2, T, projection_dim]
            if self.ctx_mode == 'adaln':
                if self.random_action_condition:
                    random_actions = torch.rand(particle_proj_int.shape[0], particle_proj_int.shape[2],
                                                self.random_action_dim, device=particle_proj_int.device)
                    c_random_action = self.random_action_proj(random_actions)
                    if len(c_random_action.shape) == 3:
                        c_random_action = c_random_action.unsqueeze(1).repeat(1, particle_proj_int.shape[1], 1,
                                                                              1)  # [bs, n, t, f]
                else:
                    c_random_action = 0

                if self.action_condition and actions is not None:
                    start_step = max(z.shape[1] - block_size, 0)
                    end_step = min(start_step + block_size, z.shape[1])
                    actions_v = actions[:, start_step:end_step]
                    if self.learn_null_action_embed and actions_mask is not None:
                        # action_mask: [batch_size, T] or [batch_size, T, 1], 1 where use action, 0 replace action
                        # Expand mask
                        if len(actions_mask.shape) == 2:
                            actions_mask_v = actions_mask[:, start_step:end_step].bool().unsqueeze(
                                -1)  # (batch_size, seq_len, 1)
                        # Expand null embedding to match
                        null_action_embeds = self.null_action_embeddings.expand(actions_v.size(0), actions_v.size(1),
                                                                                -1)

                        # Blend
                        actions_v = actions_v * actions_mask_v + null_action_embeds * (~actions_mask_v)

                    c_action = self.action_proj(actions_v)
                    if len(c_action.shape) == 3:
                        c_action = c_action.unsqueeze(1).repeat(1, particle_proj_int.shape[1], 1, 1)  # [bs, n, t, f]
                else:
                    c_action = 0
                c = self.context_proj(z_context_v)
                c = c.reshape(bs, -1, *c.shape[1:])
                if len(c.shape) == 3:
                    c = c.unsqueeze(1)  # [bs, 1, t, f]
                elif c.shape[2] != particle_proj_int.shape[1]:
                    c = c.permute(0, 2, 1, 3)  # [bs, n + 1, t, f]
                    c = c.repeat(1, particle_proj_int.shape[1], 1, 1)  # [bs, 1, t, f]
                else:
                    c = c.permute(0, 2, 1, 3)  # [bs, n + 1, t, f]
                c = c + c_action + c_random_action
                c = self.cond_activation(c)
            else:
                c = None

            if self.n_views > 1:
                # [bs * n_views, n, T, d] -> [bs, n_views, n, T, d] -> [bs, n_views * n, T, d]
                particle_proj_int = particle_proj_int.view(-1, self.n_views, particle_proj_int.shape[1],
                                                           *particle_proj_int.shape[2:])  # [bs, n_views, n, T, d]
                particle_proj_int = particle_proj_int + self.view_embeddings
                particle_proj_int = particle_proj_int.reshape(particle_proj_int.shape[0], -1,
                                                              *particle_proj_int.shape[3:])  # [bs, n_views * n, T, d]
                if c is not None:
                    c = c.reshape(-1, self.n_views * c.shape[1], *c.shape[2:])

            if c is not None and self.pos_embed_p_adaln:
                c_pe = self.pos_p_embeddings.repeat(c.shape[0], 1, c.shape[2], 1)
                c = c + c_pe

            if c is not None and self.pos_embed_objon_adaln:
                c_objon = self.objon_embeddings(z_obj_on[:, -block_size:])  # [bs, t, n, dim]
                c_objon_bg = torch.zeros(c_objon.shape[0], c_objon.shape[1], 1, c_objon.shape[-1],
                                         device=c_objon.device)
                c_objon = torch.cat([c_objon, c_objon_bg], dim=2)  # [bs, t, n + 1, dim]
                c_objon = c_objon.permute(0, 2, 1, 3)  # [bs, n + 1, t, dim]
                if self.n_views > 1:
                    c_objon = c_objon.reshape(-1, self.n_views * c_objon.shape[1], c_objon.shape[2], c_objon.shape[-1])
                    # [bs, n_views * (n + 1), t, dim]
                c = c + c_objon

            particles_trans = self.particle_transformer(particle_proj_int, c)
            if self.n_views > 1:
                # [bs, n_views * n, T, d] -> [bs * n_views, n, T, d]
                particles_trans = particles_trans.reshape(bs, -1, *particles_trans.shape[2:])
            particles_trans = particles_trans[:, :, -1]  # [bs, (n_particles + 1), projection_dim]
            # [bs, n_particles + 1, projection_dim]
            # decode transformer output
            # [bs, n_particles + 1, projection_dim]
            particle_decoder_out = self.particle_decoder(particles_trans)
            mu = particle_decoder_out['mu_offset']
            logvar = particle_decoder_out['logvar_offset']

            obj_on_a_gate = (particle_decoder_out['lobj_on_a']).sigmoid()
            obj_on_a = ((1 - obj_on_a_gate) * self.obj_on_min + obj_on_a_gate * self.obj_on_max).exp()
            obj_on_b_gate = 1 - (
                    particle_decoder_out['lobj_on_b'] * 0 + particle_decoder_out['lobj_on_a']).sigmoid()
            obj_on_b = ((1 - obj_on_b_gate) * self.obj_on_min + obj_on_b_gate * self.obj_on_max).exp()

            mu_depth = particle_decoder_out['mu_depth']
            logvar_depth = particle_decoder_out['logvar_depth']
            mu_scale = particle_decoder_out['mu_scale']
            logvar_scale = particle_decoder_out['logvar_scale']
            mu_features = particle_decoder_out['mu_features']
            logvar_features = particle_decoder_out['logvar_features']
            mu_bg_features = particle_decoder_out['mu_bg_features']
            logvar_bg_features = particle_decoder_out['logvar_bg_features']
            mu_score = particle_decoder_out['mu_score']
            logvar_score = particle_decoder_out['logvar_score']

            # reshape to [bs, t, ...]
            mu = mu.view(bs, 1, *mu.shape[1:])
            logvar = logvar.view(bs, 1, *logvar.shape[1:])
            obj_on_a = obj_on_a.view(bs, 1, *obj_on_a.shape[1:])
            obj_on_b = obj_on_b.view(bs, 1, *obj_on_b.shape[1:])
            mu_depth = mu_depth.view(bs, 1, *mu_depth.shape[1:])
            logvar_depth = logvar_depth.view(bs, 1, *logvar_depth.shape[1:])
            mu_scale = mu_scale.view(bs, 1, *mu_scale.shape[1:])
            logvar_scale = logvar_scale.view(bs, 1, *logvar_scale.shape[1:])
            mu_features = mu_features.view(bs, 1, *mu_features.shape[1:])
            logvar_features = logvar_features.view(bs, 1, *logvar_features.shape[1:])
            mu_bg_features = mu_bg_features.view(bs, 1, *mu_bg_features.shape[1:])
            logvar_bg_features = logvar_bg_features.view(bs, 1, *logvar_bg_features.shape[1:])
            if self.particle_score and mu_score is not None:
                mu_score = mu_score.view(bs, 1, *mu_score.shape[1:])
                logvar_score = logvar_score.view(bs, 1, *logvar_score.shape[1:])

            mu_scale = mu_scale + self.scale_anchor
            if self.use_z_orig:
                mu = self.particles_anchor.unsqueeze(1) + mu

            if self.predict_delta:
                mu = z[:, -1].unsqueeze(1) + mu
                # mu_scale = z_scale[:, -1].unsqueeze(1) + mu_scale
                # mu_depth = z_depth[:, -1].unsqueeze(1) + mu_depth
                # mu_features = z_features[:, -1].unsqueeze(1) + mu_features
                # mu_bg_features = z_bg_features[:, -1].unsqueeze(1) + mu_bg_features

            beta_dist = Beta(obj_on_a, obj_on_b)

            if deterministic:
                new_z = mu
                new_z_depth = mu_depth
                new_z_scale = mu_scale
                if self.features_dist == 'categorical':
                    logits = mu_features.view(*mu_features.shape[:-1], self.n_fg_categories, self.n_fg_classes)
                    # [bs, T, n_p, n_categories, n_classes]
                    probs = logits.softmax(dim=-1)  # [bs, T, n_p, n_categories, n_classes]
                    samples = torch.argmax(probs.view(-1, probs.shape[-1]), dim=-1, keepdim=True)
                    samples = F.one_hot(samples.squeeze(-1), num_classes=self.n_fg_classes)
                    samples = samples.view(probs.shape)
                    # straight-through
                    new_z_features = samples.detach() + (probs - probs.detach())
                    new_z_features = new_z_features.view(*mu_features.shape)  # [bs, T, n_p, n_categories * n_classes]

                    logits_bg = mu_bg_features.view(*mu_bg_features.shape[:-1], self.n_bg_categories, self.n_bg_classes)
                    # [bs, T, n_p, n_categories, n_classes]
                    probs_bg = logits_bg.softmax(dim=-1)  # [bs, T, n_p, n_categories, n_classes]
                    samples_bg = torch.argmax(probs_bg.view(-1, probs_bg.shape[-1]), dim=-1, keepdim=True)
                    samples_bg = F.one_hot(samples_bg.squeeze(-1), num_classes=self.n_bg_classes)
                    samples_bg = samples_bg.view(probs_bg.shape)
                    # straight-through
                    new_z_bg_features = samples_bg.detach() + (probs_bg - probs_bg.detach())
                    new_z_bg_features = new_z_bg_features.view(*mu_bg_features.shape)
                    # [bs, T, n_p, n_categories * n_classes]
                else:
                    new_z_features = mu_features
                    new_z_bg_features = mu_bg_features
                new_z_obj_on = beta_dist.mean
                if self.particle_score and mu_score is not None:
                    new_z_score = mu_score
                else:
                    new_z_score = logvar.sum(-1, keepdim=True)
            else:
                if deterministic_particles:
                    new_z = mu
                    new_z_depth = mu_depth
                    new_z_scale = mu_scale
                    if self.features_dist == 'categorical':
                        logits = mu_features.view(*mu_features.shape[:-1], self.n_fg_categories, self.n_fg_classes)
                        # [bs, T, n_p, n_categories, n_classes]
                        probs = logits.softmax(dim=-1)  # [bs, T, n_p, n_categories, n_classes]
                        samples = torch.argmax(probs.view(-1, probs.shape[-1]), dim=-1, keepdim=True)
                        samples = F.one_hot(samples.squeeze(-1), num_classes=self.n_fg_classes)
                        samples = samples.view(probs.shape)
                        # straight-through
                        new_z_features = samples.detach() + (probs - probs.detach())
                        new_z_features = new_z_features.view(
                            *mu_features.shape)  # [bs, T, n_p, n_categories * n_classes]

                        logits_bg = mu_bg_features.view(*mu_bg_features.shape[:-1], self.n_bg_categories,
                                                        self.n_bg_classes)
                        # [bs, T, n_p, n_categories, n_classes]
                        probs_bg = logits_bg.softmax(dim=-1)  # [bs, T, n_p, n_categories, n_classes]
                        samples_bg = torch.argmax(probs_bg.view(-1, probs_bg.shape[-1]), dim=-1, keepdim=True)
                        samples_bg = F.one_hot(samples_bg.squeeze(-1), num_classes=self.n_bg_classes)
                        samples_bg = samples_bg.view(probs_bg.shape)
                        # straight-through
                        new_z_bg_features = samples_bg.detach() + (probs_bg - probs_bg.detach())
                        new_z_bg_features = new_z_bg_features.view(*mu_bg_features.shape)
                        # [bs, T, n_p, n_categories * n_classes]
                    else:
                        new_z_features = mu_features
                        new_z_bg_features = mu_bg_features
                    new_z_obj_on = beta_dist.mean
                    if self.particle_score and mu_score is not None:
                        new_z_score = mu_score
                    else:
                        new_z_score = logvar.sum(-1, keepdim=True)
                else:
                    new_z = reparameterize(mu, logvar)
                    new_z_depth = reparameterize(mu_depth, logvar_depth)
                    new_z_scale = reparameterize(mu_scale, logvar_scale)
                    if self.features_dist == 'categorical':
                        logits = mu_features.view(*mu_features.shape[:-1], self.n_fg_categories, self.n_fg_classes)
                        # [bs, T, n_p, n_categories, n_classes]
                        probs = logits.softmax(dim=-1)  # [bs, T, n_p, n_categories, n_classes]
                        samples = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1)
                        samples = F.one_hot(samples.squeeze(-1), num_classes=self.n_fg_classes)
                        samples = samples.view(probs.shape)
                        # straight-through
                        new_z_features = samples.detach() + (probs - probs.detach())
                        new_z_features = new_z_features.view(*mu_features.shape)
                        # [bs, T, n_p, n_categories * n_classes]

                        logits_bg = mu_bg_features.view(*mu_bg_features.shape[:-1],
                                                        self.n_bg_categories, self.n_bg_classes)
                        # [bs, T, n_p, n_categories, n_classes]
                        probs_bg = logits_bg.softmax(dim=-1)  # [bs, T, n_p, n_categories, n_classes]
                        samples_bg = torch.multinomial(probs_bg.view(-1, probs_bg.shape[-1]), num_samples=1)
                        samples_bg = F.one_hot(samples_bg.squeeze(-1), num_classes=self.n_bg_classes)
                        samples_bg = samples_bg.view(probs.shape)
                        # straight-through
                        new_z_bg_features = samples_bg.detach() + (probs_bg - probs_bg.detach())
                        new_z_bg_features = new_z_bg_features.view(*mu_bg_features.shape)
                        # [bs, T, n_p, n_categories * n_classes]
                    else:
                        new_z_features = reparameterize(mu_features, logvar_features)
                        new_z_bg_features = reparameterize(mu_bg_features, logvar_bg_features)
                    new_z_obj_on = beta_dist.sample()
                    if self.particle_score and mu_score is not None:
                        new_z_score = reparameterize(mu_score, logvar_score)
                    else:
                        new_z_score = logvar.sum(-1, keepdim=True)

            z = torch.cat([z, new_z], dim=1)
            z_depth = torch.cat([z_depth, new_z_depth], dim=1)
            z_scale = torch.cat([z_scale, new_z_scale], dim=1)
            z_features = torch.cat([z_features, new_z_features], dim=1)
            z_bg_features = torch.cat([z_bg_features, new_z_bg_features], dim=1)
            z_obj_on = torch.cat([z_obj_on, new_z_obj_on], dim=1)
            z_score = torch.cat([z_score, new_z_score], dim=1)

        out_dict = {'z': z, 'z_scale': z_scale, 'z_obj_on': z_obj_on, 'z_depth': z_depth,
                    'z_features': z_features, 'z_bg_features': z_bg_features, 'z_context': z_context,
                    'z_score': z_score,
                    'z_context_posterior': z_context_posterior, 'mu_context_posterior': mu_context_posterior}
        return out_dict

    def forward(self, z, z_scale, z_obj_on, z_depth, z_features, z_bg_features, z_context, z_score=None, actions=None,
                actions_mask=None):
        # forward dynamics
        # z, z_scale: [bs, T, n_particles, 2]
        # z_depth, z_obj_on: [bs, T, n_particles, 1]
        # z_features: [bs, T, n_particles, in_features_dim]
        # z_bg_features: [bs, T, bg_features_dim]
        # z_bg_features: [bs, T, action_dim]
        # z_context: [bs, T, context_dim]
        bs, timestep_horizon, n_particles, _ = z.shape

        # policy: state -> context
        mu_context = logvar_context = None

        # dynamics: prev_state + context -> next_state

        # project particles
        z_v = z.reshape(bs * timestep_horizon, *z.shape[2:])
        z_scale_v = z_scale.reshape(bs * timestep_horizon, *z_scale.shape[2:])
        z_obj_on_v = z_obj_on.reshape(bs * timestep_horizon, *z_obj_on.shape[2:])
        z_depth_v = z_depth.reshape(bs * timestep_horizon, *z_depth.shape[2:])
        z_features_v = z_features.reshape(bs * timestep_horizon, *z_features.shape[2:])
        z_bg_features_v = z_bg_features.reshape(bs * timestep_horizon, *z_bg_features.shape[2:])
        z_context_v = z_context.reshape(bs * timestep_horizon, *z_context.shape[2:])
        if self.use_z_orig:
            z_orig_v = self.particles_anchor.repeat(bs * timestep_horizon, 1, 1)
        else:
            z_orig_v = None
        if z_score is not None:
            z_score_v = z_score.reshape(bs * timestep_horizon, *z_score.shape[2:])
        else:
            z_score_v = z_score

        detach_dyn_inputs = False
        if detach_dyn_inputs:
            z_v = z_v.detach()
            z_scale_v = z_scale_v.detach()
            z_obj_on_v = z_obj_on_v.detach()
            z_depth_v = z_depth_v.detach()
            z_features_v = z_features_v.detach()
            z_bg_features_v = z_bg_features_v.detach()

        particle_projection = self.particle_projection(z_v,
                                                       z_scale_v,
                                                       z_obj_on_v,
                                                       z_depth_v,
                                                       z_features_v,
                                                       z_bg_features_v,
                                                       z_context_v,
                                                       z_score_v,
                                                       z_orig_v)
        # [bs * T, n_particles + 2, projection_dim]
        particle_proj_int = particle_projection

        # unroll forward
        particle_proj_int = particle_proj_int.view(bs, timestep_horizon, *particle_proj_int.shape[1:])
        # [bs, T, n_particles + 2, projection_dim]

        particle_proj_int = particle_proj_int.permute(0, 2, 1, 3)
        # [bs, n_particles + 2, T, projection_dim]
        if self.ctx_mode == 'adaln':
            if self.random_action_condition:
                random_actions = torch.rand(particle_proj_int.shape[0], particle_proj_int.shape[2],
                                            self.random_action_dim, device=particle_proj_int.device)
                c_random_action = self.random_action_proj(random_actions)
                if len(c_random_action.shape) == 3:
                    c_random_action = c_random_action.unsqueeze(1).repeat(1, particle_proj_int.shape[1], 1,
                                                                          1)  # [bs, n, t, f]
            else:
                c_random_action = 0

            if self.action_condition and actions is not None:
                if self.learn_null_action_embed and actions_mask is not None:
                    # action_mask: [batch_size, T] or [batch_size, T, 1], 1 where use action, 0 replace action
                    # Expand mask
                    if len(actions_mask.shape) == 2:
                        actions_mask = actions_mask.bool().unsqueeze(-1)  # (batch_size, seq_len, 1)
                    # Expand null embedding to match
                    null_action_embeds = self.null_action_embeddings.expand(actions.size(0), actions.size(1), -1)

                    # Blend
                    actions = actions * actions_mask + null_action_embeds * (~actions_mask)

                c_action = self.action_proj(actions)
                if len(c_action.shape) == 3:
                    c_action = c_action.unsqueeze(1).repeat(1, particle_proj_int.shape[1], 1, 1)  # [bs, n, t, f]
            else:
                c_action = 0

            c = self.context_proj(z_context_v)
            c = c.reshape(bs, timestep_horizon, *c.shape[1:])
            if len(c.shape) == 3:
                c = c.unsqueeze(1).repeat(1, particle_proj_int.shape[1], 1, 1)  # [bs, 1, t, f]
            elif c.shape[2] != particle_proj_int.shape[1]:
                c = c.permute(0, 2, 1, 3)  # [bs, n + 1, t, f]
                c = c.repeat(1, particle_proj_int.shape[1], 1, 1)  # [bs, 1, t, f]
            else:
                c = c.permute(0, 2, 1, 3)  # [bs, n + 1, t, f]
            c = c + c_action + c_random_action
            c = self.cond_activation(c)
        else:
            c = None

        if self.n_views > 1:
            # [bs * n_views, n, T, d] -> [bs, n_views, n, T, d] -> [bs, n_views * n, T, d]
            particle_proj_int = particle_proj_int.view(-1, self.n_views, particle_proj_int.shape[1],
                                                       *particle_proj_int.shape[2:])  # [bs, n_views, n, T, d]
            particle_proj_int = particle_proj_int + self.view_embeddings
            particle_proj_int = particle_proj_int.reshape(particle_proj_int.shape[0], -1,
                                                          *particle_proj_int.shape[3:])  # [bs, n_views * n, T, d]
            if c is not None:
                c = c.reshape(-1, self.n_views * c.shape[1], *c.shape[2:])

        if c is not None and self.pos_embed_p_adaln:
            c_pe = self.pos_p_embeddings.repeat(c.shape[0], 1, c.shape[2], 1)
            c = c + c_pe

        if c is not None and self.pos_embed_objon_adaln:
            c_objon = self.objon_embeddings(z_obj_on)  # [bs, t, n, dim]
            c_objon_bg = torch.zeros(c_objon.shape[0], c_objon.shape[1], 1, c_objon.shape[-1], device=c_objon.device)
            c_objon = torch.cat([c_objon, c_objon_bg], dim=2)  # [bs, t, n + 1, dim]
            c_objon = c_objon.permute(0, 2, 1, 3)  # [bs, n + 1, t, dim]
            if self.n_views > 1:
                c_objon = c_objon.reshape(-1, self.n_views * c_objon.shape[1], c_objon.shape[2], c_objon.shape[-1])
                # [bs, n_views * (n + 1), t, dim]
            c = c + c_objon

        particles_trans = self.particle_transformer(particle_proj_int, c)
        # [bs, n_particles + 2, T, projection_dim]
        if self.n_views > 1:
            # [bs, n_views * n, T, d] -> [bs * n_views, n, T, d]
            particles_trans = particles_trans.reshape(bs, -1, *particles_trans.shape[2:])
        particles_trans = particles_trans.permute(0, 2, 1, 3)
        # [bs, T, n_particles + 2, projection_dim]

        # decode transformer output
        particles_trans = particles_trans.reshape(-1, *particles_trans.shape[2:])
        # [bs * T, n_particles + 2, projection_dim]
        particle_decoder_out = self.particle_decoder(particles_trans)
        mu = particle_decoder_out['mu_offset']
        logvar = particle_decoder_out['logvar_offset']

        obj_on_a_gate = (particle_decoder_out['lobj_on_a']).sigmoid()
        obj_on_a = ((1 - obj_on_a_gate) * self.obj_on_min + obj_on_a_gate * self.obj_on_max).exp()
        obj_on_b_gate = 1 - (
                particle_decoder_out['lobj_on_b'] * 0 + particle_decoder_out['lobj_on_a']).sigmoid()
        obj_on_b = ((1 - obj_on_b_gate) * self.obj_on_min + obj_on_b_gate * self.obj_on_max).exp()

        mu_depth = particle_decoder_out['mu_depth']
        logvar_depth = particle_decoder_out['logvar_depth']
        mu_scale = particle_decoder_out['mu_scale']
        logvar_scale = particle_decoder_out['logvar_scale']
        mu_features = particle_decoder_out['mu_features']
        logvar_features = particle_decoder_out['logvar_features']
        mu_bg_features = particle_decoder_out['mu_bg_features']
        logvar_bg_features = particle_decoder_out['logvar_bg_features']
        mu_score = particle_decoder_out['mu_score']
        logvar_score = particle_decoder_out['logvar_score']

        mu_scale = mu_scale + self.scale_anchor
        if self.use_z_orig:
            mu = self.particles_anchor + mu

        if self.predict_delta:
            mu = z_v + mu
            # mu_scale = z_scale_v + mu_scale
            # mu_depth = z_depth_v + mu_depth
            # mu_features = z_features_v + mu_features
            # mu_bg_features = z_bg_features_v + mu_bg_features

        # reshape to [bs, t, ...]
        mu = mu.view(bs, timestep_horizon, *mu.shape[1:])
        logvar = logvar.view(bs, timestep_horizon, *logvar.shape[1:])
        obj_on_a = obj_on_a.view(bs, timestep_horizon, *obj_on_a.shape[1:])
        obj_on_b = obj_on_b.view(bs, timestep_horizon, *obj_on_b.shape[1:])
        mu_depth = mu_depth.view(bs, timestep_horizon, *mu_depth.shape[1:])
        logvar_depth = logvar_depth.view(bs, timestep_horizon, *logvar_depth.shape[1:])
        mu_scale = mu_scale.view(bs, timestep_horizon, *mu_scale.shape[1:])
        logvar_scale = logvar_scale.view(bs, timestep_horizon, *logvar_scale.shape[1:])
        mu_features = mu_features.view(bs, timestep_horizon, *mu_features.shape[1:])
        logvar_features = logvar_features.view(bs, timestep_horizon, *logvar_features.shape[1:])
        mu_bg_features = mu_bg_features.view(bs, timestep_horizon, *mu_bg_features.shape[1:])
        logvar_bg_features = logvar_bg_features.view(bs, timestep_horizon, *logvar_bg_features.shape[1:])
        if self.particle_score and mu_score is not None:
            mu_score = mu_score.view(bs, timestep_horizon, *mu_score.shape[1:])
            logvar_score = logvar_score.view(bs, timestep_horizon, *logvar_score.shape[1:])

        output_dict = {'mu': mu, 'logvar': logvar, 'mu_features': mu_features, 'logvar_features': logvar_features,
                       'obj_on_a': obj_on_a.squeeze(-1), 'obj_on_b': obj_on_b.squeeze(-1), 'mu_depth': mu_depth,
                       'logvar_depth': logvar_depth, 'mu_scale': mu_scale, 'logvar_scale': logvar_scale,
                       'mu_bg_features': mu_bg_features, 'logvar_bg_features': logvar_bg_features,
                       'mu_context': mu_context, 'logvar_context': logvar_context,
                       'mu_score': mu_score, 'logvar_score': logvar_score}

        return output_dict
