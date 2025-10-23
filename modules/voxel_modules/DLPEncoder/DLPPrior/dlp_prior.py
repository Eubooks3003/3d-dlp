import torch
import torch.nn as nn
from typing import Optional, Tuple

# you already have these
from modules.point_cloud_modules.vision_modules_3d import Encoder3D
from modules.point_cloud_modules.DLPEncoder.prior_encoder.SSM3D import AlternativeSpatialSoftmax3D


# ---------- minimal dense tiler ----------
class VoxTiler:
    """
    Pads [B,C,D,H,W] so (D,H,W) are divisible by tile_size=(td,th,tw),
    returns tiles [B,C,T,td,th,tw] and meta for de-tiling / coord convert.
    """
    @staticmethod
    def pad_to_tiles(vox, tile_size: Tuple[int,int,int], pad_value: float = 0.0):
        B, C, D, H, W = vox.shape
        td, th, tw = tile_size

        def need(n, t):
            r = n % t
            return 0 if r == 0 else (t - r)

        pd = need(D, td); ph = need(H, th); pw = need(W, tw)
        pad = (0, pw, 0, ph, 0, pd)  # (wL,wR,hL,hR,dL,dR) but PyTorch uses reverse: (W_left,W_right,H_left,H_right,D_left,D_right)
        vox_p = nn.functional.pad(vox, (0, pw, 0, ph, 0, pd), value=pad_value)

        Dp, Hp, Wp = vox_p.shape[-3:]
        nz, ny, nx = Dp // td, Hp // th, Wp // tw
        T = nz * ny * nx

        # view-as-tiles (no copy)
        vox_p = vox_p.view(B, C, nz, td, ny, th, nx, tw)
        vox_p = vox_p.permute(0, 1, 2, 4, 6, 3, 5, 7)  # [B,C,nz,ny,nx,td,th,tw]
        tiles = vox_p.reshape(B, C, T, td, th, tw)      # [B,C,T,td,th,tw]

        # precompute tile starts (zyx) on the padded grid
        z0 = torch.arange(0, nz*td, td, device=vox.device)
        y0 = torch.arange(0, ny*th, th, device=vox.device)
        x0 = torch.arange(0, nx*tw, tw, device=vox.device)
        zz, yy, xx = torch.meshgrid(z0, y0, x0, indexing="ij")
        starts_zyx = torch.stack([zz, yy, xx], dim=-1).reshape(-1, 3).to(vox.dtype)  # [T,3]

        meta = dict(
            padded_shape=(Dp, Hp, Wp),
            original_shape=(D, H, W),
            tile_size=(td, th, tw),
            ntiles=(nz, ny, nx),
            starts_zyx=starts_zyx  # on the padded grid
        )
        return tiles, meta


class DLPPrior(nn.Module):
    """
    Dense-voxel Prior (no point-cloud voxelizer):
      vox [B,C,D,H,W] -> tile encoder -> 3D SSM per tile -> global KP in xyz.

    Returns:
      kp:      [B, K_sel, 3]   (x,y,z in [-1,1] over ORIGINAL (unpadded) grid)
      cov:     [B, K_sel, 3, 3]
      kp_mask: [B, K_sel]      (valid entries)
    """
    def __init__(self,
                 in_channels: int,
                 grid: Tuple[int,int,int],           # (D,H,W) ORIGINAL grid you expect
                 tile_size: Tuple[int,int,int]=(12,12,12),
                 base_ch=32, ch_mult=(1,2,3), num_res_blocks=2,
                 use_resblock=True, use_attention=False, cnn_mid_blocks=False,
                 n_kp_per_tile=1,
                 n_kp_prior=64,
                 kp_range=(-1., 1.), temperature=1.0,
                 filtering_heuristic='variance',   # 'variance' | 'random' | 'none'
                 density_channel_idx: Optional[int]=0,   # which channel to sum for tile validity
                 init_zero_bias=True, init_conv_layers=True, init_conv_std=0.02):
        super().__init__()
        assert filtering_heuristic in ('variance','random','none')

        self.grid = tuple(grid)                 # ORIGINAL (D,H,W)
        self.tile_size = tuple(tile_size)       # (td,th,tw)
        self.n_kp_per_tile = int(n_kp_per_tile)
        self.n_kp_prior = int(n_kp_prior)
        self.kp_range = kp_range
        self.temperature = float(temperature)
        self.filtering_heuristic = filtering_heuristic
        self.density_channel_idx = density_channel_idx
        
        # 3D encoder producing K channels per tile
        self.enc3d = Encoder3D(
            in_channels=in_channels,
            ch=base_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
            residual=use_resblock, dropout=0.0,
            use_attention=use_attention,
            mid_blocks=not cnn_mid_blocks,
            out_channels=self.n_kp_per_tile
        )
        self.ssm3d = AlternativeSpatialSoftmax3D(kp_range=self.kp_range, temperature=self.temperature)

        self.init_zero_bias   = init_zero_bias
        self.init_conv_layers = init_conv_layers
        self.init_conv_std    = float(init_conv_std)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if self.init_conv_layers:
                    nn.init.normal_(m.weight, 0.0, self.init_conv_std)
                if self.init_zero_bias and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if hasattr(self.enc3d, "conv_out"):
            with torch.no_grad():
                nn.init.constant_(self.enc3d.conv_out.weight, -self.init_conv_std)
                if self.enc3d.conv_out.bias is not None:
                    nn.init.constant_(self.enc3d.conv_out.bias, 0)

    # ---- axis helpers ----
    @staticmethod
    def _as_xyz(t: torch.Tensor) -> torch.Tensor:
        # swap (z,y,x) -> (x,y,z) for [...,3]
        return t[..., [2, 1, 0]]

    @staticmethod
    def _cov_zyx_to_xyz(cov: torch.Tensor) -> torch.Tensor:
        P = cov.new_tensor([[0.,0.,1.],
                            [0.,1.,0.],
                            [1.,0.,0.]])  # (z,y,x)->(x,y,z)
        return P.transpose(-1, -2) @ cov @ P

    def _local_to_global_xyz(self, kp_local_zyx: torch.Tensor, meta):
        """
        kp_local_zyx: [B,T,K,3] in [-1,1] per tile (order z,y,x)
        return: [B,T,K,3] in [-1,1] over ORIGINAL (order x,y,z)
        """
        B, T, K, _ = kp_local_zyx.shape
        (td, th, tw) = self.tile_size
        (D, H, W)    = self.grid

        starts_zyx = meta["starts_zyx"].to(kp_local_zyx.device, kp_local_zyx.dtype)  # [T,3]
        starts_zyx = starts_zyx.view(1, T, 1, 3)

        # [-1,1] -> [0, td/th/tw - 1] within tile
        lo, hi = self.kp_range
        p01 = (kp_local_zyx - lo) / (hi - lo)
        # allow a little overflow then softly clip; factor 4 sharpens slope
        p01 = self._soft_clip01((p01 - 0.5) * 4.0)
        scale_zyx = kp_local_zyx.new_tensor([td-1, th-1, tw-1]).view(1,1,1,3)
        kp_vox_zyx = starts_zyx + p01 * scale_zyx

        # no hard clamp here; normalization below already keeps us in-range
        full_zyx = kp_local_zyx.new_tensor([D-1, H-1, W-1]).view(1,1,1,3)
        kp_glob_zyx = (kp_vox_zyx / full_zyx) * (hi - lo) + lo
        return self._as_xyz(kp_glob_zyx)
    
    def _soft_clip01(self, u, eps=1e-3):
        # maps ℝ -> (0,1) with gradient at edges
        return (1 - 2*eps) * torch.sigmoid(u) + eps

    # --- helper: relaxed Top-K with STE (train-time), hard Top-K (eval) ---
    def relaxed_topk_weights(self,score: torch.Tensor,
                            k_target: int,
                            valid_mask: torch.Tensor,
                            tau: float = 0.3,
                            training: bool = True):
        """
        score, valid_mask: [B, K_tot]  (larger score is better)
        returns:
            w_keep:  [B, K_tot] mask in [0,1] (soft in train, hard in eval)
            top_idx: [B, k_target] hard top-k indices
        """
        score_masked = torch.where(valid_mask, score, score.new_full(score.shape, -1e9))
        # hard indices for both train/eval fast path
        top_idx = torch.topk(score_masked, k_target, dim=-1).indices  # [B,k]

        if not training:
            # eval: strictly hard mask
            hard = torch.zeros_like(score_masked).scatter(1, top_idx, 1.0)
            return hard, top_idx

        # train: soft weights spread some mass to non-selected for gradient flow
        w_soft = torch.softmax(score_masked / tau, dim=-1)              # [B,K_tot]
        # normalize so sum ≈ k_target
        w_soft = w_soft * (k_target / (w_soft.sum(-1, keepdim=True) + 1e-8))
        hard = torch.zeros_like(score_masked).scatter(1, top_idx, 1.0)
        # STE: hard forward, soft backward
        w_keep = hard + (w_soft - w_soft.detach())
        return w_keep, top_idx

    def encode_prior(self,
                    vox: torch.Tensor,
                    k: Optional[int] = None,
                    return_keep_weights: bool = True):
        """
        vox: [B,C,D,H,W] aligned to ORIGINAL grid (D,H,W) provided at init
        return_keep_weights:
            False -> (kp_sel, cov_sel, kp_tiles)
            True  -> (kp_sel, cov_sel, kp_tiles, w_keep_full) where w_keep_full is [B, K_tot]
        """
        assert vox.dim()==5, f"vox must be [B,C,D,H,W], got {vox.shape}"
        B, C, D, H, W = vox.shape
        assert (D, H, W) == tuple(self.grid), f"input grid {(D,H,W)} != expected {self.grid}"

        # 1) tile the dense grid
        tiles, meta = VoxTiler.pad_to_tiles(vox, self.tile_size)    # [B,C,T,td,th,tw]
        B, Cg, T, td, th, tw = tiles.shape

        # 2) validity per tile
        if self.density_channel_idx is not None and 0 <= self.density_channel_idx < Cg:
            den_tiles = tiles[:, self.density_channel_idx:self.density_channel_idx+1]    # [B,1,T,td,th,tw]
        else:
            den_tiles = tiles.sum(dim=1, keepdim=True)  # [B,1,T,td,th,tw]
        tile_mass  = den_tiles.sum(dim=(1,3,4,5))       # [B,T]
        valid_tile = tile_mass > 1e-6                   # [B,T]
        valid_bt   = valid_tile.view(B*T, 1, 1, 1, 1)

        # 3) run encoder per tile
        tiles_bt = tiles.permute(0,2,1,3,4,5).reshape(B*T, Cg, td, th, tw)  # [B*T,C,td,th,tw]
        logits = self.enc3d(tiles_bt)                                       # [B*T, K, d', h', w']

        # softened guard SSM on empty tiles (keeps tiny grad paths)
        dead = (~valid_bt).float()
        logits = logits + dead * (-20.0)

        # 4) SSM -> local zyx mean + covariance
        kp_local, cov_local = self.ssm3d(logits, probs=False, variance=True) # [B*T,K,3], [B*T,K,3,3]
        K = kp_local.shape[1]
        kp_local  = kp_local.view(B, T, K, 3)
        cov_local = cov_local.view(B, T, K, 3, 3)

        # 5) local->global (xyz)
        kp_glob_xyz = self._local_to_global_xyz(kp_local, meta)      # [B,T,K,3]
        cov_glob_xyz = self._cov_zyx_to_xyz(cov_local)               # [B,T,K,3,3]

        kp_all  = kp_glob_xyz.reshape(B, T*K, 3)        # [B, K_tot, 3]
        cov_all = cov_glob_xyz.reshape(B, T*K, 3, 3)    # [B, K_tot, 3,3]
        valid_kp = valid_tile.unsqueeze(-1).expand(B, T, K).reshape(B, T*K)   # [B, K_tot]

        # 6) score + relaxed Top-K (train), hard (eval)
        if self.filtering_heuristic == 'variance':
            # negative trace (low variance = good)
            tr = cov_all[..., 0,0] + cov_all[..., 1,1] + cov_all[..., 2,2]  # [B, K_tot]
            score = -tr
        elif self.filtering_heuristic == 'random':
            score = torch.rand_like(valid_kp.float())
        else:
            score = torch.zeros_like(valid_kp.float())

        k_target = self.n_kp_prior if k is None else int(k)
        w_keep_full, top_idx = self.relaxed_topk_weights(
            score=score,
            k_target=k_target,
            valid_mask=valid_kp,
            tau=getattr(self, 'relaxed_topk_tau', 0.3),
            training=self.training
        )  # w_keep_full: [B, K_tot], top_idx: [B, k]

        # 7) gather selected subset for the fast path (keeps your existing behavior)
        # build linear tile ids aligned with kp_all/cov_all (optional, for downstream)
        nz, ny, nx = meta["ntiles"]
        tile_lin = torch.arange(nz*ny*nx, device=vox.device).view(1, -1).expand(B, -1)  # [B,T]
        tile_lin_all = tile_lin.unsqueeze(-1).expand(B, tile_lin.size(1), K).reshape(B, -1)  # [B, K_tot]

        kp_sel_list, cov_sel_list, tileid_sel_list = [], [], []
        for b in range(B):
            idxb = top_idx[b]                                # [k_target]
            kp_sel_list.append(kp_all[b, idxb])
            cov_sel_list.append(cov_all[b, idxb])
            tileid_sel_list.append(tile_lin_all[b, idxb])

        kp_sel   = torch.stack(kp_sel_list,  dim=0)          # [B, k_target, 3]
        cov_sel  = torch.stack(cov_sel_list, dim=0)          # [B, k_target, 3,3]
        kp_tiles = torch.stack(tileid_sel_list, dim=0)       # [B, k_target]

        if return_keep_weights:
            return kp_sel, cov_sel, kp_tiles, w_keep_full    # w_keep_full is over ALL proposals
        else:
            return kp_sel, cov_sel, kp_tiles

    def forward(self, vox: torch.Tensor):
        return self.encode_prior(vox, k=self.n_kp_prior)
