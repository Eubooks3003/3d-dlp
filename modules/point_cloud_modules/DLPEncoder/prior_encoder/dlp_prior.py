import torch
import torch.nn as nn
from typing import Optional, Tuple

from modules.point_cloud_modules.DLPEncoder.prior_encoder.grid_voxelizer import GridVoxelizer
from modules.point_cloud_modules.vision_modules_3d import Encoder3D
from modules.point_cloud_modules.DLPEncoder.prior_encoder.SSM3D import AlternativeSpatialSoftmax3D


class DLPPrior(nn.Module):
    """
    Single-frame Point-Cloud Prior:
      points (+mask) -> voxel grid (7ch moments) -> 3D encoder (K channels per tile)
      -> 3D spatial softmax -> keypoint proposals (z,y,x in [-1,1]) + covariance.

    Inputs:
      points:  [B, N, 3]   (xyz in [-1,1]^3)
      mask_pc: [B, N] or None

    Outputs:
      kp:      [B, K_sel, 3]   (z,y,x in [-1,1] over the ORIGINAL (unpadded) grid)
      cov:     [B, K_sel, 3, 3]  (if variance=True in SSM)
      kp_mask: [B, K_sel] (True where kp is valid; False for padded slots when K_sel > #valid)
    """

    def __init__(self,
                 grid: Tuple[int,int,int]=(48, 48, 48),      # (D,H,W) original grid
                 tile_size: Tuple[int,int,int]=(12, 12, 12), # tile size in voxels (must divide padded dims)
                 base_ch=32, ch_mult=(1, 2, 3), num_res_blocks=2,
                 use_resblock=True, use_attention=False, cnn_mid_blocks=False,
                 n_kp_per_tile=1,            # channels (keypoints) predicted per tile
                 n_kp_prior=64,              # target number to keep after selection
                 kp_range=(-1., 1.), temperature=1.0,
                 filtering_heuristic='variance',  # 'variance' | 'random' | 'none'
                 init_zero_bias=True, init_conv_layers=True, init_conv_std=0.02):
        super().__init__()

        assert filtering_heuristic in ('variance', 'random', 'none')
        self.grid         = tuple(grid)          # (D,H,W) original
        self.tile_size    = tuple(tile_size)     # (td,th,tw)
        self.n_kp_per_tile = int(n_kp_per_tile)
        self.n_kp_prior   = int(n_kp_prior)
        self.kp_range     = kp_range
        self.temperature  = float(temperature)
        self.filtering_heuristic = filtering_heuristic

        # --- Voxelizer (emits 7 channels when with_moments=True in forward) ---
        self.grid_voxelizer = GridVoxelizer(D=self.grid[0], H=self.grid[1], W=self.grid[2], out_feat=1)

        # --- 3D Encoder -> n_kp_per_tile heatmaps per tile ---
        enc_in_channels = 7  # [density, mean_x/y/z, var_x/y/z] from with_moments=True
        self.enc3d = Encoder3D(
            in_channels=enc_in_channels,
            ch=base_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
            residual=use_resblock, dropout=0.0,
            use_attention=use_attention,
            mid_blocks=not cnn_mid_blocks,
            out_channels=self.n_kp_per_tile
        )

        # --- 3D Spatial Softmax over each tile/channel ---
        self.ssm3d = AlternativeSpatialSoftmax3D(kp_range=self.kp_range, temperature=self.temperature)

        # --- init ---
        self.init_zero_bias   = init_zero_bias
        self.init_conv_layers = init_conv_layers
        self.init_conv_std    = float(init_conv_std)
        self.init_weights()

    # -------------------------- utils --------------------------

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

    @staticmethod
    def _check_shapes(points, mask_pc):
        assert points.dim() == 3 and points.size(-1) == 3, f"points must be [B,N,3], got {tuple(points.shape)}"
        if mask_pc is not None:
            assert mask_pc.shape[:2] == points.shape[:2], f"mask_pc must be [B,N], got {tuple(mask_pc.shape)}"

    def _get_tile_starts(self, meta, device, dtype):
        Dp, Hp, Wp = meta["padded_shape"]
        return self.grid_voxelizer.get_tile_location_idx(Dp, Hp, Wp, self.tile_size, device=device, dtype=dtype)  # [T,3] zyx
    
    @staticmethod
    def _as_xyz(t: torch.Tensor) -> torch.Tensor:
        # swap (z,y,x) -> (x,y,z) for any [...,3]
        return t[..., [2, 1, 0]]

    @staticmethod
    def _cov_zyx_to_xyz(cov: torch.Tensor) -> torch.Tensor:
        # permute covariance from (z,y,x) basis to (x,y,z) basis
        # cov: [...,3,3]
        P = cov.new_tensor([[0.,0.,1.],
                            [0.,1.,0.],
                            [1.,0.,0.]])  # maps indices 0=z,1=y,2=x -> 0=x,1=y,2=z
        return P.transpose(-1, -2) @ cov @ P
        
    def _local_to_global_xyz(self, kp_local, meta):
        """
        kp_local: [B,T,K,3] in [-1,1] per-tile (order z,y,x)
        returns kp_glob_xyz in [-1,1] over the ORIGINAL (unpadded) grid (order x,y,z).
        """
        B, T, K, _ = kp_local.shape
        (td, th, tw) = self.tile_size
        (D, H, W)    = self.grid

        starts_zyx = self._get_tile_starts(meta, kp_local.device, dtype=torch.int32).to(kp_local.dtype)  # [T,3]
        starts_zyx = starts_zyx.view(1, T, 1, 3)  # [1,T,1,3]

        # [-1,1] -> [0, td/th/tw - 1] in tile coords
        lo, hi = self.kp_range
        p01 = (kp_local - lo) / (hi - lo)                                  # [B,T,K,3] in [0,1] (z,y,x)
        scale_zyx = kp_local.new_tensor([td-1, th-1, tw-1]).view(1,1,1,3)
        kp_vox_zyx = starts_zyx + p01 * scale_zyx                           # [B,T,K,3]

        # clamp in ORIGINAL (unpadded) grid
        kp_vox_zyx[..., 0].clamp_(0, D-1)   # z
        kp_vox_zyx[..., 1].clamp_(0, H-1)   # y
        kp_vox_zyx[..., 2].clamp_(0, W-1)   # x

        # normalize to [-1,1] over ORIGINAL, still zyx
        full_zyx = kp_local.new_tensor([D-1, H-1, W-1]).view(1,1,1,3)
        kp_glob_zyx = (kp_vox_zyx / full_zyx) * (hi - lo) + lo              # [B,T,K,3] (z,y,x)

        # convert to xyz for output
        kp_glob_xyz = self._as_xyz(kp_glob_zyx)                             # [B,T,K,3] (x,y,z)
        return kp_glob_xyz


    # ------------------------ main API -------------------------

    def encode_prior(self, points: torch.Tensor, mask_pc: Optional[torch.Tensor] = None,
                    k: Optional[int] = None):
        self._check_shapes(points, mask_pc)
        B, N, _ = points.shape

        # (1) voxelize (7ch moments)
        vox, meta0 = self.grid_voxelizer(points, mask=mask_pc, with_moments=True)  # [B,7,D,H,W]
        den = vox[:, 0:1]  # [B,1,D,H,W]

        # (2) tiles
        den_tiles, meta = self.grid_voxelizer.vox_to_tiles(den, tile_size=self.tile_size)  # [B,1,T,td,th,tw]
        feat_tiles, _   = self.grid_voxelizer.vox_to_tiles(vox, tile_size=self.tile_size)  # [B,7,T,td,th,tw]
        B, Cg, T, td, th, tw = feat_tiles.shape

        tiles_bt = feat_tiles.permute(0,2,1,3,4,5).reshape(B*T, Cg, td, th, tw)  # [B*T,7,td,th,tw]

        # (3) encoder on ALL tiles
        logits = self.enc3d(tiles_bt)  # [B*T, K, d', h', w']
        K = logits.shape[1]

        # (4) valid tiles by mass threshold
        tile_mass  = den_tiles.sum(dim=(1,3,4,5))          # [B,T]
        valid_tile = tile_mass > 1e-6                       # [B,T]
        valid_bt   = valid_tile.view(B*T, 1, 1, 1, 1)

        # Don't create NaNs inside SSM; keep finite but very low.
        logits = torch.where(valid_bt, logits, logits.new_full(logits.shape, -1e9))

        # (5) SSM
        kp_local, cov_local = self.ssm3d(logits, probs=False, variance=True)  # [B*T,K,3], [B*T,K,3,3]
        kp_local  = kp_local.view(B, T, K, 3)
        cov_local = cov_local.view(B, T, K, 3, 3)

        # (6) local -> global in **xyz**
        kp_glob_xyz = self._local_to_global_xyz(kp_local, meta)     # [B,T,K,3] (x,y,z)
        kp_all  = kp_glob_xyz.view(B, T*K, 3)                       # [B, K_tot, 3]

        # also reorder covariances to xyz
        cov_all = self._cov_zyx_to_xyz(cov_local).view(B, T*K, 3, 3)

        # (7) validity per keypoint (comes from its tile)
        valid_kp = valid_tile.unsqueeze(-1).expand(B, T, K).reshape(B, T*K)   # [B, K_tot]

        # (8) scores: only valid ones are eligible
        if self.filtering_heuristic == 'variance' and cov_all is not None:
            tr = cov_all[..., 0,0] + cov_all[..., 1,1] + cov_all[..., 2,2]     # [B,K_tot]
            score = -tr
        elif self.filtering_heuristic == 'random':
            score = torch.rand_like(valid_kp.float())
        else:
            score = torch.zeros_like(valid_kp.float())

        # mask invalid to -inf so they never get picked
        score = torch.where(valid_kp, score, score.new_full(score.shape, float('-inf')))  # [B,K_tot]

        # (9) per-batch dynamic top-k over *valid* only
        K_tot = T*K
        k_target = self.n_kp_prior if k is None else int(k)
        # argsort descending; take first n_valid or k_target
        sorted_score, sorted_idx = torch.sort(score, dim=-1, descending=True)  # [B, K_tot]
        n_valid = valid_kp.sum(dim=-1)                                         # [B]

        # Build per-batch selection with padding
        kp_sel_list, cov_sel_list, mask_sel_list = [], [], []
        for b in range(B):
            k_eff_b = int(min(k_target, int(n_valid[b].item())))
            idx_b_valid = sorted_idx[b, :k_eff_b]                               # strictly valid picks
            kp_b  = kp_all[b, idx_b_valid]                                      # [k_eff_b,3]
            cov_b = cov_all[b, idx_b_valid] if cov_all is not None else None
            mask_b = kp_b.new_ones((k_eff_b,), dtype=torch.bool)

            # pad if needed
            if k_eff_b < k_target:
                pad = k_target - k_eff_b
                kp_b  = torch.cat([kp_b,  kp_b.new_zeros(pad, 3)], dim=0)
                if cov_b is not None:
                    cov_b = torch.cat([cov_b, cov_b.new_zeros(pad, 3, 3)], dim=0)
                mask_b = torch.cat([mask_b, kp_b.new_zeros(pad, dtype=torch.bool)], dim=0)

            kp_sel_list.append(kp_b)
            cov_sel_list.append(cov_b if cov_all is not None else None)
            mask_sel_list.append(mask_b)

        kp_sel  = torch.stack(kp_sel_list, dim=0)                                # [B, k_target, 3]
        cov_sel = torch.stack([c if c is not None else kp_sel.new_zeros(k_target,3,3)
                            for c in cov_sel_list], dim=0) if cov_all is not None else None
        kp_mask = torch.stack(mask_sel_list, dim=0)                              # [B, k_target]

        return kp_sel, cov_sel, kp_mask



    def forward(self, points: torch.Tensor, mask_pc: Optional[torch.Tensor] = None):
        return self.encode_prior(points, mask_pc=mask_pc, k=self.n_kp_prior)
