import torch
import torch.nn as nn
from typing import Optional

from modules.point_cloud_modules.DLPEncoder.prior_encoder.grid_voxelizer import GridVoxelizer
from modules.point_cloud_modules.vision_modules_3d import Encoder3D
from modules.point_cloud_modules.DLPEncoder.prior_encoder.SSM3D import AlternativeSpatialSoftmax3D


class DLPPrior(nn.Module):
    """
    Single-frame Point-Cloud Prior:
      points (+mask) -> voxel grid -> 3D encoder (K channels) -> 3D spatial softmax
      -> K keypoint proposals (z,y,x in [-1,1]) + covariance per kp.

    Inputs:
      points:  [B, N, 3]
      mask_pc: [B, N] (1 valid, 0 pad) or None
    Outputs:
      kp:   [B, K, 3]   (z, y, x in [-1,1])
      cov:  [B, K, 3, 3]
    """

    def __init__(self,
                 grid=(48, 48, 48),          # (D, H, W)
                 tile_size=(12, 12, 12),
                 out_feat=1,                 # channels emitted by GridVoxelizer
                 base_ch=32, ch_mult=(1, 2, 3), num_res_blocks=2,
                 use_resblock=True, use_attention=False, cnn_mid_blocks=False,
                 n_kp_prior=64,
                 n_kp = 1,
                 kp_range=(-1., 1.), temperature=1.0,
                 filtering_heuristic='none',  # ['variance','random','none']
                 init_zero_bias=True, init_conv_layers=True, init_conv_std=0.02):
        super().__init__()

        assert filtering_heuristic in ('variance', 'random', 'none')
        self.grid = tuple(grid)
        self.out_feat = int(out_feat)
        self.n_kp_prior = int(n_kp_prior)
        self.n_kp = n_kp
        self.kp_range = kp_range
        self.tile_size   = tuple(tile_size)
        self.temperature = float(temperature)
        self.filtering_heuristic = filtering_heuristic

        # voxelizer
        self.grid_voxelizer = GridVoxelizer(D=self.grid[0], H=self.grid[1], W=self.grid[2],
                                            out_feat=self.out_feat)

        # 3D encoder -> K-channel saliency
        self.enc3d = Encoder3D(in_channels=self.out_feat,
                               ch=base_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               residual=use_resblock, dropout=0.0,
                               use_attention=use_attention,
                               mid_blocks=not cnn_mid_blocks,
                               out_channels=self.n_kp)

        # 3D spatial softmax
        self.ssm3d = AlternativeSpatialSoftmax3D(kp_range=self.kp_range, temperature=self.temperature)

        # init
        self.init_zero_bias = init_zero_bias
        self.init_conv_layers = init_conv_layers
        self.init_conv_std = float(init_conv_std)
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

    @staticmethod
    def _check_shapes(points, mask_pc):
        assert points.dim() == 3 and points.size(-1) == 3, \
            f"points must be [B,N,3], got {tuple(points.shape)}"
        if mask_pc is not None:
            assert mask_pc.shape[:2] == points.shape[:2], \
                f"mask_pc must be [B,N] matching points, got {tuple(mask_pc.shape)}"
    def get_global_kp_3d(self, local_kp, meta):
        """
        local_kp: [B, T, n_kp, 3] in [-1,1] relative to each tile
        meta: from vox_to_tiles
        returns: global kp in [-1,1] over the **padded** grid
        """
        B, T, K, _ = local_kp.shape
        (Dp,Hp,Wp) = meta["padded_shape"]
        (td,th,tw) = meta["tile_size"]

        # tile starts [T,3] in voxel coords
        starts = self.grid_voxelizer.get_tile_location_idx(Dp, Hp, Wp, meta["tile_size"],
                                                           device=local_kp.device, dtype=torch.int32)  # [T,3]

        # local [-1,1] → [0,1] within tile → voxel coords within tile
        lo, hi = self.kp_range
        p01 = (local_kp - lo) / (hi - lo)                               # [B,T,K,3] in [0,1]
        scale = torch.tensor([td-1, th-1, tw-1], device=local_kp.device, dtype=local_kp.dtype)
        kp_vox_local = p01 * scale.view(1,1,1,3)                        # [B,T,K,3]

        # add tile origin
        kp_vox = kp_vox_local + starts.to(local_kp.dtype).view(1, T, 1, 3)

        # voxel coords -> normalized [-1,1] over **padded** grid
        full = torch.tensor([Dp-1, Hp-1, Wp-1], device=local_kp.device, dtype=local_kp.dtype)
        kp_glob = (kp_vox / full.view(1,1,1,3)) * (hi - lo) + lo
        return kp_glob  # [B,T,K,3]

    def encode_prior(self, points: torch.Tensor, mask_pc: Optional[torch.Tensor] = None, k: Optional[int] = None):
        """
        points: [B,N,3] in [-1,1]^3
        returns: kp [B, k, 3] in [-1,1], cov [B, k, 3,3] (if requested by ssm3d)
        """
        self._check_shapes(points, mask_pc)
        B, N, _ = points.shape

        # (1) voxelize (with_moments=True -> 7 channels)
        vox, _ = self.grid_voxelizer(points, mask=mask_pc, with_moments=True)      # [B,7,D,H,W]

        # (2) tile the voxel grid (pads as needed)
        tiles, meta = self.grid_voxelizer.vox_to_tiles(vox, tile_size=self.tile_size)   # tiles: [B,7,T,td,th,tw]
        B_, Cg, T, td, th, tw = tiles.shape
        tiles_bt = tiles.permute(0,2,1,3,4,5).reshape(B*T, Cg, td, th, tw)              # [B*T,7,td,th,tw]

        # (3) 3D encoder per tile -> logits [B*T, n_kp, d',h',w']
        logits = self.enc3d(tiles_bt)

        # (4) 3D spatial softmax per tile/channel -> local KPs (in [-1,1])
        kp_local, cov_local = self.ssm3d(logits, probs=False, variance=True)            # [B*T,n_kp,3], [B*T,n_kp,3,3]
        kp_local = kp_local.view(B, T, self.n_kp, 3)
        cov_local = cov_local.view(B, T, self.n_kp, 3, 3) if cov_local is not None else None

        # (5) map to global scene coordinates (like get_global_kp in 2D)
        kp_glob = self.get_global_kp_3d(kp_local, meta)                                 # [B,T,n_kp,3]

        # (6) flatten tiles -> [B, T*n_kp, ...]
        kp_all  = kp_glob.view(B, T*self.n_kp, 3)
        cov_all = cov_local.view(B, T*self.n_kp, 3, 3) if cov_local is not None else None
        

        # (7) select to target n_kp_prior (variance/random/first-k)
        K_total = T * self.n_kp

        print("kp all: ", kp_all.shape)
        if k is None: k = min(self.n_kp_prior, K_total)
        if k < K_total:
            if (self.filtering_heuristic == 'variance') and (cov_all is not None):
                tr = cov_all[..., 0,0] + cov_all[..., 1,1] + cov_all[..., 2,2]         # [B,K_total]
                _, idx = torch.topk(tr, k=k, dim=-1, largest=False, sorted=True)       # lower trace = sharper
            elif self.filtering_heuristic == 'random':
                idx = torch.stack([torch.randperm(K_total, device=kp_all.device)[:k] for _ in range(B)], dim=0)
            else:
                idx = torch.arange(k, device=kp_all.device)[None, :].expand(B, -1)
            b = torch.arange(B, device=kp_all.device)[:, None]
            kp_all  = kp_all[b, idx]
            if cov_all is not None:
                cov_all = cov_all[b, idx]

        # debug prints (like your 2D version)
        # print("TOTAL KEYPOINTS (tiled 3D):", kp_all.shape)
        return kp_all, cov_all


    def forward(self, points: torch.Tensor, mask_pc: Optional[torch.Tensor] = None):
        return self.encode_prior(points, mask_pc=mask_pc, k=self.n_kp_prior)
