import torch
import torch.nn as nn
from typing import Optional

from modules.point_cloud_modules.prior_encoder.grid_voxelizer import GridVoxelizer
from modules.point_cloud_modules.vision_modules_3d import Encoder3D
from modules.point_cloud_modules.prior_encoder.SSM3D import AlternativeSpatialSoftmax3D


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
                 out_feat=1,                 # channels emitted by GridVoxelizer
                 base_ch=32, ch_mult=(1, 2, 3), num_res_blocks=2,
                 use_resblock=True, use_attention=False, cnn_mid_blocks=False,
                 n_kp_prior=64,
                 kp_range=(-1., 1.), temperature=1.0,
                 filtering_heuristic='none',  # ['variance','random','none']
                 init_zero_bias=True, init_conv_layers=True, init_conv_std=0.02):
        super().__init__()

        assert filtering_heuristic in ('variance', 'random', 'none')
        self.grid = tuple(grid)
        self.out_feat = int(out_feat)
        self.n_kp_prior = int(n_kp_prior)
        self.kp_range = kp_range
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
                               out_channels=self.n_kp_prior)

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

    def encode_prior(self, points: torch.Tensor, mask_pc: Optional[torch.Tensor] = None,
                 k: Optional[int] = None):
        """
        points:  [B, N, 3]
        mask_pc: [B, N] (1 valid, 0 pad) or None
        """
        self._check_shapes(points, mask_pc)
        B, N, _ = points.shape

        # (1) voxelize -> [B, Cg, D, H, W]
        vox, _ = self.grid_voxelizer(points, mask=mask_pc)

        # (2) 3D conv encoder -> [B, K, D', H', W']
        logits = self.enc3d(vox)

        # (3) 3D spatial softmax (per channel) -> means (& cov)
        kp, cov = self.ssm3d(logits, probs=False, variance=True)   # [B,K,3], [B,K,3,3]

        # (4) optional filtering (subset of K)
        if k is None:
            k = self.n_kp_prior
        if k < self.n_kp_prior:
            if self.filtering_heuristic == 'variance':
                tr = cov[..., 0, 0] + cov[..., 1, 1] + cov[..., 2, 2]     # [B,K]
                _, idx = torch.topk(tr, k=k, dim=-1, largest=False, sorted=True)
            elif self.filtering_heuristic == 'random':
                idx = torch.stack([torch.randperm(self.n_kp_prior, device=kp.device)[:k]
                                   for _ in range(B)], dim=0)            # [B,k]
            else:  # 'none' -> take first k channels
                idx = torch.arange(k, device=kp.device)[None, :].expand(B, -1)

            b = torch.arange(B, device=kp.device)[:, None]
            kp  = kp[b, idx]
            cov = cov[b, idx]

        return kp, cov

    def forward(self, points: torch.Tensor, mask_pc: Optional[torch.Tensor] = None):
        return self.encode_prior(points, mask_pc=mask_pc, k=self.n_kp_prior)
