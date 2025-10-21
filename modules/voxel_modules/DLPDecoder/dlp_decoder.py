import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.voxel_modules.DLPDecoder.bg_decoder import BgDecoder3D
from modules.voxel_modules.DLPDecoder.object_decoder import ObjectDecoderCNN3D

class DLPDecoder3D(nn.Module):
    def __init__(
        self,
        grid_dhw=(48, 48, 48),
        cdim_out=1,
        learned_feature_dim=16,
        learned_bg_feature_dim=16,
        n_kp_enc=16,
        anchor_s=0.25,
        # object decoder
        obj_res_from_fc=4,
        obj_ch_mult=(1, 2, 3),
        obj_base_ch=32,
        obj_final_cnn_ch=128,   # <- set to the feature channels your ObjectDecoderCNN3D returns
        num_res_blocks=2,
        use_resblock=True,
        # background decoder
        bg_res_from_fc=4,
        bg_ch_mult=(1, 2, 3),
        bg_base_ch=32,
        bg_final_cnn_ch=32,
        # init
        init_zero_bias=True,
        init_conv_layers=True,
        init_conv_fg_std=0.02,
        init_conv_bg_std=0.005,
    ):
        super().__init__()
        self.grid_dhw = tuple(int(x) for x in grid_dhw)
        self.D, self.H, self.W = self.grid_dhw
        self.n_kp_enc = int(n_kp_enc)
        self.anchor_s = float(anchor_s)
        self.cdim_out = int(cdim_out)

        # --- Object feature decoder (produces feature volume, NOT alpha/payload) ---
        self.obj_feat_ch = int(obj_final_cnn_ch)  # <- single source of truth for feature channels
        self.obj_dec = ObjectDecoderCNN3D(
            patch_size=(
                max(2, int(anchor_s * (self.D - 1))),
                max(2, int(anchor_s * (self.H - 1))),
                max(2, int(anchor_s * (self.W - 1))),
            ),
            num_chans_out=self.obj_feat_ch,      # decoder outputs feature channels
            bottleneck_size=learned_feature_dim,
            base_ch=obj_base_ch,
            ch_mult=obj_ch_mult,
            num_res_blocks=num_res_blocks,
            res_from_fc=obj_res_from_fc,
            use_resblock=use_resblock,
            final_cnn_ch=self.obj_feat_ch,
            init_zero_bias=init_zero_bias,
            init_conv_layers=init_conv_layers,
            init_conv_fg_std=init_conv_fg_std,
        )

        # Heads to convert features -> alpha / payload (fixed, NOT lazy)
        self.obj_alpha_head   = nn.Conv3d(self.obj_feat_ch, 1,            kernel_size=1, bias=True)
        self.obj_payload_head = nn.Conv3d(self.obj_feat_ch, self.cdim_out, kernel_size=1, bias=True)

        # --- Background decoder ---
        self.bg_dec = BgDecoder3D(
            grid_dhw=self.grid_dhw,
            cdim_out=self.cdim_out,
            learned_bg_feature_dim=learned_bg_feature_dim,
            base_ch=bg_base_ch,
            ch_mult=bg_ch_mult,
            num_res_blocks=num_res_blocks,
            res_from_fc=bg_res_from_fc,
            use_resblock=use_resblock,
            final_cnn_ch=bg_final_cnn_ch,
            init_zero_bias=init_zero_bias,
            init_conv_layers=init_conv_layers,
            init_conv_bg_std=init_conv_bg_std,
        )

        self.init_weights()

    def init_weights(self):
        # submodules
        for sub in [self.obj_dec, self.bg_dec]:
            if hasattr(sub, "init_weights"):
                sub.init_weights()
        # heads: zero-bias if requested
        for m in [self.obj_alpha_head, self.obj_payload_head]:
            if hasattr(self, "init_zero_bias") and self.init_zero_bias and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    @property
    def info(self) -> str:
        ps = getattr(self.obj_dec, "ps", ("?", "?", "?"))
        return (f"DLPDecoder3D(grid={self.grid_dhw}, cdim_out={self.cdim_out}, "
                f"n_kp={self.n_kp_enc}, anchor_s={self.anchor_s:.3f}, "
                f"obj_patch={ps}, obj_feat_ch={self.obj_feat_ch})")

    @torch.no_grad()
    def _make_sampling_grid_3d(self, z, z_scale, out_dhw=None):
        if out_dhw is None:
            D, H, W = self.grid_dhw
        else:
            D, H, W = map(int, out_dhw)
        B, k, _ = z.shape
        device, dtype = z.device, z.dtype

        xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        zs = torch.linspace(-1, 1, D, device=device, dtype=dtype)
        zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
        g = torch.stack([xx, yy, zz], dim=-1).view(1, 1, D, H, W, 3)

        c = z.view(B, k, 1, 1, 1, 3)
        h = self.anchor_s * torch.sigmoid(z_scale).clamp_min(1e-6)
        h = h.view(B, k, 1, 1, 1, 3)
        grid = (g - c) / h
        return grid.view(B * k, D, H, W, 3)

    @staticmethod
    def _grid_sample3d(inp, grid, *, mode="nearest"):
        return F.grid_sample(inp, grid, mode=mode, padding_mode="zeros", align_corners=True)

    def _alpha_payload_from_feats(self, vol_feats: torch.Tensor):
        """
        vol_feats: [B*, C_feat==self.obj_feat_ch, psd, psh, psw]
        -> alpha_loc [B*, 1, psd, psh, psw], payload_loc [B*, cdim_out, psd, psh, psw]
        """
        # sanity check in debug mode (cheap)
        if vol_feats.shape[1] != self.obj_feat_ch:
            raise RuntimeError(
                f"Object feature channels mismatch: got {vol_feats.shape[1]}, "
                f"expected {self.obj_feat_ch}. Set obj_final_cnn_ch to your ObjectDecoderCNN3D output."
            )
        alpha_loc   = torch.sigmoid(self.obj_alpha_head(vol_feats))
        payload_loc = self.obj_payload_head(vol_feats)
        return alpha_loc, payload_loc

    def _stream_composite_3d(
        self, z, z_scale, z_features, obj_on, z_depth,
        *, out_dhw=None, chunk_k=2, use_amp=True, mode="nearest", eps=1e-6
    ):
        if out_dhw is None:
            D, H, W = self.grid_dhw
        else:
            D, H, W = map(int, out_dhw)

        B, K, _ = z.shape
        device = z.device
        if obj_on.dim() == 3: obj_on = obj_on.squeeze(-1)
        if z_depth.dim() == 3: z_depth = z_depth.squeeze(-1)

        den = torch.zeros(B, 1, D, H, W, device=device, dtype=z_features.dtype)
        sig_neg_depth = torch.sigmoid(-z_depth)

        # Pass 1: denominator
        for start in range(0, K, chunk_k):
            end = min(K, start + chunk_k)
            kk = end - start

            zf_chunk = z_features[:, start:end, :].reshape(B * kk, -1)
            feats = self.obj_dec(zf_chunk)                               # [B*kk, C_feat, psd, psh, psw]
            alpha_loc, _ = self._alpha_payload_from_feats(feats)         # [B*kk, 1, psd, psh, psw]

            grid = self._make_sampling_grid_3d(z[:, start:end], z_scale[:, start:end], (D, H, W))
            a_g = self._grid_sample3d(alpha_loc, grid, mode=mode)        # [B*kk, 1, D, H, W]
            a_g = a_g.view(B, kk, 1, D, H, W)

            w_k = (obj_on[:, start:end] * sig_neg_depth[:, start:end]).view(B, kk, 1, 1, 1, 1)
            den = den + (a_g * w_k).sum(dim=1)

            del zf_chunk, feats, alpha_loc, grid, a_g, w_k
            torch.cuda.empty_cache()

        den = den + eps

        # Pass 2: numerator
        comp = torch.zeros(B, self.cdim_out, D, H, W, device=device, dtype=z_features.dtype)
        a_imp_sum = torch.zeros(B, 1, D, H, W, device=device, dtype=z_features.dtype)

        for start in range(0, K, chunk_k):
            end = min(K, start + chunk_k)
            kk = end - start

            zf_chunk = z_features[:, start:end, :].reshape(B * kk, -1)
            feats = self.obj_dec(zf_chunk)
            alpha_loc, payload_loc = self._alpha_payload_from_feats(feats)

            grid = self._make_sampling_grid_3d(z[:, start:end], z_scale[:, start:end], (D, H, W))

            if use_amp:
                from torch.cuda.amp import autocast
                with autocast(enabled=True):
                    a_g = self._grid_sample3d(alpha_loc,   grid, mode=mode)
                    y_g = self._grid_sample3d(payload_loc, grid, mode=mode)
            else:
                a_g = self._grid_sample3d(alpha_loc,   grid, mode=mode)
                y_g = self._grid_sample3d(payload_loc, grid, mode=mode)

            a_g = a_g.view(B, kk, 1, D, H, W)
            y_g = y_g.view(B, kk, self.cdim_out, D, H, W)

            w_base = (obj_on[:, start:end] * sig_neg_depth[:, start:end]).view(B, kk, 1, 1, 1, 1)
            unnorm = a_g * w_base
            w_norm = unnorm / den.unsqueeze(1)

            comp += (y_g * w_norm).sum(dim=1)
            a_imp_sum += (a_g * w_norm).sum(dim=1)

            del zf_chunk, feats, alpha_loc, payload_loc, grid, a_g, y_g, w_base, unnorm, w_norm
            torch.cuda.empty_cache()

        bg_mask = (1.0 - a_imp_sum).clamp(0, 1)
        return comp, bg_mask

    def decode_objects(self, z, z_scale, z_features, obj_on, z_depth):
        comp_obj, bg_mask = self._stream_composite_3d(
            z, z_scale, z_features, obj_on, z_depth,
            out_dhw=self.grid_dhw, chunk_k=2, use_amp=True, mode="nearest"
        )
        return {"comp_obj": comp_obj, "bg_mask": bg_mask}

    def forward(self, z, z_scale, z_features, obj_on, z_depth, z_bg_features):
        obj = self.decode_objects(z, z_scale, z_features, obj_on, z_depth)
        comp_obj = obj["comp_obj"]
        bg_mask = obj["bg_mask"]
        bg = self.bg_dec(z_bg_features)
        rec = bg_mask * bg + comp_obj
        return {"rec": rec, "bg_rec": bg, "obj_comp": comp_obj, "bg_mask": bg_mask}
