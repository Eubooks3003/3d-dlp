import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from torch.cuda.amp import autocast

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
        # object decoder (now returns alpha+payload directly)
        obj_res_from_fc=4,
        obj_ch_mult=(1, 2, 3),
        obj_base_ch=32,
        num_res_blocks=2,
        use_resblock=True,
        # background decoder
        bg_res_from_fc=4,
        bg_ch_mult=(1, 2, 3),
        bg_base_ch=32,
        bg_final_cnn_ch=32,
        # streaming / precision
        decode_k_chunk=1,
        use_amp=True,
        autocast_dtype="bfloat16",  # "bfloat16" | "float16" | None
        grid_sample_mode="nearest",
        # init
        init_zero_bias=True,
        init_conv_layers=True,
        init_conv_fg_std=0.02,
        init_conv_bg_std=0.005,
        use_checkpoint=False
    ):
        super().__init__()
        self.grid_dhw = tuple(int(x) for x in grid_dhw)
        self.D, self.H, self.W = self.grid_dhw
        self.n_kp_enc = int(n_kp_enc)
        self.anchor_s = float(anchor_s)
        self.cdim_out = int(cdim_out)

        # memory/precision knobs
        self.decode_k_chunk = int(decode_k_chunk)
        self.use_amp = bool(use_amp)
        self.autocast_dtype = (
            torch.bfloat16 if autocast_dtype == "bfloat16"
            else (torch.float16 if autocast_dtype == "float16" else None)
        )
        self.grid_sample_mode = grid_sample_mode
        self.use_checkpoint = bool(use_checkpoint)

        # keep init flags
        self.init_zero_bias = bool(init_zero_bias)
        self.init_conv_layers = bool(init_conv_layers)

        # --- Object decoder (outputs [alpha, payload]) ---
        self.obj_dec = ObjectDecoderCNN3D(
            patch_size=(
                max(2, int(self.anchor_s * (self.D - 1))),
                max(2, int(self.anchor_s * (self.H - 1))),
                max(2, int(self.anchor_s * (self.W - 1))),
            ),
            num_chans_out=self.cdim_out,            # payload channels; +1 alpha is internal to ObjectDecoderCNN3D
            bottleneck_size=learned_feature_dim,
            base_ch=obj_base_ch,
            ch_mult=obj_ch_mult,
            num_res_blocks=num_res_blocks,
            res_from_fc=obj_res_from_fc,
            use_resblock=use_resblock,
            # the class itself ignores / safely handles final_cnn_ch when emitting alpha+payload
            init_zero_bias=init_zero_bias,
            init_conv_layers=init_conv_layers,
            init_conv_fg_std=init_conv_fg_std,
        )

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
        for sub in [self.obj_dec, self.bg_dec]:
            if hasattr(sub, "init_weights"):
                sub.init_weights()

    @property
    def info(self) -> str:
        ps = getattr(self.obj_dec, "ps", ("?", "?", "?"))
        return (f"DLPDecoder3D(grid={self.grid_dhw}, cdim_out={self.cdim_out}, "
                f"n_kp={self.n_kp_enc}, anchor_s={self.anchor_s:.3f}, "
                f"obj_patch={ps}, chunk_k={self.decode_k_chunk}, amp={self.use_amp}, "
                f"autocast_dtype={'None' if self.autocast_dtype is None else str(self.autocast_dtype).split('.')[-1]})")

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

    def _obj_dec_run(self, zf_chunk):
        if self.use_checkpoint and self.training and zf_chunk.requires_grad:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self.obj_dec, zf_chunk, use_reentrant=False)
        return self.obj_dec(zf_chunk)

    @staticmethod
    def _split_alpha_payload(x):
        # x: [B*, 1+cdim_out, psd, psh, psw]
        return torch.sigmoid(x[:, :1]), x[:, 1:]

    def _stream_composite_3d(
        self, z, z_scale, z_features, obj_on, z_depth,
        *, out_dhw=None, chunk_k=None, use_amp=None, mode=None, eps=1e-6
    ):
        D, H, W = (self.grid_dhw if out_dhw is None else tuple(map(int, out_dhw)))
        B, K, _ = z.shape
        device = z.device

        if obj_on.dim() == 3: obj_on = obj_on.squeeze(-1)
        if z_depth.dim() == 3: z_depth = z_depth.squeeze(-1)

        chunk_k = self.decode_k_chunk if chunk_k is None else int(chunk_k)
        use_amp = self.use_amp if use_amp is None else bool(use_amp)
        mode = self.grid_sample_mode if mode is None else mode

        # accumulators in fp32
        den = torch.zeros(B, 1, D, H, W, device=device, dtype=torch.float32)
        comp = torch.zeros(B, self.cdim_out, D, H, W, device=device, dtype=torch.float32)
        a_imp_sum = torch.zeros_like(den)

        sig_neg_depth = torch.sigmoid(-z_depth).to(torch.float32)

        ac_dtype = self.autocast_dtype if use_amp else None
        ac = (lambda: autocast(enabled=True, dtype=ac_dtype)) if ac_dtype is not None else nullcontext

        # PASS 1: denominator
        for start in range(0, K, chunk_k):
            end = min(K, start + chunk_k)
            kk = end - start

            zf_chunk = z_features[:, start:end, :].reshape(B * kk, -1)

            with ac():
                out = self._obj_dec_run(zf_chunk)            # [B*kk, 1+cdim_out, psd, psh, psw]
                alpha_loc, _ = self._split_alpha_payload(out)
                grid = self._make_sampling_grid_3d(z[:, start:end], z_scale[:, start:end], (D, H, W))
                a_g = self._grid_sample3d(alpha_loc, grid, mode=mode)  # [B*kk, 1, D, H, W]

            a_g = a_g.view(B, kk, 1, D, H, W).to(torch.float32)
            w_k = (obj_on[:, start:end] * sig_neg_depth[:, start:end]).view(B, kk, 1, 1, 1, 1).to(torch.float32)
            den.add_((a_g * w_k).sum(dim=1))

            del zf_chunk, out, alpha_loc, grid, a_g, w_k

        den = den + eps

        # PASS 2: numerator
        for start in range(0, K, chunk_k):
            end = min(K, start + chunk_k)
            kk = end - start

            zf_chunk = z_features[:, start:end, :].reshape(B * kk, -1)

            with ac():
                out = self._obj_dec_run(zf_chunk)
                alpha_loc, payload_loc = self._split_alpha_payload(out)
                grid = self._make_sampling_grid_3d(z[:, start:end], z_scale[:, start:end], (D, H, W))
                a_g = self._grid_sample3d(alpha_loc,   grid, mode=mode)
                y_g = self._grid_sample3d(payload_loc, grid, mode=mode)

            a_g = a_g.view(B, kk, 1, D, H, W).to(torch.float32)
            y_g = y_g.view(B, kk, self.cdim_out, D, H, W).to(torch.float32)

            w_base = (obj_on[:, start:end] * sig_neg_depth[:, start:end]).view(B, kk, 1, 1, 1, 1).to(torch.float32)
            unnorm = a_g * w_base
            w_norm = unnorm / den.unsqueeze(1)

            comp.add_((y_g * w_norm).sum(dim=1))
            a_imp_sum.add_((a_g * w_norm).sum(dim=1))

            del zf_chunk, out, alpha_loc, payload_loc, grid, a_g, y_g, w_base, unnorm, w_norm

        bg_mask = (1.0 - a_imp_sum).clamp_(0, 1)
        return comp, bg_mask

    def decode_objects(self, z, z_scale, z_features, obj_on, z_depth):
        comp_obj, bg_mask = self._stream_composite_3d(
            z, z_scale, z_features, obj_on, z_depth,
            out_dhw=self.grid_dhw,
            chunk_k=self.decode_k_chunk,
            use_amp=self.use_amp,
            mode=self.grid_sample_mode
        )
        return {"comp_obj": comp_obj, "bg_mask": bg_mask}

    def forward(self, z, z_scale, z_features, obj_on, z_depth, z_bg_features):
        obj = self.decode_objects(z, z_scale, z_features, obj_on, z_depth)
        comp_obj = obj["comp_obj"]
        bg_mask = obj["bg_mask"]
        bg = self.bg_dec(z_bg_features)
        rec = bg_mask * bg + comp_obj
        return {"rec": rec, "bg_rec": bg, "obj_comp": comp_obj, "bg_mask": bg_mask}
