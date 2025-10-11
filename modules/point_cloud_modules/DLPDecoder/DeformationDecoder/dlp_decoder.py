import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from modules.point_cloud_modules.DLPDecoder.point_cloud_decoder import PointCloudDecoderParticles
from modules.point_cloud_modules.DLPDecoder.bg_decoder import BgDecoderPC

class DLPDecoder(nn.Module):
    def __init__(self,
                 # ---- generic / legacy args you already pass in ----
                 cdim=3,
                 image_size=64,
                 pad_mode='replicate',
                 dropout=0.0,
                 normalize_rgb=False,

                 # feature dims
                 learned_feature_dim=128,
                 learned_bg_feature_dim=128,
                 anchor_s=0.25,
                 n_kp_enc=16,
                 context_dim=0,

                 # (2D-era switches kept for compatibility; no CNNs in PC path)
                 use_resblock=True,
                 timestep_horizon=1,
                 decode_with_ctx=False,
                 cnn_mid_blocks=False,
                 mlp_hidden_dim=256,

                 # ---- init knobs (we KEEP these) ----
                 init_zero_bias=True,      # zero bias for Linear (and Conv if any)
                 init_conv_layers=True,    # (kept; no convs in PC path, but submods may use)
                 init_conv_fg_std=0.02,    # use as std for OBJ MLPs
                 init_conv_bg_std=0.005,   # use as std for BG MLPs

                 # ---- new PC-specific knobs ----
                 points_per_obj=256,       # M_obj
                 points_bg=2048,           # M_bg
                 predict_obj_color=True,
                 predict_bg_color=True,
                 color_activation="sigmoid",   # {'sigmoid','tanh'}
                 use_context=False,
                 sphere_sigma=0.06,
                 depth_blend="softmax",        # {'softmax','alpha'}
                 clamp_bounds=True,
                 bg_template="sphere",         # {'sphere','cube','gridxy'}
                 learn_bg_template=False):
        super(DLPDecoder, self).__init__()

        # ---- store legacy/generic attributes (safe even if unused in PC path) ----
        self.image_size = image_size
        self.feature_map_size = image_size
        self.n_kp_enc = n_kp_enc
        self.dropout = dropout
        self.learned_feature_dim = learned_feature_dim
        self.learned_bg_feature_dim = learned_bg_feature_dim
        assert learned_feature_dim > 0, "learned_feature_dim must be > 0"
        self.anchor_s = anchor_s
        self.context_dim = context_dim
        self.obj_patch_size = np.round(anchor_s * (image_size - 1)).astype(int)  # kept for BC
        self.cdim = cdim
        self.use_resblock = use_resblock
        self.decode_with_ctx = decode_with_ctx
        self.normalize_rgb = normalize_rgb
        self.timestep_horizon = (timestep_horizon + 1) if timestep_horizon > 1 else 1
        self.cnn_mid_blocks = cnn_mid_blocks
        self.mlp_hidden_dim = mlp_hidden_dim

        # ---- init knobs (we reuse them for MLPs in PC decoders) ----
        self.init_zero_bias = init_zero_bias
        self.init_conv_layers = init_conv_layers
        self.init_conv_fg_std = init_conv_fg_std   # used as std for OBJ MLP weights
        self.init_conv_bg_std = init_conv_bg_std   # used as std for BG MLP weights

        # ---- new PC-specific attributes ----
        self.points_per_obj   = points_per_obj
        self.points_bg        = points_bg
        self.predict_obj_color = predict_obj_color
        self.predict_bg_color  = predict_bg_color
        self.color_activation  = color_activation
        self.use_context       = use_context and (context_dim > 0)
        self.sphere_sigma      = sphere_sigma
        self.depth_blend       = depth_blend
        self.clamp_bounds      = clamp_bounds
        self.bg_template       = bg_template
        self.learn_bg_template = learn_bg_template

        # object decoder
        self.particle_dec = PointCloudDecoderParticles(
            features_dim=learned_feature_dim,
            points_per_obj=256,
            template="sphere",
            learn_template=False,
            predict_color=False,     # flip to True if you have color supervision
            hidden=256,
            use_rotation=True,
        )

        # bg decoder
        self.bg_dec = BgDecoderPC(
            learned_bg_feature_dim=learned_bg_feature_dim,
            points_bg=self.points_bg,                              # e.g., from config (2048, 4096, …)
            hidden=mlp_hidden_dim,
            predict_color=(cdim >= 3),
            use_context=(context_dim > 0 and decode_with_ctx),
            context_dim=(context_dim if (context_dim > 0 and decode_with_ctx) else 0),
            template="sphere",                                     # or "cube" / "gridxy"
            learn_template=False,
            color_activation=("tanh" if normalize_rgb else "sigmoid"),
        )
        # self.num_bg_upsample = self.bg_dec.num_bg_upsample
        # ---- initialize weights ----
        self.init_weights()

    def init_weights(self):
        """
        Initialization policy:
          - Use init_zero_bias to zero all Linear biases.
          - Use init_conv_fg_std as Normal std for OBJ decoder MLP weights.
          - Use init_conv_bg_std as Normal std for BG decoder MLP weights.
          - If your submodules expose their own init, call them too.
        """
        # zero biases and normal-init weights in a module-aware way
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # identify if this linear belongs to obj or bg by parent path
                # (simple heuristic: check module name string)
                name = m.__class__.__name__.lower()
                # default to obj std; we'll override if we detect BG module
                std = self.init_conv_fg_std

                # If module is under bg_dec, use bg std
                # (PyTorch doesn't give parent easily; rely on attribute presence)
                # Safer approach: switch on current outer loop context:
                pass  # we handle per-submodule below

                if self.init_zero_bias and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        # submodule-specific inits (cleaner control of stds)
        def _init_module_linear_gauss(mod: nn.Module, std: float):
            for mm in mod.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.normal_(mm.weight, 0.0, std)
                    if self.init_zero_bias and mm.bias is not None:
                        nn.init.constant_(mm.bias, 0.0)

        # initialize object MLPs with "fg std"
        _init_module_linear_gauss(self.particle_dec, self.init_conv_fg_std)
        # initialize background MLPs with "bg std"
        _init_module_linear_gauss(self.bg_dec, self.init_conv_bg_std)

        # if submodules have their own init hooks, call them
        if hasattr(self.particle_dec, "init_weights"):
            self.particle_dec.init_weights()
        if hasattr(self.bg_dec, "init_weights"):
            self.bg_dec.init_weights()
    def decode_objects_pc(self, z_kp, z_features, obj_on, z_scale=None, z_depth=None):
        """
        Returns per-object point clouds (optionally colored) and a flat concat.
        Expected output from self.obj_pc_dec:
        - points_obj: [B, K, M, 3]
        - rgb_obj:    [B, K, M, 3] or None
        - obj_weights:[B, K, 1]    (from obj_on)
        """
        # obj_on: [B,K] or [B,K,1] -> [B,K,1]
        if obj_on.dim() == 2:
            obj_on = obj_on.unsqueeze(-1)

        out = self.particle_dec(
            z_pos=z_kp,              # [B,K,3]
            z_scale=z_scale,        # [B,K,3] or [B,K,1]
            z_feat=z_features,      # [B,K,F]
            z_depth=z_depth,        # [B,K,1] or None
            z_obj_on=obj_on           # [B,K,1]
        )

        points_obj = out["points_obj"]      # [B,K,M,3]
        rgb_obj    = out.get("rgb_obj", None)   # [B,K,M,3] or None
        weights    = out.get("obj_weights", obj_on)  # [B,K,1]

        # Flatten object dimension for scene concat
        B, K, M, _ = points_obj.shape
        points_obj_flat = points_obj.reshape(B, K * M, 3)
        if rgb_obj is not None:
            rgb_obj_flat = rgb_obj.reshape(B, K * M, 3)
        else:
            rgb_obj_flat = None

        return {
            "points_obj": points_obj,               # [B,K,M,3]
            "points_obj_flat": points_obj_flat,     # [B,K*M,3]
            "rgb_obj_flat": rgb_obj_flat,           # [B,K*M,3] or None
            "obj_weights": weights,                 # [B,K,1]
        }
    def decode_bg_pc(self, z_bg_features):
        bg = self.bg_dec(z_bg_features)      # BgDecoderPC forward
        # bg['bg_points']: [B, M_bg, 3]
        # bg['bg_rgb']:    [B, M_bg, 3] or None
        return bg
    
    def decode_all(self, z, z_scale, z_features, obj_on, z_depth, z_bg_features, warmup=False):
        # z: [B,K,3], z_scale: [B,K,?], obj_on: [B,K] or [B,K,1], z_depth: [B,K,1]
        if obj_on is not None and obj_on.dim() == 2:
            obj_on = obj_on.unsqueeze(-1)  # [B,K,1]

        obj = self.decode_objects_pc(
            z_kp=z, z_features=z_features, obj_on=obj_on,
            z_scale=z_scale, z_depth=z_depth
        )
        pts_obj_flat = obj["points_obj_flat"]   # [B, K*M, 3]
        rgb_obj_flat = obj["rgb_obj_flat"]      # [B, K*M, 3] or None

        bg = self.decode_bg_pc(z_bg_features)   # {'bg_points':[B,M_bg,3], 'bg_rgb': Optional[B,M_bg,3]}
        pts_bg  = bg["bg_points"]
        rgb_bg  = bg.get("bg_rgb", None)

        pts_scene = torch.cat([pts_bg, pts_obj_flat], dim=1)  # [B, M_bg + K*M, 3]
        if (rgb_bg is not None) and (rgb_obj_flat is not None):
            rgb_scene = torch.cat([rgb_bg, rgb_obj_flat], dim=1)
        else:
            rgb_scene = None

        # points_scene: fused point cloud for the full reconstruction
        #               [B, M_total, 3] in scene coords (typically [-1,1]^3).
        #               Usually a concat of background + all object points (after transforms).

        # points_bg:    background-only point cloud [B, M_bg, 3], decoded from z_bg_features
        #               (no object compositing). Useful for ablations/visualizing bg alone.

        # points_obj:   per-object point clouds AFTER applying each particle’s pose/scale
        #               [B, K, M_obj, 3]. These are in the same global frame as points_bg.

        # rgb_scene:    per-point colors for points_scene [B, M_total, C].
        #               C is typically 3 (RGB in [0,1] or [-1,1] depending on normalize_rgb).

        # rgb_bg:       per-point colors for points_bg [B, M_bg, C]. Background colors only.

        # rgb_obj:      per-object, per-point colors flattened across objects
        #               [B, K * M_obj, C] (or [B, K, M_obj, C] before flatten).
        #               Matches points_obj ordering; use for per-object coloring.

        # obj_weights:  soft per-object visibility/importance weights
        #               [B, K, 1], derived from obj_on/depth during compositing.
        #               Helpful for diagnostics, pruning, or weighted metrics.

        return {
            "points_scene": pts_scene,
            "points_bg":    pts_bg,
            "points_obj":   obj["points_obj"],  # [B,K,M,3]
            "rgb_scene":    rgb_scene,
            "rgb_bg":       rgb_bg,
            "rgb_obj":      rgb_obj_flat,
            "obj_weights":  obj["obj_weights"], # [B,K,1]
        }


    def forward(self, z, z_scale, z_features, obj_on, z_depth, z_bg_features,
                warmup=False):
        return self.decode_all(z, z_scale, z_features, obj_on, z_depth, z_bg_features, warmup)

