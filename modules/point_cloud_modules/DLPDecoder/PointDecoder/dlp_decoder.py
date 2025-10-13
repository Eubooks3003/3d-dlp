import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

# If you keep your existing modules in the same package:
from modules.point_cloud_modules.DLPDecoder.DeformationDecoder.point_cloud_decoder import PointCloudDecoderParticles
from modules.point_cloud_modules.DLPDecoder.bg_decoder import BgDecoderPC
from modules.point_cloud_modules.DLPDecoder.PointDecoder.point_set_spawner import PointSetSpawner


# ----------------------------- #
# Utility: axis-angle -> matrix #
# ----------------------------- #
def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """
    aa: [B,K,3] axis-angle
    returns R: [B,K,3,3]
    """
    eps = 1e-9
    angle = torch.linalg.norm(aa, dim=-1, keepdim=True).clamp_min(eps)  # [B,K,1]
    axis = aa / angle
    x, y, z = axis.unbind(-1)  # each [B,K]
    c = torch.cos(angle)[..., 0]
    s = torch.sin(angle)[..., 0]
    C = 1.0 - c

    R = torch.stack([
        c + x * x * C,     x * y * C - z * s,  x * z * C + y * s,
        y * x * C + z * s, c + y * y * C,      y * z * C - x * s,
        z * x * C - y * s, z * y * C + x * s,  c + z * z * C
    ], dim=-1).reshape(*aa.shape[:2], 3, 3)
    return R


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
                 init_zero_bias=True,
                 init_conv_layers=True,
                 init_conv_fg_std=0.02,
                 init_conv_bg_std=0.005,

                 # ---- PC-specific knobs ----
                 points_per_obj=256,
                 points_bg=2048,
                 predict_obj_color=True,
                 predict_bg_color=True,
                 color_activation="sigmoid",   # {'sigmoid','tanh'}
                 use_context=False,
                 sphere_sigma=0.06,
                 depth_blend="softmax",        # {'softmax','alpha'}
                 clamp_bounds=True,
                 bg_template="sphere",         # {'sphere','cube','gridxy'}
                 learn_bg_template=False,

                 # ---- NEW: choose geometry mode ----
                 decoder_point_mode: str = "spawn",     # {'spawn','deform','hybrid'}
                 nsp_scales: int = 2,
                 points_per_scale: Tuple[int, ...] = (128, 256),
                 spawn_use_rotation: bool = True,
                 spawn_scale_activation: str = "sigmoid",
                 spawn_predict_color: bool = False):
        super().__init__()

        # Store args
        self.image_size = image_size
        self.feature_map_size = image_size
        self.n_kp_enc = n_kp_enc
        self.dropout = dropout
        self.learned_feature_dim = learned_feature_dim
        self.learned_bg_feature_dim = learned_bg_feature_dim
        assert learned_feature_dim > 0, "learned_feature_dim must be > 0"
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

        self.init_zero_bias = init_zero_bias
        self.init_conv_layers = init_conv_layers
        self.init_conv_fg_std = init_conv_fg_std
        self.init_conv_bg_std = init_conv_bg_std

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

        # Mode selection
        assert decoder_point_mode in {"spawn", "deform", "hybrid"}
        self.decoder_point_mode = decoder_point_mode
        self.nsp_scales = nsp_scales
        self.points_per_scale = tuple(points_per_scale)

        # ---- Object decoders ----
        self.particle_dec_deform = None
        self.particle_dec_spawn  = None
        print("DECODER MODE: ", decoder_point_mode)
        if decoder_point_mode in {"deform", "hybrid"}:
            # Template + deformation head (your existing path)
            self.particle_dec_deform = PointCloudDecoderParticles(
                features_dim=learned_feature_dim,
                points_per_obj=points_per_obj,
                template="sphere",
                learn_template=False,
                predict_color=(predict_obj_color and color_activation in {"sigmoid","tanh"}),
                hidden=mlp_hidden_dim,
                use_rotation=True,
                scale_activation="sigmoid",
                color_activation=color_activation,
            )

        if decoder_point_mode in {"spawn", "hybrid"}:
            # Point spawner (Point-NSP style)
            self.particle_dec_spawn = PointSetSpawner(
                features_dim=learned_feature_dim,
                n_scales=self.nsp_scales,
                points_per_scale=self.points_per_scale,
                hidden=mlp_hidden_dim,
                predict_color=(spawn_predict_color and color_activation in {"sigmoid","tanh"}),
                use_rotation=spawn_use_rotation,
                scale_activation=spawn_scale_activation,
                color_activation=color_activation,
            )

        # ---- Background decoder ----
        self.bg_dec = BgDecoderPC(
            learned_bg_feature_dim=learned_bg_feature_dim,
            points_bg=self.points_bg,
            hidden=mlp_hidden_dim,
            predict_color=(cdim >= 3 and predict_bg_color),
            use_context=(context_dim > 0 and decode_with_ctx),
            context_dim=(context_dim if (context_dim > 0 and decode_with_ctx) else 0),
            template=self.bg_template,
            learn_template=self.learn_bg_template,
            color_activation=("tanh" if normalize_rgb else "sigmoid"),
        )

        # Init
        self.init_weights()

    # -------------------- init --------------------
    def init_weights(self):
        def _init_module_linear_gauss(mod: nn.Module, std: float):
            for mm in mod.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.normal_(mm.weight, 0.0, std)
                    if self.init_zero_bias and mm.bias is not None:
                        nn.init.constant_(mm.bias, 0.0)

        # FG/BG inits
        if self.particle_dec_deform is not None:
            _init_module_linear_gauss(self.particle_dec_deform, self.init_conv_fg_std)
            if hasattr(self.particle_dec_deform, "init_weights"):
                self.particle_dec_deform.init_weights()

        if self.particle_dec_spawn is not None:
            _init_module_linear_gauss(self.particle_dec_spawn, self.init_conv_fg_std)

        _init_module_linear_gauss(self.bg_dec, self.init_conv_bg_std)
        if hasattr(self.bg_dec, "init_weights"):
            self.bg_dec.init_weights()

    # ----------------- object decode -----------------
    def decode_objects_pc(self, z_kp, z_features, obj_on, z_scale=None, z_depth=None, kp_mask=None):
        if obj_on is not None and obj_on.dim()==2:
            obj_on = obj_on.unsqueeze(-1)  # [B,K,1]

        w = (kp_mask.float().unsqueeze(-1) if kp_mask is not None else None)      # [B,K,1]
        if w is None and obj_on is not None: w = obj_on
        elif w is not None and obj_on is not None: w = w * obj_on

        # route to spawn/deform as before
        if self.decoder_point_mode == "spawn":
            out = self.particle_dec_spawn(z_pos=z_kp, z_scale=z_scale, z_feat=z_features,
                                        z_depth=z_depth, z_obj_on=w)
        elif self.decoder_point_mode == "deform":
            out = self.particle_dec_deform(z_pos=z_kp, z_scale=z_scale, z_feat=z_features,
                                        z_depth=z_depth, z_obj_on=w)
        else:
            out = self.particle_dec_spawn(z_pos=z_kp, z_scale=z_scale, z_feat=z_features,
                                        z_depth=z_depth, z_obj_on=w)

        points_obj = out["points_obj"]                        # [B,K,M,3]
        rgb_perobj = out.get("rgb_obj", None)                 # [B,K,M,3] or None

        # If your heads don’t already absorb w, apply it here for safety in losses:
        # keep coordinates as-is; carry w out so losses can weight them.
        weights = out.get("obj_weights", w if w is not None else torch.ones_like(points_obj[..., :1, 0:1]))
        rgb_obj  = None
        if rgb_perobj is not None:
            B, K, M, _ = rgb_perobj.shape
            rgb_obj = rgb_perobj.reshape(B, K*M, 3)

        B, K, M, _ = points_obj.shape
        pts_flat = points_obj.reshape(B, K*M, 3)

        return {"points_obj": points_obj,
                "points_obj_flat": pts_flat,
                "rgb_obj_flat": rgb_obj,
                "obj_weights": weights}  # [B,K,1]


    # ----------------- background decode -----------------
    def decode_bg_pc(self, z_bg_features):
        return self.bg_dec(z_bg_features)  # {'bg_points':[B,M_bg,3], 'bg_rgb': Optional[B,M_bg,3]}

    # ----------------- full decode -----------------
    def decode_all(self, z, z_scale, z_features, obj_on, z_depth, z_bg_features, kp_mask, warmup: bool = False):
        # Objects
        obj = self.decode_objects_pc(
            z_kp=z, z_features=z_features, obj_on=obj_on, z_scale=z_scale, z_depth=z_depth,  kp_mask=kp_mask
        )
        pts_obj_flat = obj["points_obj_flat"]   # [B, K*M, 3]
        rgb_obj_flat = obj["rgb_obj_flat"]      # [B, K*M, 3] or None

        # Background
        bg = self.decode_bg_pc(z_bg_features)
        pts_bg = bg["bg_points"]                # [B, M_bg, 3]
        rgb_bg = bg.get("bg_rgb", None)         # [B, M_bg, 3] or None

        # Scene concat
        pts_scene = torch.cat([pts_bg, pts_obj_flat], dim=1)  # [B, M_total, 3]
        if (rgb_bg is not None) and (rgb_obj_flat is not None):
            rgb_scene = torch.cat([rgb_bg, rgb_obj_flat], dim=1)
        else:
            rgb_scene = None

        return {
            "points_scene": pts_scene,
            "points_bg":    pts_bg,
            "points_obj":   obj["points_obj"],   # [B,K,M,3]
            "rgb_scene":    rgb_scene,
            "rgb_bg":       rgb_bg,
            "rgb_obj":      rgb_obj_flat,        # flattened K*M to match scene concat
            "obj_weights":  obj["obj_weights"],  # [B,K,1]
        }

    def forward(self, z, z_scale, z_features, obj_on, z_depth, z_bg_features, kp_mask, warmup: bool = False):
        return self.decode_all(z, z_scale, z_features, obj_on, z_depth, z_bg_features, kp_mask, warmup)