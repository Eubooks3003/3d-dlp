import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.point_cloud_modules.DLPEncoder.particle_attributes_projection import ParticleAttributesProjection3D
from modules.point_cloud_modules.DLPEncoder.particle_self_attn_transformer import ParticleSelfAttTransformer
from modules.point_cloud_modules.DLPDecoder.particle_attribute_decoder import ParticleAttributeDecoder
from utils.util_func import reparameterize

class ParticleInteractionEncoder3D(nn.Module):
    def __init__(self,
                 n_kp_enc,
                 learned_feature_dim=16,
                 learned_bg_feature_dim=16,
                 projection_dim=128,
                 pte_layers=1, pte_heads=1,
                 hidden_dim=256,
                 temporal_interaction=False,          # single-frame PC by default
                 interaction_depth=False,
                 interaction_obj_on=False,
                 interaction_features=False,
                 with_bg=True,
                 use_pc_ctx=True,                    # was use_img_input; now PointNet ctx
                 ctx_type="pointnet",                # {"pointnet","none"}
                 obj_on_min=1e-4, obj_on_max=100.0,
                 features_dist='gauss',
                 n_fg_categories=8, n_fg_classes=4,
                 n_bg_categories=4, n_bg_classes=4,
                 add_particle_temp_embed=False,
                 particle_positional_embed=True,
                 particle_score=False,
                 use_z_orig=False,
                 embed_init_std=0.2,
                 dropout=0.0,
                 attn_norm_type='rms',
                 activation='gelu',
                 # init
                 init_zero_bias=True,
                 init_conv_layers=True,
                 init_conv_fg_std=0.02):
        super().__init__()

        self.n_kp_enc = n_kp_enc
        self.learned_feature_dim = learned_feature_dim
        self.learned_bg_feature_dim = learned_bg_feature_dim
        self.features_dist = features_dist
        self.n_fg_categories = n_fg_categories
        self.n_fg_classes = n_fg_classes
        self.n_bg_categories = n_bg_categories
        self.n_bg_classes = n_bg_classes

        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.temporal_interaction = temporal_interaction
        self.interaction_depth = interaction_depth
        self.interaction_obj_on = interaction_obj_on
        self.interaction_features = interaction_features
        self.with_bg = with_bg

        self.use_pc_ctx = use_pc_ctx and (ctx_type != "none")
        self.ctx_type = ctx_type

        self.obj_on_min = obj_on_min
        self.obj_on_max = obj_on_max
        self.add_particle_temp_embed = add_particle_temp_embed
        self.use_z_orig = use_z_orig
        self.particle_score = particle_score

        # init flags
        self.init_zero_bias = init_zero_bias
        self.init_conv_layers = init_conv_layers
        self.init_conv_fg_std = init_conv_fg_std

        # (optional) canonical anchors for z_orig
        if self.use_z_orig:
            self.register_buffer('particles_anchor', torch.zeros(1, 1, self.n_kp_enc))
        else:
            self.register_buffer('particles_anchor', torch.zeros(1, 1, 1))  # dummy

        # ----- token counts -----
        n_tokens = self.n_kp_enc
        if with_bg:
            n_tokens += 1
            self.bg_embeddings = nn.Parameter(embed_init_std * torch.randn(1, 1, 1, projection_dim))
        if self.use_pc_ctx:
            n_tokens += 1
            self.ctx_embeddings = nn.Parameter(embed_init_std * torch.randn(1, 1, 1, projection_dim))

        # particle positional embeddings (same as before)
        self.with_bg = with_bg
        self.particle_positional_embed = particle_positional_embed

        if self.particle_positional_embed:
            self.particle_embeddings = nn.Parameter(
                embed_init_std * torch.randn(1, 1, n_kp_enc, projection_dim)
            )
        else:
            self.particle_embeddings = nn.Parameter(
                embed_init_std * torch.randn(1, 1, 1, projection_dim)
            )

        if self.with_bg:
            self.bg_embeddings = nn.Parameter(
                embed_init_std * torch.randn(1, 1, 1, projection_dim)
            )
        else:
            self.bg_embeddings = None
        # ----- projection from attributes to transformer tokens -----
        self.basic_particle_proj = ParticleAttributesProjection3D(
            n_particles=self.n_kp_enc,
            in_features_dim=self.learned_feature_dim,
            hidden_dim=self.hidden_dim,
            output_dim=projection_dim,
            bg_features_dim=self.learned_bg_feature_dim,
            add_ctx_token=False,
            depth=not self.interaction_depth,
            obj_on=not self.interaction_obj_on,
            base_var=False,
            bg=self.with_bg,
            particle_score=self.particle_score,
            norm_layer=True,
            use_z_orig=self.use_z_orig,
        )

        # ----- (optional) PointNet(-lite) context over the input cloud -----
        if self.use_pc_ctx:
            # Minimal global PointNet: per-point MLP -> max pool -> MLP
            self.ctx_point_mlp = nn.Sequential(
                nn.Linear(3, 64), nn.ReLU(True),
                nn.Linear(64, 128), nn.ReLU(True),
                nn.Linear(128, 256), nn.ReLU(True),
            )
            self.ctx_head = nn.Sequential(
                nn.Linear(256, hidden_dim), nn.ReLU(True),
                nn.Linear(hidden_dim, projection_dim),
            )
        else:
            self.ctx_point_mlp = None
            self.ctx_head = None

        # ----- self-attention over tokens -----
        block_size = 1 if not temporal_interaction else None  # single-step in PC setting
        self.pte = ParticleSelfAttTransformer(
            n_embed=projection_dim, n_head=pte_heads, n_layer=pte_layers,
            block_size=block_size, output_dim=projection_dim, attn_pdrop=dropout,
            resid_pdrop=dropout, hidden_dim_multiplier=4, positional_bias=False,
            activation=activation, max_particles=None, norm_type=attn_norm_type,
            init_std=embed_init_std
        )

        # ----- decode updated attributes (same head you already use) -----
        self.particle_decoder = ParticleAttributeDecoder(
            n_particles=self.n_kp_enc, input_dim=projection_dim, hidden_dim=self.hidden_dim,
            features_dim=learned_feature_dim, bg_features_dim=learned_bg_feature_dim,
            depth=self.interaction_depth, obj_on=self.interaction_obj_on,
            features=self.interaction_features, bg_features=(self.interaction_features and self.with_bg),
            features_dist=self.features_dist
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.init_zero_bias and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def _ctx_from_points(self, x_pc, mask_pc=None):
        """
        x_pc: [B, N, 3(+C)] -> use xyz only for ctx
        mask_pc: [B, N] bool
        returns [B, 1, projection_dim]
        """
        x = x_pc[..., :3]
        if mask_pc is not None:
            # zero out invalid points before per-point MLP; max-pool handles them if set to -inf
            m = mask_pc.unsqueeze(-1).float()
            x = x * m

        per_point = self.ctx_point_mlp(x)                 # [B, N, 256]
        if mask_pc is not None:
            per_point = per_point.masked_fill(~mask_pc.unsqueeze(-1), float('-inf'))
        global_feat, _ = per_point.max(dim=1)             # [B, 256]
        ctx = self.ctx_head(global_feat)                  # [B, proj]
        return ctx.unsqueeze(1)                           # [B, 1, proj]

    def encode_all(self,
                   x_pc,                 # [B, N, 3(+C)]
                   mask_pc,              # [B, N] (bool)
                   z,                    # [B, K, 3]
                   z_scale,              # [B, K, 3] or [B, K, 1]
                   z_obj_on,             # [B, K, 1] or None
                   z_depth,              # [B, K, 1] or None
                   z_features,           # [B, K, F] or None
                   z_bg_features=None,   # [B, Fbg] or None
                   z_base_var=None,      # [B, K, Dv] or None
                   z_score=None,         # [B, K, ?]  or None
                   patch_id_embed=None,  # unused here (kept for API parity)
                   deterministic=False,
                   warmup=False,
                   detach_before_proj=False):
        B, K = z.shape[:2]

        z_v          = z.detach()          if detach_before_proj else z
        z_scale_v    = z_scale.detach()    if detach_before_proj else z_scale
        z_obj_on_v   = z_obj_on.detach()   if (z_obj_on is not None and detach_before_proj) else z_obj_on
        z_depth_v    = z_depth.detach()    if (z_depth  is not None and detach_before_proj) else z_depth
        z_features_v = z_features.detach() if detach_before_proj else z_features
        z_bg_v       = z_bg_features.detach() if (z_bg_features is not None and detach_before_proj) else z_bg_features
        z_base_var_v = z_base_var.detach() if z_base_var is not None else z_base_var
        z_score_v    = z_score.detach()    if z_score is not None else z_score
        z_orig_v     = self.particles_anchor.repeat(B, 1, 1) if self.use_z_orig else None  # [B,1,K] or dummy

        # 1) per-particle projection → tokens [B, K, proj]
        tokens = self.basic_particle_proj(z=z_v,
                                          z_scale=z_scale_v,
                                          z_obj_on=z_obj_on_v,
                                          z_depth=z_depth_v,
                                          z_features=z_features_v,
                                          z_bg_features=z_bg_v,
                                          z_base_var=z_base_var_v,
                                          z_score=z_score_v,
                                          z_orig=z_orig_v)
        # tokens: [B, 1(or T), N, proj] where N = K (+1 if bg)
        B, T, N, D = tokens.shape

        # build positional embeddings matching N
        if self.particle_positional_embed:
            if self.particle_embeddings.shape[2] == 1:
                p_emb = self.particle_embeddings.repeat(B, T, N, 1)
            else:
                # foreground part
                fg_emb = self.particle_embeddings.repeat(B, T, 1, 1)  # [B,T,K,proj]
                if self.with_bg and self.bg_embeddings is not None and fg_emb.shape[2] + 1 == N:
                    bg_emb = self.bg_embeddings.repeat(B, T, 1, 1)     # [B,T,1,proj]
                    p_emb = torch.cat([fg_emb, bg_emb], dim=2)          # [B,T,K+1,proj]
                else:
                    # fallback: slice or pad to N
                    if fg_emb.shape[2] >= N:
                        p_emb = fg_emb[:, :, :N, :]
                    else:
                        # pad zeros if ever needed
                        pad = torch.zeros(B, T, N - fg_emb.shape[2], D, device=fg_emb.device, dtype=fg_emb.dtype)
                        p_emb = torch.cat([fg_emb, pad], dim=2)
        else:
            p_emb = 0.0

        tokens = tokens + p_emb
        tokens = tokens.squeeze(1)                  # [B, N_main, proj], N_main = K + (1 if with_bg else 0)

        # Build attention sequence = [fg..., (bg), ctx]
        seq_attn = tokens
        if self.use_pc_ctx:
            ctx = self._ctx_from_points(x_pc, mask_pc)                 # [B,1,proj]
            ctx = ctx + self.ctx_embeddings.repeat(B, 1, 1, 1).squeeze(1)  # [B,1,proj]
            seq_attn = torch.cat([seq_attn, ctx], dim=1)               # [B, N_main+1, proj]

        # Self-attention over tokens (time=1)
        seq_attn_1t = seq_attn.unsqueeze(1)                            # [B,1,N_attn,proj]
        seq_out = self.pte(seq_attn_1t).squeeze(1)                     # [B,N_attn,proj]

        # === IMPORTANT: pass what the decoder expects ===
        # decoder sees [fg..., (bg), ctx] so it can compute fg_particles = n - 2 (bg+ctx) or n - 1 (ctx only)
        n_for_decoder = seq_out.shape[1]          # == N_main (+1 if ctx)
        seq_for_dec = seq_out[:, :n_for_decoder]  # [B, N_main(+1), proj]
        seq_for_dec_t = seq_for_dec.unsqueeze(1)  # [B,1,N_dec,proj]

        # Configure the decoder:
        # If `with_bg=True` AND you appended ctx → decoder will use fg = N_dec-2
        # If `with_bg=False` but you appended ctx → decoder will use fg = N_dec-1
        dec = self.particle_decoder(seq_for_dec_t)

        # ---- unpack & sample like your 2D interaction did ----
        mu_depth = dec['mu_depth']       # [B,1,K,1] or None
        logvar_depth = dec['logvar_depth']
        if self.interaction_depth:
            z_depth_new = reparameterize(mu_depth, logvar_depth) if not deterministic else mu_depth
        else:
            z_depth_new = None

        mu_features = dec['mu_features']                 # [B,1,K,F] or None
        logvar_features = dec['logvar_features']
        mu_bg_features = dec['mu_bg_features']           # [B,1, Fbg] if with_bg
        logvar_bg_features = dec['logvar_bg_features']

        if self.interaction_features:
            mu_features = (z_features.unsqueeze(1) if z_features is not None else 0) + mu_features
            if self.features_dist == 'categorical':
                # (same categorical sampling you already had; omitted for brevity)
                raise NotImplementedError("categorical features not shown; reuse your existing code here.")
            else:
                z_features_new = (reparameterize(mu_features, logvar_features)
                                  if not deterministic else mu_features)
            if self.with_bg:
                mu_bg_features = (z_bg_features.unsqueeze(1) if z_bg_features is not None else 0) + mu_bg_features
                z_bg_features_new = (reparameterize(mu_bg_features, logvar_bg_features)
                                     if not deterministic else mu_bg_features)
            else:
                z_bg_features_new = None
        else:
            z_features_new = None
            z_bg_features_new = None

        lobj_on_a = dec['lobj_on_a']    # [B,1,K,1]
        lobj_on_b = dec['lobj_on_b']
        if self.interaction_obj_on:
            a_gate = lobj_on_a.sigmoid()
            a = ((1 - a_gate) * self.obj_on_min + a_gate * self.obj_on_max).exp()
            b_gate = 1 - (lobj_on_b * 0 + lobj_on_a).sigmoid()
            b = ((1 - b_gate) * self.obj_on_min + b_gate * self.obj_on_max).exp()
            beta = torch.distributions.Beta(a, b)
            mu_obj_on = beta.mean
            z_obj_on_new = beta.rsample() if not deterministic else beta.mean
        else:
            a = b = mu_obj_on = z_obj_on_new = None

        return {
            'mu_depth': mu_depth.squeeze(1) if mu_depth is not None else None,
            'logvar_depth': logvar_depth.squeeze(1) if logvar_depth is not None else None,
            'z_depth': z_depth_new.squeeze(1) if z_depth_new is not None else None,

            'mu_features': mu_features.squeeze(1) if mu_features is not None else None,
            'logvar_features': logvar_features.squeeze(1) if logvar_features is not None else None,
            'z_features': z_features_new.squeeze(1) if self.interaction_features else None,

            'mu_bg_features': mu_bg_features.squeeze(1) if mu_bg_features is not None else None,
            'logvar_bg_features': logvar_bg_features.squeeze(1) if logvar_bg_features is not None else None,
            'z_bg_features': z_bg_features_new.squeeze(1) if self.interaction_features and self.with_bg else None,

            'obj_on_a': a.squeeze(1) if a is not None else None,
            'obj_on_b': b.squeeze(1) if b is not None else None,
            'z_obj_on': z_obj_on_new.squeeze(1) if z_obj_on_new is not None else None,
            'mu_obj_on': mu_obj_on.squeeze(1) if mu_obj_on is not None else None,

            # pass-through for convenience
            'z': z, 'z_scale': z_scale,
        }

    def forward(self, x_pc, mask_pc, z, z_scale, z_obj_on, z_depth, z_features,
                z_bg_features=None, z_base_var=None, z_score=None, patch_id_embed=None,
                deterministic=False, warmup=False):
        return self.encode_all(x_pc, mask_pc, z, z_scale, z_obj_on, z_depth, z_features,
                               z_bg_features, z_base_var, z_score, patch_id_embed,
                               deterministic=deterministic, warmup=warmup)
