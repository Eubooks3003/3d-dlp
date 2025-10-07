import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.point_cloud_modules.DLPEncoder.particle_self_attn_transformer import RMSNorm

class ParticleAttributesProjection3D(nn.Module):
    """
    3D version of ParticleAttributesProjection.
    Expects z and z_scale to be 3D vectors. Depth channel is disabled by default.
    var_projection adapts to the provided last-dim of z_base_var.
    """
    def __init__(self,
                 n_particles: int,
                 in_features_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 bg_features_dim: int,
                 add_ctx_token: bool = False,
                 base_dim: int = 32,
                 depth: bool = False,             # <-- off by default in PC
                 obj_on: bool = True,
                 base_var: bool = False,
                 bg: bool = True,
                 activation: str = 'gelu',
                 init_std: float = 0.2,
                 cat_particle_num: bool = False,
                 norm_layer: bool = True,
                 particle_score: bool = False,
                 mask_inputs: bool = True,
                 use_z_orig: bool = False,
                 obj_on_film: bool = False,
                 mask_obj_on: bool = False):
        super().__init__()
        self.n_particles = n_particles
        self.in_features_dim = in_features_dim
        self.bg_features_dim = bg_features_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # what to include
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

        act = nn.GELU if activation == 'gelu' else nn.ReLU
        Norm = RMSNorm if norm_layer else nn.Identity

        # how many base blocks we will concatenate
        self.base_dim = base_dim
        n_entities = 0

        # 3D position & 3D scale are always present
        n_entities += 1  # pos
        self.pos_projection = nn.Sequential(
            nn.Linear(3, hidden_dim), Norm(hidden_dim), act(),
            nn.Linear(hidden_dim, base_dim)
        )

        n_entities += 1  # scale
        self.scale_projection = nn.Sequential(
            nn.Linear(3, hidden_dim), Norm(hidden_dim), act(),
            nn.Linear(hidden_dim, base_dim)
        )

        # learned per-particle features
        n_entities += 1
        self.features_projection = nn.Sequential(
            nn.Linear(in_features_dim, hidden_dim), Norm(hidden_dim), act(),
            nn.Linear(hidden_dim, base_dim)
        )

        # optional: obj_on
        if self.with_obj_on:
            n_entities += 1
            if self.obj_on_film:
                self.obj_on_projection = nn.Sequential(
                    nn.Linear(1, hidden_dim), act(),
                    nn.Linear(hidden_dim, 2 * hidden_dim)
                )
                # initialize FiLM head like your 2D code
                nn.init.constant_(self.obj_on_projection[-1].weight, 0.0)
                nn.init.constant_(self.obj_on_projection[-1].bias[:hidden_dim], 1.0)
                nn.init.constant_(self.obj_on_projection[-1].bias[hidden_dim:], 0.0)
            else:
                self.obj_on_projection = nn.Sequential(
                    nn.Linear(1, hidden_dim), Norm(hidden_dim), act(),
                    nn.Linear(hidden_dim, base_dim)
                )

        # optional: depth (off by default in 3D PC)
        if self.with_depth:
            n_entities += 1
            self.depth_projection = nn.Sequential(
                nn.Linear(1, hidden_dim), Norm(hidden_dim), act(),
                nn.Linear(hidden_dim, base_dim)
            )

        # optional: variance (dim inferred at runtime)
        if self.with_var:
            # create a placeholder; actual Linear will be lazily built on first call
            self.var_projection = None
            self._var_in_dim = None  # set on first forward
            n_entities += 1

        # optional: particle score
        if self.with_score:
            n_entities += 1
            self.score_projection = nn.Sequential(
                nn.Linear(1, hidden_dim), Norm(hidden_dim), act(),
                nn.Linear(hidden_dim, base_dim)
            )

        # optional: original anchor + offset (in 3D)
        if self.use_z_orig:
            # concatenate [z_orig(3), z_offset(3)] -> 6
            n_entities += 1
            self.origin_projection = nn.Sequential(
                nn.Linear(6, hidden_dim), Norm(hidden_dim), act(),
                nn.Linear(hidden_dim, base_dim)
            )

        # optional: particle index embedding
        if self.cat_particle_num:
            n_entities += 1
            self.particle_num_embed = nn.Parameter(0.02 * torch.randn(1, n_particles, base_dim))

        self.particle_dim = base_dim * n_entities

        # final projector (with/without FiLM on obj_on)
        if self.obj_on_film and self.with_obj_on:
            self.particle_projection_0 = nn.Sequential(nn.Linear(self.particle_dim, hidden_dim), Norm(hidden_dim))
            self.particle_projection = nn.Sequential(act(), nn.Linear(hidden_dim, output_dim))
        else:
            self.particle_projection = nn.Sequential(
                nn.Linear(self.particle_dim, hidden_dim), Norm(hidden_dim), act(),
                nn.Linear(hidden_dim, output_dim)
            )

        # bg token
        if self.with_bg:
            self.bg_projection = nn.Sequential(
                nn.Linear(bg_features_dim, hidden_dim), Norm(hidden_dim), act(),
                nn.Linear(hidden_dim, output_dim)
            )

        # optional ctx token (kept for API parity; your PC ctx can live upstream)
        if self.add_ctx_token:
            self.ctx_embedding = nn.Parameter(init_std * torch.randn(1, 1, 1, output_dim))

        # input masks (extended to 3D)
        if self.with_obj_on and self.mask_inputs:
            self.pos_mask     = nn.Parameter(2.0 * torch.ones(3))
            self.scale_mask   = nn.Parameter(0.1 * torch.ones(3))
            self.features_mask= nn.Parameter(init_std * torch.randn(in_features_dim))
            if self.mask_obj_on:
                self.obj_on_mask = nn.Parameter(torch.zeros(1))
        if self.with_depth and self.with_obj_on and self.mask_inputs:
            self.depth_mask = nn.Parameter(init_std * torch.randn(1))
        if self.use_z_orig and self.mask_inputs:
            # mask for [z_orig(3), z_offset(3)]
            self.orig_mask = nn.Parameter(2.0 * torch.ones(6))

    def _lazy_build_var_proj(self, in_dim, hidden_dim, base_dim, Norm, act):
        # Build var_projection the first time we see z_base_var with a concrete last-dim
        self.var_projection = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), Norm(hidden_dim), act(),
            nn.Linear(hidden_dim, base_dim)
        )
        self._var_in_dim = in_dim

    def forward(self,
                z,            # [B, K, 3]
                z_scale,      # [B, K, 3]
                z_obj_on,     # [B, K, 1] or None
                z_depth,      # [B, K, 1] or None
                z_features,   # [B, K, F]
                z_bg_features=None,  # [B, Fbg]
                z_base_var=None,     # [B, K, Dv] (Dv arbitrary)
                z_score=None,        # [B, K, 1] or None
                z_orig=None):        # [B, K, 3] anchors if use_z_orig=True
        B, K, _ = z.shape
        device = z.device

        # build [z_orig, z_offset] if requested
        if self.use_z_orig and z_orig is not None:
            z_offset = z - z_orig
            z_orig_tot = torch.cat([z_orig, z_offset], dim=-1)  # [B,K,6]
        else:
            z_orig_tot = None

        # input masking based on obj_on
        if self.with_obj_on and self.mask_inputs and (z_obj_on is not None):
            z_gate = (z_obj_on > 0.2).float()                # [B,K,1]
            z      = z_gate * z + (1 - z_gate) * self.pos_mask
            z_scale= z_gate * z_scale + (1 - z_gate) * self.scale_mask
            z_features = z_gate * z_features + (1 - z_gate) * self.features_mask
            if self.use_z_orig and z_orig_tot is not None:
                z_orig_tot = z_gate * z_orig_tot + (1 - z_gate) * self.orig_mask
            if self.mask_obj_on:
                z_obj_on = z_gate * z_obj_on + (1 - z_gate) * self.obj_on_mask

        # per-field projections
        z_proj         = self.pos_projection(z)              # [B,K,base]
        z_scale_proj   = self.scale_projection(z_scale)      # [B,K,base]
        z_features_proj= self.features_projection(z_features)# [B,K,base]

        chunks = [z_proj, z_scale_proj, z_features_proj]

        if self.with_obj_on and z_obj_on is not None:
            if self.obj_on_film:
                z_obj_on_proj = self.obj_on_projection(z_obj_on)   # [B,K,2*hidden]
            else:
                z_obj_on_proj = self.obj_on_projection(z_obj_on)   # [B,K,base]
                chunks.append(z_obj_on_proj)

        if self.with_depth and z_depth is not None:
            if self.with_obj_on and self.mask_inputs:
                z_depth = (z_obj_on > 0.2).float() * z_depth + (1 - (z_obj_on > 0.2).float()) * self.depth_mask
            z_depth_proj = self.depth_projection(z_depth)    # [B,K,base]
            chunks.append(z_depth_proj)

        if self.with_var and (z_base_var is not None):
            # build var head lazily to match Dv
            if (self.var_projection is None) or (self._var_in_dim != z_base_var.shape[-1]):
                Norm = RMSNorm if self.norm_layer else nn.Identity
                act  = nn.GELU if isinstance(self.features_projection[1], nn.GELU) else nn.ReLU
                self._lazy_build_var_proj(z_base_var.shape[-1], self.hidden_dim, self.base_dim, Norm, act)
            z_var_proj = self.var_projection(z_base_var)     # [B,K,base]
            chunks.append(z_var_proj)

        if self.with_score and (z_score is not None):
            z_score_proj = self.score_projection(z_score)    # [B,K,base]
            chunks.append(z_score_proj)

        if self.use_z_orig and (z_orig_tot is not None):
            z_orig_proj = self.origin_projection(z_orig_tot) # [B,K,base]
            chunks.append(z_orig_proj)

        if self.cat_particle_num:
            p_embed = self.particle_num_embed.repeat(B, 1, 1)      # [B,K,base]
            chunks.append(p_embed)

        z_all = torch.cat(chunks, dim=-1)                    # [B,K, particle_dim]

        # final projection (+ optional FiLM on obj_on)
        if self.obj_on_film and self.with_obj_on and (z_obj_on is not None):
            oscale, oshift = z_obj_on_proj.chunk(2, dim=-1)  # [B,K,H], [B,K,H]
            z_all_proj = self.particle_projection(oscale * self.particle_projection_0(z_all) + oshift)
        else:
            z_all_proj = self.particle_projection(z_all)      # [B,K,output_dim]

        # append bg token if requested
        if self.with_bg and (z_bg_features is not None):
            z_bg_proj = self.bg_projection(z_bg_features)     # [B, output_dim]
            z_all_proj = torch.cat([z_all_proj, z_bg_proj.unsqueeze(1)], dim=1)  # [B,K+1,D]

        # keep ctx token support (optional; usually done upstream in PC)
        if self.add_ctx_token:
            z_all_proj = torch.cat([z_all_proj, self.ctx_embedding.repeat(B, 1, 1, 1).squeeze(1)], dim=1)

        # return shape expected by your transformer wrapper:
        # [B, T=1,  n_tokens, D]
        return z_all_proj.unsqueeze(1)
