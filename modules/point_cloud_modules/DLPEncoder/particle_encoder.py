import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util_func import reparameterize

import numpy as np

from modules.point_cloud_modules.DLPEncoder.prior_encoder.dlp_prior import DLPPrior
from modules.point_cloud_modules.DLPEncoder.particle_attribute_encoder import ParticleAttributeEncoderPoint
from modules.point_cloud_modules.DLPEncoder.particle_features_encoder import ParticleFeaturesEncoderPoint

class ParticleEncoder(nn.Module):
    def __init__(self, cdim=3, image_size=64,
                 pad_mode='replicate', dropout=0.0, n_kp_per_patch=1, n_kp_prior=20,
                 patch_size=16, n_kp_enc=20, n_kp_dec=None, learned_feature_dim=16,
                 kp_range=(-1, 1), kp_activation="tanh", anchor_s=0.25,
                 use_resblock=True, embed_init_std=0.2, projection_dim=128, timestep_horizon=1,
                 filtering_heuristic='none', obj_ch_mult_prior=(1, 2),
                 obj_ch_mult=(1, 2, 3), obj_base_ch=32, obj_final_cnn_ch=32, num_res_blocks=2,
                 interaction_features=False, interaction_obj_on=False, interaction_depth=True,
                 temporal_interaction=True, cnn_mid_blocks=False, mlp_hidden_dim=256,
                 embed_prior_patch_pos=False, add_particle_temp_embed=False,
                 features_dist='gauss', n_fg_categories=8, n_fg_classes=4,
                 use_null_features_embed=True, obj_on_min=1e-4, obj_on_max=100.0, warmup_n_kp_ratio=0.35,
                 # initialization
                 init_zero_bias=True,  # zero bias for conv and linear layers
                 init_ssm_last_layer=True,  # spatial softmax initialization
                 init_conv_layers=True,  # initialize conv layers with normal dist
                 init_conv_fg_std=0.02,  # std for conv fg normal dist
                 separate_depth_features=False,
                 depth_feature_dim=16,):
        super(ParticleEncoder, self).__init__()
        """
        DLP Foreground Module – Extracts objects from an image using keypoints and learned features. 
        Combines posterior CNN for full image processing and prior CNN for patch-based keypoint proposals.
        
        Args:
        cdim (int, default=3): Number of channels in the input image.
        image_size (int, default=64): Resolution of the input image (assumes square images).
        pad_mode (str, default='replicate'): Padding mode for CNNs, options are 'zeros' or 'replicate'.
        dropout (float, default=0.0): Dropout rate for CNNs (not used in practice).
        n_kp_per_patch (int, default=1): Number of keypoints proposed per patch.
        n_kp_prior (int, default=20): Number of keypoints filtered from prior proposals.
        patch_size (int, default=16): Size of patches for the prior keypoint proposal network.
        n_kp_enc (int, default=20): Number of posterior keypoints to learn.
        n_kp_dec (int, optional): Number of keypoints for decoder (if different from encoder).
        learned_feature_dim (int, default=16): Dimensionality of latent visual features for glimpses.
        kp_range (tuple, default=(-1, 1)): Range for keypoints; options are (-1, 1) or (0, 1).
        kp_activation (str, default='tanh'): Activation function for keypoints; 'tanh' for range (-1, 1), 'sigmoid' for range (0, 1).
        anchor_s (float, default=0.25): Glimpse size as a ratio of image size (e.g., 0.25 → glimpse size is 0.25 * image_size).
        use_resblock (bool, default=True): Whether to use residual blocks in CNNs.
        embed_init_std (float, default=0.2): Standard deviation for initializing learned tokens.
        projection_dim (int, default=128): Dimensionality of embeddings for transformer input.
        timestep_horizon (int, default=1): Maximum timesteps the model processes at once.
        filtering_heuristic (str, default='none'): Method for filtering prior keypoints. Options: 'distance', 'variance', 'random', 'none'.
        obj_ch_mult (tuple, default=(1, 2, 3)): Multiplicative factors for object feature channels at each CNN stage.
        obj_base_ch (int, default=32): Base number of channels in object feature extractor.
        obj_final_cnn_ch (int, default=32): Number of channels in the final object CNN layer.
        num_res_blocks (int, default=2): Number of residual blocks in object feature extractor.
        interaction_features (bool, default=False): Whether to compute interaction-based features.
        interaction_obj_on (bool, default=False): Whether to include "object-on" features for interactions.
        interaction_depth (bool, default=True): Whether to compute depth information for interactions.
        temporal_interaction (bool, default=True): Whether to model temporal interactions between features.
        cnn_mid_blocks (bool, default=False): Whether to include intermediate blocks in the CNN.
        mlp_hidden_dim (int, default=256): Hidden dimensionality for MLP layers.
        embed_prior_patch_pos (bool, default=False): Whether to embed positional information for prior patches.
        add_particle_temp_embed (bool, default=False): Whether to add temporal embeddings to particles.
        features_dist (str, default='gauss'): Distribution type for keypoint features. Options: 'gauss'.
        n_fg_categories (int, default=8): Number of foreground categories for classification.
        n_fg_classes (int, default=4): Number of foreground classes for classification.
        use_null_features_embed (bool, default=True): Whether to use a learned embedding for filtered-out particles.
        obj_on_min (float, default=1e-4): Minimum concentration value in Beta dist for transparency" probabilities.
        obj_on_max (float, default=100.0): Maximum concentration value in Beta dist for transparency" probabilities.
        """
        self.image_size = image_size
        self.dropout = dropout
        self.kp_range = kp_range
        self.n_kp_per_patch = n_kp_per_patch
        self.n_kp_enc = n_kp_enc
        self.n_kp_dec = self.n_kp_enc if n_kp_dec is None else n_kp_dec
        self.n_kp_prior = n_kp_prior
        self.kp_activation = kp_activation
        self.patch_size = patch_size
        self.anchor_patch_s = patch_size / image_size
        self.features_dim = int(image_size // (2 ** (len(obj_ch_mult) - 1)))
        self.learned_feature_dim = learned_feature_dim
        self.features_dist = features_dist
        self.n_fg_categories = n_fg_categories
        self.n_fg_classes = n_fg_classes
        assert learned_feature_dim > 0, "learned_feature_dim must be greater than 0"
        self.anchor_s = anchor_s
        self.obj_patch_size = np.round(anchor_s * (image_size - 1)).astype(int)
        self.cdim = cdim
        self.use_resblock = use_resblock
        self.embed_init_std = embed_init_std
        self.projection_dim = projection_dim
        self.timestep_horizon = (timestep_horizon + 1) if timestep_horizon > 1 else 1
        self.num_patches = int((image_size // self.patch_size) ** 2)
        self.interaction_features = interaction_features
        self.interaction_depth = interaction_depth
        self.interaction_obj_on = interaction_obj_on
        self.temporal_interaction = temporal_interaction
        self.add_particle_temp_embed = add_particle_temp_embed
        self.cnn_mid_blocks = cnn_mid_blocks
        self.mlp_hidden_dim = mlp_hidden_dim
        self.embed_prior_patch_pos = embed_prior_patch_pos
        self.obj_on_min = obj_on_min
        self.obj_on_max = obj_on_max
        self.use_null_features_embed = use_null_features_embed
        self.warmup_n_kp_ratio = warmup_n_kp_ratio
        # initialization
        self.init_zero_bias = init_zero_bias  # zero bias for conv and linear layers
        self.init_ssm_last_layer = init_ssm_last_layer  # spatial softmax initialization
        self.init_conv_layers = init_conv_layers  # initialize conv layers with normal dist
        self.init_conv_fg_std = init_conv_fg_std  # std for conv fg normal dist
        
        self.separate_depth_features = separate_depth_features
        self.depth_feature_dim = depth_feature_dim

        print("Creating DLP Prior with cdim: ", cdim)
        # TODO: Make all parameters configurable if necessary
        self.prior_encoder = DLPPrior(
            grid=(48, 48, 48),          # (D,H,W) of the voxel volume
            out_feat=4,                 # voxelizer output channels (e.g., density, rgb bins, etc.)
            base_ch=32,                 # 3D encoder base width
            ch_mult=(1, 2, 3),          # per-level multipliers
            num_res_blocks=2,
            use_resblock=True,
            use_attention=False,
            cnn_mid_blocks=False,       # False -> keep mid blocks in Encoder3D
            n_kp_prior=n_kp_prior,      # total K keypoint channels/proposals
            kp_range=(-1., 1.),         # coordinate range for SSM3D
            temperature=1.0,            # softmax temperature for SSM3D
            filtering_heuristic='none', # 'none' | 'variance' | 'distance' | 'random'
            init_zero_bias=True,
            init_conv_layers=True,
            init_conv_std=0.02
        )


        # attribute encoder - anchor (z_a), offset (z_o), scale (z_s)
        anchor_s_att = patch_size / image_size
        k_neighbors_attr = 128
        k_neighbors_feat = 128

        extra_feats = 0   # per-point features beyond xyz from the dataset
        # TODO: Make this gettable via the dataloader 

        # --- Particle attributes (offset/scale/obj_on/depth) ---
        self.particle_attribute_enc_pc = ParticleAttributeEncoderPoint(
            n_particles=self.n_kp_prior,
            in_feat=extra_feats,
            k_neighbors=k_neighbors_attr,
            hidden=mlp_hidden_dim,
            kp_activation=kp_activation,                 # 'tanh' or 'sigmoid'
            max_offset=1.0,
            with_scale=True,
            with_obj_on=(not self.interaction_obj_on),
            with_depth=(not self.interaction_depth),
            base_radius=anchor_s,                        # canonical crop radius in [-1,1] units
            clamp_after_st=True,
            obj_on_min=self.obj_on_min,
            obj_on_max=self.obj_on_max,
            init_zero_bias=init_zero_bias,
        )

        # --- Particle appearance/features (PointNet-like) ---
        output_logvar = (not self.interaction_features and self.features_dist != 'categorical')

        print("Output logvar for features: ", output_logvar)

        self.particle_features_enc = ParticleFeaturesEncoderPoint(
            anchor_size=anchor_s,                        # used to set canonical crop scale
            features_dim=learned_feature_dim,
            in_feat=extra_feats,
            k_neighbors=k_neighbors_feat,
            output_logvar=output_logvar,
            features_dist=self.features_dist,            # usually 'gauss' here
            hidden=mlp_hidden_dim,
            interaction_features=self.interaction_features,
            use_null_features_embed=self.use_null_features_embed,
            embed_init_std=embed_init_std,
            base_radius=anchor_s,
            clamp_after_st=True,
        )
        # embed the source patch of the particles
        # if self.embed_prior_patch_pos:
        #     self.patch_id_embed = nn.Parameter(self.embed_init_std * torch.randn(1, self.n_kp_prior, mlp_hidden_dim))
        # else:
        #     self.patch_id_embed = None
        # patch_centers = self.prior_encoder.get_patch_centers().unsqueeze(0) * (
        #         self.kp_range[1] - self.kp_range[0]) + self.kp_range[0]
        # # append null particle
        # patch_centers = torch.cat([patch_centers, torch.zeros(1, 1, 2)], dim=1)
        # if self.n_kp_enc != self.n_kp_dec and self.interaction_features and self.use_null_features_embed:
        #     self.null_feature_embed = nn.Parameter(self.embed_init_std * torch.randn(1, 1, self.learned_feature_dim))
        #     self.null_depth_feature_embed = nn.Parameter(self.embed_init_std * torch.randn(1, 1, self.depth_feature_dim)) if self.separate_depth_features and cdim == 4 else None
        # self.register_buffer('patch_centers', patch_centers)
        # self.register_buffer('mu_scale_prior', torch.tensor(np.log(self.anchor_s / (1 - self.anchor_s + 1e-5))))
        self.init_weights()

    def init_weights(self):
        self.prior_encoder.init_weights()
        # initialize ssm and other conv ins the same
        # conv1 = self.prior_encoder.enc.conv_in
        # conv2 = self.particle_attribute_enc.cnn.conv_in
        # conv3 = self.particle_features_enc.cnn.conv_in
        # if conv1.weight.shape == conv2.weight.shape:
        #     with torch.no_grad():
        #         conv2.weight.copy_(conv1.weight)
        #         if conv1.bias is not None and conv2.bias is not None:
        #             conv2.bias.copy_(conv1.bias)
        # if conv1.weight.shape == conv3.weight.shape:
        #     with torch.no_grad():
        #         conv3.weight.copy_(conv1.weight)
        #         if conv1.bias is not None and conv3.bias is not None:
        #             conv3.bias.copy_(conv1.bias)

    def encode_prior(self, x, mask):
        return self.prior_encoder(x, mask)

    def encode_pos_scale_with_prior(self, x, mask=None, deterministic=False, warmup=False, timesteps=None):
        """
        Point-cloud version.
        x:     [B, N, 3]
        mask:  [B, N] or None
        Returns the same dict keys as the 2D version, but positions are 3D.
        """
        assert x.dim() == 3 and x.size(-1) == 3, f"expected points [B,N,3], got {tuple(x.shape)}"
        B = x.shape[0]

        # ---- 1) Prior proposals from point cloud ----
        # kp_p:   [B, K, 3] in [-1,1]^3 (global normalized coords)
        # var_kp: [B, K, 3, 3] covariance per proposal
        kp_p, var_kp = self.encode_prior(x, mask)       # this calls self.prior_encoder(points, mask)

        # Base (deterministic) proposal position: z_base == mu (like 2D code)
        mu      = kp_p                                  # [B, K, 3]
        logvar  = torch.zeros_like(mu)                  # [B, K, 3] (deterministic chamfer KL)
        z_base  = mu                                    # [B, K, 3]

        # ---- 2) Particle attribute encoder (offset/scale/obj_on/depth) ----
        # If you already implemented a PC attribute encoder, call it.
        attr = self.particle_attribute_enc_pc(
            x, z_base, mask=mask,
            timesteps=timesteps, deterministic=deterministic
        )
        mu_offset     = attr['mu']                  # [B, K, 3]
        logvar_offset = attr['logvar']              # [B, K, 3]
        mu_scale      = attr['mu_scale']            # [B, K, 3] (3D scales)
        logvar_scale  = attr['logvar_scale']        # [B, K, 3]
        # optional heads
        if not self.interaction_obj_on:
            lobj_on_a   = attr['lobj_on_a']
            lobj_on_b   = attr['lobj_on_b']
            obj_on_a    = attr['obj_on_a']
            obj_on_b    = attr['obj_on_b']
            mu_obj_on   = attr['mu_obj_on']
            z_obj_on    = attr['z_obj_on']
        else:
            obj_on_a = obj_on_b = z_obj_on = mu_obj_on = None

        if not self.interaction_depth:
            mu_depth      = attr['mu_depth']        # you can define this as radial depth or leave None
            logvar_depth  = attr['logvar_depth']
            z_depth       = mu_depth if deterministic else reparameterize(mu_depth, logvar_depth)
        else:
            mu_depth = logvar_depth = z_depth = None

        # ---- 3) Combine base + offsets, reparameterize if stochastic ----
        mu_tot    = z_base + mu_offset                  # [B, K, 3]
        logvar_tot= logvar_offset
        # Optional prior on scale; keep as-is if you don’t have a 3D prior for scale yet.
        # Example if you keep the same API: mu_scale = self.mu_scale_prior + mu_scale  # (make mu_scale_prior 3D if used)

        if deterministic:
            z_offset = mu_offset
            z_scale  = mu_scale
        else:
            z_offset = reparameterize(mu_offset, logvar_offset)
            z_scale  = reparameterize(mu_scale,  logvar_scale)

        z = z_base + z_offset                           # [B, K, 3]

        # ---- 4) Variance-based utilities / filtering (3D) ----
        # total_var: use trace of covariance as scalar uncertainty
        # var_kp: [B,K,3,3] -> [B,K]
        total_var = var_kp.diagonal(dim1=-2, dim2=-1).sum(-1)

        # z_base_var: keep the full covariance flattened (like 2D had extra stats)
        z_base_var = var_kp.view(B, var_kp.shape[1], 9) # [B,K,9]
        # simple "confidence" proxy from attribute logvar (if you have it), else zeros
        confidence_score = logvar_offset.norm(dim=-1, keepdim=True) if logvar_offset is not None else torch.zeros(B, kp_p.shape[1], 1, device=x.device)
        z_base_var = torch.cat([z_base_var, confidence_score], dim=-1)  # [B,K,10]

        # Optional: stable channel id (replaces old patch id)
        z_base_id = torch.arange(kp_p.shape[1], device=x.device)[None, :, None].expand(B, -1, 1)  # [B,K,1]

        # Optionally keep a null patch embedding out for now
        patch_id_embed = None

        # A tiny score head (kept for API parity)
        mu_score     = (z_base_var.mean(-1, keepdim=True) / 30.0) * 2 - 1   # shape [B,K,1]
        logvar_score = math.log(0.2 ** 2) * torch.ones_like(mu_score)
        z_score      = mu_score

        # ---- 5) (Optional) Select top-K by variance if encoder budget < prior proposals ----
        if self.n_kp_enc < self.n_kp_prior:
            n_filter = self.n_kp_enc if not warmup else min(self.n_kp_enc, int(self.warmup_n_kp_ratio * self.n_kp_prior))
            _, embed_ind = torch.topk(total_var, k=n_filter, dim=-1, largest=False)
            b_idx = torch.arange(B, device=x.device)[:, None]

            # index everything consistently
            mu_tot        = mu_tot[b_idx, embed_ind]
            z_base        = z_base[b_idx, embed_ind]
            z_base_var    = z_base_var[b_idx, embed_ind]
            z_base_id     = z_base_id[b_idx, embed_ind]
            mu_offset     = mu_offset[b_idx, embed_ind]
            logvar_offset = logvar_offset[b_idx, embed_ind]
            z             = z[b_idx, embed_ind]
            z_offset      = z_offset[b_idx, embed_ind]
            z_scale       = z_scale[b_idx, embed_ind]
            mu_scale      = mu_scale[b_idx, embed_ind]
            mu_score      = mu_score[b_idx, embed_ind]
            logvar_score  = logvar_score[b_idx, embed_ind]

            if logvar_scale is not None:
                logvar_scale = logvar_scale[b_idx, embed_ind]
            if not self.interaction_obj_on and obj_on_a is not None:
                obj_on_a   = obj_on_a[b_idx, embed_ind]
                obj_on_b   = obj_on_b[b_idx, embed_ind]
                mu_obj_on  = mu_obj_on[b_idx, embed_ind]
                z_obj_on   = z_obj_on[b_idx, embed_ind]
            if not self.interaction_depth and mu_depth is not None:
                z_depth      = z_depth[b_idx, embed_ind]
                mu_depth     = mu_depth[b_idx, embed_ind]
                logvar_depth = logvar_depth[b_idx, embed_ind]

        print("Z Base Var: ", z_base_var.mean())
        out_dict = {
            'mu': mu, 'logvar': logvar, 'z_base': z_base, 'z': z, 'mu_tot': mu_tot,
            'patch_id_embed': patch_id_embed,
            'mu_scale': mu_scale, 'logvar_scale': logvar_scale, 'z_scale': z_scale,
            'mu_depth': mu_depth, 'logvar_depth': logvar_depth, 'z_depth': z_depth,
            'mu_offset': mu_offset, 'logvar_offset': logvar_offset, 'z_offset': z_offset,
            'kp_p': kp_p, 'var_kp': var_kp,
            'z_base_var': z_base_var, 'total_var': total_var,
            'obj_on_a': obj_on_a, 'obj_on_b': obj_on_b, 'z_obj_on': z_obj_on, 'mu_obj_on': mu_obj_on,
            'z_base_id': z_base_id, 'mu_score': mu_score, 'logvar_score': logvar_score, 'z_score': z_score
        }
        return out_dict


    def sample_gauss(self, mu, logvar, deterministic, interaction_features):
        if deterministic or interaction_features or (logvar is None):
            return mu
        return reparameterize(mu, logvar)

    def sample_categorical_logits(self, logits, deterministic, n_cat, n_cls):
        # logits: [..., n_cat*n_cls]
        shape = logits.shape
        probs = logits.view(*shape[:-1], n_cat, n_cls).softmax(dim=-1)
        if deterministic:
            idx = torch.argmax(probs.view(-1, n_cls), dim=-1, keepdim=True)    # [M,1]
            onehot = F.one_hot(idx.squeeze(-1), num_classes=n_cls)
            onehot = onehot.view(*probs.shape).to(logits.dtype)
        else:
            # multinomial over the last dim (classes) per category
            draws = torch.multinomial(
                probs.view(-1, n_cls), num_samples=1, replacement=True
            )  # [M,1]
            onehot = F.one_hot(draws.squeeze(-1), num_classes=n_cls)
            onehot = onehot.view(*probs.shape).to(logits.dtype)
        # straight-through
        st = onehot.detach() + (probs - probs.detach())
        return st.view(*shape)  # back to [..., n_cat*n_cls]

    def gate_with_null(self, mu, gate, null_embed):
        # gate: broadcastable zeros/ones
        if null_embed is None:
            return mu  # no-op if you didn't set one
        if null_embed.shape[-1] != mu.shape[-1]:
            pad = mu.new_zeros(*null_embed.shape[:-1], mu.shape[-1] - null_embed.shape[-1])
            null_embed = torch.cat([null_embed, pad], dim=-1)
        return gate * mu + (1.0 - gate) * null_embed
    def encode_appearance(self,
                        points,                # [B, N, 3(+F)]
                        z,                     # [B, K, 3]  (keypoint centers)
                        z_scale,               # [B, K, 3]  (logits; optional – can pass None)
                        deterministic=False,
                        obj_on=None,           # [B, K] or [B, K, 1]
                        mask_pc=None):         # [B, N] (bool) True=valid

        """
        Point-cloud wrapper for appearance encoding.

        - Delegates to self.particle_features_enc (your ParticleFeaturesEncoderPoint).
        - That module already handles:
            * KNN neighborhood gathering
            * 3D "spatial transform" via translate+scale into canonical cube
            * PointNet-like encoding + pooling
            * optional obj_on gating (using its own null_feature_embed)
            * sampling (unless interaction_features=True)
        - We simply standardize the returned dict to mirror the 2D contract.
        """

        enc_out = self.particle_features_enc(
            points=points,
            kp=z,
            z_scale=z_scale,
            deterministic=deterministic,
            obj_on=obj_on,
            mask=mask_pc
        )

        mu_features     = enc_out['mu_features']            # [B,K,F]
        logvar_features = enc_out['logvar_features']        # [B,K,F] or None
        z_features      = enc_out['z_features']             # [B,K,F]


        return {
            'mu_features':           mu_features,
            'logvar_features':       logvar_features,
            'z_features':            z_features,
            'mu_features_total':     mu_features,
            'logvar_features_total': logvar_features,
            'z_features_total':      z_features,
        }


    def encode_all(self,
                points: torch.Tensor,           # [B, N, 3(+F)]
                mask_pc: torch.Tensor = None,   # [B, N] (bool) True=valid
                deterministic: bool = False,
                warmup: bool = False):
        """
        Single-frame point-cloud pipeline:
        1) propose positions/scales via prior (+ attribute head)
        2) optionally variance-filter if enc<dec and using null-embed
        3) encode appearance for the selected set
        4) return a compact dict (no time dims)
        """
        assert points.dim() == 3 and points.size(-1) >= 3, \
            f"expected [B,N,3(+F)], got {tuple(points.shape)}"
        if mask_pc is not None:
            assert mask_pc.shape[:2] == points.shape[:2], \
                f"mask_pc must be [B,N], got {tuple(mask_pc.shape)}"

        B = points.size(0)

        # ---- 1) stage-1: positions & scales from prior+attributes ----
        s1 = self.encode_pos_scale_with_prior(points,
                                            mask=mask_pc,
                                            deterministic=deterministic,
                                            warmup=warmup)

        # unpack stage-1 we actually use
        kp_p         = s1['kp_p']                  # [B,K,3]
        var_kp       = s1['var_kp']                # [B,K,*]
        z_base       = s1['z_base']                # [B,K,3]
        z            = s1['z']                     # [B,K,3]
        mu_tot       = s1['mu_tot']                # [B,K,3]
        z_base_var   = s1['z_base_var']            # [B,K,Dv]
        mu_scale     = s1['mu_scale']              # [B,K,3]
        logvar_scale = s1.get('logvar_scale', None)
        z_scale      = s1['z_scale']               # [B,K,3]

        # optional extras (presence/depth etc.)
        obj_on_a     = s1.get('obj_on_a', None)
        obj_on_b     = s1.get('obj_on_b', None)
        mu_obj_on    = s1.get('mu_obj_on', None)
        z_obj_on     = s1.get('z_obj_on', None)

        mu_depth     = s1.get('mu_depth', None)
        logvar_depth = s1.get('logvar_depth', None)
        z_depth      = s1.get('z_depth', None)

        # ---- 2) variance-based filtering if you plan to encode fewer particles later ----
        if self.n_kp_enc != self.n_kp_dec and self.interaction_features and self.use_null_features_embed:
            total_var = z_base_var.sum(-1)                                     # [B,K]
            n_filter  = self.n_kp_dec if not warmup else min(
                self.n_kp_dec, int(self.warmup_n_kp_ratio * self.n_kp_enc)
            )
            _, keep_idx = torch.topk(total_var, k=n_filter, dim=-1, largest=False)   # [B,n_filter]
            bidx = torch.arange(B, device=points.device)[:, None]

            # subset for appearance stage
            z_app       = z[bidx, keep_idx].contiguous()        # [B,n_filter,3]
            z_scale_app = z_scale[bidx, keep_idx].contiguous()  # [B,n_filter,3]

            # ---- 3) stage-2: appearance on subset ----
            s2 = self.encode_appearance(points, z_app, z_scale_app,
                                        deterministic=deterministic,
                                        obj_on=None,
                                        mask_pc=mask_pc)

            # scatter back into full K slots using nulls/zeros
            Fdim = s2['mu_features'].size(-1)
            if self.use_null_features_embed:
                mu_features_full = self.null_feature_embed.repeat(B, self.n_kp_enc, 1)
            else:
                mu_features_full = torch.zeros(B, self.n_kp_enc, Fdim, device=points.device)
            z_features_full = mu_features_full.clone()
            logvar_features_full = None
            if s2['logvar_features'] is not None:
                logvar_features_full = torch.zeros(B, self.n_kp_enc, Fdim, device=points.device)

            mu_features_full[bidx, keep_idx] = s2['mu_features']
            z_features_full[bidx, keep_idx]  = s2['z_features']
            if logvar_features_full is not None:
                logvar_features_full[bidx, keep_idx] = s2['logvar_features']

            # keep stage-1 tensors consistent with the selected set
            take = keep_idx
            z           = z[bidx, take]
            z_scale     = z_scale[bidx, take]
            z_base      = z_base[bidx, take]
            mu_tot      = mu_tot[bidx, take]
            kp_p        = kp_p[bidx, take]
            var_kp      = var_kp[bidx, take]
            z_base_var  = z_base_var[bidx, take]
            mu_scale    = mu_scale[bidx, take]
            if logvar_scale is not None:
                logvar_scale = logvar_scale[bidx, take]

            mu_features     = mu_features_full
            z_features      = z_features_full
            logvar_features = logvar_features_full
            print("Encoded appearance Logvar features is None? ", logvar_features is None)
        else:
            # ---- 3) stage-2: appearance for all proposals ----
            s2 = self.encode_appearance(points, z, z_scale,
                                        deterministic=deterministic,
                                        obj_on=None,
                                        mask_pc=mask_pc)
            mu_features     = s2['mu_features']           # [B,K,F]
            logvar_features = s2['logvar_features']       # [B,K,F] or None
            z_features      = s2['z_features']            # [B,K,F]

            print("Encoded all appearance Logvar features is None? ", logvar_features is None)

        # ---- 4) final compact dict (point-cloud only; no BG) ----
        return {
            # positions
            'pos_anchor':   z_base,            # [B,K,3]
            'pos':          z,                 # [B,K,3]
            'pos_mu':       mu_tot,            # [B,K,3]
            'pos_logvar':   s1['logvar_offset'],  # [B,K,3] if produced

            # scales
            'scale_mu':     mu_scale,          # [B,K,3]
            'scale_logvar': logvar_scale,      # [B,K,3] or None
            'scale':        z_scale,           # [B,K,3]

            # features
            'feat_mu':      mu_features,       # [B,K,F]
            'feat_logvar':  logvar_features,   # [B,K,F] or None
            'feat':         z_features,        # [B,K,F]

            # prior proposal info
            'kp_p':         kp_p,              # [B,K,3]
            'kp_var':       var_kp,            # [B,K,*]
            'kp_score':     z_base_var.sum(-1, keepdim=True),  # [B,K,1]

            # optional extras if your stage-1 produced them
            'obj_on_a':     obj_on_a,
            'obj_on_b':     obj_on_b,
            'mu_obj_on':    mu_obj_on,
            'z_obj_on':     z_obj_on,
            'mu_depth':     mu_depth,
            'logvar_depth': logvar_depth,
            'z_depth':      z_depth,
            'z_base_var':   z_base_var,      # [B,K,Dv]
        }



    def forward(self, x, mask, deterministic=False, warmup=False):
        output_dict = self.encode_all(x, mask, deterministic, warmup)
        return output_dict
