import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util_func import reparameterize

import numpy as np

from modules.voxel_modules.DLPEncoder.DLPPrior.dlp_prior import DLPPrior
from modules.voxel_modules.DLPEncoder.particle_attribute_encoder import ParticleAttributeEncoder3D
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
        in_channels = 7
        self.prior_encoder = DLPPrior(
            in_channels=in_channels,
            grid=(48, 48, 48),           # TODO: Remove this directly query from vox
            tile_size=(12, 12, 12),      # must divide the padded grid; 48 works perfectly
            base_ch=32,
            ch_mult=(1, 2, 3),
            num_res_blocks=2,
            use_resblock=True,
            use_attention=False,
            cnn_mid_blocks=False,        # False -> keep mid blocks in Encoder3D
            n_kp_per_tile=1,             # K channels per tile
            n_kp_prior=n_kp_prior,       # final number to return (<= tiles * n_kp_per_tile)
            kp_range=(-1., 1.),
            temperature=1.0,             # try 0.25–0.5 for sharper peaks
            filtering_heuristic='variance',  # 'variance' | 'random' | 'none'
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
        self.attr_enc_3d = ParticleAttributeEncoder3D(
            anchor_size=0.25,                 # not used since crop_size is given
            grid_dhw=(48, 48, 48),
            n_particles=self.n_kp_prior,      # or self.n_kp_enc — whichever you use downstream
            ch_in=in_channels,                # was `in_channels`
            crop_size=(16, 16, 16),

            base_ch=32, ch_mult=(1, 2, 3), num_res_blocks=2,
            use_resblock=True, cnn_mid_blocks=False, use_attention=False,
            hidden_dim=512, activation='gelu',
            kp_activation='tanh', max_offset=1.0,

            obj_on=True,                      # was `with_obj_on`
            depth=False,                      # set True if you want depth head
            scale=True,                       # was `with_scale`

            timestep_horizon=1, add_particle_temp_embed=False, init_std=0.2,
            init_zero_bias=True, init_conv_layers=True, init_conv_fg_std=0.02,
            obj_on_min=1e-4, obj_on_max=100.0
        )

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

    def encode_prior(self, dense):
        return self.prior_encoder(dense)


    def encode_pos_scale_with_prior(self, x, dense, mask=None, deterministic=False, warmup=False, timesteps=None):
        """
        Voxel-context version aligned with the RGB path.
        x:     [B, N, 3]
        dense: [B, C, D, H, W]
        """
        B = x.shape[0]

        # --- prior proposals ---
        kp_p, var_kp = self.encode_prior(dense)              # kp_p:[B,K,3], var_kp:[B,K,3,3]
        K = kp_p.shape[1]

        # keep grads, break aliasing exactly like RGB
        mu     = kp_p.clone()                                 # [B,K,3]
        logvar = torch.zeros_like(mu)                         # [B,K,3]
        z_base = mu + 0.0 * logvar                            # [B,K,3]

        # --- posterior offsets & scales (attribute encoder) ---
        particle_stats_dict = self.attr_enc_3d(
            x_dense=dense,
            kp_xyz=z_base.clone(),                            # keep grads, avoid in-place aliasing
            z_scale=None,
            timesteps=timesteps,
            deterministic=deterministic
        )

        mu_offset     = particle_stats_dict['mu']             # [B,K,3]
        logvar_offset = particle_stats_dict['logvar']         # [B,K,3]
        mu_scale      = particle_stats_dict.get('mu_scale', None)   # [B,K,3] or None
        logvar_scale  = particle_stats_dict.get('logvar_scale', None)

        if not self.interaction_obj_on:
            # match RGB keys/behavior
            lobj_on_a  = particle_stats_dict.get('lobj_on_a', None) # kept for parity (may be None)
            lobj_on_b  = particle_stats_dict.get('lobj_on_b', None)
            obj_on_a   = particle_stats_dict.get('obj_on_a',  None) # [B,K,1] or None
            obj_on_b   = particle_stats_dict.get('obj_on_b',  None)
            mu_obj_on  = particle_stats_dict.get('mu_obj_on', None) # [B,K,1] or None
            z_obj_on   = particle_stats_dict.get('z_obj_on',  None) # [B,K,1] or None
        else:
            obj_on_a = obj_on_b = z_obj_on = mu_obj_on = None

        if not self.interaction_depth:
            mu_depth     = particle_stats_dict.get('mu_depth', None)     # [B,K,1] or None
            logvar_depth = particle_stats_dict.get('logvar_depth', None) # [B,K,1] or None
            if mu_depth is not None and logvar_depth is not None:
                z_depth = mu_depth if deterministic else reparameterize(mu_depth, logvar_depth)
            else:
                z_depth = None
        else:
            mu_depth = logvar_depth = z_depth = None

        # --- final position & scale (match RGB logic) ---
        mu_tot    = z_base + mu_offset                         # [B,K,3]
        logvar_tot = logvar_offset

        if mu_scale is not None:
            # if you have a learned prior term (parity with RGB)
            if hasattr(self, 'mu_scale_prior') and self.mu_scale_prior is not None:
                mu_scale = self.mu_scale_prior + mu_scale

        if deterministic:
            z_offset = mu_offset
            z_scale  = mu_scale
        else:
            z_offset = reparameterize(mu_offset, logvar_offset)
            z_scale  = (reparameterize(mu_scale, logvar_scale)
                        if (mu_scale is not None and logvar_scale is not None) else None)

        z = z_base + z_offset                                  # [B,K,3]

        # --- variance features / scores (match RGB) ---
        # var_kp: [B,K,3,3] -> [B,K,9]
        z_base_var = var_kp.reshape(B, K, -1).detach()         # scoring only; safe to detach
        confidence_score = logvar_offset.detach()              # scoring only; safe to detach
        z_base_var = torch.cat([z_base_var, confidence_score], dim=-1)  # [B,K,12]

        z_base_id = torch.arange(K, device=z_base.device)[None, :, None].repeat(B, 1, 1)  # [B,K,1]

        if getattr(self, 'embed_prior_patch_pos', False):
            # keep API parity; voxel path typically doesn't use this
            patch_id_embed = self.patch_id_embed.repeat(mu_tot.shape[0], 1, 1)
        else:
            patch_id_embed = None

        mu_score     = (z_base_var.sum(-1, keepdim=True) / 30.0) * 2 - 1  # [B,K,1]
        logvar_score = torch.full_like(mu_score, math.log(0.2 ** 2))      # [B,K,1]
        z_score      = mu_score

        total_var = z_base_var.sum(-1)                                    # [B,K]

        # --- variance filtering (identical structure to RGB) ---
        if self.n_kp_enc < self.n_kp_prior:
            n_filter = (self.n_kp_enc if not warmup
                        else min(self.n_kp_enc, int(self.warmup_n_kp_ratio * self.n_kp_prior)))

            # Build a validity mask: mark proposals that are non-empty / meaningful.
            # Two quick heuristics (use either or both):
            #  1) position not all-zeros
            #  2) covariance not (near-)zero
            pos_valid = (kp_p.abs().sum(dim=-1) > 1e-6)                      # [B,K]
            cov_diag  = var_kp.diagonal(dim1=-2, dim2=-1)                    # [B,K,3]
            cov_valid = (cov_diag.sum(dim=-1) > 1e-9)                        # [B,K]
            valid     = (pos_valid | cov_valid)                              # [B,K]

            # Penalize invalids so they are NEVER selected as low-variance
            big = torch.finfo(total_var.dtype).max / 4.0
            total_var_masked = torch.where(valid, total_var, torch.full_like(total_var, big))

            # Optional tiny jitter to break ties deterministically across GPUs
            total_var_masked = total_var_masked + 1e-9 * torch.arange(total_var_masked.shape[-1],
                                                                    device=total_var_masked.device).float()

            # keep low-var among valid ones
            _, embed_ind = torch.topk(total_var_masked, k=n_filter, dim=-1, largest=False)
            b_idx = torch.arange(B, device=x.device)[:, None]

            def take(t):
                return (t[b_idx, embed_ind] if (t is not None) else None)

            mu_tot        = take(mu_tot)
            z_base        = take(z_base)
            z_base_var    = take(z_base_var)
            z_base_id     = take(z_base_id)
            mu_offset     = take(mu_offset)
            logvar_offset = take(logvar_offset)
            z             = take(z)
            z_offset      = take(z_offset)
            z_scale       = take(z_scale)
            mu_scale      = take(mu_scale)
            mu_score      = take(mu_score)
            logvar_score  = take(logvar_score)
            z_score       = take(z_score)
            kp_p          = take(kp_p)
            var_kp        = take(var_kp)
            total_var     = take(total_var)

            if logvar_scale is not None:
                logvar_scale = take(logvar_scale)
            if not self.interaction_obj_on:
                obj_on_a   = take(obj_on_a)
                obj_on_b   = take(obj_on_b)
                mu_obj_on  = take(mu_obj_on)
                z_obj_on   = take(z_obj_on)
            if not self.interaction_depth:
                z_depth      = take(z_depth)
                mu_depth     = take(mu_depth)
                logvar_depth = take(logvar_depth)
            if patch_id_embed is not None:
                patch_id_embed = take(patch_id_embed)

        # print("KP FROM PARTICLE ENCODER: ", kp_p)
        out_dict = {
            'mu': mu, 'logvar': logvar,
            'z_base': z_base, 'z': z, 'mu_tot': mu_tot,
            'patch_id_embed': patch_id_embed,
            'mu_scale': mu_scale, 'logvar_scale': logvar_scale, 'z_scale': z_scale,
            'mu_depth': mu_depth, 'logvar_depth': logvar_depth, 'z_depth': z_depth,
            'mu_offset': mu_offset, 'logvar_offset': logvar_offset, 'z_offset': z_offset,
            'kp_p': kp_p, 'var_kp': var_kp,
            'z_base_var': z_base_var, 'total_var': total_var,
            'obj_on_a': obj_on_a, 'obj_on_b': obj_on_b, 'z_obj_on': z_obj_on, 'mu_obj_on': mu_obj_on,
            'z_base_id': z_base_id,
            'mu_score': mu_score, 'logvar_score': logvar_score, 'z_score': z_score,
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
                dense=None,
                mask_pc: torch.Tensor = None,   # [B, N] (bool) True=valid
                deterministic: bool = False,
                warmup: bool = False):

        assert points.dim() == 3 and points.size(-1) >= 3, \
            f"expected [B,N,3(+F)], got {tuple(points.shape)}"
        if mask_pc is not None:
            assert mask_pc.shape[:2] == points.shape[:2], \
                f"mask_pc must be [B,N], got {tuple(mask_pc.shape)}"

        B = points.size(0)

        # ---- stage-1: pos & scale ----
        s1 = self.encode_pos_scale_with_prior(points, dense,
                                            mask=mask_pc,
                                            deterministic=deterministic,
                                            warmup=warmup)

        # unpack
        kp_p         = s1['kp_p']
        var_kp       = s1['var_kp']
        z_base       = s1['z_base']
        z            = s1['z']
        mu_tot       = s1['mu_tot']
        z_base_var   = s1['z_base_var']

        mu_scale     = s1['mu_scale']
        logvar_scale = s1.get('logvar_scale', None)
        z_scale      = s1['z_scale']

        mu_offset     = s1['mu_offset']
        logvar_offset = s1['logvar_offset']
        z_offset      = s1['z_offset']

        obj_on_a   = s1.get('obj_on_a', None)
        obj_on_b   = s1.get('obj_on_b', None)
        mu_obj_on  = s1.get('mu_obj_on', None)
        z_obj_on   = s1.get('z_obj_on', None)

        mu_depth     = s1.get('mu_depth', None)
        logvar_depth = s1.get('logvar_depth', None)
        z_depth      = s1.get('z_depth', None)

        # expected presence (for gating)
        if (obj_on_a is not None) and (obj_on_b is not None):
            p_on = (obj_on_a / (obj_on_a + obj_on_b + 1e-6)).clamp(0, 1)   # [B,K,1]
        else:
            p_on = None

        # ---- (optional) variance filtering if you later shrink K for appearance ----
        if self.n_kp_enc != self.n_kp_dec and self.interaction_features and self.use_null_features_embed:
            total_var = z_base_var.sum(-1)                                  # [B,K]
            n_filter = self.n_kp_dec if not warmup else min(
                self.n_kp_dec, int(self.warmup_n_kp_ratio * self.n_kp_enc)
            )
            _, keep_idx = torch.topk(total_var, k=n_filter, dim=-1, largest=False)
            bidx = torch.arange(B, device=points.device)[:, None]

            z_app         = z[bidx, keep_idx].contiguous()
            z_scale_app   = z_scale[bidx, keep_idx].contiguous()
            p_on_app      = p_on[bidx, keep_idx] if p_on is not None else None  # [B,nf,1]

            # ---- stage-2: appearance on subset ----
            s2 = self.encode_appearance(points, z_app, z_scale_app,
                                        deterministic=deterministic,
                                        obj_on=p_on_app,     # gate with expected presence
                                        mask_pc=mask_pc)

            # scatter back into full K slots
            Fdim = s2['mu_features'].size(-1)
            mu_features_full = torch.zeros(B, z.shape[1], Fdim, device=points.device, dtype=s2['mu_features'].dtype)
            z_features_full  = torch.zeros_like(mu_features_full)
            logvar_features_full = (torch.zeros_like(mu_features_full)
                                    if s2['logvar_features'] is not None else None)

            mu_features_full[bidx, keep_idx] = s2['mu_features']
            z_features_full[bidx, keep_idx]  = s2['z_features']
            if logvar_features_full is not None:
                logvar_features_full[bidx, keep_idx] = s2['logvar_features']

            # reduce stage-1 to the kept subset for consistency
            z           = z_app
            z_scale     = z_scale_app
            z_base      = z_base[bidx, keep_idx]
            mu_tot      = mu_tot[bidx, keep_idx]
            kp_p        = kp_p[bidx, keep_idx]
            var_kp      = var_kp[bidx, keep_idx]
            z_base_var  = z_base_var[bidx, keep_idx]
            mu_scale    = mu_scale[bidx, keep_idx] if mu_scale is not None else None
            if logvar_scale is not None:
                logvar_scale = logvar_scale[bidx, keep_idx]
            mu_features     = mu_features_full
            z_features      = z_features_full
            logvar_features = logvar_features_full
            if p_on is not None:
                p_on = p_on_app

        else:
            # ---- stage-2: appearance for all K proposals ----
            s2 = self.encode_appearance(points, z, z_scale,
                                        deterministic=deterministic,
                                        obj_on=p_on,        # may be None -> ungated
                                        mask_pc=mask_pc)
            mu_features     = s2['mu_features']
            logvar_features = s2['logvar_features']
            z_features      = s2['z_features']

        # ---- final compact dict (PC path) ----
        return {
            # anchors / positions
            'mu_anchor':       z_base,
            'logvar_anchor':   torch.zeros_like(z_base),
            'z_base':          z_base,
            'z':               z,

            # offsets
            'mu_offset':       mu_offset,
            'logvar_offset':   logvar_offset,
            'z_offset':        z_offset,
            'mu_tot':          mu_tot,

            # scales
            'mu_scale':        mu_scale,
            'logvar_scale':    logvar_scale,
            'z_scale':         z_scale,

            # features
            'mu_features':     mu_features,
            'logvar_features': logvar_features,
            'z_features':      z_features,

            # depth-specific feature keys (not used in PC path)
            'z_depth_features':      None,
            'mu_depth_features':     None,
            'logvar_depth_features': None,

            # crops (not available in PC path)
            'cropped_objects': None,

            # prior proposal info
            'kp_p':        kp_p,
            'var_kp':      var_kp,
            'z_base_var':  z_base_var,

            # optional scores (kept compatible with RGB)
            'mu_score':     (z_base_var.sum(-1, keepdim=True) / 30.0) * 2 - 1,
            'logvar_score': torch.full((B, z_base.shape[1], 1),
                                    math.log(0.2 ** 2),
                                    device=z_base.device,
                                    dtype=z_base.dtype),
            'z_score':      (z_base_var.sum(-1, keepdim=True) / 30.0) * 2 - 1,

            # presence & depth (if produced)
            'obj_on_a':   obj_on_a,
            'obj_on_b':   obj_on_b,
            'mu_obj_on':  mu_obj_on,
            'z_obj_on':   z_obj_on,

            'mu_depth':     mu_depth,
            'logvar_depth': logvar_depth,
            'z_depth':      z_depth,

            # no kp_mask anymore
            'patch_id_embed': None,
        }



    def forward(self, x, dense, mask,deterministic=False, warmup=False):
        output_dict = self.encode_all(x, dense, mask, deterministic, warmup)
        return output_dict
