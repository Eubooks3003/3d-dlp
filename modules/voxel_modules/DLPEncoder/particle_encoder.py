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
from modules.voxel_modules.voxel_grid import tile_ids_norm_xyz_to_dhw

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
                 depth_feature_dim=16,
                 # Voxel
                 voxel_grid_whd=(48,48,48)):
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
            in_channels=self.cdim,
            grid=voxel_grid_whd,           # TODO: Remove this directly query from vox
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

        self.training = True
        extra_feats = 0   # per-point features beyond xyz from the dataset
        self.voxel_grid_whd = voxel_grid_whd
        # TODO: Make this gettable via the dataloader 

        # --- Particle attributes (offset/scale/obj_on/depth) ---
        self.attr_enc_3d = ParticleAttributeEncoder3D(
            anchor_size=0.25,                 # not used since crop_size is given
            grid_dhw=voxel_grid_whd,
            n_particles=self.n_kp_prior,      # or self.n_kp_enc — whichever you use downstream
            ch_in=self.cdim,                # was `in_channels`
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

    def _relaxed_topk_weights(self, score, k, valid_mask, tau=0.3, training=True):
        """
        score, valid_mask: [B,K]  (larger score is better)
        returns:
        w_keep:  [B,K] in [0,1] (soft in train, hard one-hot in eval)
        top_idx: [B,k] hard indices for fast paths
        """
        score_masked = torch.where(valid_mask, score, score.new_full(score.shape, -1e9))
        top_idx = torch.topk(score_masked, k, dim=-1).indices
        if not training:
            hard = torch.zeros_like(score_masked).scatter(1, top_idx, 1.0)
            return hard, top_idx
        w_soft = torch.softmax(score_masked / tau, dim=-1)                   # [B,K]
        w_soft = w_soft * (k / (w_soft.sum(-1, keepdim=True) + 1e-8))        # sum ≈ k
        hard  = torch.zeros_like(score_masked).scatter(1, top_idx, 1.0)
        w_keep = hard + (w_soft - w_soft.detach())                            # STE
        return w_keep, top_idx

    def _soft_assign(self, points_xyz, kp_xyz, tau=0.07):
        """
        Soft point→KP ownership.
        points_xyz: [B,N,3], kp_xyz: [B,K,3]  →  [B,N,K]
        """
        d2 = ((points_xyz[:, :, None, :] - kp_xyz[:, None, :, :])**2).sum(-1)  # [B,N,K]
        return torch.softmax(-d2 / tau, dim=-1)

    def encode_pos_scale_with_prior(self, x, dense, mask=None, deterministic=False, warmup=False, timesteps=None):
        """
        Voxel-context version aligned with the RGB path.
        x:     [B, N, 3]
        dense: [B, C, D, H, W]
        """
        B = x.shape[0]

        # --- prior proposals (train returns relaxed Top-K weights) ---
        if self.training:
            kp_p, var_kp, kp_tiles, prior_keep_w = self.encode_prior(dense)
        else:
            kp_p, var_kp, kp_tiles = self.encode_prior(dense)
            prior_keep_w = torch.ones(kp_p.shape[:2], device=kp_p.device, dtype=kp_p.dtype)
        K = kp_p.shape[1]

        # keep grads, break aliasing like RGB
        mu     = kp_p.clone()                           # [B,K,3]
        logvar = torch.zeros_like(mu)                   # [B,K,3]
        z_base = mu + 0.0 * logvar                      # [B,K,3]

        # --- posterior offsets & scales ---
        particle_stats_dict = self.attr_enc_3d(
            x_dense=dense,
            kp_xyz=z_base.clone(),                      # avoid in-place aliasing
            z_scale=None,
            timesteps=timesteps,
            deterministic=deterministic
        )

        mu_offset     = particle_stats_dict['mu']                   # [B,K,3]
        logvar_offset = particle_stats_dict['logvar']               # [B,K,3]
        mu_scale      = particle_stats_dict.get('mu_scale', None)   # [B,K,3] or None
        logvar_scale  = particle_stats_dict.get('logvar_scale', None)

        if not self.interaction_obj_on:
            lobj_on_a  = particle_stats_dict.get('lobj_on_a', None)
            lobj_on_b  = particle_stats_dict.get('lobj_on_b', None)
            obj_on_a   = particle_stats_dict.get('obj_on_a',  None) # [B,K,1]
            obj_on_b   = particle_stats_dict.get('obj_on_b',  None)
            mu_obj_on  = particle_stats_dict.get('mu_obj_on', None) # [B,K,1]
            z_obj_on   = particle_stats_dict.get('z_obj_on',  None) # [B,K,1]
        else:
            obj_on_a = obj_on_b = z_obj_on = mu_obj_on = None

        if not self.interaction_depth:
            mu_depth     = particle_stats_dict.get('mu_depth', None)     # [B,K,1]
            logvar_depth = particle_stats_dict.get('logvar_depth', None) # [B,K,1]
            z_depth = (mu_depth if deterministic or (mu_depth is None or logvar_depth is None)
                    else reparameterize(mu_depth, logvar_depth))
        else:
            mu_depth = logvar_depth = z_depth = None

        # final z, z_scale
        mu_tot     = z_base + mu_offset
        logvar_tot = logvar_offset
        if mu_scale is not None and hasattr(self, 'mu_scale_prior') and self.mu_scale_prior is not None:
            mu_scale = self.mu_scale_prior + mu_scale
        z_offset = (mu_offset if deterministic else reparameterize(mu_offset, logvar_offset))
        z_scale  = (mu_scale  if deterministic or (mu_scale is None or logvar_scale is None)
                    else reparameterize(mu_scale, logvar_scale))
        z = z_base + z_offset

        # --- variance features / scores (for ranking only) ---
        z_base_var = var_kp.reshape(B, K, -1).detach()              # [B,K,9]
        confidence_score = logvar_offset.detach()                   # [B,K,3]
        z_base_var = torch.cat([z_base_var, confidence_score], dim=-1)  # [B,K,12]
        z_base_id  = torch.arange(K, device=z_base.device)[None, :, None].repeat(B, 1, 1)

        mu_score     = (z_base_var.sum(-1, keepdim=True) / 30.0) * 2 - 1
        logvar_score = torch.full_like(mu_score, math.log(0.2 ** 2))
        z_score      = mu_score
        total_var    = z_base_var.sum(-1)                            # [B,K]

        # --- optional variance filtering: train-soft / eval-hard ---
        if self.n_kp_enc < self.n_kp_prior:
            n_filter = (self.n_kp_enc if not warmup
                        else min(self.n_kp_enc, int(self.warmup_n_kp_ratio * self.n_kp_prior)))

            # validity (don’t select degenerate proposals)
            pos_valid = (kp_p.abs().sum(dim=-1) > 1e-6)               # [B,K]
            cov_diag  = var_kp.diagonal(dim1=-2, dim2=-1)             # [B,K,3]
            cov_valid = (cov_diag.sum(dim=-1) > 1e-9)                 # [B,K]
            valid     = (pos_valid | cov_valid)

            # rank by low variance
            score = -total_var                                        # [B,K]

            if self.training:
                w_var, _ = self._relaxed_topk_weights(
                    score, n_filter, valid,
                    tau=getattr(self, "relaxed_topk_tau", 0.3),
                    training=True
                )                                                     # [B,K]
                # combine with prior Top-K softness so both gates contribute
                keep_w = (w_var * prior_keep_w).clamp_min(0)          # [B,K]
                # NOTE: we do NOT slice tensors in training; downstream can use keep_w to weight per-K losses.
            else:
                # eval: slice hard
                _, embed_ind = torch.topk(score.masked_fill(~valid, float("-inf")),
                                        k=n_filter, dim=-1, largest=True)
                b_idx = torch.arange(B, device=x.device)[:, None]
                def take(t):
                    return (t[b_idx, embed_ind] if (t is not None) else None)
                mu_tot        = take(mu_tot);        z_base        = take(z_base)
                z_base_var    = take(z_base_var);    z_base_id     = take(z_base_id)
                mu_offset     = take(mu_offset);     logvar_offset = take(logvar_offset)
                z             = take(z);             z_offset      = take(z_offset)
                z_scale       = take(z_scale);       mu_scale      = take(mu_scale)
                mu_score      = take(mu_score);      logvar_score  = take(logvar_score)
                z_score       = take(z_score);       kp_p          = take(kp_p)
                var_kp        = take(var_kp);        total_var     = take(total_var)
                kp_tiles      = take(kp_tiles)
                if logvar_scale is not None: logvar_scale = take(logvar_scale)
                if not self.interaction_obj_on:
                    obj_on_a   = take(obj_on_a);   obj_on_b   = take(obj_on_b)
                    mu_obj_on  = take(mu_obj_on);  z_obj_on   = take(z_obj_on)
                if not self.interaction_depth:
                    z_depth      = take(z_depth);   mu_depth     = take(mu_depth)
                    logvar_depth = take(logvar_depth)
                if getattr(self, 'embed_prior_patch_pos', False):
                    patch_id_embed = take(patch_id_embed) if 'patch_id_embed' in locals() else None
        else:
            keep_w = prior_keep_w  # all K active

        out = {
            'mu': mu, 'logvar': logvar,
            'z_base': z_base, 'z': z, 'mu_tot': mu_tot,
            'patch_id_embed': None,
            'mu_scale': mu_scale, 'logvar_scale': logvar_scale, 'z_scale': z_scale,
            'mu_depth': mu_depth, 'logvar_depth': logvar_depth, 'z_depth': z_depth,
            'mu_offset': mu_offset, 'logvar_offset': logvar_offset, 'z_offset': z_offset,
            'kp_p': kp_p, 'var_kp': var_kp,
            'z_base_var': z_base_var, 'total_var': total_var,
            'obj_on_a': obj_on_a, 'obj_on_b': obj_on_b, 'z_obj_on': z_obj_on, 'mu_obj_on': mu_obj_on,
            'z_base_id': z_base_id,
            'mu_score': mu_score, 'logvar_score': logvar_score, 'z_score': z_score,
            'kp_tiles': kp_tiles,
        }
        # expose keep weights for downstream weighting (optional)
        if self.training:
            out['prior_keep_w'] = keep_w if 'keep_w' in locals() else prior_keep_w
        return out




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
                        z,                     # [B, K, 3]
                        z_scale,               # [B, K, 3] or None
                        deterministic=False,
                        obj_on=None,           # [B,K] or [B,K,1]
                        mask_pc=None,
                        kp_tiles=None,         # [B,K] (tile ids) for eval/fast path
                        soft_weights=None):    # [B,N,K] optional soft ownership (train)

        """
        Point-cloud wrapper for appearance encoding.

        If soft_weights is provided (training), we pool with soft assignments.
        Otherwise we fall back to discrete tile/KNN path (eval).
        """

        if soft_weights is None and self.training:
            # build soft assignment on-the-fly if the caller didn’t pass one
            tau = getattr(self, 'ownership_tau', 0.10)
            soft_weights = self._soft_assign(points[..., :3], z, tau=tau)  # [B,N,K]

        if soft_weights is not None:
            enc_out = self.particle_features_enc(
                points=points,
                kp=z,
                z_scale=z_scale,
                deterministic=deterministic,
                obj_on=obj_on,
                mask=mask_pc,
                soft_weights=soft_weights,       # NEW: your ParticleFeaturesEncoderPoint should consume this
                point_tile_ids=None,
                kp_tile_ids=None,
            )
        else:
            # eval / ablation: discrete fast path
            point_tile_ids = tile_ids_norm_xyz_to_dhw(
                points[..., :3],
                grid_DHW=self.voxel_grid_whd,
                tile_DHW=(12, 12, 12)
            )
            enc_out = self.particle_features_enc(
                points=points,
                kp=z,
                z_scale=z_scale,
                deterministic=deterministic,
                obj_on=obj_on,
                mask=mask_pc,
                point_tile_ids=point_tile_ids,
                kp_tile_ids=kp_tiles,
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

        assert points.dim() == 3 and points.size(-1) >= 3, f"expected [B,N,3(+F)], got {tuple(points.shape)}"
        if mask_pc is not None:
            assert mask_pc.shape[:2] == points.shape[:2], f"mask_pc must be [B,N], got {tuple(mask_pc.shape)}"

        B, N = points.shape[:2]
        dbg = getattr(self, "debug_prints", False)

        # ---- stage-1: pos & scale (prior + attributes) ----
        s1 = self.encode_pos_scale_with_prior(points, dense,
                                            mask=mask_pc,
                                            deterministic=deterministic,
                                            warmup=warmup)

        # unpack (stage-1, FULL K)
        kp_p         = s1['kp_p']           # [B,K,3]
        var_kp       = s1['var_kp']         # [B,K,*]
        z_base       = s1['z_base']         # [B,K,3]
        z            = s1['z']              # [B,K,3]
        mu_tot       = s1['mu_tot']         # [B,K,3]
        z_base_var   = s1['z_base_var']     # [B,K,Dv]

        mu_scale     = s1['mu_scale']       # [B,K,3]
        logvar_scale = s1.get('logvar_scale', None)
        z_scale      = s1['z_scale']        # [B,K,3]

        mu_offset     = s1['mu_offset']     # [B,K,3]
        logvar_offset = s1['logvar_offset'] # [B,K,3]
        z_offset      = s1['z_offset']      # [B,K,3]

        obj_on_a   = s1.get('obj_on_a', None)  # [B,K,1] or logits
        obj_on_b   = s1.get('obj_on_b', None)
        mu_obj_on  = s1.get('mu_obj_on', None)
        z_obj_on   = s1.get('z_obj_on', None)  # [B,K,1] (what VoxelDLP.forward reads)

        mu_depth     = s1.get('mu_depth', None)
        logvar_depth = s1.get('logvar_depth', None)
        z_depth      = s1.get('z_depth', None)

        kp_tiles     = s1.get('kp_tiles', None)

        # presence prob (for gating)
        if (obj_on_a is not None) and (obj_on_b is not None):
            p_on = (obj_on_a / (obj_on_a + obj_on_b + 1e-6)).clamp(0, 1)  # [B,K,1]
        else:
            p_on = None

        # ---------- soft point→KP ownership for training ----------
        if self.training:
            tau = getattr(self, "ownership_tau", 0.10)
            soft_w = self._soft_assign(points[..., :3], z, tau=tau)  # [B,N,K]
            if mask_pc is not None:
                # add -big to masked points before softmax (assume _soft_assign returns probs; if it returns logits, move this inside)
                # If _soft_assign already returns normalized weights, renormalize after zeroing masked rows:
                soft_w = soft_w * mask_pc.unsqueeze(-1).to(soft_w.dtype)      # zero out pad rows
                denom = soft_w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                soft_w = soft_w / denom
            use_tiles = None
        else:
            soft_w = None
            use_tiles = kp_tiles

        # ---- optional: appearance on subset, but KEEP full-K tensors ----
        use_subset = (self.n_kp_enc != self.n_kp_dec) and self.interaction_features and self.use_null_features_embed
        K = z.shape[1]

        if use_subset:
            total_var = z_base_var.sum(-1)  # [B,K]
            n_filter = self.n_kp_dec if not warmup else min(self.n_kp_dec, int(self.warmup_n_kp_ratio * self.n_kp_enc))
            n_filter = max(1, n_filter)

            if self.training:
                # relaxed top-k weights (no slicing)
                pos_valid = (kp_p.abs().sum(dim=-1) > 1e-6)                  # [B,K]
                cov_diag  = var_kp.diagonal(dim1=-2, dim2=-1)                # [B,K,3]
                cov_valid = (cov_diag.sum(dim=-1) > 1e-9)                    # [B,K]
                valid     = (pos_valid | cov_valid)

                score = -total_var                                          # low variance = good
                keep_w_app, _ = self._relaxed_topk_weights(
                    score, n_filter, valid,
                    tau=getattr(self, "relaxed_topk_tau", 0.3),
                    training=True
                )  # [B,K] mixture over K

                s1['keep_w_app'] = keep_w_app

                # full-K appearance; let encode_appearance optionally use keep_w_app if it wants
                s2 = self.encode_appearance(points, z, z_scale,
                                            deterministic=deterministic,
                                            obj_on=p_on,
                                            mask_pc=mask_pc,
                                            kp_tiles=use_tiles,
                                            soft_weights=soft_w)
                mu_features     = s2['mu_features']   # [B,K,F]
                logvar_features = s2['logvar_features']
                z_features      = s2['z_features']

            else:
                # hard top-k for speed, but keep full-K tensors by scattering
                _, keep_idx = torch.topk(total_var, k=n_filter, dim=-1, largest=False)  # [B,nf]
                bidx = torch.arange(B, device=points.device)[:, None]

                z_app       = z[bidx, keep_idx].contiguous()        # [B,nf,3]
                z_scale_app = z_scale[bidx, keep_idx].contiguous()  # [B,nf,3]
                p_on_app    = p_on[bidx, keep_idx] if p_on is not None else None

                # gather soft_w per batch for the kept KPs if available
                if soft_w is None:
                    soft_w_app = None
                else:
                    # soft_w: [B,N,K] -> [B,N,nf] per batch gather
                    # build an index tensor for advanced gather
                    gather_idx = keep_idx.unsqueeze(1).expand(B, N, n_filter)  # [B,N,nf]
                    soft_w_app = torch.gather(soft_w, dim=2, index=gather_idx)

                s2 = self.encode_appearance(points, z_app, z_scale_app,
                                            deterministic=deterministic,
                                            obj_on=p_on_app,
                                            mask_pc=mask_pc,
                                            kp_tiles=use_tiles,
                                            soft_weights=soft_w_app)

                # scatter back into full K slots
                Fdim = s2['mu_features'].size(-1)
                mu_features = torch.zeros(B, K, Fdim, device=points.device, dtype=s2['mu_features'].dtype)
                z_features  = torch.zeros_like(mu_features)
                logvar_features = (torch.zeros_like(mu_features)
                                if s2['logvar_features'] is not None else None)

                mu_features[bidx, keep_idx] = s2['mu_features']
                z_features[bidx, keep_idx]  = s2['z_features']
                if logvar_features is not None:
                    logvar_features[bidx, keep_idx] = s2['logvar_features']

                if dbg:
                    print(f"[encode_all] eval subset: computed appearance on {n_filter}/{K} and scattered back to full K")

        else:
            # full-K appearance
            s2 = self.encode_appearance(points, z, z_scale,
                                        deterministic=deterministic,
                                        obj_on=p_on,        # may be None
                                        mask_pc=mask_pc,
                                        kp_tiles=use_tiles,
                                        soft_weights=soft_w)
            mu_features     = s2['mu_features']   # [B,K,F]
            logvar_features = s2['logvar_features']
            z_features      = s2['z_features']

        # ---- return dict matching VoxelDLP.forward() keys ----
        return {
            # anchors / positions
            'z_base':          z_base,            # [B,K,3]
            'z':               z,                 # [B,K,3]
            'mu_tot':          mu_tot,            # [B,K,3]
            'mu_offset':       mu_offset,         # [B,K,3]
            'logvar_offset':   logvar_offset,     # [B,K,3]

            # scales
            'mu_scale':        mu_scale,          # [B,K,3]
            'logvar_scale':    logvar_scale,      # [B,K,3] or None
            'z_scale':         z_scale,           # [B,K,3]

            # features
            'mu_features':     mu_features,       # [B,K,F]
            'logvar_features': logvar_features,   # [B,K,F] or None
            'z_features':      z_features,        # [B,K,F]

            # presence / depth
            'obj_on_a':        obj_on_a,
            'obj_on_b':        obj_on_b,
            'mu_obj_on':       mu_obj_on,
            'z_obj_on':        z_obj_on,          # [B,K,1] or None

            'mu_depth':        mu_depth,
            'logvar_depth':    logvar_depth,
            'z_depth':         z_depth,

            # prior proposal info
            'kp_p':            kp_p,              # [B,K,3]
            'var_kp':          var_kp,            # [B,K,*]
            'z_base_var':      z_base_var,        # [B,K,Dv]

            # BG features (not used yet)
            'mu_bg_features':     None,
            'logvar_bg_features': None,
            'z_bg_features':      None,

            # optional training diagnostics
            **({'keep_w_app': s1.get('keep_w_app')} if (self.training and ('keep_w_app' in s1)) else {}),
            'patch_id_embed': None,
            'z_offset':        z_offset,          # keep if you use it elsewhere
        }




    def forward(self, x, dense, mask,deterministic=False, warmup=False):
        output_dict = self.encode_all(x, dense, mask, deterministic, warmup)
        return output_dict
