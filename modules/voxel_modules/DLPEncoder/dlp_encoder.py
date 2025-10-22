import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.voxel_modules.DLPEncoder.particle_encoder import ParticleEncoder
# from modules.point_cloud_modules.DLPEncoder.bg_encoder import BgEncoderPC
from modules.voxel_modules.DLPEncoder.bg_encoder import BgEncoder3D
from modules.point_cloud_modules.DLPEncoder.particle_interaction_encoder import ParticleInteractionEncoder3D

class DLPEncoder(nn.Module):
    def __init__(self,
                 # Input configuration
                 cdim=3,  # Number of input image channels
                 image_size=64,  # Input image size (assumed square)
                 n_views=1,  # number of input views (e.g., multiple cameras)
                 pad_mode='replicate',  # Padding mode for CNNs
                 dropout=0.0,  # Dropout rate (not typically used)

                 # Keypoint and patch configuration
                 n_kp_per_patch=1,  # Number of keypoints per patch
                 n_kp_prior=20,  # Number of keypoints to filter from proposals
                 patch_size=16,  # Patch size for keypoint proposal network
                 n_kp_enc=20,  # Number of posterior keypoints to learn
                 n_kp_dec=None,  # Number of keypoints for decoder (if different from encoder)
                 warmup_n_kp_ratio=0.35,
                 mask_bg_in_enc=True,  # before encoding the bg, mask with the particles' obj_on

                 # Feature dimensions
                 learned_feature_dim=16,  # Dimension of learned visual features
                 learned_bg_feature_dim=16,  # Dimension of background features
                 kp_range=(-1, 1),  # Range for keypoint coordinates
                 kp_activation="tanh",  # Activation for keypoint coordinates
                 anchor_s=0.25,  # Glimpse size ratio

                 # Network architecture
                 use_resblock=True,  # Use residual blocks
                 embed_init_std=0.02,  # Standard deviation for embedding initialization
                 projection_dim=128,  # Embedding dimension for transformer

                 # Transformer configuration
                 timestep_horizon=1,  # Maximum timesteps to process at once
                 pte_layers=1,  # Number of particle transformer encoder layers
                 pte_heads=1,  # Number of particle transformer encoder heads
                 context_dim=16,  # Context latent dimension
                 filtering_heuristic='none',  # Method to filter prior keypoints
                 attn_norm_type='rms',  # Normalization type for attention

                 # Object encoder configuration
                 obj_ch_mult_prior=(1, 2,),  # Channel multipliers for prior patch encoder (kp proposals)
                 obj_ch_mult=(1, 2, 3),  # Channel multipliers for object encoder
                 obj_base_ch=32,  # Base channels for object encoder
                 obj_final_cnn_ch=32,  # Final CNN channels for object encoder
                 cnn_mid_blocks=False,  # Use middle blocks in CNN
                 mlp_hidden_dim=256,  # Hidden dimension for MLPs
                 pte_inner_dim=256,  # Inner dimension for particle transformer

                 # Background decoder configuration
                 bg_ch_mult=(1, 2, 3),  # Channel multipliers for background encoder
                 bg_base_ch=32,  # Base channels for background encoder
                 bg_final_cnn_ch=32,  # Final CNN channels for background encoder
                 num_res_blocks=2,  # Number of residual blocks

                 # Interaction configuration
                 ctx_pool_mode='none',  # Mode for pooling context features
                 interaction_depth=True,  # Enable depth interaction between particles
                 interaction_obj_on=False,  # Enable transparency interaction
                 interaction_features=True,  # Enable feature interaction
                 particle_score=False,  # Use particle confidence scores

                 # Embedding options
                 add_particle_temp_embed=False,  # Add temporal embeddings to particles
                 particle_positional_embed=True,  # Add positional embeddings to particles

                 # Context modeling
                 ctx_enc=None,
                 causal_ctx=True,  # Use causal attention for context
                 pte_ctx_layers=1,  # Number of context transformer layers
                 pte_ctx_heads=1,  # Number of context transformer heads
                 ctx_dist='gauss',  # Distribution type for context
                 n_ctx_categories=4,  # Number of context categories
                 n_ctx_classes=4,  # Number of context classes per category
                 global_ctx_pool=False,  # learn global latent context in addition to per-particle context
                 pool_ctx_dim=256,  # pool dimension for the global ctx latent
                 n_pool_ctx_categories=8,  # Number of global context categories (if categorical)
                 n_pool_ctx_classes=4,  # Number of global context classes per category
                 global_local_fuse_mode='none',  # concatenate/add global and local z_ctx to condition the dynamics
                 condition_local_on_global=True,  # condition z_context on z_context_global

                 # Distribution configuration
                 features_dist='gauss',  # Distribution type for features
                 n_fg_categories=8,  # Number of foreground categories, 'categorical' dist
                 n_fg_classes=4,  # Number of foreground classes per category, 'categorical' dist
                 n_bg_categories=4,  # Number of background categories, 'categorical' dist
                 n_bg_classes=4,  # Number of background classes per category, 'categorical' dist
                 obj_on_min=1e-4,  # Minimum concentration in Beta dist transparency value
                 obj_on_max=100,  # Maximum concentration in Beta dist transparency value
                 use_z_orig=True,  # Use original patch center coordinates as features

                 # initialization
                 init_zero_bias=True,  # zero bias for conv and linear layers
                 init_ssm_last_layer=True,  # spatial softmax initialization
                 init_conv_layers=True,  # initialize conv layers with normal dist
                 init_conv_fg_std=0.02,  # std for conv fg normal dist
                 init_conv_bg_std=0.005,  # std for conv bg normal dist (<fg -> prioritize fg in learning)
                 #RGBD 
                 separate_depth_features=False, # use separate feature encoder for RGB and Depth channels
                 depth_feature_dim=16, # feature dimension for depth channel
                 # Voxel
                 voxel_grid_whd=(48,48,48),
                 ):
        """
        DLP Encoder Module

        A neural network module that extracts object-centric representations from images using
        the Deep Latent Particles (DLP) approach. This encoder processes images to identify
        and represent objects as particles with learned attributes.

        Args:
            cdim (int): Number of input image channels. Defaults to 3.
            image_size (int): Size of input images (assumed square). Defaults to 64.
            pad_mode (str): Padding mode for CNNs ('zeros' or 'replicate'). Defaults to 'replicate'.
            dropout (float): Dropout rate for CNNs (typically unused). Defaults to 0.0.
            n_kp_per_patch (int): Number of keypoints to extract per patch. Defaults to 1.
            n_kp_prior (int): Number of keypoints to filter from proposals. Defaults to 20.
            patch_size (int): Size of patches for keypoint proposal network. Defaults to 16.
            n_kp_enc (int): Number of posterior keypoints to learn. Defaults to 20.
            n_kp_dec (Optional[int]): Number of keypoints for decoder. If None, equals n_kp_enc. Defaults to None.
            learned_feature_dim (int): Dimension of learned visual features. Defaults to 16.
            learned_bg_feature_dim (int): Dimension of background features. Defaults to 16.
            kp_range (tuple): Range for keypoint coordinates, either (-1, 1) or (0, 1). Defaults to (-1, 1).
            kp_activation (str): Activation for keypoint coordinates ('tanh' or 'sigmoid'). Defaults to 'tanh'.
            anchor_s (float): Glimpse size as ratio of image_size. Defaults to 0.25.
            use_resblock (bool): Use residual blocks in network. Defaults to True.
            embed_init_std (float): Standard deviation for embedding initialization. Defaults to 0.02.
            projection_dim (int): Embedding dimension for transformer. Defaults to 128.
            timestep_horizon (int): Maximum number of timesteps to process at once. Defaults to 1.
            pte_layers (int): Number of particle transformer encoder layers. Defaults to 1.
            pte_heads (int): Number of particle transformer encoder heads. Defaults to 1.
            context_dim (int): Dimension of context latent space. Defaults to 16.
            filtering_heuristic (str): Method to filter prior keypoints. Defaults to 'none'.
            attn_norm_type (str): Normalization type for attention blocks. Defaults to 'rms'.
            obj_ch_mult_prior (tuple): Channel multipliers for prior patch encoder. Defaults to (1, 2, 3).
            obj_ch_mult (tuple): Channel multipliers for object encoder. Defaults to (1, 2, 3).
            obj_base_ch (int): Base channels for object encoder. Defaults to 32.
            obj_final_cnn_ch (int): Final CNN channels for object encoder. Defaults to 32.
            cnn_mid_blocks (bool): Use middle blocks in CNN. Defaults to False.
            mlp_hidden_dim (int): Hidden dimension for MLPs. Defaults to 256.
            pte_inner_dim (int): Inner dimension for particle transformer. Defaults to 256.
            bg_ch_mult (tuple): Channel multipliers for background encoder. Defaults to (1, 2, 3).
            bg_base_ch (int): Base channels for background encoder. Defaults to 32.
            bg_final_cnn_ch (int): Final CNN channels for background encoder. Defaults to 32.
            num_res_blocks (int): Number of residual blocks. Defaults to 2.
            ctx_pool_mode (str): Mode for pooling context features. Defaults to 'none'.
            interaction_depth (bool): Enable modeling depth by interaction between particles. Defaults to True.
            interaction_obj_on (bool): Enable modeling transparency by interaction. Defaults to False.
            interaction_features (bool): Enable modeling features by interaction. Defaults to True.
            particle_score (bool): Use particle confidence scores. Defaults to False.
            add_particle_temp_embed (bool): Add temporal embeddings to particles. Defaults to False.
            particle_positional_embed (bool): Add positional embeddings to particles. Defaults to True.
            causal_ctx (bool): Use causal attention for context. Defaults to True.
            pte_ctx_layers (int): Number of context transformer layers. Defaults to 1.
            pte_ctx_heads (int): Number of context transformer heads. Defaults to 1.
            ctx_dist (str): Distribution type for context ('gauss' or 'categorical'). Defaults to 'gauss'.
            n_ctx_categories (int): Number of context categories if categorical. Defaults to 4.
            n_ctx_classes (int): Number of context classes per category. Defaults to 4.
            features_dist (str): Distribution type for features ('gauss' or 'categorical'). Defaults to 'gauss'.
            n_fg_categories (int): Number of foreground categories if categorical. Defaults to 8.
            n_fg_classes (int): Number of foreground classes per category. Defaults to 4.
            n_bg_categories (int): Number of background categories if categorical. Defaults to 4.
            n_bg_classes (int): Number of background classes per category. Defaults to 4.
            obj_on_min (float): Minimum concentration value in Beta dist for transparency value. Defaults to 1e-4.
            obj_on_max (float): Maximum concentration value in Beta dist transparency value. Defaults to 100.
            use_z_orig (bool): Use original patch center coordinates. Defaults to True.

        Notes:
            The encoder operates in several stages:
            1. Patch Processing: Divides input image into patches and processes each
            2. Keypoint Proposal: Generates candidate keypoints using spatial softmax
            3. Feature Extraction: Learns visual features around each keypoint
            4. Particle Interaction: Models relationships between particles
            5. Context Modeling: Captures dynamics for the latent context (if enabled)

            The module supports both Gaussian and categorical distributions for
            features and context variables.

        The architecture uses a combination of CNNs and transformers:
            - CNNs for initial feature extraction from patches
            - Transformer encoders for modeling particle interactions
            - Separate pathways for foreground and background processing
            - Optional causal attention for temporal modeling
        """
        super(DLPEncoder, self).__init__()
        self.cdim = cdim
        self.image_size = image_size
        self.n_views = n_views
        self.dropout = dropout
        self.kp_range = kp_range
        self.n_kp_per_patch = n_kp_per_patch
        self.n_kp_enc = n_kp_enc
        self.n_kp_prior = n_kp_prior
        self.n_kp_dec = self.n_kp_enc if n_kp_dec is None else n_kp_dec
        self.warmup_n_kp_ratio = warmup_n_kp_ratio
        self.kp_activation = kp_activation
        self.patch_size = patch_size
        self.anchor_patch_s = patch_size / image_size
        self.features_dim = int(image_size // (2 ** (len(bg_ch_mult) - 1)))
        self.learned_feature_dim = learned_feature_dim
        self.learned_bg_feature_dim = learned_bg_feature_dim
        assert learned_feature_dim > 0, "learned_feature_dim must be greater than 0"
        self.features_dist = features_dist
        self.n_fg_categories = n_fg_categories
        self.n_fg_classes = n_fg_classes
        self.n_bg_categories = n_bg_categories
        self.n_bg_classes = n_bg_classes

        # self.context_dist = ctx_dist
        # self.n_ctx_categories = n_ctx_categories
        # self.n_ctx_classes = n_ctx_classes
        self.context_dim = context_dim
        # # global latent context
        # self.global_ctx_pool = global_ctx_pool
        # self.pool_ctx_dim = pool_ctx_dim
        # self.n_pool_ctx_categories = n_pool_ctx_categories
        # self.n_pool_ctx_classes = n_pool_ctx_classes
        # if self.context_dist == 'categorical':
        #     self.pool_ctx_dim = int(self.n_pool_ctx_categories * self.n_pool_ctx_classes)
        # self.global_local_fuse_mode = global_local_fuse_mode
        # self.condition_local_on_global = condition_local_on_global
        self.mask_bg_in_enc = mask_bg_in_enc  # before encoding the bg, mask with the particles' obj_on
        self.anchor_s = anchor_s
        self.obj_patch_size = np.round(anchor_s * (image_size - 1)).astype(int)
        self.obj_on_min = obj_on_min
        self.obj_on_max = obj_on_max
        self.use_resblock = use_resblock
        self.embed_init_std = embed_init_std
        self.projection_dim = projection_dim
        self.timestep_horizon = (timestep_horizon + 1) if timestep_horizon > 1 else 1
        self.num_patches = int((image_size // self.patch_size) ** 2)
        self.attn_norm_type = attn_norm_type
        self.use_z_orig = use_z_orig
        self.interaction_depth = interaction_depth
        self.interaction_obj_on = interaction_obj_on
        self.interaction_features = interaction_features
        self.use_particle_inter_enc = (self.interaction_features or self.interaction_depth or self.interaction_obj_on)
        self.add_particle_temp_embed = add_particle_temp_embed
        self.temporal_interaction = False  # True=allow to attend over timesteps

        self.use_ctx_enc = (self.context_dim > 0)
        # self.ctx_pool_mode = ctx_pool_mode
        # self.causal_ctx = causal_ctx
        self.particle_score = particle_score
        self.cnn_mid_blocks = cnn_mid_blocks
        self.mlp_hidden_dim = mlp_hidden_dim

        # initialization
        self.init_zero_bias = init_zero_bias  # zero bias for conv and linear layers
        self.init_ssm_last_layer = init_ssm_last_layer  # spatial softmax initialization
        self.init_conv_layers = init_conv_layers  # initialize conv layers with normal dist
        self.init_conv_fg_std = init_conv_fg_std  # std for conv fg normal dist
        self.init_conv_bg_std = init_conv_bg_std  # std for conv bg normal dist

        #RGBD
        self.separate_depth_features = separate_depth_features
        self.depth_feature_dim = depth_feature_dim

        self.register_buffer('scale_anchor', torch.tensor(np.log(anchor_s / (1 - anchor_s + 1e-5))))
        use_norm_layer = True  # norm layer in the pre-attention projections modules
        self.particle_enc = ParticleEncoder(cdim=cdim,
                                            image_size=image_size,
                                            pad_mode=pad_mode,
                                            n_kp_per_patch=self.n_kp_per_patch,
                                            n_kp_prior=self.n_kp_prior,
                                            patch_size=self.patch_size, n_kp_enc=self.n_kp_enc, n_kp_dec=self.n_kp_dec,
                                            learned_feature_dim=learned_feature_dim,
                                            kp_range=kp_range, kp_activation=kp_activation, anchor_s=anchor_s,
                                            use_resblock=use_resblock, embed_init_std=embed_init_std,
                                            projection_dim=projection_dim, timestep_horizon=timestep_horizon,
                                            filtering_heuristic=filtering_heuristic,
                                            obj_ch_mult_prior=obj_ch_mult_prior,
                                            obj_ch_mult=obj_ch_mult,
                                            obj_base_ch=obj_base_ch,
                                            obj_final_cnn_ch=obj_final_cnn_ch, num_res_blocks=num_res_blocks,
                                            interaction_features=interaction_features,
                                            interaction_obj_on=interaction_obj_on,
                                            interaction_depth=interaction_depth,
                                            temporal_interaction=self.temporal_interaction,
                                            cnn_mid_blocks=cnn_mid_blocks,
                                            mlp_hidden_dim=mlp_hidden_dim, embed_prior_patch_pos=False,
                                            add_particle_temp_embed=self.add_particle_temp_embed,
                                            features_dist=self.features_dist, n_fg_categories=n_fg_categories,
                                            n_fg_classes=n_fg_classes, obj_on_min=self.obj_on_min,
                                            obj_on_max=self.obj_on_max, warmup_n_kp_ratio=self.warmup_n_kp_ratio,
                                            init_zero_bias=init_zero_bias,
                                            init_ssm_last_layer=init_ssm_last_layer,
                                            init_conv_layers=init_conv_layers,
                                            init_conv_fg_std=init_conv_fg_std,
                                            separate_depth_features=separate_depth_features,
                                            depth_feature_dim=depth_feature_dim)
        extra_point_feats = 0 # TODO: link this up with the verison in particle Encoder and make them all come form the same config
        self.prior_encoder = self.particle_enc.prior_encoder
        in_channels = self.cdim
        self.bg_encoder = BgEncoder3D(
            in_channels=in_channels,
            grid_dhw=voxel_grid_whd,   
            learned_feature_dim=learned_bg_feature_dim,
            features_dist=('categorical' if self.features_dist == 'categorical' else 'gauss'),
            interaction_features=False,                      # bg is usually not interacted
            base_ch=bg_base_ch,
            ch_mult=bg_ch_mult,
            num_res_blocks=num_res_blocks,
            use_resblock=use_resblock,
            use_attention=False,
            cnn_mid_blocks=cnn_mid_blocks,
            final_cnn_ch=bg_final_cnn_ch,
            timestep_horizon=self.timestep_horizon,
            add_particle_temp_embed=self.add_particle_temp_embed,
            mlp_hidden_dim=mlp_hidden_dim,
            init_zero_bias=init_zero_bias,
            init_conv_layers=init_conv_layers,
            init_conv_bg_std=init_conv_bg_std,
        )


        # patch_centers = self.prior_encoder.get_patch_centers().unsqueeze(0) * (
        #         self.kp_range[1] - self.kp_range[0]) + self.kp_range[0]
        # # append null particle
        # patch_centers = torch.cat([patch_centers, torch.zeros(1, 1, 2)], dim=1)
        # # self.patch_centers = patch_centers
        # self.register_buffer('patch_centers', patch_centers)
        # particle_anchors = patch_centers[:, :-1]  # [1, 1, n_kp_enc], no need for (0,0)-the bg
        # particle_anchors = particle_anchors.unsqueeze(-2).repeat(1, 1, self.n_kp_per_patch, 1).view(1, -1, 2)

        if self.use_particle_inter_enc:
            self.particle_inter_enc = ParticleInteractionEncoder3D(
                # core sizes
                n_kp_enc=n_kp_enc,
                learned_feature_dim=learned_feature_dim,
                learned_bg_feature_dim=learned_bg_feature_dim,
                projection_dim=projection_dim,
                hidden_dim=mlp_hidden_dim,          # or `hidden_dim` if that's your var name

                # transformer config
                pte_layers=pte_layers,
                pte_heads=pte_heads,
                attn_norm_type=attn_norm_type,
                dropout=0.0,
                activation='gelu',
                temporal_interaction=self.temporal_interaction,

                # what to refine
                interaction_features=interaction_features,
                interaction_depth=interaction_depth,
                interaction_obj_on=interaction_obj_on,

                # tokens / embeddings
                particle_positional_embed=particle_positional_embed,
                add_particle_temp_embed=self.add_particle_temp_embed,
                particle_score=False, # TODO: this has to be set to false for shape size for now
                with_bg=True,               # include bg token

                # PC context (no image CNN)
                use_pc_ctx=True,
                ctx_type="pointnet",

                # feature distribution (FG/BG)
                features_dist=self.features_dist,
                n_fg_categories=n_fg_categories,
                n_fg_classes=n_fg_classes,
                n_bg_categories=n_bg_categories,
                n_bg_classes=n_bg_classes,

                # obj_on shaping + anchors
                obj_on_min=self.obj_on_min,
                obj_on_max=self.obj_on_max,
                use_z_orig=self.use_z_orig,

                # init
                embed_init_std=embed_init_std,
                init_zero_bias=init_zero_bias,
                init_conv_layers=init_conv_layers,
                init_conv_fg_std=init_conv_fg_std,
            )


        else:
            self.particle_inter_enc = None

        self.ctx_enc = ctx_enc

        # if self.use_ctx_enc:
        #     self.ctx_enc = ParticleContextSharedEncoder(n_kp_enc=n_kp_enc, dropout=dropout,
        #                                                 learned_feature_dim=learned_feature_dim,
        #                                                 learned_bg_feature_dim=learned_bg_feature_dim,
        #                                                 embed_init_std=embed_init_std, projection_dim=pte_inner_dim,
        #                                                 timestep_horizon=timestep_horizon, pte_layers=pte_ctx_layers,
        #                                                 pte_heads=pte_ctx_heads,
        #                                                 attn_norm_type=attn_norm_type,
        #                                                 context_dim=context_dim,
        #                                                 hidden_dim=pte_inner_dim,
        #                                                 ctx_pool_mode=self.ctx_pool_mode,
        #                                                 bg=True,
        #                                                 particle_positional_embed=particle_positional_embed,
        #                                                 particle_score=self.particle_score,
        #                                                 causal=self.causal_ctx, norm_layer=use_norm_layer,
        #                                                 shared_logvar=False, ctx_dist=ctx_dist,
        #                                                 n_ctx_categories=n_ctx_categories, n_ctx_classes=n_ctx_classes,
        #                                                 particle_anchors=particle_anchors, use_z_orig=self.use_z_orig,
        #                                                 global_ctx_pool=self.global_ctx_pool,
        #                                                 ctx_pool_dim=self.pool_ctx_dim,
        #                                                 n_pool_ctx_categories=self.n_pool_ctx_categories,
        #                                                 n_pool_ctx_classes=self.n_pool_ctx_classes,
        #                                                 global_local_fuse_mode=global_local_fuse_mode,
        #                                                 condition_local_on_global=condition_local_on_global)
        # else:
        #     self.ctx_enc = None

        self.init_weights()

    def init_weights(self):
        self.particle_enc.init_weights()
        self.bg_encoder.init_weights()
        self.prior_encoder.init_weights()
        # if self.with_ctx:
        #     self.ctx_enc.init_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                pass
                # nn.init.normal_(m.weight, 0, 0.01)
                # if m.bias is not None:
                #    nn.init.constant_(m.bias, 0)
            #         # print(m.__repr__())
            #     elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            #         nn.init.constant_(m.weight, 1)
            #         nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5), mode='fan_in')
                # nn.init.normal_(m.weight, 0, 0.02)
                # if m.bias is not None:
                #    nn.init.constant_(m.bias, 0)
                # use pytorch's default
                pass

    @torch.no_grad()
    def get_bg_mask_from_particle_glimpses_3d(
        self,
        *,
        z: torch.Tensor,                    # [B,K,3] (xyz in [-1,1])
        grid_dhw: tuple,                    # (D,H,W)
        z_scale: torch.Tensor = None,       # [B,K,3] logits or None
        z_obj_on: torch.Tensor = None,      # [B,K] or [B,K,1] or None
        anchor_s: float = 0.25,
        gate_threshold: float = 0.2,
        detach_grad: bool = True,
        shape: str = "box",                 # "box" | "ellipsoid"
    ) -> torch.Tensor:
        """
        Returns background mask in voxel space: [B,1,D,H,W] with 1=background, 0=covered by any particle.
        Robust to K mismatch between z and z_obj_on (pads or trims gate to match z.shape[1]).
        """
        if detach_grad:
            z = z.detach()
            if z_scale is not None: z_scale = z_scale.detach()
            if z_obj_on is not None: z_obj_on = z_obj_on.detach()

        B, K_in, _ = z.shape
        D, H, W = map(int, grid_dhw)

        # ---- grid in xyz order mapped to [-1,1] ----
        xs = torch.linspace(-1, 1, W, device=z.device, dtype=z.dtype)
        ys = torch.linspace(-1, 1, H, device=z.device, dtype=z.dtype)
        zs = torch.linspace(-1, 1, D, device=z.device, dtype=z.dtype)
        zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")   # z,y,x
        grid_xyz = torch.stack([xx, yy, zz], dim=-1)             # [D,H,W,3] (xyz)

        # ---- centers & half-extends (xyz) ----
        # half-extent per axis in [-1,1] units
        if z_scale is None:
            half = torch.full((B, K_in, 3), fill_value=anchor_s, device=z.device, dtype=z.dtype)
        else:
            half = anchor_s * torch.sigmoid(z_scale)             # [B,K,3]

        # ---- inside test per kp ----
        # expand for broadcasting
        c = z.view(B, K_in, 1, 1, 1, 3)                          # [B,K,1,1,1,3]
        h = half.view(B, K_in, 1, 1, 1, 3)                       # [B,K,1,1,1,3]
        g = grid_xyz.view(1, 1, D, H, W, 3)                      # [1,1,D,H,W,3]

        if shape == "ellipsoid":
            # ((x-cx)/hx)^2 + ((y-cy)/hy)^2 + ((z-cz)/hz)^2 <= 1
            num = (g - c) / (h.clamp_min(1e-6))
            inside = (num ** 2).sum(dim=-1) <= 1.0               # [B,K,D,H,W]
        else:  # "box"
            inside = (torch.abs(g - c) <= h).all(dim=-1)         # [B,K,D,H,W]

        # ---- gate alignment to K_in ----
        if z_obj_on is None:
            gate = torch.ones(B, K_in, device=z.device, dtype=z.dtype)
        else:
            gate = z_obj_on
            if gate.dim() == 3:   # [B,K,1] -> [B,K]
                gate = gate.squeeze(-1)
            # binarize or soften
            gate = (gate > gate_threshold).to(z.dtype)
            # match K
            K_gate = gate.shape[1]
            if K_gate != K_in:
                if K_gate > K_in:
                    gate = gate[:, :K_in]
                else:
                    pad = K_in - K_gate
                    gate = torch.cat([gate, torch.ones(B, pad, device=z.device, dtype=z.dtype)], dim=1)
        gate = gate.view(B, K_in, 1, 1, 1)                        # [B,K,1,1,1]

        # ---- aggregate objects and make bg mask ----
        obj = inside.float() * gate                               # [B,K,D,H,W]
        obj = obj.sum(dim=1, keepdim=True).clamp(0, 1)            # [B,1,D,H,W]
        bg_mask = 1.0 - obj                                       # [B,1,D,H,W]
        return bg_mask


    def encode_all(self,
                x: torch.Tensor,                  # [B,N,3(+F)]
                dense=None,
                mask_pc: torch.Tensor = None,     # [B,N] (bool) True=valid
                deterministic: bool = False,
                warmup: bool = False,
                actions=None, actions_mask=None, lang_embed=None,
                x_goal=None, deterministic_goal=True):
        """
        Point-cloud, single-frame DLPEncoder:
        1) particle encoding (prior -> attributes -> appearance)
        2) optional BG encoding (with PC mask derived from particle anchors/scales)
        3) pack a flat dict; no time dims
        """
        assert x.dim() == 3 and x.size(-1) >= 3, f"expected points [B,N,>=3], got {tuple(x.shape)}"
        print("dense.shape:: ", dense.shape)
        if mask_pc is not None:
            assert mask_pc.dim() == 2 and mask_pc.shape[:2] == x.shape[:2], \
                f"mask_pc must be [B,N], got {tuple(mask_pc.shape)}"
        B = x.size(0)

        # ---- 1) particles ----
        # your ParticleEncoder(PC) returns the compact PC dict we agreed on
        p = self.particle_enc(x, dense, mask_pc, deterministic=deterministic, warmup=warmup)

        # unify mandatory bits
        z_base       = p.get('pos_anchor', p.get('z_base'))        # [B,K,3]
        z            = p.get('pos',        p.get('z'))             # [B,K,3]
        mu_tot       = p.get('pos_mu',     p.get('mu_tot', z))     # [B,K,3]
        pos_logvar   = p.get('pos_logvar', p.get('logvar_offset',
                        torch.zeros_like(z)))                    # [B,K,3]

        z_scale      = p.get('scale',      p.get('z_scale'))       # [B,K,3]
        mu_scale     = p.get('scale_mu',   p.get('mu_scale', z_scale))
        logvar_scale = p.get('scale_logvar', p.get('logvar_scale', None))

        z_features      = p.get('feat',       p.get('z_features', None))   # [B,K,F]
        mu_features     = p.get('feat_mu',    p.get('mu_features', z_features))
        logvar_features = p.get('feat_logvar', p.get('logvar_features', None))
        # prior metadata
        kp_p        = p.get('kp_p',        p.get('kp_prior', None))        # [B,K,3]
        var_kp      = p.get('kp_var',      p.get('var_kp',   None))        # [B,K,*]
        z_base_var  = p.get('z_base_var',  None)                           # [B,K,?]

        patch_id_embed = p.get('patch_id_embed', None)

        # optional extras (pass-through)
        z_obj_on    = p.get('z_obj_on', None)     # [B,K] or [B,K,1]
        obj_on_a    = p.get('obj_on_a', None)
        obj_on_b    = p.get('obj_on_b', None)
        mu_obj_on   = p.get('mu_obj_on', None)

        mu_depth     = p.get('mu_depth', None)
        mu_offset = p.get('mu_offset', None)
        logvar_depth = p.get('logvar_depth', None)
        z_depth      = p.get('z_depth', None)

        # ---- 2) background (optional) ----
        mu_bg_features = logvar_bg_features = z_bg_features = None
        
        # build a PC bg mask from anchors/scales if helper exists on this class
        if self.mask_bg_in_enc:
            bg_mask_vox = self.get_bg_mask_from_particle_glimpses_3d(
                z=z,                               # [B,K,3] xyz in [-1,1]
                z_obj_on=z_obj_on,                 # [B,K] or [B,K,1]
                grid_dhw=dense.shape[-3:],         # (D,H,W)
                z_scale=z_scale,                   # [B,K,3] logits or None
                anchor_s=self.anchor_s,            # mirror your 2D anchor
                gate_threshold=0.2,
                detach_grad=True,
                shape="box",                       # or "ellipsoid"
            )
        else:
            bg_mask_vox = None

        # BgEncoderPC forward signature should accept (points, mask, bg_mask, deterministic)
        bg_out = self.bg_encoder(dense, bg_mask_vox, deterministic=deterministic)
        
        mu_bg_features   = bg_out['mu_bg']
        logvar_bg_features = bg_out.get('logvar_bg', None)
        z_bg_features    = bg_out['z_bg']

        if self.use_particle_inter_enc:
            inter = self.particle_inter_enc(
                x, mask_pc, z, z_scale, z_obj_on, z_depth, z_features,
                z_bg_features=z_bg_features, z_base_var=z_base_var, z_score=None,
                deterministic=deterministic, warmup=warmup
            )
            # then selectively override:
            if self.interaction_features:
                mu_features      = inter['mu_features']      
                logvar_features   = inter['logvar_features']
                z_features       = inter['z_features']
                mu_bg_features   = inter['mu_bg_features']   
                logvar_bg_features= inter['logvar_bg_features']
                z_bg_features    = inter['z_bg_features']
            if self.interaction_depth:
                mu_depth         = inter['mu_depth']         
                logvar_depth      = inter['logvar_depth']
                z_depth          = inter['z_depth']
            if self.interaction_obj_on:
                obj_on_a         = inter['obj_on_a']         
                obj_on_b          = inter['obj_on_b']
                mu_obj_on        = inter['mu_obj_on']        
                z_obj_on          = inter['z_obj_on']


        # ---- 3) pack & return ----
        return {
            # positions
            'z_base': z_base,                  # [B,K,3] anchor/proposal
            'z':      z,                       # [B,K,3] refined/sample
            'mu_tot': mu_tot,                  # [B,K,3] mean after offset
            'mu_offset': mu_offset,
            'logvar_offset': pos_logvar,       # [B,K,3] (kept name for compat)

            # scales
            'z_scale':      z_scale,           # [B,K,3]
            'mu_scale':     mu_scale,          # [B,K,3]
            'logvar_scale': logvar_scale,      # [B,K,3] or None

            # features
            'z_features':       z_features,    # [B,K,F]
            'mu_features':      mu_features,   # [B,K,F]
            'logvar_features':  logvar_features,  # [B,K,F] or None

            # prior proposal metadata
            'kp_p':        kp_p,               # [B,K,3]
            'var_kp':      var_kp,             # [B,K,*]
            'z_base_var':  z_base_var,         # [B,K,?]
            'patch_id_embed': patch_id_embed,

            'z_obj_on':    z_obj_on,
            'obj_on_a':    obj_on_a,
            'obj_on_b':    obj_on_b,
            'mu_obj_on':   mu_obj_on,
            'mu_depth':    mu_depth,
            'logvar_depth':logvar_depth,
            'z_depth':     z_depth,

            'mu_bg_features':    mu_bg_features,
            'logvar_bg_features':logvar_bg_features,
            'z_bg_features':     z_bg_features,
        }



    def forward(self, x, dense, mask_pc, deterministic=False, warmup=False, actions=None, actions_mask=None, lang_embed=None,
                x_goal=None):
        output_dict = self.encode_all(x, dense, mask_pc, deterministic, warmup, actions=actions, actions_mask=actions_mask,
                                      lang_embed=lang_embed, x_goal=x_goal)
        return output_dict
