"""
DLP Policy
Map latent context from pre-trained DLP world-model to GT actions.
"""

import torch
import torch.nn as nn
from sympy.categories.baseclasses import Class

from models import DLP
from modules.vision_modules import rgb_to_minusoneone
from modules.modules import ParticlePool, RMSNorm, ParticleSpatioTemporalTransformer
from utils.util_func import get_config
from utils.loss_functions import calc_reconstruction_loss


class DLPTransformerPolicy(nn.Module):
    def __init__(self, n_particles, dropout=0.1, in_dim=7, embed_init_std=0.02,
                 projection_dim=512, pte_layers=1, pte_heads=1,
                 attn_norm_type='rms', hidden_dim=256,
                 activation='gelu', n_views=1, particle_positional_embed=True,
                 pos_embed_p_adaln=True,  # pos embeddings for particles using adaln
                 action_dim=7,
                 tanh_on_output=False,
                 timestep_horizon=1,
                 ptokens_adaln=False,
                 causal=False
                 ):
        super(DLPTransformerPolicy, self).__init__()
        self.n_particles = n_particles
        self.dropout = dropout
        self.in_dim = in_dim
        self.embed_init_std = embed_init_std
        self.projection_dim = projection_dim
        self.timestep_horizon = timestep_horizon if causal else 1
        # self.timestep_horizon = timestep_horizon
        self.attn_norm_type = attn_norm_type
        self.hidden_dim = hidden_dim
        self.n_views = n_views
        self.activation = activation
        self.pos_embed_p_adaln = pos_embed_p_adaln
        self.action_dim = action_dim
        self.tanh_on_output = tanh_on_output
        self.ptokens_adaln = ptokens_adaln

        # entities positional embeddings
        if particle_positional_embed:
            self.particle_embeddings = nn.Parameter(
                self.embed_init_std * torch.randn(1, 1, 1, self.n_particles, projection_dim))
        else:
            self.particle_embeddings = nn.Parameter(self.embed_init_std * torch.randn(1, 1, 1, 1, projection_dim))

        if self.n_views > 1:
            self.view_embeddings = nn.Parameter(
                self.embed_init_std * torch.randn(1, 1, self.n_views, 1, projection_dim))
        else:
            self.view_embeddings = None

        if self.pos_embed_p_adaln:
            n_particles += (self.n_views - 1) * (self.n_particles + 1)
            self.pos_p_embeddings = nn.Parameter(
                self.embed_init_std * torch.randn(1, 1, n_particles + 1, self.hidden_dim))

        self.action_token = nn.Parameter(self.embed_init_std * torch.randn(1, 1, 1, self.projection_dim))
        if self.ptokens_adaln:
            n_particles = self.n_particles + (self.n_views - 1) * self.n_particles
            self.query_tokens = nn.Parameter(
                self.embed_init_std * torch.randn(1, 1, n_particles, self.hidden_dim))
            self.c_action = nn.Parameter(
                self.embed_init_std * torch.randn(1, 1, 1, self.hidden_dim))

        self.projection = nn.Sequential(nn.Linear(self.in_dim, self.hidden_dim),
                                        RMSNorm(self.hidden_dim),
                                        nn.GELU(),
                                        nn.Linear(self.hidden_dim, self.hidden_dim))

        block_size = self.timestep_horizon
        pte_context_cond = self.pos_embed_p_adaln or self.ptokens_adaln
        self.pte = ParticleSpatioTemporalTransformer(n_embed=self.projection_dim, n_head=pte_heads,
                                                     n_layer=pte_layers,
                                                     block_size=block_size,
                                                     output_dim=self.projection_dim, attn_pdrop=dropout,
                                                     resid_pdrop=dropout,
                                                     hidden_dim_multiplier=4, positional_bias=False,
                                                     activation='gelu',
                                                     max_particles=None, norm_type=attn_norm_type,
                                                     particles_first=False, init_std=embed_init_std,
                                                     causal=causal and (self.timestep_horizon > 1),
                                                     context_cond=pte_context_cond,
                                                     residual_modulation=pte_context_cond,
                                                     context_gate=pte_context_cond,
                                                     pos_embed_t_adaln=(self.timestep_horizon > 1))

        self.action_decoder = nn.Sequential(nn.Linear(self.projection_dim, self.hidden_dim),
                                            nn.GELU(),
                                            nn.Linear(self.hidden_dim, self.action_dim),
                                            nn.Tanh() if self.tanh_on_output else nn.Identity())

        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, z_context):
        # z_context: [bs, T-1, n_views * n_particles, dim]
        bs, timestep_horizon = z_context.shape[0], z_context.shape[1]
        if self.timestep_horizon > 1:
            z_context_v = z_context[:, :self.timestep_horizon].contiguous()
            z_context_v = z_context_v.reshape(bs, z_context_v.shape[1], self.n_views, -1,
                                                                       z_context.shape[-1])
        else:
            z_context_v = z_context.reshape(bs * timestep_horizon, 1, self.n_views, -1,
                                            z_context.shape[-1])
        # z_context = z_context[:, -self.timestep_horizon:].contiguous()

        particle_projection = self.projection(z_context_v)  # [bs, T-1, n_views * n_particles, proj_dim]
        # if self.timestep_horizon > 1:
        #     particle_projection = particle_projection.reshape(bs, timestep_horizon, self.n_views, -1,
        #                                                       particle_projection.shape[-1])
        # else:
        #     particle_projection = particle_projection.reshape(bs * timestep_horizon, 1, self.n_views, -1,
        #                                                       particle_projection.shape[-1])
        # [bs * (T-1), 1, n_views, n_particles, proj_dim]
        # add entity pos embeddings
        if self.particle_embeddings.shape[3] == 1:
            p_embeddings = self.particle_embeddings.repeat(particle_projection.shape[0], 1, 1, self.n_particles, 1)
        else:
            p_embeddings = self.particle_embeddings.repeat(particle_projection.shape[0], 1, 1, 1, 1)
        particle_projection = particle_projection + p_embeddings
        # view embeddings
        if self.n_views > 1:
            # add view embeddings
            particle_projection = particle_projection + self.view_embeddings

        particle_projection = particle_projection.reshape(particle_projection.shape[0],
                                                          particle_projection.shape[1],
                                                          -1,
                                                          particle_projection.shape[-1])
        # [bs * t, 1, n_views * n, d]

        if self.ptokens_adaln:
            particle_tokens = self.query_tokens.repeat(particle_projection.shape[0], particle_projection.shape[1], 1, 1)
            # [bs, T, n_particles, dim]
            assert particle_tokens.shape[2] == particle_projection.shape[2]
            # add action token
            action_tokens = self.action_token.repeat(particle_tokens.shape[0], particle_tokens.shape[1], 1, 1)
            particle_tokens = torch.cat([particle_tokens, action_tokens], dim=-2)
            # [bs, t, n_views * n + 1, d]
            # prepare condition
            c = torch.cat(
                [particle_projection,
                 self.c_action.repeat(particle_projection.shape[0], particle_projection.shape[1], 1, 1)], dim=-2)
            if self.pos_embed_p_adaln:
                pos_embed = self.pos_p_embeddings[:, :, :particle_tokens.shape[2]].repeat(particle_tokens.shape[0],
                                                                                          particle_tokens.shape[1],
                                                                                          1, 1)
                c = c + pos_embed
            particle_projection = particle_tokens
        else:
            # add action token
            action_tokens = self.action_token.repeat(particle_projection.shape[0], particle_projection.shape[1], 1, 1)
            particle_projection = torch.cat([particle_projection, action_tokens], dim=-2)
            # [bs * t, 1, n_views * n + 1, d]

            # prepare condition
            if self.pos_embed_p_adaln:
                c = self.pos_p_embeddings[:, :, :particle_projection.shape[2]].repeat(particle_projection.shape[0],
                                                                                      particle_projection.shape[1],
                                                                                      1, 1)
            else:
                c = None

        particles_out = self.pte(particle_projection, c=c)
        # [bs * t, 1, n_views * n + 1, d]
        action_out = particles_out[:, :, -1]  # [bs * t, 1, d]
        if self.timestep_horizon > 1 and timestep_horizon > self.timestep_horizon:
            action_out = action_out.reshape(bs, self.timestep_horizon, -1)
        else:
            action_out = action_out.reshape(bs, timestep_horizon, -1)
        action_out = self.action_decoder(action_out)
        # [bs, t, action_dim]
        actions = action_out
        if self.timestep_horizon > 1:
            i = 1
            while actions.shape[1] < timestep_horizon:
                # collect all actions
                z_context_v = z_context[:, i:i + self.timestep_horizon].reshape(bs, self.timestep_horizon,
                                                                                self.n_views, -1, z_context.shape[-1])
                particle_projection = self.projection(z_context_v)  # [bs, T-1, n_views * n_particles, proj_dim]
                # [bs * (T-1), 1, n_views, n_particles, proj_dim]
                # add entity pos embeddings
                if self.particle_embeddings.shape[3] == 1:
                    p_embeddings = self.particle_embeddings.repeat(particle_projection.shape[0], 1, 1, self.n_particles,
                                                                   1)
                else:
                    p_embeddings = self.particle_embeddings.repeat(particle_projection.shape[0], 1, 1, 1, 1)
                particle_projection = particle_projection + p_embeddings
                # view embeddings
                if self.n_views > 1:
                    # add view embeddings
                    particle_projection = particle_projection + self.view_embeddings

                particle_projection = particle_projection.reshape(particle_projection.shape[0],
                                                                  particle_projection.shape[1],
                                                                  -1,
                                                                  particle_projection.shape[-1])
                # [bs * t, 1, n_views * n, d]

                if self.ptokens_adaln:
                    particle_tokens = self.query_tokens.repeat(particle_projection.shape[0],
                                                               particle_projection.shape[1], 1, 1)
                    # [bs, T, n_particles, dim]
                    assert particle_tokens.shape[2] == particle_projection.shape[2]
                    # add action token
                    action_tokens = self.action_token.repeat(particle_tokens.shape[0], particle_tokens.shape[1], 1, 1)
                    particle_tokens = torch.cat([particle_tokens, action_tokens], dim=-2)
                    # [bs, t, n_views * n + 1, d]
                    # prepare condition
                    c = torch.cat(
                        [particle_projection,
                         self.c_action.repeat(particle_projection.shape[0], particle_projection.shape[1], 1, 1)],
                        dim=-2)
                    if self.pos_embed_p_adaln:
                        pos_embed = self.pos_p_embeddings[:, :, :particle_tokens.shape[2]].repeat(
                            particle_tokens.shape[0],
                            particle_tokens.shape[1],
                            1, 1)
                        c = c + pos_embed
                    particle_projection = particle_tokens
                else:
                    # add action token
                    action_tokens = self.action_token.repeat(particle_projection.shape[0], particle_projection.shape[1],
                                                             1, 1)
                    particle_projection = torch.cat([particle_projection, action_tokens], dim=-2)
                    # [bs * t, 1, n_views * n + 1, d]

                    # prepare condition
                    if self.pos_embed_p_adaln:
                        c = self.pos_p_embeddings[:, :, :particle_projection.shape[2]].repeat(
                            particle_projection.shape[0],
                            particle_projection.shape[1],
                            1, 1)
                    else:
                        c = None

                particles_out = self.pte(particle_projection, c=c)
                # [bs * t, 1, n_views * n + 1, d]
                action_out = particles_out[:, -1:, -1]  # [bs * t, 1, d]
                # action_out = action_out.reshape(bs, timestep_horizon, -1)
                action_out = self.action_decoder(action_out)
                # [bs, t, action_dim]
                actions = torch.cat([actions, action_out], dim=1)
                i += 1
                # print(i)

        return actions


def load_dlp_from_config(conf_path, ckpt_path=None):
    # load config
    try:
        config = get_config(conf_path)
    except FileNotFoundError:
        raise SystemExit("config file not found")
    # hparams = config  # to save a copy of the hyper-parameters
    ch = config['ch']  # image channels
    image_size = config['image_size']
    n_views = config.get('n_views', 1)
    # model
    timestep_horizon = config['timestep_horizon']
    num_static_frames = config['num_static_frames']
    pad_mode = config['pad_mode']
    n_kp_per_patch = config['n_kp_per_patch']  # kp per patch in prior, best to leave at 1
    n_kp_prior = config['n_kp_prior']  # number of prior kp to filter for the kl
    n_kp_enc = config['n_kp_enc']  # total posterior kp
    patch_size = config['patch_size']  # prior patch size
    anchor_s = config['anchor_s']  # posterior patch/glimpse ratio of image size
    # visual latent features
    features_dist = config.get('features_dist', 'gauss')
    learned_feature_dim = config['learned_feature_dim']
    learned_bg_feature_dim = config.get('learned_bg_feature_dim', learned_feature_dim)
    n_fg_categories = config.get('n_fg_categories', 8)  # Number of foreground feature categories (if categorical)
    n_fg_classes = config.get('n_fg_classes', 4)  # Number of foreground feature classes per category
    n_bg_categories = config.get('n_bg_categories', 4)  # Number of background feature categories
    n_bg_classes = config.get('n_bg_classes', 4)
    # latent context
    context_dist = config.get('context_dist', 'gauss')
    context_dim = config['context_dim']
    ctx_pool_mode = config.get("ctx_pool_mode", "none")
    n_ctx_categories = config.get('n_ctx_categories', 8)  # Number of context feature categories (if categorical)
    n_ctx_classes = config.get('n_ctx_classes', 4)  # Number of context feature classes per category
    dropout = config['dropout']
    use_resblock = config['use_resblock']
    # priors
    scale_std = config['scale_std']
    offset_std = config['offset_std']
    obj_on_alpha = config['obj_on_alpha']  # transparency beta distribution "a"
    obj_on_beta = config['obj_on_beta']  # transparency beta distribution "b"
    # transformer - PINT
    pint_enc_layers = config['pint_enc_layers']
    pint_enc_heads = config['pint_enc_heads']
    pint_ctx_layers = config['pint_ctx_layers']
    pint_ctx_heads = config['pint_ctx_heads']
    pint_dyn_layers = config['pint_dyn_layers']
    pint_dyn_heads = config['pint_dyn_heads']
    pint_dim = config['pint_dim']

    predict_delta = config['predict_delta']  # dynamics module predicts the delta from previous step

    normalize_rgb = config['normalize_rgb']
    obj_res_from_fc = config["obj_res_from_fc"]
    obj_ch_mult = config["obj_ch_mult"]
    obj_ch_mult_prior = config.get("obj_ch_mult_prior", obj_ch_mult)
    obj_base_ch = config["obj_base_ch"]
    obj_final_cnn_ch = config["obj_final_cnn_ch"]
    bg_res_from_fc = config["bg_res_from_fc"]
    bg_ch_mult = config["bg_ch_mult"]
    bg_base_ch = config["bg_base_ch"]
    bg_final_cnn_ch = config["bg_final_cnn_ch"]
    num_res_blocks = config["num_res_blocks"]
    cnn_mid_blocks = config.get('cnn_mid_blocks', False)
    mlp_hidden_dim = config.get('mlp_hidden_dim', 256)

    # actions
    action_condition = config.get('action_condition', False)
    action_dim = config.get('action_dim', 0)
    null_action_embed = config.get('null_action_embed', False)

    random_action_condition = config.get('random_action_condition', False)
    random_action_dim = config.get('random_action_dim', 0)

    # language
    language_condition = config.get('language_condition', False)
    language_embed_dim = config.get('language_embed_dim', 0)
    language_max_len = config.get('language_max_len', 32)

    # image goal condition
    img_goal_condition = config.get('image_goal_condition', False)

    # model
    model = DLP(cdim=ch,  # Number of input image channels
                image_size=image_size,  # Input image size (assumed square)
                normalize_rgb=normalize_rgb,  # If True, normalize RGB to [-1, 1], else keep [0, 1]
                n_views=n_views,  # number of input views (e.g., multiple cameras)

                # Keypoint and patch configuration
                n_kp_per_patch=n_kp_per_patch,  # Number of proposal/prior keypoints to extract per patch
                patch_size=patch_size,  # Size of patches for keypoint proposal network
                anchor_s=anchor_s,  # Glimpse size ratio relative to image size
                n_kp_enc=n_kp_enc,  # Number of posterior keypoints to learn
                n_kp_prior=n_kp_prior,  # Number of keypoints to filter from prior proposals

                # Network configuration
                pad_mode=pad_mode,  # Padding mode for CNNs ('zeros' or 'replicate')
                dropout=dropout,  # Dropout rate for transformers

                # Feature representation
                features_dist=features_dist,  # Distribution type for features ('gauss' or 'categorical')
                learned_feature_dim=learned_feature_dim,  # Dimension of learned visual features
                learned_bg_feature_dim=learned_bg_feature_dim,
                # Background feature dimension (if None, equals learned_feature_dim)
                n_fg_categories=n_fg_categories,  # Number of foreground feature categories (if categorical)
                n_fg_classes=n_fg_classes,  # Number of foreground feature classes per category
                n_bg_categories=n_bg_categories,  # Number of background feature categories
                n_bg_classes=n_bg_classes,  # Number of background feature classes per category

                # Prior distributions parameters
                scale_std=scale_std,  # Prior standard deviation for scale
                offset_std=offset_std,  # Prior standard deviation for offset
                obj_on_alpha=obj_on_alpha,  # Alpha parameter for transparency Beta distribution
                obj_on_beta=obj_on_beta,  # Beta parameter for transparency Beta distribution

                # Object decoder architecture
                obj_res_from_fc=obj_res_from_fc,  # Initial resolution for object encoder-decoder
                obj_ch_mult_prior=obj_ch_mult_prior,  # Channel multipliers for prior patch encoder (kp proposals)
                obj_ch_mult=obj_ch_mult,  # Channel multipliers for object encoder-decoder
                obj_base_ch=obj_base_ch,  # Base channels for object encoder-decoder
                obj_final_cnn_ch=obj_final_cnn_ch,  # Final CNN channels for object encoder-decoder

                # Background decoder architecture
                bg_res_from_fc=bg_res_from_fc,  # Initial resolution for background encoder-decoder
                bg_ch_mult=bg_ch_mult,  # Channel multipliers for background encoder-decoder
                bg_base_ch=bg_base_ch,  # Base channels for background encoder-decoder
                bg_final_cnn_ch=bg_final_cnn_ch,  # Final CNN channels for background encoder-decoder

                # Network architecture options
                use_resblock=use_resblock,  # Use residual blocks in encoders-decoders
                num_res_blocks=num_res_blocks,  # Number of residual blocks per resolution
                cnn_mid_blocks=cnn_mid_blocks,  # Use middle blocks in CNN
                mlp_hidden_dim=mlp_hidden_dim,  # Hidden dimension for MLPs

                # Particle interaction transformer (PINT) configuration
                pint_enc_layers=pint_enc_layers,  # Number of PINT encoder layers
                pint_enc_heads=pint_enc_heads,  # Number of PINT encoder attention heads

                # Dynamics configuration
                timestep_horizon=timestep_horizon,  # Number of timesteps to predict ahead
                n_static_frames=num_static_frames,  # Number of initial frames for static KL optimization
                predict_delta=predict_delta,  # Predict position deltas instead of absolute positions
                context_dim=context_dim,  # Context latent dimension (if None, equals learned_feature_dim)
                ctx_dist=context_dist,  # Context distribution type ('gauss' or 'categorical')
                n_ctx_categories=n_ctx_categories,  # Number of context categories (if categorical)
                n_ctx_classes=n_ctx_classes,  # Number of context classes per category
                ctx_pool_mode=ctx_pool_mode,  # Context pooling mode ('none' = per-particle context)

                # Context and dynamics transformer configuration
                pint_dyn_layers=pint_dyn_layers,  # Number of dynamics transformer layers
                pint_dyn_heads=pint_dyn_heads,  # Number of dynamics transformer heads
                pint_dim=pint_dim,  # Hidden dimension for PINT
                pint_ctx_layers=pint_ctx_layers,  # Number of context transformer layers
                pint_ctx_heads=pint_ctx_heads,

                # external conditioning
                action_condition=action_condition,  # condition on actions
                action_dim=action_dim,  # dimension of input actions
                null_action_embed=null_action_embed,
                random_action_condition=random_action_condition,
                random_action_dim=random_action_dim,
                # learn a "no-input-action" embedding, to learn on action-free videos as well
                language_condition=language_condition,  # condition on language embedding
                language_embed_dim=language_embed_dim,  # embedding dimension for each token
                language_max_len=language_max_len,  # maximum tokens per prompt
                img_goal_condition=img_goal_condition,  # condition the future on image goal
                )
    if ckpt_path is not None:
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))
            print("loaded dlp model from checkpoint")
        except:
            print("dlp model checkpoint not found")

    return model


class DLPPolicy(nn.Module):
    def __init__(self, policy_conf_path):
        super().__init__()
        self.config = get_config(policy_conf_path)
        self.dlp_config = get_config(self.config['path_to_dlp_hparams'])
        self.dlp_model = None
        self.n_views = self.config['n_views']
        # self.normalize_actions = False
        self.action_dim = self.config['action_dim']
        self.tanh_on_output = self.config['tanh_on_output']
        self.policy_type = self.config['policy_type']
        self.hidden_dim = self.config['hidden_dim']
        self.n_layers = self.config['n_layers']
        self.n_heads = self.config.get('n_heads', 2 ** self.n_layers)
        self.loss_type = self.config['recon_loss_type']
        self.n_particles = (self.dlp_config['n_kp_prior'] + 1) * self.n_views
        self.context_dim = self.dlp_config['context_dim']
        self.timestep_horizon = self.config['timestep_horizon']
        self.causal_policy = self.config.get('causal_policy', False)
        self.input_dim = self.n_particles * self.context_dim
        if self.policy_type == 'mlp':
            self.policy = nn.Sequential()
            self.policy.add_module(f'in_layer', nn.Linear(self.input_dim, self.hidden_dim))
            self.policy.add_module(f'in_activation', nn.GELU())
            for i in range(self.n_layers):
                if i == self.n_layers - 1:
                    self.policy.add_module(f'out_layer', nn.Linear(self.hidden_dim, self.action_dim))
                    if self.tanh_on_output:
                        self.policy.add_module(f'out_activation', nn.Tanh())
                else:
                    self.policy.add_module(f'hid_layer_{i + 1}', nn.Linear(self.hidden_dim, self.hidden_dim))
                    self.policy.add_module(f'in_activation', nn.GELU())
        elif self.policy_type == 'maxpool-mlp':
            self.policy = nn.Sequential()
            self.policy.add_module(f'in_layer', nn.Linear(self.context_dim, self.hidden_dim))
            self.policy.add_module(f'in_activation', nn.GELU())
            self.policy.add_module(f'in_pool', ParticlePool(pool_mode='max', pool_dim=-2, keepdim=False))
            for i in range(self.n_layers):
                if i == self.n_layers - 1:
                    self.policy.add_module(f'out_layer', nn.Linear(self.hidden_dim, self.action_dim))
                    if self.tanh_on_output:
                        self.policy.add_module(f'out_activation', nn.Tanh())
                else:
                    self.policy.add_module(f'hid_layer_{i + 1}', nn.Linear(self.hidden_dim, self.hidden_dim))
                    self.policy.add_module(f'in_activation', nn.GELU())
        elif self.policy_type == 'norm-mlp':
            base_dim = 16
            total_dim = self.n_particles * base_dim
            self.policy = nn.Sequential()
            self.policy.add_module(f'in_layer', nn.Linear(self.context_dim, 512))
            self.policy.add_module(f'in_norm', RMSNorm(512))
            self.policy.add_module(f'in_activation', nn.GELU())
            self.policy.add_module(f'in_layer_2', nn.Linear(512, base_dim))
            self.policy.add_module(f'in_flatten', nn.Flatten(start_dim=-2, end_dim=-1))
            self.policy.add_module(f'in_proj', nn.Linear(total_dim, self.hidden_dim))
            for i in range(self.n_layers):
                if i == self.n_layers - 1:
                    self.policy.add_module(f'out_layer', nn.Linear(self.hidden_dim, self.action_dim))
                    if self.tanh_on_output:
                        self.policy.add_module(f'out_activation', nn.Tanh())
                else:
                    self.policy.add_module(f'hid_layer_{i + 1}', nn.Linear(self.hidden_dim, self.hidden_dim))
                    self.policy.add_module(f'in_activation', nn.GELU())
        elif self.policy_type == 'attention':
            self.policy = DLPTransformerPolicy(n_particles=self.dlp_config['n_kp_prior'] + 1, dropout=0.1,
                                               in_dim=self.context_dim, embed_init_std=0.02,
                                               projection_dim=self.hidden_dim, pte_layers=self.n_layers,
                                               pte_heads=self.n_heads,
                                               attn_norm_type='rms', hidden_dim=self.hidden_dim,
                                               activation='gelu', n_views=self.n_views, particle_positional_embed=True,
                                               pos_embed_p_adaln=True,
                                               action_dim=self.action_dim,
                                               tanh_on_output=self.tanh_on_output,
                                               timestep_horizon=self.timestep_horizon, causal=self.causal_policy)
        elif self.policy_type == 'attention_tokens':
            self.policy = DLPTransformerPolicy(n_particles=self.dlp_config['n_kp_prior'] + 1, dropout=0.1,
                                               in_dim=self.context_dim, embed_init_std=0.02,
                                               projection_dim=self.hidden_dim, pte_layers=self.n_layers,
                                               pte_heads=self.n_heads,
                                               attn_norm_type='rms', hidden_dim=self.hidden_dim,
                                               activation='gelu', n_views=self.n_views, particle_positional_embed=True,
                                               pos_embed_p_adaln=True,
                                               action_dim=self.action_dim,
                                               tanh_on_output=self.tanh_on_output,
                                               timestep_horizon=self.timestep_horizon,
                                               ptokens_adaln=True, causal=self.causal_policy)

        else:
            raise NotImplementedError(f'policy type {self.policy_type} unsupported!')
        self.init_from_ckpt()
        print(f'policy type: {self.policy_type}')

    def init_from_ckpt(self):
        pretrained_path = self.config['path_to_policy_ckpt']
        if pretrained_path is not None:
            self.load_dlp_model(load_pretrained=False)
            self.load_state_dict(torch.load(pretrained_path, map_location=torch.device('cpu')))
            print("loaded policy model from checkpoint")
            # if self.dlp_model is None:
            #     print(f'dlp model is None, trying to load from ckpt')
            #     self.load_dlp_model()
            self.dlp_model.eval()
            self.dlp_model.requires_grad_(False)
        else:
            print("poicy model checkpoint not found, initializing and loading dlp model")
            self.load_dlp_model()

    def load_dlp_model(self, load_pretrained=True):
        cpath = self.config['path_to_dlp_ckpt'] if load_pretrained else None
        dlp_model = load_dlp_from_config(self.config['path_to_dlp_hparams'], cpath)
        self.dlp_model = dlp_model
        self.dlp_model.eval()
        self.dlp_model.requires_grad_(False)

    def encode_particles_and_context(self, x, x_goal, deterministic=False):
        enc_dict = self.dlp_model.encode_all(x, deterministic, x_goal=x_goal)
        # unpack encoder output: [bs, T, ...]
        z = enc_dict['z']
        z_features = enc_dict['z_features']
        z_obj_on = enc_dict['obj_on']
        z_depth = enc_dict['z_depth']
        z_scale = enc_dict['z_scale']
        z_bg_features = enc_dict['z_bg_features']
        z_base_var = enc_dict['z_base_var']  # for particle filtering
        z_context = enc_dict['z_context'][:, 1:]  # [bs, T - 1, ...]

        z_dict = {'z': z, 'z_scale': z_scale, 'z_depth': z_depth, 'z_obj_on': z_obj_on, 'z_features': z_features,
                  'z_bg_features': z_bg_features, 'z_base_var': z_base_var, 'z_context': z_context}

        return z_context, z_dict

    def context_to_actions(self, z_context):
        # z_context: [bs, T-1, n_views * n_particles, dim]
        if self.policy_type == 'mlp':
            # flatten
            z_context = z_context.view(z_context.shape[0], z_context.shape[1], -1)
            # [bs, T-1, n_views * n_particles * dim]
        # elif self.policy_type == 'maxpool-mlp' or self.policy_type == 'norm-mlp':
        #     # no need for flattening
        #     z_context = z_context
        # else:
        #     raise NotImplementedError(f'unsupported policy type {self.policy_type}')
        actions = self.policy(z_context)
        # [bs, T-1, action_dim]
        return actions

    def render(self, z_dict):
        # render rgb frames
        z_base_var = z_dict.get('z_base_var', None)
        z = z_dict['z']
        z_scale = z_dict['z_scale']
        z_features = z_dict['z_features']
        z_bg_features = z_dict['z_bg_features']
        z_obj_on = z_dict['z_obj_on']
        z_depth = z_dict['z_depth']
        z_context = z_dict['z_context']
        if self.dlp_model.filter_particles_in_decoder:
            if z_base_var is not None:
                filter_key = z_base_var.sum(-1) if (self.dlp_model.n_kp_enc != self.dlp_model.n_kp_dec) else None
            else:
                filter_key = (1 - z_obj_on).sum(-1) if len(z_obj_on.shape) == 4 else (1 - z_obj_on)
        else:
            filter_key = None
        dec_dict = self.dlp_model.decode_all(z, z_scale, z_features, z_obj_on, z_depth, z_bg_features, z_context,
                                             filter_key=filter_key)
        rec_rgb = dec_dict['rec_rgb']
        return rec_rgb, dec_dict

    def sample(self, x, x_goal=None, n_steps=1, render=False, deterministic=False, use_posterior_ctx=True,
               posterior_ctx_img=False):
        # x: [bs * n_views, T, ch, h, w] -> actions: [bs, n_steps, action_dim]
        # if self.n_views > 1:
        #     # expect: [bs, T, n_views, ...]
        #     x = x.permute(0, 2, 1, 3, 4, 5)
        #     x = x.reshape(-1, *x.shape[2:])  # [bs * n_views, T, ...]
        #     if x_goal is not None:
        #         x_goal = x_goal.reshape(-1, *x_goal.shape[2:])  # [bs * n_views, ...]
        if use_posterior_ctx:
            # go through the posterior: first sample from the prior, then re-encode through the posterior
            if posterior_ctx_img:
                # this means we actually render the images and then re-encode them
                rec_rgb, z_dict = self.dlp_model.sample_from_x(x, num_steps=n_steps, deterministic=deterministic,
                                                               return_z=True,
                                                               x_goal=x_goal, decode=True, n_pred_eq_gt=False)
                # rec_rgb: [bs * n_views, T + n_steps, ch, h, w]
                # we need the last (n_steps + 1) steps to get the inverse dynamics actions
                rec_rgb = rec_rgb[:, -(n_steps + 1):].contiguous()
                z_context, z_dict = self.encode_particles_and_context(rec_rgb, x_goal, deterministic=True)
            else:
                # we are staying in the particles space, just generating one step-more and then re-encoding through
                # the inverse dynamics posterior
                rec_rgb, z_dict = self.dlp_model.sample_from_x(x, num_steps=n_steps + 1, deterministic=deterministic,
                                                               return_z=True,
                                                               x_goal=x_goal, decode=render, n_pred_eq_gt=False,
                                                               return_context_posterior=True)
                # z_context = z_dict['z_context_posterior']  # [bs * n_views, T + n_steps, ...]
                z_context = z_dict['mu_context_posterior']  # [bs * n_views, T + n_steps, ...]
            z_context = z_context[:, -n_steps:].contiguous()  # [bs * n_views, n_steps, n_particles, context_dim]
        else:
            rec_rgb, z_dict = self.dlp_model.sample_from_x(x, num_steps=n_steps, deterministic=deterministic,
                                                           return_z=True,
                                                           x_goal=x_goal, decode=render, n_pred_eq_gt=False)
            z_context = z_dict['z_context']  # [bs * n_views, T + n_steps, n_particles, context_dim]
            z_context = z_context[:, -n_steps:].contiguous()  # [bs * n_views, n_steps, n_particles, context_dim]
        if self.n_views > 1:
            z_context = z_context.view(-1, self.n_views, *z_context.shape[1:])
            # [bs, n_views, n_steps, n_particles + 1, dim]
            z_context = z_context.permute(0, 2, 1, 3, 4)  # [bs, n_steps, n_views, n_particles + 1, dim]
            z_context = z_context.reshape(z_context.shape[0], z_context.shape[1], -1, z_context.shape[-1])
            # [bs, n_steps, n_views * (n_particles + 1), dim]
        # convert to actions
        actions = self.context_to_actions(z_context)
        sample_out = {'actions': actions, 'z_dict': z_dict, 'rec': rec_rgb}
        return sample_out

    def act(self, x, x_goal, n_steps=1, deterministic=False, use_posterior_ctx=True, posterior_ctx_img=False):
        sample_out = self.sample(x, x_goal, n_steps=n_steps, deterministic=deterministic,
                                 use_posterior_ctx=use_posterior_ctx, posterior_ctx_img=posterior_ctx_img)
        actions = sample_out['actions']
        return actions

    def forward(self, x, x_goal, deterministic=True):
        # x: images [bs * n_views, T, ch, h, w] -> a: actions: [bs, T - 1, action_dim]
        # deterministic: if true, z = mu, no **posterior** sampling (posterior typically has small variance)
        # if self.n_views > 1:
        #     # expect: [bs, T, n_views, ...]
        #     x = x.permute(0, 2, 1, 3, 4, 5)
        #     x = x.reshape(-1, *x.shape[2:])  # [bs * n_views, T, ...]
        #     if x_goal is not None:
        #         x_goal = x_goal.reshape(-1, *x_goal.shape[2:])  # [bs * n_views, ...]
        if len(x.shape) == 4:
            # x: [bs, ch, h, w]
            batch_size = x.size(0)
            timestep_horizon = 1
            x = x.unsqueeze(1)
        else:
            # x: [bs, T + 1, ch, h, w]
            batch_size, timestep_horizon = x.size(0), x.size(1)
        if self.dlp_model.normalize_rgb:
            x = rgb_to_minusoneone(x)
            if x_goal is not None:
                x_goal = rgb_to_minusoneone(x_goal)
        # only encode and get context, no need for dynamics or reconstruction
        # get latent actions from DLP
        z_context, z_dict = self.encode_particles_and_context(x, x_goal, deterministic)
        # z_context: [bs * n_views, T - 1, n_particles + 1, dim]
        if self.n_views > 1:
            z_context = z_context.view(-1, self.n_views, *z_context.shape[1:])
            # [bs, n_views, T - 1, n_particles + 1, dim]
            z_context = z_context.permute(0, 2, 1, 3, 4)  # [bs, T - 1, n_views, n_particles + 1, dim]
            z_context = z_context.reshape(z_context.shape[0], z_context.shape[1], -1, z_context.shape[-1])
            # [bs, T - 1, n_views * (n_particles + 1), dim]
        # convert to actions
        actions = self.context_to_actions(z_context)  # [bs, T - 1, action_dim]
        return actions

    def calc_loss(self, gt_actions, pred_actions, loss_type=None):
        # gt_actions, pred_actions: [bs, T - 1, action_dim]
        if loss_type is None:
            loss_type = self.loss_type
        bs, ts, dim = gt_actions.shape
        loss = calc_reconstruction_loss(gt_actions.view(bs * ts, -1), pred_actions.view(bs * ts, -1),
                                        loss_type=loss_type, reduction='none')
        loss = loss.view(bs, ts, -1).sum(-1).mean()
        # norm_f = 1 / ts
        # loss = norm_f * loss
        return loss


if __name__ == '__main__':
    # path_to_config = 'C:/Users/tabad/Projects/gdlp-reloaded/configs/ogbench_policy.json'
    path_to_config = 'C:/Users/tabad/Projects/gdlp-reloaded/configs/frankakitchen_policy.json'
    # path_to_config = 'C:/Users/tabad/Projects/gdlp-reloaded/configs/panda_policy.json'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    model = DLPPolicy(path_to_config).to(device)
    print(f'causal policy: {model.causal_policy}')

    # demo
    ch = 3
    image_size = 128
    # image_size = 64
    batch_size = 2
    timesteps = model.timestep_horizon
    # n_views = 2
    n_views = 1
    render = True
    n_sample_steps = 2
    x = torch.rand(batch_size * n_views, timesteps + 1, ch, image_size, image_size, device=device)
    x_goal = torch.rand(batch_size * n_views, 1, ch, image_size, image_size, device=device)
    gt_actions = torch.randn(batch_size, timesteps, model.action_dim, device=device)
    print(f'x: {x.shape}, x_goal: {x_goal.shape}, gt_actions: {gt_actions.shape}')
    print("----------------------------------")
    actions_output = model(x, x_goal, deterministic=False)
    print(f'model(x, x_goal) output:')
    print(f'actions: {actions_output.shape}')
    # ---- output ---- #
    """
    model(x, x_goal) output:
    actions: torch.Size([2, 3, 3])
    """
    # ---- end output ---- #
    print("----------------------------------")
    loss = model.calc_loss(gt_actions, actions_output)
    print(f'model.calc_loss(gt_actions, actions_output) output:')
    print(f'loss: {loss}')
    # ---- output ---- #
    """
    model.calc_loss(gt_actions, actions_output) output:
    loss: 0.7443069219589233
    """
    # ---- end output ---- #
    print("----------------------------------")
    print(f'sample(x, x_goal, n_steps={n_sample_steps}, render={render}) output:')
    sample_output = model.sample(x, x_goal, n_steps=n_sample_steps, render=render, deterministic=True)
    for k in sample_output.keys():
        if sample_output[k] is not None and not isinstance(sample_output[k], dict):
            print(f'{k}: {sample_output[k].shape}')
        if sample_output[k] is not None and isinstance(sample_output[k], dict):
            print(f'{k}:')
            z_dict = sample_output[k]
            for j in z_dict.keys():
                if z_dict[j] is not None and not isinstance(z_dict[j], dict):
                    print(f'{j}: {z_dict[j].shape}')
    # ---- output ---- #
    """
    sample(x, x_goal, n_steps=2, render=False) output:
    actions: torch.Size([2, 2, 3])
    z_dict:
    z_pos: torch.Size([4, 6, 64, 2])
    z_scale: torch.Size([4, 6, 64, 2])
    z_obj_on: torch.Size([4, 6, 64, 1])
    z_depth: torch.Size([4, 6, 64, 1])
    z_features: torch.Size([4, 6, 64, 4])
    z_context: torch.Size([4, 5, 65, 7])
    z_bg_features: torch.Size([4, 6, 2])
    z_ids: torch.Size([4, 6, 64])
    z_score: torch.Size([4, 6, 64, 1])
    z_goal_proj: torch.Size([4, 1, 65, 512])
    ----------------------------------------
    sample(x, x_goal, n_steps=2, render=True) output:
    actions: torch.Size([2, 2, 3])
    z_dict:
    z_pos: torch.Size([4, 6, 64, 2])
    z_scale: torch.Size([4, 6, 64, 2])
    z_obj_on: torch.Size([4, 6, 64, 1])
    z_depth: torch.Size([4, 6, 64, 1])
    z_features: torch.Size([4, 6, 64, 4])
    z_context: torch.Size([4, 5, 65, 7])
    z_bg_features: torch.Size([4, 6, 2])
    z_ids: torch.Size([4, 6, 64])
    z_score: torch.Size([4, 6, 64, 1])
    z_goal_proj: torch.Size([4, 1, 65, 512])
    rec: torch.Size([4, 6, 3, 128, 128])
    """
    # ---- end output ---- #
    print("----------------------------------")
    print(f'act(x, x_goal, n_steps=1) output:')
    act_output = model.act(x, x_goal, n_steps=1, deterministic=True)
    print(f'actions: {act_output.shape}')
    # ---- output ---- #
    """
    act(x, x_goal, n_steps=1) output:
    actions: torch.Size([2, 1, 3])
    """
    # ---- end output ---- #
