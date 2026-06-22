"""
AdaLNPINTDenoiser
-----------------

Denoising transformer over (action, proprioception, particles) trajectories.

The action and proprio (gripper) channels can be represented either as a
single token each, or split into three semantic sub-tokens (pos / rot / grip)
with separate projections and decoder heads -- the 3DDA-style multi-entity
decomposition that helps the model disentangle position, rotation, and gripper
prediction.

Two flags control robot-token construction:
  * split_action_tokens: if True, BOTH the action AND the proprio (when
    gripper_dim>0) are decoded as three semantic sub-tokens. If False (default),
    action is a single monolithic token and proprio (if present) is a single
    monolithic token -- the legacy 3D behavior, preserved for backwards
    compatibility with existing checkpoints.
  * gripper_dim > 0: if True, proprioception is included in the sequence.
    The number of proprio tokens (1 or 3) is governed by split_action_tokens.

Transformer sequence at each timestep:

    [lang? | cond? | action_tokens... | proprio_tokens... | bg? | particles...]

where action_tokens is 1 or 3 tokens and proprio_tokens is 0, 1, or 3 tokens.

Default component sizes (RLBench keypose, absolute EEF control with rot6d,
matching the 2D close_jar_keypose_multientity config):
    action_dim=10:  act_pos_dim=3, act_rot_dim=6, act_grip_dim=1
    gripper_dim=10: prop_pos_dim=3, prop_rot_dim=6, prop_grip_dim=1

The flat input/output tensor layout is unchanged regardless of split mode:
    x: [batch_size, T, action_dim + gripper_dim + bg_dim + (n_particles * features_dim)]

Note (vs 2D): 2D defaults split_action_tokens=None to (gripper_dim > 0) and
always splits proprio when gripper_dim>0. The 3D port defaults None to False
and couples proprio split to the same flag, so existing 3D checkpoints
(single gripper_projection) keep loading.
"""

import torch
from torch import nn
from diffuser.models.transformer_modules import (
    AdaLNParticleTransformer,
    SinusoidalPosEmb,
    RandomOrLearnedSinusoidalPosEmb,
)

class AdaLNPINTDenoiser(nn.Module):
    """
    AdaLNPINTDenoiser

    Implements a denoising model based on an Adaptive Layer Normalized Particle Interaction
    Transformer. It processes sequences of particle state features concatenated with action
    information and conditioned on a time signal.

    Parameters:
        features_dim (int): Dimensionality of each particle's feature vector.
        action_dim (int): Dimensionality of the action vector.
        hidden_dim (int): Hidden dimension used in projection layers.
        projection_dim (int): Dimension of the latent space in the transformer.
        n_head (int): Number of attention heads in the transformer.
        n_layer (int): Number of transformer layers.
        block_size (int): Time horizon (number of time steps).
        dropout (float): Dropout probability for transformer components.
        predict_delta (bool): If True, the model predicts a delta change rather than an absolute value.
        positional_bias (bool): If True, applies positional bias in the transformer.
        max_particles (int or None): Maximum number of particles (for relative positional bias).
        learned_sinusoidal_cond (bool): If True, use a learned sinusoidal embedding for time conditioning.
        random_fourier_features (bool): If True, use fixed random Fourier features.
        learned_sinusoidal_dim (int): Dimensionality for the learned sinusoidal (or Fourier) features.
        multiview (bool): If True, use separate encodings for multi-view particle inputs.
        gripper_dim (int): Dimensionality of gripper state. If > 0, gripper is treated as
            additional token(s). Typical format: [pos(3), rot_6d(6), open(1)] = 10 dims.
        act_pos_dim, act_rot_dim, act_grip_dim (int): action sub-component dims, used only
            when split_action_tokens=True. Must sum to action_dim.
        prop_pos_dim, prop_rot_dim, prop_grip_dim (int): proprio sub-component dims, used
            only when split_action_tokens=True and gripper_dim>0. Must sum to gripper_dim.
        split_action_tokens (bool or None): if True, split action and proprio into three
            semantic sub-tokens (pos/rot/grip). Default None -> False (single tokens,
            matching legacy 3D behavior).
    """
    def __init__(self, features_dim=2, action_dim=3, hidden_dim=256, projection_dim=256,
                 n_head=8, n_layer=6, block_size=50, dropout=0.1,
                 predict_delta=False, positional_bias=True, max_particles=4,
                 learned_sinusoidal_cond=False, random_fourier_features=False,
                 learned_sinusoidal_dim=16, multiview=False, gripper_dim=0, bg_dim=0,
                 act_pos_dim=3, act_rot_dim=6, act_grip_dim=1,
                 prop_pos_dim=3, prop_rot_dim=6, prop_grip_dim=1,
                 split_action_tokens=None,
                 lang_dim=0, use_cond_tokens=False, **kwargs):
        super(AdaLNPINTDenoiser, self).__init__()

        self.features_dim = features_dim
        self.action_dim = action_dim
        self.gripper_dim = gripper_dim
        self.bg_dim = bg_dim
        self.lang_dim = lang_dim  # CLIP hidden size (e.g. 512); 0 disables language conditioning
        self.use_cond_tokens = use_cond_tokens  # 3DDA-style: current-keypose cond as separate context tokens
        self.predict_delta = predict_delta
        self.projection_dim = projection_dim
        self.max_particles = max_particles
        self.multiview = multiview

        self.act_pos_dim = act_pos_dim
        self.act_rot_dim = act_rot_dim
        self.act_grip_dim = act_grip_dim
        self.prop_pos_dim = prop_pos_dim
        self.prop_rot_dim = prop_rot_dim
        self.prop_grip_dim = prop_grip_dim
        # Default to legacy 3D behavior (single tokens) so existing checkpoints
        # keep loading. Set explicitly to True to enable 3DDA-style multi-entity
        # decomposition (3 action sub-tokens + 3 proprio sub-tokens).
        if split_action_tokens is None:
            split_action_tokens = False
        self.split_action_tokens = bool(split_action_tokens)
        self.use_proprio = gripper_dim > 0
        if self.split_action_tokens:
            assert act_pos_dim + act_rot_dim + act_grip_dim == action_dim, (
                f"action sub-dims {act_pos_dim}+{act_rot_dim}+{act_grip_dim} "
                f"must sum to action_dim={action_dim}")
            if self.use_proprio:
                assert prop_pos_dim + prop_rot_dim + prop_grip_dim == gripper_dim, (
                    f"proprio sub-dims {prop_pos_dim}+{prop_rot_dim}+{prop_grip_dim} "
                    f"must sum to gripper_dim={gripper_dim}")
        # block_size is the time horizon

        # Define an intermediate time embedding dimension.
        time_dim = projection_dim * 4

        # Decide whether to use random/learned Fourier features for time conditioning.
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            # Fourier feature output is concatenated with the original scalar, so add 1.
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(projection_dim)
            fourier_dim = projection_dim

        # Time embedding network.
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, projection_dim)
        )

        # Particle feature projection network.
        self.particle_projection = nn.Sequential(
            nn.Linear(self.features_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.projection_dim)
        )

        def _make_proj(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.projection_dim),
            )

        def _make_dec(out_dim):
            return nn.Sequential(
                nn.Linear(self.projection_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )

        # Action side: 1 token (single head) or 3 tokens (pos/rot/grip heads).
        if self.split_action_tokens:
            self.a_pos_projection = _make_proj(act_pos_dim)
            self.a_rot_projection = _make_proj(act_rot_dim)
            self.a_grip_projection = _make_proj(act_grip_dim)
            self.a_pos_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
            self.a_rot_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
            self.a_grip_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
            self.a_pos_decoder = _make_dec(act_pos_dim)
            self.a_rot_decoder = _make_dec(act_rot_dim)
            self.a_grip_decoder = _make_dec(act_grip_dim)
        else:
            self.action_projection = _make_proj(action_dim)
            self.action_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
            self.action_decoder = _make_dec(action_dim)

        # Proprio (gripper) side: 0 tokens, 1 token (legacy), or 3 tokens (pos/rot/grip).
        if self.use_proprio:
            if self.split_action_tokens:
                self.p_pos_projection = _make_proj(prop_pos_dim)
                self.p_rot_projection = _make_proj(prop_rot_dim)
                self.p_grip_projection = _make_proj(prop_grip_dim)
                self.p_pos_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
                self.p_rot_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
                self.p_grip_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
                self.p_pos_decoder = _make_dec(prop_pos_dim)
                self.p_rot_decoder = _make_dec(prop_rot_dim)
                self.p_grip_decoder = _make_dec(prop_grip_dim)
            else:
                self.gripper_projection = _make_proj(self.gripper_dim)
                self.gripper_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
                self.gripper_decoder = _make_dec(self.gripper_dim)

        # Background-feature projection (if bg_dim > 0). bg is denoised as its own
        # transformer token rather than passed through unchanged.
        if self.bg_dim > 0:
            self.bg_projection = _make_proj(self.bg_dim)
            self.bg_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
            self.bg_decoder = _make_dec(self.bg_dim)

        # Language token projection (if lang_dim > 0). CLIP embedding dim -> projection_dim.
        if self.lang_dim > 0:
            self.lang_projection = _make_proj(self.lang_dim)
            self.lang_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))

        # Instantiate the AdaLN Particle Transformer.
        self.particle_transformer = AdaLNParticleTransformer(
            self.projection_dim, n_head, n_layer, block_size, self.projection_dim,
            attn_pdrop=dropout, resid_pdrop=dropout,
            hidden_dim_multiplier=4,
            positional_bias=positional_bias,
            activation='gelu', max_particles=max_particles
        )

        # Decoder networks for particle outputs.
        self.particle_decoder = nn.Sequential(
            nn.Linear(self.projection_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.features_dim)
        )

        # Particle encoding: either shared or view-specific for multi-view inputs.
        if self.multiview:
            self.view1_encoding = nn.Parameter(0.02 * torch.randn(1, 1, 1, projection_dim))
            self.view2_encoding = nn.Parameter(0.02 * torch.randn(1, 1, 1, projection_dim))
        else:
            self.particle_encoding = nn.Parameter(0.02 * torch.randn(1, 1, 1, projection_dim))

        # Conditioning-token encodings: 3DDA-style separate context tokens for the
        # current keypose. Distinct from the trajectory encodings so the model can
        # tell "this token is current scene context" vs "this token is the
        # next-keypose prediction." Reuse projection layers (gripper/bg/particle)
        # since the modalities are identical.
        if self.use_cond_tokens:
            if self.use_proprio:
                if self.split_action_tokens:
                    self.cond_p_pos_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
                    self.cond_p_rot_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
                    self.cond_p_grip_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
                else:
                    self.cond_gripper_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
            if self.bg_dim > 0:
                self.cond_bg_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
            if self.multiview:
                self.cond_view1_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
                self.cond_view2_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
            else:
                self.cond_particle_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))

    def forward(self, x, cond, time, return_attention=False, lang=None, lang_mask=None):
        """
        Forward pass for the denoiser.

        Args:
            x (torch.Tensor): Input tensor of shape:
                - If gripper_dim == 0: [batch_size, T, action_dim + bg_dim + particle_feature_dim]
                - If gripper_dim > 0:  [batch_size, T, action_dim + gripper_dim + bg_dim + particle_feature_dim]
                The flat layout is: [actions, gripper_state (optional), bg (optional), particles].
                Layout is unchanged whether split_action_tokens is on or off; splitting only
                affects the internal token construction.
            cond: Conditioning dict. When use_cond_tokens=True, cond[0] is a
                concatenated tensor [bs, gripper_dim + bg_dim + n_particles*features_dim]
                holding the *current* keypose's modalities. The model projects these
                into separate context tokens (with distinct learnable encodings) and
                prepends them after lang in the transformer sequence -- they are
                non-denoised cross-attention context, not predicted. When
                use_cond_tokens=False, cond is unused (legacy visuomotor path uses
                apply_conditioning to inpaint slot 0 of the trajectory instead).
            time (torch.Tensor): Tensor of time indices with shape [batch_size]. These are embedded via time_mlp.
            return_attention (bool): If True, returns the attention weights along with the output.
            lang (torch.Tensor, optional): CLIP-encoded instruction tokens of shape
                [batch_size, L_lang, lang_dim]. Treated as additional (non-denoised) tokens
                concatenated at the front of each timestep's token sequence.

        Returns:
            torch.Tensor or tuple:
                - If return_attention is False: output tensor of shape [batch_size, T, output_dim],
                  where output_dim = action_dim + gripper_dim + bg_dim + (n_particles * features_dim).
                - If return_attention is True: (output, attention_dict)
        """
        # ---------------------------------------------------------------------
        # Reshape input: separate actions, gripper state, bg, and particle features.
        # x: [bs, T, action_dim + gripper_dim + bg_dim + particle_feature_dim]
        bs, T, f = x.size()
        actions = x[:, :, :self.action_dim]  # [bs, T, action_dim]

        if self.gripper_dim > 0:
            gripper_state = x[:, :, self.action_dim:self.action_dim + self.gripper_dim]  # [bs, T, gripper_dim]
        else:
            gripper_state = None

        # Extract bg_features if present (projected into a transformer token below).
        if self.bg_dim > 0:
            bg_start = self.action_dim + self.gripper_dim
            bg_features = x[:, :, bg_start:bg_start + self.bg_dim]  # [bs, T, bg_dim]
        else:
            bg_features = None

        # particle_start_idx accounts for actions, gripper_state, and bg_features.
        particle_start_idx = self.action_dim + self.gripper_dim + self.bg_dim

        # Reshape remaining features into particles of shape [bs, T, n_particles, features_dim].
        x_particles = x[:, :, particle_start_idx:].view(bs, T, -1, self.features_dim)

        # ---------------------------------------------------------------------
        # Project particles (with optional per-view encoding).
        state_particles = self.particle_projection(x_particles)
        if self.multiview:
            n_particles = state_particles.size(2) // 2
            particles_view1 = state_particles[:, :, :n_particles, :] + self.view1_encoding.repeat(bs, T, n_particles, 1)
            particles_view2 = state_particles[:, :, n_particles:, :] + self.view2_encoding.repeat(bs, T, n_particles, 1)
            new_state_particles = torch.cat([particles_view1, particles_view2], dim=2)
        else:
            new_state_particles = state_particles + self.particle_encoding.repeat(bs, T, state_particles.size(2), 1)

        # Project bg features if present (bg is denoised as its own token).
        if self.bg_dim > 0 and bg_features is not None:
            bg_emb = self.bg_projection(bg_features)  # [bs, T, projection_dim]
            bg_emb = bg_emb + self.bg_encoding.repeat(bs, T, 1)

        # ---------------------------------------------------------------------
        # Action tokens: 1 (single head) or 3 (pos/rot/grip heads).
        if self.split_action_tokens:
            ap1 = self.act_pos_dim
            ar1 = ap1 + self.act_rot_dim
            ag1 = ar1 + self.act_grip_dim  # == action_dim
            a_pos_tok = self.a_pos_projection(actions[:, :, :ap1]) + self.a_pos_encoding.repeat(bs, T, 1)
            a_rot_tok = self.a_rot_projection(actions[:, :, ap1:ar1]) + self.a_rot_encoding.repeat(bs, T, 1)
            a_grip_tok = self.a_grip_projection(actions[:, :, ar1:ag1]) + self.a_grip_encoding.repeat(bs, T, 1)
            action_tok_list = [a_pos_tok.unsqueeze(2), a_rot_tok.unsqueeze(2), a_grip_tok.unsqueeze(2)]
            anchor = a_pos_tok  # used by the transformer's AdaLN anchor path
        else:
            action_particle = self.action_projection(actions) + self.action_encoding.repeat(bs, T, 1)
            action_tok_list = [action_particle.unsqueeze(2)]
            anchor = action_particle

        # Proprio tokens: 0 (no gripper), 1 (single legacy), or 3 (pos/rot/grip).
        proprio_tok_list = []
        if self.use_proprio and gripper_state is not None:
            if self.split_action_tokens:
                pp1 = self.prop_pos_dim
                pr1 = pp1 + self.prop_rot_dim
                pg1 = pr1 + self.prop_grip_dim  # == gripper_dim
                p_pos_tok = self.p_pos_projection(gripper_state[:, :, :pp1]) + self.p_pos_encoding.repeat(bs, T, 1)
                p_rot_tok = self.p_rot_projection(gripper_state[:, :, pp1:pr1]) + self.p_rot_encoding.repeat(bs, T, 1)
                p_grip_tok = self.p_grip_projection(gripper_state[:, :, pr1:pg1]) + self.p_grip_encoding.repeat(bs, T, 1)
                proprio_tok_list = [p_pos_tok.unsqueeze(2), p_rot_tok.unsqueeze(2), p_grip_tok.unsqueeze(2)]
            else:
                gripper_emb = self.gripper_projection(gripper_state) + self.gripper_encoding.repeat(bs, T, 1)
                proprio_tok_list = [gripper_emb.unsqueeze(2)]

        n_action_toks = len(action_tok_list)
        n_proprio_toks = len(proprio_tok_list)

        # ---------------------------------------------------------------------
        # Token sequence: [action_tokens..., proprio_tokens..., bg?, particles].
        tokens = [*action_tok_list, *proprio_tok_list]
        proprio_start_token_idx = n_action_toks if n_proprio_toks > 0 else None

        bg_token_idx = None
        if self.bg_dim > 0 and bg_features is not None:
            bg_token_idx = n_action_toks + n_proprio_toks
            tokens.append(bg_emb.unsqueeze(2))

        tokens.append(new_state_particles)
        x_cat = torch.cat(tokens, dim=2)
        particle_start_token_idx = n_action_toks + n_proprio_toks + (1 if bg_token_idx is not None else 0)

        # ---------------------------------------------------------------------
        # Language conditioning: prepend projected CLIP tokens to the per-timestep
        # token sequence (broadcast across T, since the instruction is time-invariant).
        # Language tokens are non-denoised context and are sliced out before decoding.
        n_lang = 0
        if self.lang_dim > 0 and lang is not None:
            # lang: [bs, L_lang, lang_dim]
            lang_proj = self.lang_projection(lang)                  # [bs, L_lang, projection_dim]
            lang_proj = lang_proj + self.lang_encoding              # broadcast over batch/length
            lang_proj = lang_proj.unsqueeze(1).expand(-1, T, -1, -1)  # [bs, T, L_lang, projection_dim]
            x_cat = torch.cat([lang_proj, x_cat], dim=2)
            n_lang = lang_proj.shape[2]
            if proprio_start_token_idx is not None:
                proprio_start_token_idx = proprio_start_token_idx + n_lang
            if bg_token_idx is not None:
                bg_token_idx = bg_token_idx + n_lang
            particle_start_token_idx = particle_start_token_idx + n_lang

        # ---------------------------------------------------------------------
        # Conditioning tokens: 3DDA-style separate context for the *current*
        # keypose. cond[0] = concat(gripper, bg, particles) at the conditioning
        # frame. Project each modality through the existing trajectory
        # projections (weight reuse), tag with cond-specific encodings, broadcast
        # across T (time-invariant context), and insert between lang and
        # trajectory tokens.
        n_cond = 0
        if self.use_cond_tokens and cond is not None and 0 in cond:
            cond_vec = cond[0]  # [bs, gripper_dim + bg_dim + n_particles*features_dim]
            cond_pos = 0
            cond_token_list = []

            if self.gripper_dim > 0:
                cond_grip = cond_vec[:, cond_pos:cond_pos + self.gripper_dim]
                cond_pos += self.gripper_dim
                if self.split_action_tokens:
                    cpp1 = self.prop_pos_dim
                    cpr1 = cpp1 + self.prop_rot_dim
                    cpg1 = cpr1 + self.prop_grip_dim
                    cp_pos = self.p_pos_projection(cond_grip[:, :cpp1]) + self.cond_p_pos_encoding[:, 0]
                    cp_rot = self.p_rot_projection(cond_grip[:, cpp1:cpr1]) + self.cond_p_rot_encoding[:, 0]
                    cp_grip = self.p_grip_projection(cond_grip[:, cpr1:cpg1]) + self.cond_p_grip_encoding[:, 0]
                    cond_token_list.extend([
                        cp_pos.unsqueeze(1), cp_rot.unsqueeze(1), cp_grip.unsqueeze(1),
                    ])  # each [bs, 1, projection_dim]
                else:
                    cond_grip_emb = self.gripper_projection(cond_grip)  # [bs, projection_dim]
                    cond_grip_emb = cond_grip_emb + self.cond_gripper_encoding[:, 0]
                    cond_token_list.append(cond_grip_emb.unsqueeze(1))  # [bs, 1, projection_dim]

            if self.bg_dim > 0:
                cond_bg = cond_vec[:, cond_pos:cond_pos + self.bg_dim]
                cond_pos += self.bg_dim
                cond_bg_emb = self.bg_projection(cond_bg)
                cond_bg_emb = cond_bg_emb + self.cond_bg_encoding[:, 0]
                cond_token_list.append(cond_bg_emb.unsqueeze(1))

            cond_particles_flat = cond_vec[:, cond_pos:]  # [bs, n_particles*features_dim]
            cond_particles = cond_particles_flat.view(bs, -1, self.features_dim)
            cond_part_emb = self.particle_projection(cond_particles)  # [bs, n_particles, projection_dim]
            if self.multiview:
                n_p = cond_part_emb.size(1) // 2
                cond_part_view1 = cond_part_emb[:, :n_p] + self.cond_view1_encoding[:, 0]
                cond_part_view2 = cond_part_emb[:, n_p:] + self.cond_view2_encoding[:, 0]
                cond_part_emb = torch.cat([cond_part_view1, cond_part_view2], dim=1)
            else:
                cond_part_emb = cond_part_emb + self.cond_particle_encoding[:, 0]
            cond_token_list.append(cond_part_emb)

            cond_tokens = torch.cat(cond_token_list, dim=1)  # [bs, n_cond, projection_dim]
            n_cond = cond_tokens.size(1)
            cond_tokens = cond_tokens.unsqueeze(1).expand(-1, T, -1, -1)  # [bs, T, n_cond, projection_dim]

            # Insert between lang (positions [0:n_lang]) and trajectory tokens.
            x_cat = torch.cat([x_cat[:, :, :n_lang], cond_tokens, x_cat[:, :, n_lang:]], dim=2)
            if proprio_start_token_idx is not None:
                proprio_start_token_idx = proprio_start_token_idx + n_cond
            if bg_token_idx is not None:
                bg_token_idx = bg_token_idx + n_cond
            particle_start_token_idx = particle_start_token_idx + n_cond

        # Build per-token padding mask (B, n_tokens). Lang tokens come first in
        # x_cat, so we pad lang positions with lang_mask validity and mark all
        # non-lang positions as valid (1). Used by the transformer to ignore
        # CLIP padding in self-attention.
        attn_mask = None
        if n_lang > 0 and lang_mask is not None:
            n_nonlang = x_cat.size(2) - n_lang
            ones = torch.ones(bs, n_nonlang, device=x.device, dtype=lang_mask.dtype)
            attn_mask = torch.cat([lang_mask, ones], dim=1)

        # Time embedding: project time indices and add to all tokens.
        t_embed = self.time_mlp(time)  # [bs, projection_dim]
        x_proj = x_cat + t_embed[:, None, None, :]  # Broadcast addition.

        # Permute to match transformer input shape: [bs, n_tokens, T, projection_dim]
        x_proj = x_proj.permute(0, 2, 1, 3)

        # ---------------------------------------------------------------------
        # Apply the particle transformer.
        if return_attention:
            particles_trans, attention_dict = self.particle_transformer(x_proj, anchor, t_embed,
                                                                         return_attention=return_attention,
                                                                         attn_mask=attn_mask)
        else:
            particles_trans = self.particle_transformer(x_proj, anchor, t_embed, attn_mask=attn_mask)
        # Permute back to [bs, T, n_tokens, projection_dim].
        particles_trans = particles_trans.permute(0, 2, 1, 3)

        # ---------------------------------------------------------------------
        # Decode transformer output.
        # Token layout: [lang? | cond? | action_tokens... | proprio_tokens... | bg? | particles].
        # Lang and cond tokens are non-denoised context and are dropped at decode.
        action_token_start = n_lang + n_cond

        parts = []
        if self.split_action_tokens:
            parts.append(self.a_pos_decoder(particles_trans[:, :, action_token_start + 0, :]))
            parts.append(self.a_rot_decoder(particles_trans[:, :, action_token_start + 1, :]))
            parts.append(self.a_grip_decoder(particles_trans[:, :, action_token_start + 2, :]))
        else:
            parts.append(self.action_decoder(particles_trans[:, :, action_token_start, :]))

        if self.use_proprio and proprio_start_token_idx is not None:
            if self.split_action_tokens:
                parts.append(self.p_pos_decoder(particles_trans[:, :, proprio_start_token_idx + 0, :]))
                parts.append(self.p_rot_decoder(particles_trans[:, :, proprio_start_token_idx + 1, :]))
                parts.append(self.p_grip_decoder(particles_trans[:, :, proprio_start_token_idx + 2, :]))
            else:
                parts.append(self.gripper_decoder(particles_trans[:, :, proprio_start_token_idx, :]))

        if self.bg_dim > 0 and bg_token_idx is not None:
            parts.append(self.bg_decoder(particles_trans[:, :, bg_token_idx, :]))  # [bs, T, bg_dim]

        particle_decoder_out = self.particle_decoder(particles_trans[:, :, particle_start_token_idx:, :])
        particle_decoder_out = particle_decoder_out.view(bs, T, -1)  # Flatten particle outputs.
        parts.append(particle_decoder_out)

        x_out = torch.cat(parts, dim=-1)

        if return_attention:
            return x_out, attention_dict
        else:
            return x_out

# ------------------------------------------------------------------------------
# Test block
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    batch_size = 32
    timessteps = 5

    # Test without gripper token
    print("=" * 60)
    print("Test 1: Without gripper token (gripper_dim=0)")
    print("=" * 60)
    model = AdaLNPINTDenoiser(features_dim=10, action_dim=3, hidden_dim=256, projection_dim=256,
                        n_head=8, n_layer=6, block_size=timessteps, dropout=0.1,
                        predict_delta=False, positional_bias=False, max_particles=None,
                        learned_sinusoidal_cond=False, random_fourier_features=False, learned_sinusoidal_dim=16,
                        gripper_dim=0)
    in_particles = torch.randn(batch_size, timessteps, 240)
    actions = torch.randn(batch_size, timessteps, 3)
    t = torch.randint(0, 1000, (batch_size,), device=in_particles.device).long()

    # Concatenate actions and particle features.
    x = torch.cat([actions, in_particles], dim=-1)
    model_out = model(x, cond=None, time=t, return_attention=False)
    print("Input shape:", x.shape)
    print("Output shape:", model_out.shape)
    assert model_out.shape == x.shape, "Output shape should match input shape"

    # Test with gripper token (legacy, single-token mode)
    print("\n" + "=" * 60)
    print("Test 2: With gripper token, single-token mode (gripper_dim=10, split=False)")
    print("=" * 60)
    gripper_dim = 10  # pos(3) + rot_6d(6) + open(1)
    model_with_gripper = AdaLNPINTDenoiser(
        features_dim=10, action_dim=3, hidden_dim=256, projection_dim=256,
        n_head=8, n_layer=6, block_size=timessteps, dropout=0.1,
        predict_delta=False, positional_bias=False, max_particles=None,
        learned_sinusoidal_cond=False, random_fourier_features=False, learned_sinusoidal_dim=16,
        gripper_dim=gripper_dim
    )
    gripper_state = torch.randn(batch_size, timessteps, gripper_dim)

    # Concatenate actions, gripper state, and particle features.
    x_with_gripper = torch.cat([actions, gripper_state, in_particles], dim=-1)
    model_out_with_gripper = model_with_gripper(x_with_gripper, cond=None, time=t, return_attention=False)
    print("Input shape:", x_with_gripper.shape)
    print("Output shape:", model_out_with_gripper.shape)
    assert model_out_with_gripper.shape == x_with_gripper.shape, "Output shape should match input shape"

    # Test split-token (multi-entity) path: action_dim=10 [pos(3)+rot6d(6)+open(1)],
    # gripper_dim=10 same layout. Matches RLBench keypose multientity configs.
    print("\n" + "=" * 60)
    print("Test 3: Split tokens (action_dim=10, gripper_dim=10, split_action_tokens=True)")
    print("=" * 60)
    action_dim_split = 10  # pos(3) + rot6d(6) + open(1)
    gripper_dim_split = 10
    model_split = AdaLNPINTDenoiser(
        features_dim=10, action_dim=action_dim_split, hidden_dim=256, projection_dim=256,
        n_head=8, n_layer=6, block_size=timessteps, dropout=0.1,
        predict_delta=False, positional_bias=False, max_particles=None,
        learned_sinusoidal_cond=False, random_fourier_features=False, learned_sinusoidal_dim=16,
        gripper_dim=gripper_dim_split,
        act_pos_dim=3, act_rot_dim=6, act_grip_dim=1,
        prop_pos_dim=3, prop_rot_dim=6, prop_grip_dim=1,
        split_action_tokens=True,
    )
    actions_split = torch.randn(batch_size, timessteps, action_dim_split)
    gripper_state_split = torch.randn(batch_size, timessteps, gripper_dim_split)
    x_split = torch.cat([actions_split, gripper_state_split, in_particles], dim=-1)
    model_out_split = model_split(x_split, cond=None, time=t, return_attention=False)
    print("Input shape:", x_split.shape)
    print("Output shape:", model_out_split.shape)
    assert model_out_split.shape == x_split.shape, "Output shape should match input shape"

    # Test split-token + use_cond_tokens (3DDA-style cond context).
    print("\n" + "=" * 60)
    print("Test 4: Split tokens + use_cond_tokens=True (keypose mode)")
    print("=" * 60)
    bg_dim = 2
    model_cond = AdaLNPINTDenoiser(
        features_dim=10, action_dim=action_dim_split, hidden_dim=256, projection_dim=256,
        n_head=8, n_layer=6, block_size=timessteps, dropout=0.1,
        predict_delta=False, positional_bias=False, max_particles=None,
        learned_sinusoidal_cond=False, random_fourier_features=False, learned_sinusoidal_dim=16,
        gripper_dim=gripper_dim_split, bg_dim=bg_dim,
        act_pos_dim=3, act_rot_dim=6, act_grip_dim=1,
        prop_pos_dim=3, prop_rot_dim=6, prop_grip_dim=1,
        split_action_tokens=True, use_cond_tokens=True,
    )
    bg_traj = torch.randn(batch_size, timessteps, bg_dim)
    x_cond = torch.cat([actions_split, gripper_state_split, bg_traj, in_particles], dim=-1)
    cond_vec = torch.randn(batch_size, gripper_dim_split + bg_dim + 24 * 10)
    cond = {0: cond_vec}
    model_out_cond = model_cond(x_cond, cond=cond, time=t, return_attention=False)
    print("Input shape:", x_cond.shape)
    print("Output shape:", model_out_cond.shape)
    assert model_out_cond.shape == x_cond.shape, "Output shape should match input shape"

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
