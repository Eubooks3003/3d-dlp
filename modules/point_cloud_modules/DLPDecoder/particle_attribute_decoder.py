import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class ParticleAttributeDecoder(nn.Module):
    def __init__(self, n_particles, input_dim, hidden_dim, features_dim, bg_features_dim=None,
                 depth=False, obj_on=False, features=False, bg_features=False,
                 offset_logvar=False,
                 activation='gelu', dropout=0.0, shared_logvar=False,
                 output_ctx_logvar=True, features_dist='gauss'):
        super().__init__()
        # decoder to map back from PTE's inner dim to the particle's original dimension
        self.n_particles = n_particles
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.features_dist = features_dist
        self.features_dim = features_dim
        self.bg_features_dim = bg_features_dim
        self.offset_logvar = offset_logvar
        self.with_depth = depth
        self.with_obj_on = obj_on
        self.with_features = features
        self.with_bg_features = bg_features
        self.use_fg_backbone = (self.with_obj_on or self.with_depth or self.with_features)
        self.shared_logvar = shared_logvar
        self.output_ctx_logvar = output_ctx_logvar
        activation_f = nn.GELU if activation == 'gelu' else nn.ReLU
        if self.use_fg_backbone:
            self.fg_backbone = nn.Identity()
            if self.with_obj_on:
                self.obj_on_head = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                                 activation_f(),
                                                 nn.Linear(hidden_dim, 1)
                                                 )  # log_a, log_b
            if self.with_depth:
                self.depth_head = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                                activation_f(),
                                                nn.Linear(hidden_dim, 2)
                                                )  # mu_z, logvar_z
            if self.with_features:
                output_feat_dim = 2 * features_dim if (features_dist != 'categorical') else features_dim
                self.features_head = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                                   activation_f(),
                                                   nn.Linear(hidden_dim, output_feat_dim)
                                                   )  # mu_features, logvar_features
        if self.with_bg_features:
            output_bg_feat_dim = 2 * bg_features_dim if (features_dist != 'categorical') else bg_features_dim
            self.bg_backbone = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                             activation_f(),
                                             )
            self.bg_features_head = nn.Linear(hidden_dim, output_bg_feat_dim)  # mu_features, logvar_features

        self.init_weights()

    def init_weights(self):
        if self.with_features and self.features_dist != 'categorical':
            nn.init.constant_(self.features_head[-1].weight[:self.features_dim], 0.0)
            nn.init.constant_(self.features_head[-1].bias[:self.features_dim], 0.0)
            nn.init.constant_(self.features_head[-1].weight[self.features_dim:], 0.0)
            nn.init.constant_(self.features_head[-1].bias[self.features_dim:], math.log(0.001 ** 2))
        if self.with_bg_features and self.features_dist != 'categorical':
            nn.init.constant_(self.bg_features_head.weight[:self.bg_features_dim], 0.0)
            nn.init.constant_(self.bg_features_head.bias[:self.bg_features_dim], 0.0)
            nn.init.constant_(self.bg_features_head.weight[self.bg_features_dim:], 0.0)
            nn.init.constant_(self.bg_features_head.bias[self.bg_features_dim:], math.log(0.001 ** 2))

    def forward(self, x):
        # x: [bs, n_particles, input_dim]
        # bs, n_particles, in_dim = x.shape
        bs, ts, n_particles = x.shape[0], x.shape[1], x.shape[2]
        # the following assumes fg_particles + bg_particle + context particle
        fg_particles = n_particles - 2 if self.with_bg_features else n_particles - 1
        if self.use_fg_backbone:
            x_fg = x[:, :, :fg_particles]
            fg_features = self.fg_backbone(x_fg)
            if self.with_depth:
                depth = self.depth_head(fg_features)
                mu_depth, logvar_depth = torch.chunk(depth, 2, dim=-1)
            else:
                mu_depth = logvar_depth = None
            if self.with_obj_on:
                obj_on = self.obj_on_head(fg_features)
                lobj_on_a = lobj_on_b = obj_on
            else:
                lobj_on_a = lobj_on_b = None
            if self.with_features:
                features = self.features_head(fg_features)
                if self.features_dist != 'categorical':
                    mu_features, logvar_features = torch.chunk(features, 2, dim=-1)
                else:
                    mu_features = logvar_features = features
            else:
                mu_features = logvar_features = None
        else:
            mu_depth = logvar_depth = None
            lobj_on_a = lobj_on_b = None
            mu_features = logvar_features = None

        if self.with_bg_features:
            x_bg = x[:, :, fg_particles]
            bg_features = self.bg_backbone(x_bg)
            bg_features = self.bg_features_head(bg_features)
            if self.features_dist != 'categorical':
                mu_bg_features, logvar_bg_features = torch.chunk(bg_features, 2, dim=-1)
            else:
                mu_bg_features = logvar_bg_features = bg_features
        else:
            mu_bg_features = logvar_bg_features = None

        decoder_out = {'mu_depth': mu_depth, 'logvar_depth': logvar_depth,
                       'lobj_on_a': lobj_on_a, 'lobj_on_b': lobj_on_b,
                       'mu_features': mu_features, 'logvar_features': logvar_features,
                       'mu_bg_features': mu_bg_features, 'logvar_bg_features': logvar_bg_features}

        return decoder_out