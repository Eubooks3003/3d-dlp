import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.util_func import modulate
class ParticleSelfAttTransformer(nn.Module):
    def __init__(self, n_embed, n_head, n_layer, block_size, output_dim, attn_pdrop=0.1, resid_pdrop=0.1,
                 hidden_dim_multiplier=4, positional_bias=False, activation='gelu', max_particles=None,
                 norm_type='rms', n_registers=0, init_std=0.02):
        super().__init__()
        self.positional_bias = positional_bias
        self.max_particles = max_particles  # for positional bias
        self.n_registers = n_registers  # "vision transformers need registers", balances the attention matrix
        # input embedding stem
        if self.positional_bias:
            self.pos_emb = nn.Identity()
        else:
            self.pos_emb = nn.Parameter(init_std * torch.randn(1, block_size, n_embed))
        if self.n_registers > 0:
            self.registers = nn.Parameter(init_std * torch.randn(1, self.n_registers, 1, n_embed))
        else:
            self.registers = None
        # transformer
        self.blocks = nn.Sequential(*[SelfBlock(n_embed, n_head, block_size, attn_pdrop,
                                                resid_pdrop, hidden_dim_multiplier,
                                                positional_bias, activation=activation, max_particles=max_particles,
                                                norm_type=norm_type)
                                      for _ in range(n_layer)])
        # decoder head
        if norm_type == 'rms':
            norm_layer = RMSNorm
        elif norm_type == 'pn':
            norm_layer = ParticleNorm
        else:
            norm_layer = nn.LayerNorm
        self.ln_f = norm_layer(n_embed)
        self.head = nn.Linear(n_embed, output_dim, bias=False)

        self.block_size = block_size
        self.n_embed = n_embed
        self.n_layer = n_layer
        # print(f"particle transformer # parameters: {sum(p.numel() for p in self.parameters())}")

    def get_block_size(self):
        return self.block_size

    def init_weights(self):
        # initialize layers
        pass
        # self.apply(self._init_weights)
        # if self.positional_bias:
        #     for m in self.blocks:
        #         m.attn.rel_pos_bias.reset_parameters()

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        # elif isinstance(module, ParticleTransformer):
        #     if not self.positional_bias:
        #         torch.nn.init.normal_(module.pos_emb, mean=0.0, std=std)

    def forward(self, x):
        # x: [b, t, n, f]
        x = x.permute(0, 2, 1, 3)  # [b, n, t, f]
        b, n, t, f = x.size()
        # n is the number of particles
        assert t <= self.block_size, f"Cannot forward, model block size is exhausted: t:{t}, block_size: {self.block_size}"
        assert f == self.n_embed, "invalid particle feature dim"

        if self.n_registers > 0 and self.registers is not None:
            registers = self.registers.repeat(b, 1, t, 1)
            x = torch.cat([x, registers], dim=1)  # [b, n+n_reg, t, f]

        if not self.positional_bias:
            position_embeddings = self.pos_emb[:, None, :t, :]
            x = x + position_embeddings
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if self.n_registers > 0:
            logits, _ = logits.split([logits.shape[1] - self.n_registers, self.n_registers], dim=1)
        logits = logits.permute(0, 2, 1, 3)  # [b, t, n, f]

        return logits


class ParticleNorm(nn.Module):
    """
    experimental particle normalization module, not used in the code but left here for research
    """

    def __init__(self, particle_dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.particle_dim = particle_dim
        self.a = nn.Parameter(torch.ones(1, 1, 1, self.particle_dim))
        self.g = nn.Parameter(torch.ones(1, 1, 1, self.particle_dim))
        self.s = nn.Parameter(torch.zeros(1, 1, 1, self.particle_dim))

    def forward(self, x):
        # [bs, n_particles, T, dim]
        # if self.particle_dim > 1:
        #     dims = (1, 3)
        # else:
        dims = (1,)
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, unbiased=False, keepdim=True)
        if len(x.shape) == 3:
            d_n = (x - self.a.squeeze(2) * mean) / (var + self.eps).sqrt()
            out = d_n * self.g.squeeze(2) + self.s.squeeze(2)
        else:
            d_n = (x - self.a * mean) / (var + self.eps).sqrt()
            out = d_n * self.g + self.s
        return out


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # F.normalize: x = x / (x ** 2).sum(-1, keepdim=True).sqrt()
        return F.normalize(x, dim=-1) * self.scale * self.g


class SelfBlock(nn.Module):
    """ self-attention Transformer block """

    def __init__(self, n_embed, n_head, block_size, attn_pdrop=0.1, resid_pdrop=0.1, hidden_dim_multiplier=4,
                 positional_bias=False, activation='gelu', max_particles=None, norm_type='ln', context_cond=False,
                 residual_modulation=False, context_gate=False, attn_scale=1.0):
        super().__init__()
        self.max_particles = max_particles
        if norm_type == 'rms':
            norm_layer = RMSNorm
        elif norm_type == 'pn':
            norm_layer = ParticleNorm
        else:
            norm_layer = nn.LayerNorm
        self.ln1 = norm_layer(n_embed)
        self.ln2 = norm_layer(n_embed)
        self.attn = ParticleSelfAttention(n_embed, n_head, block_size, attn_pdrop, resid_pdrop,
                                          positional_bias=positional_bias, max_particles=max_particles)
        self.attn_scale = attn_scale
        self.mlp = MLP(n_embed, resid_pdrop, hidden_dim_multiplier, activation=activation)
        self.context_cond = context_cond
        self.residual_modulation = residual_modulation
        self.context_gate = context_gate
        self.c_multiplier = 6 if context_gate else 4
        if self.context_cond:
            self.c_proj = nn.Linear(n_embed, self.c_multiplier * n_embed)
            nn.init.constant_(self.c_proj.weight, 0.0)
            if self.residual_modulation:
                nn.init.constant_(self.c_proj.bias, 0.0)
            else:
                nn.init.constant_(self.c_proj.bias[:2 * n_embed], 1.0)  # identity
                nn.init.constant_(self.c_proj.bias[2 * n_embed: 4 * n_embed], 0.0)  # zero shift
                if self.context_gate:
                    nn.init.constant_(self.c_proj.bias[4 * n_embed:], 0.0)  # zero gate

    def forward(self, x, c=None):
        if self.context_cond and c is not None:
            c_proj = self.c_proj(c).chunk(self.c_multiplier, dim=-1)
            scale_a, scale_b, shift_a, shift_b = c_proj[0], c_proj[1], c_proj[2], c_proj[3]
            if self.context_gate:
                gate_a, gate_b = c_proj[4], c_proj[5]
            else:
                gate_a = gate_b = 1.0
            x = x + self.attn_scale * gate_a * self.attn(
                modulate(self.ln1(x), scale_a, shift_a, self.residual_modulation))
            x = x + gate_b * self.mlp(modulate(self.ln2(x), scale_b, shift_b, self.residual_modulation))
        else:
            x = x + self.attn_scale * self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
        return x


class ParticleSelfAttention(nn.Module):
    """
    A particle-based multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embed, n_head, block_size, attn_pdrop=0.1, resid_pdrop=0.1,
                 positional_bias=False, max_particles=None, linear_bias=False, torch_attn=False):
        super().__init__()
        assert n_embed % n_head == 0
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.torch_attn = torch_attn
        # if self.torch_attn:
        #     self.attn_net = nn.MultiheadAttention(embed_dim=n_embed, num_heads=n_head, dropout=attn_pdrop,
        #                                           bias=linear_bias, batch_first=True)
        #     # key, query, value projections for all heads
        #     self.key = None
        #     self.query = None
        #     self.value = None
        #     # regularization
        #     self.attn_drop = None
        #     # output projection
        #     self.proj = nn.Identity()  # already part of attn_net
        # else:
        # self.attn_net = None
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embed, n_embed, bias=linear_bias)
        self.query = nn.Linear(n_embed, n_embed, bias=linear_bias)
        self.value = nn.Linear(n_embed, n_embed, bias=linear_bias)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop) if not self.torch_attn else nn.Identity()
        # output projection
        self.proj = nn.Linear(n_embed, n_embed, bias=linear_bias)

        self.resid_drop = nn.Dropout(resid_pdrop)
        self.n_head = n_head
        self.positional_bias = positional_bias
        self.max_particles = max_particles
        if self.positional_bias:
            self.rel_pos_bias = SimpleRelativePositionalBias(block_size, n_head, max_particles=max_particles)
        else:
            self.rel_pos_bias = nn.Identity()

    def forward(self, x):
        B, N, T, C = x.size()  # batch size, n_particles, sequence length, embedding dimensionality (n_embd)
        # if self.torch_attn:
        #     if self.positional_bias:
        #         raise NotImplementedError(f'torch_attn: {self.torch_attn}, can not use positional bias')
        #         # if self.max_particles is not None:
        #         #     bias_t, bias_p = self.rel_pos_bias(T, num_particles=N)
        #         #     bias_t = bias_t.view(1, bias_t.shape[1], 1, T, 1, T)
        #         #     bias_p = bias_p.view(1, bias_p.shape[1], N, 1, N, 1)
        #         #     mask = bias_t + bias_p
        #         # else:
        #         #     bias_t, _ = self.rel_pos_bias(T)
        #         #     bias_t = bias_t.view(1, bias_t.shape[1], 1, T, 1, T)
        #         #     mask = bias_t
        #     x = x.reshape(B, N * T, C)
        #     y, _ = self.attn_net(query=x, key=x, value=x, need_weights=False)
        # else:
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, N * T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, N * T, hs)
        q = self.query(x).view(B, N * T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, N * T, hs)
        v = self.value(x).view(B, N * T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, N * T, hs)

        if self.torch_attn:
            y = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=False,
                                               dropout_p=self.attn_pdrop if self.training else 0.0)

        else:
            # causal self-attention; Self-attend: (B, nh, N * T, hs) x (B, nh, hs, N  *T) -> (B, nh, N * T, N *T )
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, N * T, N * T)
            if self.positional_bias:
                att = att.view(B, -1, N, T, N, T)  # (B, nh, N, T, N, T)
                if self.max_particles is not None:
                    bias_t, bias_p = self.rel_pos_bias(T, num_particles=N)
                    bias_t = bias_t.view(1, bias_t.shape[1], 1, T, 1, T)
                    bias_p = bias_p.view(1, bias_p.shape[1], N, 1, N, 1)
                    att = att + bias_t + bias_p
                else:
                    bias_t, _ = self.rel_pos_bias(T)
                    bias_t = bias_t.view(1, bias_t.shape[1], 1, T, 1, T)
                    att = att + bias_t
                att = att.view(B, -1, N * T, N * T)  # (B, nh, N * T, N * T)
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v  # (B, nh, N*T, N*T) x (B, nh, N*T, hs) -> (B, nh, N*T, hs)

        y = y.transpose(1, 2).contiguous().view(B, N * T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        y = y.view(B, N, T, -1)
        return y


class SimpleRelativePositionalBias(nn.Module):
    # adapted from https://github.com/facebookresearch/mega
    def __init__(self, max_positions, num_heads=1, max_particles=None, layer_norm=False):
        super().__init__()
        self.max_positions = max_positions
        self.num_heads = num_heads
        self.max_particles = max_particles
        self.rel_pos_bias = nn.Parameter(torch.Tensor(2 * max_positions - 1, self.num_heads))
        self.ln_t = nn.LayerNorm([2 * max_positions - 1, self.num_heads]) if layer_norm else nn.Identity()

        if self.max_particles is not None:
            self.particle_rel_pos_bias = nn.Parameter(torch.Tensor(2 * max_particles - 1, self.num_heads))
            self.ln_p = nn.LayerNorm([2 * max_particles - 1, self.num_heads]) if layer_norm else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.rel_pos_bias, mean=0.0, std=std)
        if self.max_particles is not None:
            nn.init.normal_(self.particle_rel_pos_bias, mean=0.0, std=std)

    def get_particle_rel_position(self, num_particles):
        if self.max_particles is None:
            return 0.0
        if num_particles > self.max_particles:
            raise ValueError('Num particles {} going beyond max particles {}'.format(num_particles, self.max_particles))

        # seq_len * 2 -1
        in_ln = self.ln_p(self.particle_rel_pos_bias)
        b = in_ln[(self.max_particles - num_particles):(self.max_particles + num_particles - 1)]
        # seq_len * 3 - 1
        t = F.pad(b, (0, 0, 0, num_particles))
        # (seq_len * 3 - 1) * seq_len
        t = torch.tile(t, (num_particles, 1))
        t = t[:-num_particles]
        # seq_len x (3 * seq_len - 2)
        t = t.view(num_particles, 3 * num_particles - 2, b.shape[-1])
        r = (2 * num_particles - 1) // 2
        start = r
        end = t.size(1) - r
        t = t[:, start:end]  # [seq_len, seq_len, n_heads]
        t = t.permute(2, 0, 1).unsqueeze(0)  # [1, n_heads, seq_len, seq_len]
        return t

    def forward(self, seq_len, num_particles=None):
        if seq_len > self.max_positions:
            raise ValueError('Sequence length {} going beyond max length {}'.format(seq_len, self.max_positions))

        # seq_len * 2 -1
        in_ln = self.ln_t(self.rel_pos_bias)
        b = in_ln[(self.max_positions - seq_len):(self.max_positions + seq_len - 1)]
        # seq_len * 3 - 1
        t = F.pad(b, (0, 0, 0, seq_len))
        # (seq_len * 3 - 1) * seq_len
        t = torch.tile(t, (seq_len, 1))
        t = t[:-seq_len]
        # seq_len x (3 * seq_len - 2)
        t = t.view(seq_len, 3 * seq_len - 2, b.shape[-1])
        r = (2 * seq_len - 1) // 2
        start = r
        end = t.size(1) - r
        t = t[:, start:end]  # [seq_len, seq_len, n_heads]
        t = t.permute(2, 0, 1).unsqueeze(0)  # [1, n_heads, seq_len, seq_len]
        p = None
        if num_particles is not None and self.max_particles is not None:
            p = self.get_particle_rel_position(num_particles)  # [1, n_heads, n_part, n_part]
            t = t[:, :, None, :, None, :]
            p = p[:, :, :, None, :, None]
        return t, p

class MLP(nn.Module):
    def __init__(self, n_embed, resid_pdrop=0.1, hidden_dim_multiplier=4, activation='gelu'):
        super().__init__()
        self.fc_1 = nn.Linear(n_embed, hidden_dim_multiplier * n_embed)
        if activation == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU(True)
        self.proj = nn.Linear(hidden_dim_multiplier * n_embed, n_embed)
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, x):
        x = self.dropout(self.proj(self.act(self.fc_1(x))))
        return x