import math

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, reduce

def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d


# ------------------------------------------------------------------------------
# Sinusoidal Positional Embedding Modules
# ------------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


# ------------------------------------------------------------------------------
# Relative Positional Bias
# ------------------------------------------------------------------------------

class SimpleRelativePositionalBias(nn.Module):
    """
    Simple relative positional bias module for transformer attention.
    Adapted from https://github.com/facebookresearch/mega


    This module generates learnable biases that depend on the relative positions
    between tokens (or particles) in the sequence. It supports both temporal biases and,
    if specified, an additional bias along a particle dimension.

    Args:
        max_positions (int): Maximum sequence length (number of positions).
        num_heads (int): Number of attention heads.
        max_particles (int, optional): Maximum number of particles to consider for a separate
                                       bias term. If None, only temporal bias is used.
        layer_norm (bool): If True, apply layer normalization to the bias parameters.
    
    Example:
        >>> bias_module = SimpleRelativePositionalBias(max_positions=128, num_heads=8, max_particles=32)
        >>> t_bias, p_bias = bias_module(seq_len=64, num_particles=16)
    """
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

        in_ln = self.ln_p(self.particle_rel_pos_bias)
        b = in_ln[(self.max_particles - num_particles):(self.max_particles + num_particles - 1)]
        t = F.pad(b, (0, 0, 0, num_particles))
        t = torch.tile(t, (num_particles, 1))
        t = t[:-num_particles]
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

        in_ln = self.ln_t(self.rel_pos_bias)
        b = in_ln[(self.max_positions - seq_len):(self.max_positions + seq_len - 1)]
        t = F.pad(b, (0, 0, 0, seq_len))
        t = torch.tile(t, (seq_len, 1))
        t = t[:-seq_len]
        t = t.view(seq_len, 3 * seq_len - 2, b.shape[-1])
        r = (2 * seq_len - 1) // 2
        start = r
        end = t.size(1) - r
        t = t[:, start:end]  # [seq_len, seq_len, n_heads]
        t = t.permute(2, 0, 1).unsqueeze(0)  # [1, n_heads, seq_len, seq_len]
        p = None
        if num_particles is not None and self.max_particles is not None:
            p = self.get_particle_rel_position(num_particles)  # [1, n_heads, n_particles, n_particles]
            t = t[:, :, None, :, None, :]
            p = p[:, :, :, None, :, None]
        return t, p

    def extra_repr(self) -> str:
        return 'max positions={}'.format(self.max_positions)


# ------------------------------------------------------------------------------
# Particle-Based Attention Modules
# ------------------------------------------------------------------------------

class CausalParticleAttention(nn.Module):
    """
    Particle-based multi-head masked self-attention layer with relative positional bias.

    This module implements multi-head attention over particle representations.
    It supports three types of attention:
      - 'self': standard self-attention.
      - 'cross': attention over a conditioning input `c`.
      - 'hybrid': concatenates self and conditioning inputs.
    
    Optionally, relative positional biases can be added to the attention scores, and a causal
    mask can be applied to restrict attention to previous time steps.

    Args:
        n_embed (int): Dimensionality of the input embeddings.
        n_head (int): Number of attention heads.
        block_size (int): Maximum sequence length (used for constructing the causal mask).
        attn_pdrop (float): Dropout probability on attention weights.
        resid_pdrop (float): Dropout probability on the output projection.
        positional_bias (bool): If True, add relative positional biases.
        max_particles (int, optional): Maximum number of particles for particle bias. Defaults to None.
        linear_bias (bool): If True, include bias terms in the linear projections.
        att_type (str): Type of attention: 'self', 'cross', or 'hybrid'.
        causal_att (bool): If True, apply a causal (triangular) mask to attention scores.

    Example:
        >>> attn = CausalParticleAttention(n_embed=256, n_head=8, block_size=64,
        ...      positional_bias=True, max_particles=32, att_type='self', causal_att=True)
        >>> x = torch.randn(2, 16, 64, 256)  # (batch, particles, time, embedding)
        >>> output = attn(x)
    """
    def __init__(self, n_embed, n_head, block_size, attn_pdrop=0.1, resid_pdrop=0.1,
                 positional_bias=True, max_particles=None, linear_bias=False, att_type='self', causal_att=False):
        super().__init__()
        assert n_embed % n_head == 0
        assert att_type in ['hybrid', 'cross', 'self']
        self.att_type = att_type
        # Key, query, value projections for all heads
        self.key = nn.Linear(n_embed, n_embed, bias=linear_bias)
        self.query = nn.Linear(n_embed, n_embed, bias=linear_bias)
        self.value = nn.Linear(n_embed, n_embed, bias=linear_bias)
        # Dropout layers for regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # Output projection
        self.proj = nn.Linear(n_embed, n_embed, bias=linear_bias)
        # Register a causal mask (lower triangular) for self-attention
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, 1, block_size, 1, block_size))
        self.n_head = n_head
        self.positional_bias = positional_bias
        self.max_particles = max_particles
        self.causal_att = causal_att
        if self.positional_bias:
            self.rel_pos_bias = SimpleRelativePositionalBias(block_size, n_head, max_particles=max_particles)
        else:
            self.rel_pos_bias = nn.Identity()

    def forward(self, x, c=None, return_attention=False, attn_mask=None):
        B, N, T, C = x.size()  # batch size, n_particles, sequence length, embedding dim
        # Determine inputs for query and key/value projections based on attention type.
        query_input = x
        if self.att_type == 'hybrid':
            key_value_input = torch.cat([x, c], dim=1)
            key_value_N = key_value_input.shape[1]
            key_value_T = key_value_input.shape[2]
        elif self.att_type == 'cross':
            key_value_input = c
            key_value_N = key_value_input.shape[1]
            key_value_T = key_value_input.shape[2]
        else:   # self-attention
            key_value_input = x
            key_value_N = N
            key_value_T = T

        # Compute projections and reshape for multi-head attention.
        k = self.key(key_value_input).view(B, key_value_N * key_value_T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(query_input).view(B, N * T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(key_value_input).view(B, key_value_N * key_value_T, self.n_head, C // self.n_head).transpose(1, 2)

        # Scaled dot-product attention.
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if self.positional_bias:
            assert self.att_type == 'self'
            att = att.view(B, -1, N, T, N, T)  # reshape to separate particles and time
            if self.max_particles is not None:
                bias_t, bias_p = self.rel_pos_bias(T, num_particles=N)
                bias_t = bias_t.view(1, bias_t.shape[1], 1, T, 1, T)
                bias_p = bias_p.view(1, bias_p.shape[1], N, 1, N, 1)
                att = att + bias_t + bias_p
            else:
                bias_t, _ = self.rel_pos_bias(T)
                bias_t = bias_t.view(1, bias_t.shape[1], 1, T, 1, T)
                att = att + bias_t
            att = att.view(B, -1, N * T, N * T)
        # Apply causal masking if enabled.
        if self.causal_att:
            assert self.att_type == 'self'
            att = att.view(B, -1, N, T, N, T)
            att = att.masked_fill(self.mask[:, :, :, :T, :, :T] == 0, float('-inf'))
            att = att.view(B, -1, N * T, N * T)

        # Padding mask over key positions: attn_mask is (B, N) token-level validity,
        # broadcast across key-time so softmax ignores invalid tokens entirely.
        if attn_mask is not None and self.att_type == 'self':
            key_mask = attn_mask.unsqueeze(-1).expand(B, N, T).reshape(B, N * T)
            att = att.masked_fill(key_mask.view(B, 1, 1, N * T) == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        if return_attention:
            attention_matrix = att
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, N*T, key_value_N*key_value_T) x (B, nh, key_value_N*key_value_T, hs) -> (B, nh, N*T, hs)
        y = y.transpose(1, 2).contiguous().view(B, N * T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        y = y.view(B, N, T, -1)
        if return_attention:
            return y, attention_matrix
        else:
            return y


# ------------------------------------------------------------------------------
# Feed-Forward Network (MLP)
# ------------------------------------------------------------------------------

class MLP(nn.Module):
    """
    Feed-forward multi-layer perceptron (MLP) used within transformer blocks.

    This module is a two-layer fully-connected network with an activation function (GELU or ReLU)
    between the layers and dropout applied after the activation.

    Args:
        n_embed (int): Input and output embedding dimensionality.
        resid_pdrop (float): Dropout probability applied after activation.
        hidden_dim_multiplier (int): Factor to multiply `n_embed` for the hidden layer size.
        activation (str): Activation function to use ('gelu' or 'relu').
    
    Example:
        >>> mlp = MLP(n_embed=256, hidden_dim_multiplier=4)
        >>> x = torch.randn(32, 10, 256)
        >>> output = mlp(x)  # shape: (32, 10, 256)
    """
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


# ------------------------------------------------------------------------------
# Final Adaptive Modulation Layers
# ------------------------------------------------------------------------------

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class FinalLayer(nn.Module):
    """
    Final adaptive modulation layer.
    """
    def __init__(self, dim):
        super().__init__()
        self.ln_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.gamma = nn.Linear(dim, dim)
        self.beta = nn.Linear(dim, dim)
        # Zero-out modulation layers to start as identity
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias) 

    def forward(self, x, c):
        scale = self.gamma(c)
        shift = self.beta(c)
        x = modulate(self.ln_final(x), shift, scale)
        return x


# ------------------------------------------------------------------------------
# Adaptive Layer-Normalized Particle Interaction Transformer Block
# ------------------------------------------------------------------------------

class AdaLNPINTBlock(nn.Module):
    """
    Adaptive Layer Normalized Particle Interaction Transformer (AdaLN-PINT) block.

    This block extends a standard transformer block by incorporating adaptive layer
    normalization. The normalization parameters are conditioned on an external input `c`
    and are used to modulate both the attention and MLP sub-layers.

    Args:
        n_embed (int): Embedding dimensionality.
        n_head (int): Number of attention heads.
        block_size (int): Maximum sequence length.
        attn_pdrop (float): Dropout probability for attention weights.
        resid_pdrop (float): Dropout probability for residual connections.
        hidden_dim_multiplier (int): Multiplier for the hidden dimension in the MLP.
        positional_bias (bool): If True, add relative positional biases.
        activation (str): Activation function for the MLP ('gelu' or 'relu').
        max_particles (int, optional): Maximum number of particles for biasing. Defaults to None.
        att_type (str): Type of attention ('self', 'cross', or 'hybrid').
        causal_att (bool): If True, apply causal masking in attention.
    
    Example:
        >>> adaln_block = AdaLNPINTBlock(n_embed=256, n_head=8, block_size=64, positional_bias=True)
        >>> x = torch.randn(2, 16, 64, 256)
        >>> c = torch.randn(2, 256)
        >>> output = adaln_block(x, c)
    """
    def __init__(self, n_embed, n_head, block_size, attn_pdrop=0.1, resid_pdrop=0.1, hidden_dim_multiplier=4,
                 positional_bias=False, activation='gelu', max_particles=None, att_type='self', causal_att=False):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embed, elementwise_affine=False, eps=1e-6)
        self.attn = CausalParticleAttention(n_embed, n_head, block_size, attn_pdrop, resid_pdrop,
                                             positional_bias=positional_bias,
                                             max_particles=max_particles, att_type=att_type, causal_att=causal_att)
        self.ln_2 = nn.LayerNorm(n_embed, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(n_embed, resid_pdrop, hidden_dim_multiplier, activation=activation)
        self.gamma_1 = nn.Linear(n_embed, n_embed)
        self.beta_1 = nn.Linear(n_embed, n_embed)
        self.gamma_2 = nn.Linear(n_embed, n_embed)
        self.beta_2 = nn.Linear(n_embed, n_embed)
        self.scale_1 = nn.Linear(n_embed, n_embed)
        self.scale_2 = nn.Linear(n_embed, n_embed)

        nn.init.zeros_(self.gamma_1.weight)
        nn.init.zeros_(self.beta_1.weight)
        nn.init.zeros_(self.gamma_1.bias)
        nn.init.zeros_(self.beta_1.bias)  

        nn.init.zeros_(self.gamma_2.weight)
        nn.init.zeros_(self.beta_2.weight)
        nn.init.zeros_(self.gamma_2.bias)
        nn.init.zeros_(self.beta_2.bias)  

        nn.init.zeros_(self.scale_1.weight)
        nn.init.zeros_(self.scale_2.weight)
        nn.init.zeros_(self.scale_1.bias)
        nn.init.zeros_(self.scale_2.bias)  

    def forward(self, x, c, attn_mask=None):
        scale_msa = self.gamma_1(c)
        shift_msa = self.beta_1(c)
        scale_mlp = self.gamma_2(c)
        shift_mlp = self.beta_2(c)
        gate_msa = self.scale_1(c).unsqueeze(1)
        gate_mlp = self.scale_2(c).unsqueeze(1)
        x = self.attn(modulate(self.ln_1(x), shift_msa, scale_msa), attn_mask=attn_mask) * gate_msa + x
        return self.mlp(modulate(self.ln_2(x), shift_mlp, scale_mlp)) * gate_mlp + x
    

# ------------------------------------------------------------------------------
# Adaptive Layer Normalized Particle Transformer (Top-Level Model)
# ------------------------------------------------------------------------------

class AdaLNParticleTransformer(nn.Module):
    """
    Adaptive Layer Normalized Particle Transformer.

    This is a full transformer model designed for particle-based data. It stacks multiple
    AdaLN-PINT blocks and applies either shared or per-particle positional embeddings.
    The network supports conditioning via external embeddings (such as action and temporal cues)
    which modulate the normalization layers in each block.

    Args:
        n_embed (int): Embedding dimensionality for particle features.
        n_head (int): Number of attention heads.
        n_layer (int): Number of transformer blocks.
        block_size (int): Maximum sequence length (number of time steps).
        output_dim (int): Dimensionality of the final output.
        attn_pdrop (float): Dropout probability for attention weights.
        resid_pdrop (float): Dropout probability for residual connections.
        hidden_dim_multiplier (int): Multiplier for the hidden dimension in MLPs.
        positional_bias (bool): If True, use per-particle positional bias; otherwise, use shared positional embeddings.
        activation (str): Activation function for MLPs ('gelu' or 'relu').
        max_particles (int, optional): Maximum number of particles. Defaults to None.
        causal_att (bool): If True, apply causal masking in attention.
    
    Example:
        >>> transformer = AdaLNParticleTransformer(
        ...     n_embed=256, n_head=8, n_layer=6, block_size=64, output_dim=10,
        ...     positional_bias=False, max_particles=16, causal_att=True)
        >>> x = torch.randn(2, 16, 64, 256)  # (batch, particles, time, features)
        >>> action_embed = torch.randn(2, 64, 256)
        >>> t_embed = torch.randn(2, 256)
        >>> logits = transformer(x, action_embed, t_embed)
    """
    def __init__(self, n_embed, n_head, n_layer, block_size, output_dim, attn_pdrop=0.1, resid_pdrop=0.1,
                 hidden_dim_multiplier=4, positional_bias=False,
                 activation='gelu', max_particles=None, causal_att=False):
        super().__init__()
        self.positional_bias = positional_bias
        self.max_particles = max_particles
        if self.positional_bias: 
            # When positional_bias is True, a per-particle bias is used, so no extra embedding is added.
            self.pos_emb = nn.Identity()
        else:
            # Otherwise, use a shared learned positional embedding for all particles at each timestep.
            self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embed))
        # Build the stacked transformer blocks.
        self.blocks = nn.Sequential(*[AdaLNPINTBlock(n_embed, n_head, block_size, attn_pdrop,
                                                       resid_pdrop, hidden_dim_multiplier,
                                                       positional_bias, activation=activation,
                                                       max_particles=max_particles, causal_att=causal_att)
                                      for _ in range(n_layer)])
        # Final modulation and output projection.
        self.ln_f = FinalLayer(n_embed)
        self.head = nn.Linear(n_embed, output_dim, bias=False)

        self.block_size = block_size
        self.n_embed = n_embed
        self.n_layer = n_layer

        # Initialize weights.
        self.apply(self._init_weights)
        if self.positional_bias:
            for m in self.blocks:
                m.attn.rel_pos_bias.reset_parameters()
        print(f"particle transformer # parameters: {sum(p.numel() for p in self.parameters())}")

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, AdaLNParticleTransformer):
            if not self.positional_bias:
                torch.nn.init.normal_(module.pos_emb, mean=0.0, std=std)

    def forward(self, x, action_embed, t_embed, return_attention=False, attn_mask=None):
        b, n, t, f = x.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        assert f == self.n_embed, "invalid particle feature dim"
        if t_embed is not None:
            c = action_embed + t_embed.unsqueeze(1)  # combine conditioning signals
        else:
            c = action_embed
        if return_attention:
            attention_dict = {}
        if not self.positional_bias:
            # If not using per-particle positional bias, add shared positional embeddings.
            position_embeddings = self.pos_emb[:, None, :t, :]  # (1, 1, t, n_embed)
            x = x + position_embeddings
        for i, block in enumerate(self.blocks):
            if return_attention:
                x, attention_matrix = block(x, return_attention=True)
                attention_dict[f'layer_{i}'] = attention_matrix
            else:
                x = block(x, c, attn_mask=attn_mask)
        x = self.ln_f(x, c)
        logits = self.head(x)

        if return_attention:
            return logits, attention_dict
        else:
            return logits