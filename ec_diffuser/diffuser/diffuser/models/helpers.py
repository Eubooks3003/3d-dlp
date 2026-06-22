import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
import pdb

import diffuser.utils as utils

#-----------------------------------------------------------------------------#
#---------------------------------- modules ----------------------------------#
#-----------------------------------------------------------------------------#

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

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)

#-----------------------------------------------------------------------------#
#--------------------------------- attention ---------------------------------#
#-----------------------------------------------------------------------------#

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b (h c) d -> b h c d', h=self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim = -1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = einops.rearrange(out, 'b h c d -> b (h c) d')
        return self.to_out(out)

#-----------------------------------------------------------------------------#
#---------------------------------- sampling ---------------------------------#
#-----------------------------------------------------------------------------#

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)

def apply_conditioning(x, conditions, action_dim, action_conditions=None, pin_obs=True):
    """Pin parts of x at specified timesteps.

    - `conditions[t] = val`  -> pins x[:, t, action_dim:] to val (obs/gripper/bg).
    - `action_conditions[t] = val` -> pins x[:, t, :action_dim] to val (action).

    With `action_conditions = {0: a0_value}` the model never has to learn a0
    (it's overwritten at every denoising step), so the loss-weight schedule
    that skips a0 (`weights[1:, :action_dim] = action_weight`) becomes correct.

    `pin_obs=False` skips the obs/gripper/bg pin entirely. Used by keypose-v2
    mode where `conditions[0]` is the *current* keypose's modalities passed to
    the model as separate context tokens, NOT inpainted into trajectory slot 0
    (slot 0 is the next-keypose prediction in that framing).
    """
    if pin_obs:
        for t, val in conditions.items():
            x[:, t, action_dim:] = val.clone()
    if action_conditions is not None:
        for t, val in action_conditions.items():
            x[:, t, :action_dim] = val.clone()
    return x


#-----------------------------------------------------------------------------#
#---------------------------------- losses -----------------------------------#
#-----------------------------------------------------------------------------#

class WeightedLoss(nn.Module):

    def __init__(self, weights, action_dim):
        super().__init__()
        self.register_buffer('weights', weights)
        self.action_dim = action_dim

    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        a0_loss = (loss[:, 0, :self.action_dim] / self.weights[0, :self.action_dim]).mean()
        H = loss.shape[1]
        # Last-timestep action loss (hardest step to predict).
        aH_loss = (loss[:, H - 1, :self.action_dim] / self.weights[0, :self.action_dim]).mean()
        # Mean action loss across all executable timesteps (t=1..H-1).
        a_exec_loss = loss[:, 1:, :self.action_dim].mean()
        # Observation-side loss (everything past action dims = gripper + bg + particles).
        # Unweighted so the number is comparable across configs.
        obs_loss = loss[:, :, self.action_dim:].mean()
        return weighted_loss, {
            'a0_loss': a0_loss,
            'aH_loss': aH_loss,
            'a_exec_loss': a_exec_loss,
            'obs_loss': obs_loss,
        }

class ValueLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, pred, targ):
        loss = self._loss(pred, targ).mean()

        if len(pred) > 1:
            corr = np.corrcoef(
                utils.to_np(pred).squeeze(),
                utils.to_np(targ).squeeze()
            )[0,1]
        else:
            corr = np.NaN

        info = {
            'mean_pred': pred.mean(), 'mean_targ': targ.mean(),
            'min_pred': pred.min(), 'min_targ': targ.min(),
            'max_pred': pred.max(), 'max_targ': targ.max(),
            'corr': corr,
        }

        return loss, info

class WeightedL1(WeightedLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class WeightedL2(WeightedLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

class ValueL1(ValueLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class ValueL2(ValueLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

class ChamferLoss(nn.Module):

    def __init__(self, action_weight, action_dim, obs_dim, multiview=False):
        super().__init__()
        self.a0_weight = action_weight
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.multiview = multiview

    def forward(self, preds, targ):
        '''
            preds, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        actions = preds[:, :, :self.action_dim]
        action_targs = targ[:, :, :self.action_dim]
        action_loss = F.mse_loss(actions, action_targs, reduction='none')
        action_loss = action_loss.mean(dim=-1)
        a0_loss = (action_loss[:, 0]).mean()
        action_loss[:, 0] *= self.a0_weight
        action_loss = action_loss.mean()
        
        obs = preds[:, :, self.action_dim:].view(preds.shape[0], preds.shape[1], -1, self.obs_dim)
        obs_targ = targ[:, :, self.action_dim:].view(targ.shape[0], targ.shape[1], -1, self.obs_dim)
        bs, horizon, n_entities, _ = obs.shape
        tot_chamfer = 0
        for i in range(horizon):
            if self.multiview:
                view_1_chamfer = self.get_chamfer_loss(obs_targ[:, i, :n_entities // 2], obs[:, i, :n_entities // 2])
                view_2_chamfer = self.get_chamfer_loss(obs_targ[:, i, n_entities // 2:], obs[:, i, n_entities // 2:])
                tot_chamfer += (view_1_chamfer + view_2_chamfer).mean()
            else:
                tot_chamfer += self.get_chamfer_loss(obs_targ[:, i], obs[:, i]).mean()
        
        chamfer_loss = tot_chamfer / horizon

        return action_loss + chamfer_loss, {'a0_loss': a0_loss}

    def get_chamfer_loss(self, gts, preds):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins, 1)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins, 1)
        return loss_1 + loss_2

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind_x = torch.arange(0, num_points_x, device=x.device, dtype=torch.long)
        diag_ind_y = torch.arange(0, num_points_y, device=y.device, dtype=torch.long)
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(
            zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = rx.transpose(2, 1) + ry - 2 * zz
        return P


class ChamferLossV2(nn.Module):

    def __init__(self, action_weight, action_dim, obs_dim, multiview=False,
                 chamfer_metric='l2_simple', target_weight=3, mask=False):
        super().__init__()
        self.a0_weight = action_weight
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.multiview = multiview
        self.chamfer_metric = chamfer_metric
        self.target_weight = target_weight
        self.mask = mask

    def forward(self, preds, targ):
        """
            preds, targ : tensor
                [ batch_size x horizon x transition_dim ]
        """
        bs, horizon, transition_dim = preds.shape
        n_views = 2 if self.multiview else 1
        is_state = not (self.obs_dim == 10)

        # TODO: maybe disregard initial state and final goal frame in loss computation

        # compute action loss
        actions = preds[:, :, :self.action_dim]
        action_targs = targ[:, :, :self.action_dim]
        # action_loss = F.mse_loss(actions, action_targs, reduction='none')
        action_loss = F.l1_loss(actions, action_targs, reduction='none')
        action_loss = action_loss.mean(dim=-1)
        a0_loss = (action_loss[:, 1]).mean()
        action_loss[:, 1] *= self.a0_weight
        action_loss = action_loss.mean()

        # compute particle Chamfer loss
        obs = preds[:, :, self.action_dim:].reshape(bs * horizon * n_views, -1, self.obs_dim)
        obs_targ = targ[:, :, self.action_dim:].reshape(bs * horizon * n_views, -1, self.obs_dim)
        n_entities = obs.shape[1]

        if is_state:
            obs_identity_features = obs[..., -n_entities:].clone()
            obs_targ_identity_features = obs_targ[..., -n_entities:].clone()
        else:
            obs_identity_features = obs[..., 5:9]
            obs_targ_identity_features = obs_targ[..., 5:9]
            if self.mask:
                # calculate mask based on transparency  # TODO: debug masking mechanism
                obs_transparency = obs[..., -1]
                obs_mask = torch.where(obs_transparency < 0, True, False)
                P_obs_mask = obs_mask.unsqueeze(-1).expand(-1, -1, n_entities)
                obs_targ_transparency = obs_targ[..., -1]
                obs_targ_mask = torch.where(obs_targ_transparency < 0, True, False)
                P_obs_targ_mask = obs_targ_mask.unsqueeze(-1).expand(-1, -1, n_entities).transpose(-1, -2)
                P_mask = torch.logical_or(P_obs_mask, P_obs_targ_mask)

        P = batch_pairwise_dist(obs_identity_features, obs_targ_identity_features, self.chamfer_metric)
        if not is_state and self.mask:
            P.masked_fill_(P_mask, float('inf'))  # disregard particles that don't represent objects in min operation

        # compute dist from target to generated obs
        min_identity_dists, min_indices = torch.min(P, 1)
        aligned_obs = torch.gather(obs, 1, min_indices.unsqueeze(-1).expand(-1, -1, self.obs_dim))
        if is_state or not self.mask:
            # particle_dist1 = F.mse_loss(aligned_obs, obs_targ, reduction='none')
            particle_dist1 = F.l1_loss(aligned_obs, obs_targ, reduction='none')
        else:
            # particle_dist1 = torch.square(aligned_obs - obs_targ)
            particle_dist1 = torch.abs(aligned_obs - obs_targ)
            particle_dist1[obs_targ_mask] = 0  # filter particles' contribution to reward based on target obs mask
            n_object_particles = (torch.sum(~obs_targ_mask, 1))
            n_object_particles = torch.maximum(n_object_particles, torch.ones_like(n_object_particles))  # make sure we don't divide by zero
            particle_dist1 = torch.sum(particle_dist1, 1) / n_object_particles  # normalize based on number of unfiltered particles

        # compute dist from generated to target obs
        min_identity_dists, min_indices = torch.min(P, 2)
        aligned_obs_targ = torch.gather(obs_targ, 1, min_indices.unsqueeze(-1).expand(-1, -1, self.obs_dim))
        if is_state or not self.mask:
            # particle_dist2 = F.mse_loss(aligned_obs_targ, obs, reduction='none')
            particle_dist2 = F.l1_loss(aligned_obs_targ, obs, reduction='none')
        else:
            # particle_dist2 = torch.square(aligned_obs_targ - obs)
            particle_dist2 = torch.abs(aligned_obs_targ - obs)
            particle_dist2[obs_mask] = 0  # filter particles' contribution to reward based on obs mask
            n_object_particles = (torch.sum(~obs_mask, 1))
            n_object_particles = torch.maximum(n_object_particles, torch.ones_like(n_object_particles))  # make sure we don't divide by zero
            particle_dist2 = torch.sum(particle_dist2, 1) / n_object_particles  # normalize based on number of unfiltered particles

        # compute loss
        chamfer_dist = (self.target_weight * particle_dist1 + particle_dist2) / (self.target_weight + 1) # TODO: maybe give more weight to dist from target to generated (particle_dist1) since target is ground truth and should be an anchor
        chamfer_loss = chamfer_dist.mean()

        loss = action_loss + chamfer_loss

        return loss, {'a0_loss': a0_loss}


def batch_pairwise_dist(x, y, metric='l2_simple'):
    assert metric in ['l2', 'l2_simple', 'l1', 'cosine'], f'metric {metric} unrecognized'
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    if metric == 'cosine':
        dist_func = torch.nn.functional.cosine_similarity
        P = -dist_func(x.unsqueeze(2), y.unsqueeze(1), dim=-1, eps=1e-8)
    elif metric == 'l1':
        P = torch.abs(x.unsqueeze(2) - y.unsqueeze(1)).sum(-1)
    elif metric == 'l2_simple':
        P = ((x.unsqueeze(2) - y.unsqueeze(1)) ** 2).sum(-1)
    else:
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind_x = torch.arange(0, num_points_x, device=x.device)
        diag_ind_y = torch.arange(0, num_points_y, device=y.device)
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = rx.transpose(2, 1) + ry - 2 * zz
    return P


Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2,
    'chamfer': ChamferLoss,
    'chamferv2': ChamferLossV2,
    'value_l1': ValueL1,
    'value_l2': ValueL2,
}

def get_weighted_colors_dict(upweight=5):
    weight_vector = torch.ones(10)
    weight_vector[5:9] *= upweight
    weight_vector = torch.tile(weight_vector, (48, 1)).flatten()
    weight_dict = {}
    for i in range(weight_vector.shape[0]):
        weight_dict[i] = weight_vector[i].item()
    
    return weight_dict


if __name__ == '__main__':

    bs = 32
    n_points_x = 10
    n_points_y = 15
    loss_fn = ChamferLoss(10, 3)
    x = torch.randn(bs, 16, 483)
    y = torch.randn(bs, 16, 483)
    loss = loss_fn(x, y)
    print(loss)


    # x = torch.randn(bs, n_points_x, dim)
    # y = torch.randn(bs, n_points_y, dim)
    # P = loss_fn.batch_pairwise_dist(x, y)
    # chamfer_loss = loss_fn.get_chamfer_loss(x, y)
    # print(f'P: {P.shape}, max: {P.max()}, min: {P.min()}')
    # print(f'chamfer_loss: {chamfer_loss.shape}, max: {chamfer_loss.max()}, min: {chamfer_loss.min()}')