from collections import namedtuple
import numpy as np
import torch
from torch import nn
import pdb

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)


Sample = namedtuple('Sample', 'trajectories values chains')


@torch.no_grad()
def default_sample_fn(model, x, cond, t, lang=None, lang_mask=None, action_cond=None):
    model_mean, _, model_log_variance = model.p_mean_variance(
        x=x, cond=cond, t=t, lang=lang, lang_mask=lang_mask, action_cond=action_cond)
    model_std = torch.exp(0.5 * model_log_variance)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    values = torch.zeros(len(x), device=x.device)
    return model_mean + model_std * noise, values

@torch.no_grad()
def sample_fn_return_attn(model, x, cond, t, lang=None, lang_mask=None, action_cond=None):
    model_mean, _, model_log_variance, att_dict = model.p_mean_variance_return_attn(
        x=x, cond=cond, t=t, lang=lang, lang_mask=lang_mask, action_cond=action_cond)
    model_std = torch.exp(0.5 * model_log_variance)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    values = torch.zeros(len(x), device=x.device)
    return model_mean + model_std * noise, values, att_dict


def sort_by_values(x, values):
    inds = torch.argsort(values, descending=True)
    x = x[inds]
    values = values[inds]
    return x, values


def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None, obs_only=False, action_only=False,
        gripper_dim=0, bg_dim=0, keypose_mode=False,
    ):
        super().__init__()
        self.horizon = horizon
        self.obs_only = obs_only
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.gripper_dim = gripper_dim
        self.bg_dim = bg_dim
        # When True, conditions[0] is consumed by the model as separate context
        # tokens (use_cond_tokens path) and apply_conditioning skips slot-0 obs
        # pinning. Slot 0 of the trajectory is the next-keypose prediction.
        self.keypose_mode = keypose_mode
        # transition_dim includes: actions + gripper_state (optional) + bg_features (optional) + observations
        self.transition_dim = observation_dim + action_dim + gripper_dim + bg_dim
        self.model = model

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        if action_only:
            loss_weights = {i: 0.0 for i in range(self.transition_dim - self.action_dim)}

        if self.obs_only:
            self.loss_fn = nn.L1Loss()

        elif loss_type in ['chamfer', 'chamferv2']:
            self.loss_fn = Losses[loss_type](action_weight, self.action_dim, model.features_dim, model.multiview)
        else:
                    # get loss coefficients and initialize objective
            loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
            self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## Action loss weighting:
        ##  - keypose_mode: every slot is a real prediction (no pinned slot 0),
        ##    so action_weight applies uniformly across all slots.
        ##  - legacy: slot 0 is pinned by apply_conditioning, so weight only slot 1+.
        if getattr(self, 'keypose_mode', False):
            loss_weights[:, :self.action_dim] = action_weight
        else:
            loss_weights[1:, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, lang=None, lang_mask=None, action_cond=None):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t, lang=lang, lang_mask=lang_mask))
        x_recon = apply_conditioning(x_recon, cond, self.action_dim, action_conditions=action_cond, pin_obs=not self.keypose_mode)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_mean_variance_return_attn(self, x, cond, t, lang=None, lang_mask=None, action_cond=None):
        noise, att_dict = self.model(x, cond, t, return_attention=True, lang=lang, lang_mask=lang_mask)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim, action_conditions=action_cond, pin_obs=not self.keypose_mode)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, att_dict

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_chain=False, sample_fn=default_sample_fn, return_attention=False, sort_by_value=True, lang=None, lang_mask=None, action_cond=None, **sample_kwargs):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim, action_conditions=action_cond, pin_obs=not self.keypose_mode)

        chain = [x] if return_chain else None
        att_dict = None
        if return_attention:
            sample_fn = sample_fn_return_attn

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            if return_attention:
                x, values, att_dict = sample_fn(self, x, cond, t, lang=lang, lang_mask=lang_mask, action_cond=action_cond, **sample_kwargs)
            else:
                x, values = sample_fn(self, x, cond, t, lang=lang, lang_mask=lang_mask, action_cond=action_cond, **sample_kwargs)
            x = apply_conditioning(x, cond, self.action_dim, action_conditions=action_cond, pin_obs=not self.keypose_mode)

            progress.update({'t': i, 'vmin': values.min().item(), 'vmax': values.max().item()})
            if return_chain: chain.append(x)

        progress.stamp()
        if sort_by_value:
            x, values = sort_by_values(x, values)
        if return_chain: chain = torch.stack(chain, dim=1)
        if return_attention: return Sample(x, values, chain), att_dict
        return Sample(x, values, chain)

    @torch.no_grad()
    def conditional_sample(self, cond, horizon=None, sort_by_value=True, return_attention=False, lang=None, lang_mask=None, action_cond=None, **sample_kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)
        return self.p_sample_loop(shape, cond, sort_by_value=sort_by_value, return_attention=return_attention, lang=lang, lang_mask=lang_mask, action_cond=action_cond, **sample_kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t, action_cond=None, lang=None, lang_mask=None):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim, action_conditions=action_cond, pin_obs=not self.keypose_mode) # a, 0, 1

        x_recon = self.model(x_noisy, cond, t, lang=lang, lang_mask=lang_mask)  # a' 0' 1'

        x_recon = apply_conditioning(x_recon, cond, self.action_dim, action_conditions=action_cond, pin_obs=not self.keypose_mode)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            if self.obs_only:
                loss = self.loss_fn(x_recon, noise)
                info = {}
            else:
                loss, info = self.loss_fn(x_recon, noise)
        else:
            if self.obs_only:
                loss = self.loss_fn(x_recon, x_start)
                info = {}
            else:
                loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, cond, action_cond=None, lang=None, lang_mask=None):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t, action_cond=action_cond, lang=lang, lang_mask=lang_mask)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond, *args, **kwargs)

class ValueDiffusion(GaussianDiffusion):

    def p_losses(self, x_start, cond, target, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        pred = self.model(x_noisy, cond, t)

        loss, info = self.loss_fn(pred, target)
        return loss, info

    def forward(self, x, cond, t):
        return self.model(x, cond, t)

