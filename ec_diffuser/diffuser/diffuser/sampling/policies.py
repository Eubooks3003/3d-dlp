from collections import namedtuple
import torch
import einops
import pdb

import diffuser.utils as utils
from diffuser.datasets.preprocessing import get_policy_preprocess_fn
import numpy as np
from diffuser.models import sample_fn_return_attn, default_sample_fn


Trajectories = namedtuple('Trajectories', 'actions observations values')


class LanguageConditionedPolicy:
    """
    Policy for language-conditioned diffusion. Owns a frozen CLIP text encoder
    (lives on CPU by default), encodes the instruction once at set_instruction()
    and stashes the token tensor. Every __call__ reuses that tensor on the
    diffusion device.
    """
    def __init__(self, diffusion_model, normalizer, preprocess_fns,
                 clip_model_name: str = "openai/clip-vit-base-patch32",
                 lang_pooled: bool = False,
                 max_lang_tokens: int = 32,
                 **sample_kwargs):
        from diffuser.models.lang import CLIPTextEncoder

        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        self.gripper_dim = getattr(diffusion_model.model, "gripper_dim", 0)
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.sample_kwargs = sample_kwargs

        self.lang_pooled = lang_pooled
        self.max_lang_tokens = int(max_lang_tokens)
        self._encoder = CLIPTextEncoder(
            model_name=clip_model_name, device="cpu", return_pooled=lang_pooled,
        )
        self._lang_tokens = None
        self._lang_mask = None

    @property
    def device(self):
        return next(self.diffusion_model.parameters()).device

    def set_instruction(self, text: str):
        """Encode once. Call at env reset; reuse for every denoising step after."""
        if self.lang_pooled:
            emb = self._encoder.encode([text])        # (1, clip_dim)
            self._lang_tokens = emb.unsqueeze(1).float()  # (1, 1, clip_dim)
            self._lang_mask = torch.ones((1, 1), dtype=torch.float32)
        else:
            tokens, mask = self._encoder.encode([text])   # (1, 77, D), (1, 77)
            valid = int(mask.sum().item())
            valid = max(1, min(valid, self.max_lang_tokens))
            out = torch.zeros((1, self.max_lang_tokens, tokens.shape[-1]), dtype=torch.float32)
            out[0, :valid] = tokens[0, :valid].float()
            self._lang_tokens = out                       # (1, max_lang_tokens, clip_dim)
            out_mask = torch.zeros((1, self.max_lang_tokens), dtype=torch.float32)
            out_mask[0, :valid] = 1.0
            self._lang_mask = out_mask                    # (1, max_lang_tokens)

    def __call__(self, conditions, batch_size=1, verbose=True, return_attention=False,
                 gripper_state=None):
        if self._lang_tokens is None:
            raise RuntimeError("LanguageConditionedPolicy: call set_instruction(str) before __call__")

        # Mirror GoalConditionedPolicy condition formatting (see below for the
        # duplicated logic -- kept inline to avoid deep refactor).
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        x0 = conditions[0]
        K = getattr(self.diffusion_model.model, "max_particles", None)
        D = getattr(self.diffusion_model.model, "features_dim", None)

        if x0.ndim == 3:
            multi_input = True
        elif x0.ndim == 2 and (K is not None) and (D is not None) and x0.shape == (K, D):
            multi_input = False
        elif x0.ndim == 2:
            multi_input = True
        else:
            multi_input = False

        # Reuse the existing formatter on a GoalConditionedPolicy stub.
        conditions = GoalConditionedPolicy._format_conditions(
            self, conditions, batch_size, multi_input=multi_input, gripper_state=gripper_state
        )

        # Broadcast lang to match batch, move to device.
        lang = self._lang_tokens.to(self.device)  # (1, L, D)
        lang_mask = self._lang_mask.to(self.device) if self._lang_mask is not None else None
        if lang.shape[0] != batch_size:
            lang = lang.expand(batch_size, -1, -1).contiguous()
            if lang_mask is not None:
                lang_mask = lang_mask.expand(batch_size, -1).contiguous()

        if return_attention:
            samples, att_dict = self.diffusion_model(
                conditions, verbose=verbose, sort_by_value=False,
                return_attention=return_attention, lang=lang, lang_mask=lang_mask,
                **self.sample_kwargs)
            att_dict = {k: utils.to_np(v) for k, v in att_dict.items()}
        else:
            samples = self.diffusion_model(
                conditions, verbose=verbose, sort_by_value=False,
                return_attention=return_attention, lang=lang, lang_mask=lang_mask,
                **self.sample_kwargs)
        trajectories = utils.to_np(samples.trajectories)

        obs_start_idx = self.action_dim + self.gripper_dim
        normed_observations = trajectories[:, :, obs_start_idx:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')

        actions = trajectories[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')

        if self.gripper_dim > 0:
            normed_gripper = trajectories[:, :, self.action_dim:self.action_dim + self.gripper_dim]
            gripper_out = self.normalizer.unnormalize(normed_gripper, 'gripper_state')
        else:
            gripper_out = None

        if not multi_input:
            action = actions[0, 0]
        else:
            action = actions[:, 0]

        trajectories = Trajectories(actions, observations, samples.values)
        if return_attention:
            return action, trajectories, att_dict
        return action, trajectories


class GoalConditionedPolicy:
    def __init__(self, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        # Get gripper_dim from the model if available
        self.gripper_dim = getattr(diffusion_model.model, "gripper_dim", 0)
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.sample_kwargs = sample_kwargs

    def __call__(self, conditions, batch_size=1, verbose=True, return_attention=False,
                 gripper_state=None):
        """
        Args:
            conditions: dict with observation conditions (e.g., {0: initial_obs, horizon-1: goal_obs})
            batch_size: number of samples to generate
            verbose: whether to print progress
            return_attention: whether to return attention weights
            gripper_state: optional dict with gripper state conditions matching conditions keys
                          e.g., {0: initial_gripper, horizon-1: goal_gripper}
                          Each value should be (gripper_dim,) or (B, gripper_dim)
        """
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        x0 = conditions[0]
        K = getattr(self.diffusion_model.model, "max_particles", None)
        D = getattr(self.diffusion_model.model, "features_dim", None)

        if x0.ndim == 3:
            # (B,K,D)
            multi_input = True
        elif x0.ndim == 2 and (K is not None) and (D is not None) and x0.shape == (K, D):
            # (K,D) single env particles
            multi_input = False
        elif x0.ndim == 2:
            # (B, obs_dim) flattened batch
            multi_input = True
        else:
            # (obs_dim,) single
            multi_input = False

        conditions = self._format_conditions(conditions, batch_size, multi_input=multi_input,
                                             gripper_state=gripper_state)

        if return_attention:
            samples, att_dict = self.diffusion_model(conditions, verbose=verbose, sort_by_value=False, return_attention=return_attention, **self.sample_kwargs)
            att_dict = {k: utils.to_np(v) for k, v in att_dict.items()}
        else:
            samples = self.diffusion_model(conditions, verbose=verbose, sort_by_value=False, return_attention=return_attention, **self.sample_kwargs)
        trajectories = utils.to_np(samples.trajectories)

        # Extract components from trajectory: [actions, gripper_state (optional), observations]
        obs_start_idx = self.action_dim + self.gripper_dim
        normed_observations = trajectories[:, :, obs_start_idx:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')

        ## extract action [ batch_size x horizon x transition_dim ]
        actions = trajectories[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')

        ## extract gripper state if present
        if self.gripper_dim > 0:
            normed_gripper = trajectories[:, :, self.action_dim:self.action_dim + self.gripper_dim]
            gripper_out = self.normalizer.unnormalize(normed_gripper, 'gripper_state')
        else:
            gripper_out = None

        ## extract first action per env if conditions contains multiple envs
        if not multi_input:
            action = actions[0, 0]
        else:
            action = actions[:, 0]

        trajectories = Trajectories(actions, observations, samples.values)
        if return_attention:
            return action, trajectories, att_dict
        return action, trajectories

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size, multi_input=False, gripper_state=None):
        """
        Format conditions for the diffusion model.

        Args:
            conditions: dict with observation conditions
            batch_size: number of samples
            multi_input: whether input is batched
            gripper_state: optional dict with gripper state for each condition key
        """
        K = getattr(self.diffusion_model.model, "max_particles", None)
        D = getattr(self.diffusion_model.model, "features_dim", None)

        def to_BKD(x):
            # returns either (B,K,D) if possible, else returns x unchanged
            if (K is None) or (D is None):
                return x

            if x.ndim == 1:
                # (K*D,) -> (1,K,D) if matches
                if x.shape[0] == K * D:
                    return x.reshape(1, K, D)
                return x

            if x.ndim == 2:
                # (K,D) single env particles
                if x.shape == (K, D):
                    return x.reshape(1, K, D)
                # (B, K*D) flattened batch
                if x.shape[1] == K * D:
                    return x.reshape(x.shape[0], K, D)
                return x

            if x.ndim == 3:
                # already (B,K,D)
                return x

            return x

        def to_flat(x):
            # (B,K,D) -> (B, K*D)
            if (K is None) or (D is None):
                return x
            if x.ndim == 3 and x.shape[-2] == K and x.shape[-1] == D:
                return x.reshape(x.shape[0], K * D)
            return x

        # 1) make sure observations/goals are (B,K,D) for the particle normalizer
        conditions = {k: to_BKD(v) for k, v in conditions.items()}

        # 2) normalize in particle space
        conditions = utils.apply_dict(self.normalizer.normalize, conditions, "observations")

        # 3) flatten back because diffusion model was trained on flattened obs
        conditions = {k: to_flat(v) for k, v in conditions.items()}

        # 4) Handle gripper state if present
        if gripper_state is not None and self.gripper_dim > 0:
            # Normalize gripper state
            for k, gs in gripper_state.items():
                if isinstance(gs, np.ndarray):
                    # Ensure batch dimension
                    if gs.ndim == 1:
                        gs = gs.reshape(1, -1)
                    # Normalize
                    gs_normed = self.normalizer.normalize(gs, 'gripper_state')
                    gripper_state[k] = gs_normed

            # Concatenate gripper state with observations in conditions
            # Format: [gripper_state, observations]
            for k in conditions.keys():
                if k in gripper_state:
                    gs = gripper_state[k]
                    if isinstance(gs, np.ndarray):
                        gs = torch.from_numpy(gs).float()
                    obs = conditions[k]
                    if isinstance(obs, np.ndarray):
                        obs = torch.from_numpy(obs).float()
                    conditions[k] = torch.cat([gs, obs], dim=-1)

        # 5) torch + repeat if single-env
        conditions = utils.to_torch(conditions, dtype=torch.float32)
        if not multi_input:
            conditions = utils.apply_dict(
                einops.repeat,
                conditions,
                "b d -> (repeat b) d", repeat=batch_size,
            )
        return conditions
