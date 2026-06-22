import os
import copy
import numpy as np
import torch
import einops
import pdb

from .arrays import batch_to_device, set_global_device, to_np, to_device, apply_dict
from .timer import Timer
from .cloud import sync_logs
import wandb
from tqdm import tqdm
from dlp_utils import log_rgb_voxels

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        bucket=None,
        accelerator=None,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        # EMA model lives on the local device only; keep it on rank-0's GPU
        # since rendering / RLBench eval rollouts only run on the main process.
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.accelerator = accelerator
        self.dataset = dataset
        # Build a raw DataLoader. Under Accelerate we need to pass the raw
        # DataLoader (not a `cycle` generator) to `accelerator.prepare`, then
        # cycle it after prepare returns the wrapped (DistributedSampler-aware)
        # version.
        raw_dl = torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        )
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        if self.accelerator is not None:
            # `prepare` wraps model in DDP, makes the optimizer device-aware,
            # and swaps the dataloader's sampler for a DistributedSampler so
            # each rank sees a disjoint shard of the dataset per epoch.
            self.model, self.optimizer, raw_dl = self.accelerator.prepare(
                self.model, self.optimizer, raw_dl
            )
            # Each rank uses its own GPU (cuda:LOCAL_RANK). Push the global
            # DEVICE pointer so `to_device`/`batch_to_device` follow.
            self.device = self.accelerator.device
            set_global_device(self.device)
        else:
            self.device = next(diffusion_model.parameters()).device

        self.dataloader = cycle(raw_dl)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference

        self.reset_parameters()
        self.step = 0

    def _unwrap(self, model):
        """Return the underlying model, stripping the DDP wrapper if present."""
        if self.accelerator is not None:
            return self.accelerator.unwrap_model(model)
        return model

    @property
    def is_main_process(self):
        return True if self.accelerator is None else self.accelerator.is_main_process

    def reset_parameters(self):
        self.ema_model.load_state_dict(self._unwrap(self.model).state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self._unwrap(self.model))

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps, front_bg=None, side_bg=None, latent_rep_model=None):
        timer = Timer()
        # Resolve the loss callable through the unwrapped model so that DDP's
        # forward/backward hooks still fire (they're attached to __call__ /
        # forward, not to .loss). `accelerator.accumulate(self.model)` below
        # is what gates gradient sync to the final accumulation micro-step.
        model_loss = self._unwrap(self.model).loss

        for step in range(int(n_train_steps)):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch)

                if self.accelerator is not None:
                    # The accumulate context skips DDP gradient sync on all
                    # but the final micro-step, then steps the optimizer on
                    # that final step. Loss scaling by 1/N is also handled.
                    with self.accelerator.accumulate(self.model):
                        loss, infos = model_loss(*batch)
                        self.accelerator.backward(loss)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                else:
                    loss, infos = self.model.loss(*batch)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()

            if self.accelerator is None:
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Side effects (EMA update, save, log, render) only on the main
            # process. EMA lives on rank 0's GPU; all ranks already see the
            # same gradients post-DDP-sync so the model state is identical.
            if self.is_main_process:
                if self.step % self.update_ema_every == 0:
                    self.step_ema()

                if self.step % self.save_freq == 0:
                    label = self.step // self.label_freq * self.label_freq
                    self.save(label)

                if self.step % self.log_freq == 0:
                    infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                    print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}', flush=True)

                    log_dict = {'step': self.step, 'loss': loss, **infos}
                    wandb.log(log_dict)

                if self.step == 0 and self.sample_freq:
                    self.render_reference(self.n_reference, front_bg=front_bg, side_bg=side_bg)

            self.step += 1

    def evaluate(self, n_eval_steps):
        '''
            evaluate model on validation set
        '''
        self.model.eval()
        self.ema_model.eval()
        with torch.no_grad():
            all_losses = []
            all_infos = []
            for step in tqdm(range(n_eval_steps), desc='eval'):
                batch = next(self.dataloader)
                batch = batch_to_device(batch)
                loss, infos = self.model.loss(*batch)
                all_losses.append(loss.item())
                all_infos.append(infos)
            for key in infos:
                mean_val = np.mean([info[key].cpu().numpy() for info in all_infos])
                print(f'{key}: {mean_val:8.4f}')
                wandb.log({'step':self.step, key: mean_val})
            print(f'loss: {np.mean(all_losses):8.4f}')
            wandb.log({'step':self.step, 'loss': np.mean(all_losses)})

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        # Only the main process writes ckpts. Other ranks would clobber the file.
        if not self.is_main_process:
            return
        data = {
            'step': self.step,
            # Strip the DDP wrapper so the saved state_dict has the original
            # parameter names (single-GPU and multi-GPU ckpts stay compatible).
            'model': self._unwrap(self.model).state_dict(),
            'ema': self.ema_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        ckpt_dir = os.path.join(self.logdir, 'ckpt')
        os.makedirs(ckpt_dir, exist_ok=True)
        savepath = os.path.join(ckpt_dir, f'state_{epoch}_step{self.step}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        import glob
        ckpt_dir = os.path.join(self.logdir, 'ckpt')

        # Try new location + naming first (ckpt/state_{epoch}_step{step}.pt)
        pattern = os.path.join(ckpt_dir, f'state_{epoch}_step*.pt')
        matches = glob.glob(pattern)
        if matches:
            loadpath = sorted(matches)[-1]
        else:
            # Try new naming in old location (state_{epoch}_step{step}.pt)
            pattern = os.path.join(self.logdir, f'state_{epoch}_step*.pt')
            matches = glob.glob(pattern)
            if matches:
                loadpath = sorted(matches)[-1]
            else:
                # Fall back to old naming convention (state_{epoch}.pt)
                loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')

        data = torch.load(loadpath)
        print(f'[ utils/training ] Loaded model from {loadpath}', flush=True)

        self.step = data['step']
        # Load into the unwrapped module so the DDP-wrapped parent stays
        # consistent (DDP shares the same parameter tensors).
        self._unwrap(self.model).load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
        if 'optimizer' in data:
            self.optimizer.load_state_dict(data['optimizer'])

    def load_latest(self):
        '''
            finds the latest checkpoint in logdir/ckpt and loads it.
            returns True if a checkpoint was loaded, False otherwise.
        '''
        import glob as glob_mod
        ckpt_dir = os.path.join(self.logdir, 'ckpt')
        pattern = os.path.join(ckpt_dir, 'state_*_step*.pt')
        matches = glob_mod.glob(pattern)
        if not matches:
            return False

        def _get_step(path):
            return int(os.path.basename(path).split('_step')[1].replace('.pt', ''))

        latest = max(matches, key=_get_step)
        data = torch.load(latest, map_location='cpu')
        print(f'[ utils/training ] Resuming from {latest}', flush=True)

        self.step = data['step']
        self._unwrap(self.model).load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
        if 'optimizer' in data:
            self.optimizer.load_state_dict(data['optimizer'])
        return True

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10, front_bg=None, side_bg=None):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=False, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        # Trajectory format: [actions, gripper_state, bg_features, observations]
        gripper_dim = getattr(self.dataset, 'gripper_dim', 0)
        bg_dim = getattr(self.dataset, 'bg_dim', 0)
        action_dim = self.dataset.action_dim

        # Extract bg_features if present
        bg_features_seq = None
        if bg_dim > 0:
            bg_start_idx = action_dim + gripper_dim
            bg_end_idx = bg_start_idx + bg_dim
            normed_bg = trajectories[:, :, bg_start_idx:bg_end_idx]
            bg_features_seq = self.dataset.normalizer.unnormalize(normed_bg, 'bg_features')

        obs_start_idx = action_dim + gripper_dim + bg_dim
        normed_observations = trajectories[:, :, obs_start_idx:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        savepath = os.path.join(self.logdir, f'_sample-reference.ply')
        self.renderer.composite(
            savepath, observations,
            front_bg=front_bg, side_bg=side_bg,
            bg_features_seq=bg_features_seq,
            log_bg=(bg_dim > 0), log_full=(bg_dim > 0)
        )

    def render_samples(self, batch_size=2, n_samples=2, front_bg=None, side_bg=None):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions)

            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + gripper_dim + bg_dim + observation_dim) ]
            samples = self.ema_model(conditions)
            trajectories = to_np(samples.trajectories)

            ## Extract dimensions
            gripper_dim = getattr(self.dataset, 'gripper_dim', 0)
            bg_dim = getattr(self.dataset, 'bg_dim', 0)
            action_dim = self.dataset.action_dim

            ## Extract bg_features if present
            bg_features_seq = None
            if bg_dim > 0:
                bg_start_idx = action_dim + gripper_dim
                bg_end_idx = bg_start_idx + bg_dim
                normed_bg = trajectories[:, :, bg_start_idx:bg_end_idx]
                # Get condition bg_features for prepending
                # Conditions include [gripper_state, bg_features, observations]
                cond_gripper_dim = gripper_dim
                cond_bg_start = cond_gripper_dim
                cond_bg_end = cond_bg_start + bg_dim
                normed_cond_bg = to_np(batch.conditions[0])[:, cond_bg_start:cond_bg_end][:, None]
                # Prepend condition bg to trajectory bg
                normed_bg = np.concatenate([
                    np.repeat(normed_cond_bg, n_samples, axis=0),
                    normed_bg
                ], axis=1)
                bg_features_seq = self.dataset.normalizer.unnormalize(normed_bg, 'bg_features')

            ## [ n_samples x horizon x observation_dim ]
            obs_start_idx = action_dim + gripper_dim + bg_dim
            normed_observations = trajectories[:, :, obs_start_idx:]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]
            # Extract just observations from conditions (skip gripper and bg)
            cond_obs_start = gripper_dim + bg_dim
            normed_cond_obs = normed_conditions[:, :, cond_obs_start:]

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_cond_obs, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.png')
            self.renderer.composite(
                savepath, observations,
                front_bg=front_bg, side_bg=side_bg,
                bg_features_seq=bg_features_seq,
                log_bg=(bg_dim > 0), log_full=(bg_dim > 0)
            )
    

    @torch.no_grad()
    def eval_mimicgen_rollouts(
        self,
        make_env_fn,
        dlp_model,
        calib_h5_path,
        n_episodes=5,
        max_steps=500,
        bounds_xyz=((-2,2), (-2,2), (-0.2,2.5)),
        grid_dhw=(64,64,64),
        cams=("agentview","sideview","robot0_eye_in_hand"),
        pixel_stride=2,
        goal_from_env_fn=None,
        goal_provider=None,         # NEW: DatasetGoalProvider for init_state + goal pairing
        random_init=False,          # If True, use random env reset instead of dataset init states
        task=None,                  # Task name for task-specific voxel bounds (e.g., "threading", "hammer_cleanup")

        save_videos=True,
        video_dir=None,
        video_fps=20,
        video_cams=("agentview",),

        renderer_3d=None,
        render_debug=True,
        render_debug_steps=(0,),
        exe_steps=1,  # NEW: number of actions to execute from each plan before replanning
        log_imagined_states=True,  # NEW: decode and log diffuser's predicted future states
        log_imagined_episode=0,  # which episode to log imagined states for
        log_imagined_plan_idx=0,  # which plan (replan) within the episode to log
    ):
        """
        True success eval by stepping MimicGen.
        - make_env_fn: () -> env
        - dlp_model: your 3D DLP (already loaded)
        - calib_h5_path: hdf5 with meta/cameras/*
        - goal_from_env_fn: (env, raw_obs) -> raw_goal_obs (optional, legacy)
        - goal_provider: DatasetGoalProvider for paired init_state + goal tokens (recommended)
        - exe_steps: number of actions to execute from each predicted trajectory before replanning
                     (action chunking). Default=1 means replan every step.
        - log_imagined_states: if True, decode and log the diffuser's predicted future observations
                               to wandb using the 3D renderer. This allows verifying that the
                               imagined states reconstruct into something reasonable.
        - log_imagined_episode: which episode to log imagined states for (default 0 = first episode)
        - log_imagined_plan_idx: which plan within the episode to log (default 0 = first plan)
        """
        from diffuser.envs.mimicgen_dlp_wrapper import MimicGenDLPWrapper

        device = next(self.ema_model.parameters()).device
        dlp_model = dlp_model.to(device).eval()

        import os
        import numpy as np
        import imageio.v2 as imageio

        def _to_uint8(img):
            img = np.asarray(img)
            if img.ndim == 4 and img.shape[0] == 1:
                img = img[0]
            # CHW -> HWC if needed
            if img.ndim == 3 and img.shape[0] in (1,3,4) and img.shape[-1] not in (1,3,4):
                img = np.transpose(img, (1,2,0))
            if img.dtype != np.uint8:
                if img.max() <= 1.5:
                    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)
            return img

        def _frame_from_raw_obs(raw_obs, cams_to_use):
            frames = []
            for cam in cams_to_use:
                k = f"{cam}_image"
                if k not in raw_obs:
                    continue
                frames.append(_to_uint8(raw_obs[k]))
            if not frames:
                return None
            # concat multi-view horizontally
            if len(frames) == 1:
                return frames[0]
            # make same height if needed (simple crop)
            h = min(f.shape[0] for f in frames)
            frames = [f[:h] for f in frames]
            return np.concatenate(frames, axis=1)
        def _render_tokens_debug(tag, obs_vec_flat, horizon_step=None):
            if renderer_3d is None:
                return
            if obs_vec_flat is None:
                raise RuntimeError("obs_vec_flat is None")

            # obs_vec_flat is 1D = K*Dtok
            K = getattr(envw, "K", None)
            Dtok = getattr(envw, "Dtok", None)

            print("K: ", K)
            print("Dtok: ", Dtok)


            # TODO: Actually make this read from something
            if K is None:
                K = 16
            if Dtok is None:
                # hard fail-fast: you should set this; but infer if possible
                if obs_vec_flat.size % K != 0:
                    raise RuntimeError(f"Cannot infer Dtok: len(obs_vec_flat)={obs_vec_flat.size} not divisible by K={K}")
                Dtok = obs_vec_flat.size // K
            
            print("obs vec flat: ", obs_vec_flat.shape)
            toks = np.asarray(obs_vec_flat, dtype=np.float32).reshape(1, K, Dtok)  # [1,K,Dtok]

            # Use monotonic step for wandb: prefer global trainer step if provided.
            step_to_log = 200
            print("toks: ", toks.shape)
            renderer_3d.render(
                toks[0],                              # renderer accepts [K,Dtok] or flat; yours accepts either
                tag=tag,
                step=step_to_log,
                base="eval_debug"
            )


        if save_videos:
            if video_dir is None:
                video_dir = os.path.join(self.logdir, "eval_videos", f"step_{self.step}")
            os.makedirs(video_dir, exist_ok=True)


        successes = []
        returns = []
        lengths = []

        # Print eval configuration
        print("=" * 60)
        print(f"[eval_mimicgen_rollouts] Starting {n_episodes} episodes, max_steps={max_steps}")
        print(f"[eval_mimicgen_rollouts] Goal provider: {'DatasetGoalProvider' if goal_provider is not None else 'None (legacy mode)'}")
        print(f"[eval_mimicgen_rollouts] Init mode: {'RANDOM' if random_init else 'DATASET (fixed init states)'}")
        print(f"[eval_mimicgen_rollouts] ACTION CHUNKING: exe_steps={exe_steps}, horizon={self.dataset.horizon}")
        if exe_steps > 1:
            print(f"[eval_mimicgen_rollouts] Will execute {exe_steps} actions per plan before replanning")
        else:
            print(f"[eval_mimicgen_rollouts] WARNING: exe_steps=1 means replanning every step (no chunking)")
        if log_imagined_states and renderer_3d is not None:
            print(f"[eval_mimicgen_rollouts] IMAGINED STATE LOGGING: enabled for ep={log_imagined_episode}, plan={log_imagined_plan_idx}")
        elif log_imagined_states and renderer_3d is None:
            print(f"[eval_mimicgen_rollouts] IMAGINED STATE LOGGING: requested but renderer_3d is None, skipping")
        print("=" * 60, flush=True)

        # Create env and wrapper ONCE to avoid OpenGL context corruption
        env = make_env_fn()
        envw = MimicGenDLPWrapper(
            env=env,
            dlp_model=dlp_model,
            device=device,
            cams=cams,
            grid_dhw=grid_dhw,
            pixel_stride=pixel_stride,
            calib_h5_path=calib_h5_path,
            get_goal_raw_obs_fn=goal_from_env_fn,
            goal_provider=goal_provider,  # NEW: dataset-based goal provider
            random_init=random_init,      # NEW: random vs dataset init
            normalize_to_unit_cube=False,
            task=task,                    # Task name for task-specific voxel bounds
        )

        for ep in range(n_episodes):
            print(f"\n[eval] Episode {ep+1}/{n_episodes}")

            obs_vec = envw.reset()
            # envw.print_params_like_h5_script(envw.last_raw_obs)   

            if render_debug and renderer_3d is not None:
                # _render_tokens_debug(tag=f"ep_{ep:02d}/reset_obs", obs_vec_flat=obs_vec, horizon_step=0)

                vox = envw.last_vox  # np [C,D,H,W]
                print("GT VOX: ", vox.shape )
                    # avg_rgb
                log_rgb_voxels(
                    name=f"eval_debug/ep_{ep:02d}/gt_vox_rgb",
                    rgb_vol=vox,          # [3,D,H,W]
                    alpha_vol=None,
                    KPx=None,
                    step=int(200),
                    mode="splat",
                    topk=60000,
                    alpha_thresh=0.05,
                    pad=2.0,
                    show_axes=True,
                )

                # ====== LIVE vs PREPROCESSED COMPARISON (same initial state!) ======
                if goal_provider is not None and hasattr(goal_provider, 'get_first_frame_tokens'):
                    preproc_toks = goal_provider.get_first_frame_tokens()  # (K, Dtok) from pkl
                    # live_toks = envw.last_toks if envw.last_toks is not None else obs_vec.reshape(16, 12)
                    if envw.last_toks is not None:
                        live_toks = envw.last_toks
                        print(f" Using last_toks from envw")
                    else:
                        K, Dtok = preproc_toks.shape
                        live_toks = obs_vec.reshape(K, Dtok)
                        print(f" Reshaping to ({K}, {Dtok})")

                    print(f"\n[LIVE vs PREPROCESSED - SAME INITIAL STATE]")
                    print(f"  Preprocessed tokens: range=[{preproc_toks.min():.4f}, {preproc_toks.max():.4f}], mean={preproc_toks.mean():.4f}, std={preproc_toks.std():.4f}")
                    print(f"  Live tokens:         range=[{live_toks.min():.4f}, {live_toks.max():.4f}], mean={live_toks.mean():.4f}, std={live_toks.std():.4f}")

                    diff = np.abs(preproc_toks - live_toks)
                    print(f"  DIFF: max={diff.max():.4f}, mean={diff.mean():.4f}, std={diff.std():.4f}")

                    if diff.max() > 0.1:
                        print(f"  *** SIGNIFICANT MISMATCH DETECTED ***")
                        # Per-dimension breakdown
                        dim_names = ["z_x", "z_y", "z_z", "scale_x", "scale_y", "scale_z", "depth", "obj_on", "feat_0", "feat_1", "feat_2", "feat_3"]
                        print(f"  Per-dimension diff (all 12 dims):")
                        for d in range(min(len(dim_names), preproc_toks.shape[1])):
                            p_col = preproc_toks[:, d]
                            l_col = live_toks[:, d]
                            col_diff = np.abs(p_col - l_col)
                            print(f"    {dim_names[d]:8s}: preproc=[{p_col.min():7.3f}, {p_col.max():7.3f}] live=[{l_col.min():7.3f}, {l_col.max():7.3f}] diff_max={col_diff.max():.4f}")
                    else:
                        print(f"  ✓ Tokens match well!")

                    # ====== VISUAL COMPARISON: Decode both token sets to voxels ======
                    if renderer_3d is not None and hasattr(renderer_3d, 'render_volume'):
                        try:
                            import torch
                            # Get bg_features for full reconstruction visualization
                            bg_features = envw.last_bg_features if hasattr(envw, 'last_bg_features') else None

                            # Decode preprocessed tokens to voxels (with bg_features for full reconstruction)
                            preproc_dec = renderer_3d.render_volume(preproc_toks, bg_features=bg_features)
                            preproc_fg = preproc_dec['fg_only']
                            preproc_rec = preproc_dec['rec_rgb']
                            preproc_bg = preproc_dec.get('bg_only')
                            log_rgb_voxels(
                                name=f"eval_debug/ep_{ep:02d}/preproc_decoded_fg",
                                rgb_vol=preproc_fg.cpu().numpy(),
                                alpha_vol=None, KPx=None,
                                step=int(200 + ep),
                                mode="splat", topk=60000, alpha_thresh=0.05, pad=2.0, show_axes=True,
                            )
                            log_rgb_voxels(
                                name=f"eval_debug/ep_{ep:02d}/preproc_decoded_rec",
                                rgb_vol=preproc_rec.cpu().numpy(),
                                alpha_vol=None, KPx=None,
                                step=int(200 + ep),
                                mode="splat", topk=60000, alpha_thresh=0.05, pad=2.0, show_axes=True,
                            )
                            # Log background-only if available
                            if preproc_bg is not None:
                                log_rgb_voxels(
                                    name=f"eval_debug/ep_{ep:02d}/preproc_decoded_bg",
                                    rgb_vol=preproc_bg.cpu().numpy(),
                                    alpha_vol=None, KPx=None,
                                    step=int(200 + ep),
                                    mode="splat", topk=60000, alpha_thresh=0.05, pad=2.0, show_axes=True,
                                )

                            # Decode live tokens to voxels (with bg_features for full reconstruction)
                            live_dec = renderer_3d.render_volume(live_toks, bg_features=bg_features)
                            live_fg = live_dec['fg_only']
                            live_rec = live_dec['rec_rgb']
                            live_bg = live_dec.get('bg_only')
                            log_rgb_voxels(
                                name=f"eval_debug/ep_{ep:02d}/live_decoded_fg",
                                rgb_vol=live_fg.cpu().numpy(),
                                alpha_vol=None, KPx=None,
                                step=int(200 + ep),
                                mode="splat", topk=60000, alpha_thresh=0.05, pad=2.0, show_axes=True,
                            )
                            log_rgb_voxels(
                                name=f"eval_debug/ep_{ep:02d}/live_decoded_rec",
                                rgb_vol=live_rec.cpu().numpy(),
                                alpha_vol=None, KPx=None,
                                step=int(200 + ep),
                                mode="splat", topk=60000, alpha_thresh=0.05, pad=2.0, show_axes=True,
                            )
                            # Log background-only if available
                            if live_bg is not None:
                                log_rgb_voxels(
                                    name=f"eval_debug/ep_{ep:02d}/live_decoded_bg",
                                    rgb_vol=live_bg.cpu().numpy(),
                                    alpha_vol=None, KPx=None,
                                    step=int(200 + ep),
                                    mode="splat", topk=60000, alpha_thresh=0.05, pad=2.0, show_axes=True,
                                )

                            decoded_vox = envw.decoded_vox
                            log_rgb_voxels(
                                name=f"eval_debug/ep_{ep:02d}/decoded_vox",
                                rgb_vol=decoded_vox.cpu().numpy(),
                                alpha_vol=None, KPx=None,
                                step=int(200 + ep),
                                mode="splat", topk=60000, alpha_thresh=0.05, pad=2.0, show_axes=True,
                            )

                            vox = envw.vox
                            log_rgb_voxels(
                                name=f"eval_debug/ep_{ep:02d}/vox_from_encode",
                                rgb_vol=vox.cpu().numpy(),
                                alpha_vol=None, KPx=None,
                                step=int(200 + ep),
                                mode="splat", topk=60000, alpha_thresh=0.05, pad=2.0, show_axes=True,
                            )

                            # Also log the raw GT voxels from simulator for comparison
                            # (already logged above as gt_vox_rgb, but log again here for side-by-side)
                            log_rgb_voxels(
                                name=f"eval_debug/ep_{ep:02d}/gt_vox_from_sim",
                                rgb_vol=vox,
                                alpha_vol=None, KPx=None,
                                step=int(200 + ep),
                                mode="splat", topk=60000, alpha_thresh=0.05, pad=2.0, show_axes=True,
                            )
                            print(f"  [VISUAL DEBUG] Logged decoded voxels for preproc and live tokens")
                        except Exception as e:
                            print(f"  [VISUAL DEBUG] Failed to render token comparison: {e}")
                    # ====== END VISUAL COMPARISON ======
                # ====== END LIVE vs PREPROCESSED ======

                # ====== TOKEN DISTRIBUTION DIAGNOSTIC ======
                if hasattr(envw, 'goal_vec') and envw.goal_vec is not None:
                    K_goal = self.dataset.fields.observations.shape[2]
                    Dtok_goal = self.dataset.fields.observations.shape[3]
                    goal_toks = envw.goal_vec.reshape(K_goal, Dtok_goal)  # (K, Dtok)
                    obs_toks = envw.last_toks if envw.last_toks is not None else obs_vec.reshape(K_goal, Dtok_goal)
                    print(f"\n[TOKEN DISTRIBUTION DIAGNOSTIC]")
                    print(f"  Goal tokens (from dataset):  range=[{goal_toks.min():.4f}, {goal_toks.max():.4f}], mean={goal_toks.mean():.4f}, std={goal_toks.std():.4f}")
                    print(f"  Obs tokens (live encoded):   range=[{obs_toks.min():.4f}, {obs_toks.max():.4f}], mean={obs_toks.mean():.4f}, std={obs_toks.std():.4f}")

                    # Check per-dimension statistics
                    print(f"  Per-dimension comparison (first 4 dims):")
                    for d in range(min(4, goal_toks.shape[1])):
                        g_col = goal_toks[:, d]
                        o_col = obs_toks[:, d]
                        print(f"    dim {d}: goal=[{g_col.min():.3f}, {g_col.max():.3f}] obs=[{o_col.min():.3f}, {o_col.max():.3f}]")
                # ====== END TOKEN DIAGNOSTIC ======

                # if you want to also see the goal:
                # if getattr(envw, "goal_vec", None) is not None:
                #     _render_tokens_debug(tag=f"ep_{ep:02d}/reset_goal", obs_vec_flat=envw.goal_vec, horizon_step=0)

            ep_ret = 0.0
            plan_idx = 0  # track how many times we've replanned this episode
            frames = []
            if save_videos and envw.last_raw_obs is not None:
                fr = _frame_from_raw_obs(envw.last_raw_obs, video_cams)
                if fr is not None:
                    frames.append(fr)
            # ACTION CHUNKING: track planned actions buffer
            action_buffer = None  # will hold (H, a_dim) tensor of planned actions
            action_idx = 0  # which action in buffer to execute next
            a_dim = self.dataset.action_dim
            z_scale = getattr(self.dataset, 'action_z_scale', 1.0)

            # Helper to normalize observations
            def norm_obs(v):
                v_norm = self.dataset.normalizer.normalize(v[None], "observations")[0]
                return torch.from_numpy(v_norm).float().to(device)

            # Helper to normalize gripper state
            def norm_gripper(v):
                v_norm = self.dataset.normalizer.normalize(v[None], "gripper_state")[0]
                return torch.from_numpy(v_norm).float().to(device)

            # Helper to normalize bg_features
            def norm_bg(v):
                v_norm = self.dataset.normalizer.normalize(v[None], "bg_features")[0]
                return torch.from_numpy(v_norm).float().to(device)

            # Check if we should use gripper observations and bg_features
            use_gripper_obs = getattr(self.dataset, 'use_gripper_obs', False)
            gripper_dim = getattr(self.dataset, 'gripper_dim', 0)
            use_bg_obs = getattr(self.dataset, 'use_bg_obs', False)
            bg_dim = getattr(self.dataset, 'bg_dim', 0)

            t = 0
            while t < max_steps:
                # Check if we need to replan (no buffer, or exhausted exe_steps actions)
                need_replan = (action_buffer is None) or (action_idx >= exe_steps) or (action_idx >= action_buffer.shape[0])

                if need_replan:
                    # ====== PLANNING PHASE ======
                    # Build current observation condition
                    # Condition format: [gripper_state (optional), bg_features (optional), observations]
                    cond_parts = []

                    # Add gripper state if enabled
                    if use_gripper_obs and gripper_dim > 0:
                        gripper_cond = envw.get_gripper_cond(horizon=self.dataset.horizon)
                        if gripper_cond is not None and 0 in gripper_cond:
                            gripper_norm = norm_gripper(gripper_cond[0])
                            cond_parts.append(gripper_norm)

                    # Add bg_features if enabled
                    if use_bg_obs and bg_dim > 0:
                        bg_cond = envw.get_bg_cond(horizon=self.dataset.horizon)
                        if bg_cond is not None and 0 in bg_cond:
                            bg_norm = norm_bg(bg_cond[0])
                            cond_parts.append(bg_norm)

                    # Add observations
                    obs_norm = norm_obs(obs_vec)
                    cond_parts.append(obs_norm)

                    # Concatenate all parts
                    cond_0 = torch.cat(cond_parts, dim=-1)[None, :]

                    # No goal conditioning — only condition on initial observation.
                    # GoalDataset trains with conditions={0: obs_0} only (no H-1 key),
                    # so inference must match: adding a zero condition at H-1 would
                    # force the last observation to the mean state at every denoising
                    # step, distorting the predicted trajectory.
                    cond = {
                        0: cond_0,
                    }

                    if t == 0:
                        print(f"\n[EVAL] ep={ep}: No goal conditioning (matches training)")
                        print(f"  cond_0 shape: {cond_0.shape}")

                    # Sample new trajectory from diffusion model
                    sample = self.ema_model(cond, verbose=False)
                    traj = sample.trajectories[0]  # (H, action_dim + obs_dim)
                    action_buffer = traj[:, :a_dim].detach().cpu().numpy().astype(np.float32)  # (H, a_dim)
                    action_idx = 0

                    # Print planning info
                    n_actions_to_exec = min(exe_steps, action_buffer.shape[0])
                    print(f"[PLAN] t={t}: Generated {action_buffer.shape[0]}-step plan, will execute {n_actions_to_exec} actions")
                    if t < 10:  # Show full planned trajectory for first few plans
                        print(f"  Planned Z trajectory (normalized): {[f'{action_buffer[i, 2]:.3f}' for i in range(min(5, action_buffer.shape[0]))]}")

                    # ====== LOG IMAGINED STATES ======
                    # Decode and visualize the diffuser's predicted future observations
                    if log_imagined_states and renderer_3d is not None and ep == log_imagined_episode and plan_idx == log_imagined_plan_idx:
                        print(f"\n[IMAGINED STATES] Logging decoded predictions for ep={ep}, plan={plan_idx}")
                        # Extract predicted observations (skip action_dim + gripper_dim + bg_dim)
                        obs_start_idx = a_dim + gripper_dim + bg_dim
                        pred_obs_norm = traj[:, obs_start_idx:].detach().cpu().numpy()  # (H, obs_dim)
                        # Unnormalize observations
                        pred_obs = self.dataset.normalizer.unnormalize(pred_obs_norm, "observations")  # (H, obs_dim)

                        # Extract bg_features from trajectory if available
                        pred_bg_features = None
                        if use_bg_obs and bg_dim > 0:
                            bg_start_idx = a_dim + gripper_dim
                            bg_end_idx = bg_start_idx + bg_dim
                            pred_bg_norm = traj[:, bg_start_idx:bg_end_idx].detach().cpu().numpy()  # (H, bg_dim)
                            pred_bg_features = self.dataset.normalizer.unnormalize(pred_bg_norm, "bg_features")  # (H, bg_dim)

                        # ====== DIAGNOSTIC: Check for clipping and roundtrip loss ======
                        H = pred_obs.shape[0]
                        t0_normalized = pred_obs_norm[0]
                        print(f"  [DIAG] t=0 normalized range: [{t0_normalized.min():.4f}, {t0_normalized.max():.4f}]")
                        print(f"  [DIAG] t=0 would be clipped: {t0_normalized.max() > 1.0001 or t0_normalized.min() < -1.0001}")

                        # Check a later timestep for comparison
                        mid_t = min(12, H - 1)
                        if mid_t > 0:
                            tmid_normalized = pred_obs_norm[mid_t]
                            print(f"  [DIAG] t={mid_t} normalized range: [{tmid_normalized.min():.4f}, {tmid_normalized.max():.4f}]")
                            print(f"  [DIAG] t={mid_t} would be clipped: {tmid_normalized.max() > 1.0001 or tmid_normalized.min() < -1.0001}")

                        # Compare pred_obs[0] with original observation
                        diff_t0 = np.abs(pred_obs[0] - obs_vec)
                        print(f"  [DIAG] pred_obs[0] vs obs_vec diff: max={diff_t0.max():.6f}, mean={diff_t0.mean():.6f}")
                        if diff_t0.max() > 0.01:
                            print(f"  [DIAG] WARNING: Significant roundtrip loss detected at t=0!")
                            # Find which dimensions have the largest differences
                            top_diff_idx = np.argsort(diff_t0)[-5:][::-1]
                            print(f"  [DIAG] Top 5 differing dims: {top_diff_idx}, diffs: {diff_t0[top_diff_idx]}")
                        # ====== END DIAGNOSTIC ======

                        # Log a subset of timesteps (start, middle, end)
                        timesteps_to_log = [0, H // 4, H // 2, 3 * H // 4, H - 1]
                        timesteps_to_log = sorted(set(t_idx for t_idx in timesteps_to_log if 0 <= t_idx < H))

                        for t_idx in timesteps_to_log:
                            obs_at_t = pred_obs[t_idx]  # (obs_dim,)
                            bg_at_t = pred_bg_features[t_idx] if pred_bg_features is not None else None
                            tag = f"imagined/ep_{ep:02d}_plan_{plan_idx:02d}/t_{t_idx:03d}_of_{H}"
                            try:
                                renderer_3d.render(
                                    obs_at_t,
                                    bg_features=bg_at_t,
                                    tag=tag,
                                    step=self.step,
                                    base="imagined_states",
                                    log_fg=True,
                                    log_bg=(bg_at_t is not None),
                                    log_full=(bg_at_t is not None),
                                )
                                print(f"  Logged imagined state at horizon t={t_idx}/{H}")
                            except Exception as e:
                                print(f"  Failed to log imagined state t={t_idx}: {e}")

                        # Also log the current observation for comparison
                        try:
                            current_bg = envw.last_bg_features if hasattr(envw, 'last_bg_features') else None
                            renderer_3d.render(
                                obs_vec,
                                bg_features=current_bg,
                                tag=f"imagined/ep_{ep:02d}_plan_{plan_idx:02d}/cond_t0_current",
                                step=self.step,
                                base="imagined_states",
                                log_fg=True,
                                log_bg=(current_bg is not None),
                                log_full=(current_bg is not None),
                            )
                            print(f"  Logged conditioning: current observation")
                        except Exception as e:
                            print(f"  Failed to log current obs condition: {e}")
                        print()

                    plan_idx += 1  # increment plan counter

                # ====== EXECUTION PHASE ======
                # Get current action from buffer
                a_norm = action_buffer[action_idx]

                # Unnormalize action
                a = self.dataset.normalizer.unnormalize(a_norm[None], "actions")[0]

                # Unscale Z if action_z_scale was applied during training
                if z_scale != 1.0:
                    a[2] /= z_scale

                # Step environment
                obs_vec, r, done, info = envw.step(a)

                # Print execution info
                if t < 10:
                    print(f"[EXEC] t={t}: action_idx={action_idx}/{exe_steps} | pos=[{a[0]:.3f}, {a[1]:.3f}, {a[2]:.3f}] grip={a[6]:.2f}")
                    if t == 0:
                        norm = self.dataset.normalizer.normalizers.get('actions', None)
                        if norm is not None:
                            if hasattr(norm, 'mins') and hasattr(norm, 'maxs'):
                                print(f"  normalizer mins: {norm.mins.flatten()}")
                                print(f"  normalizer maxs: {norm.maxs.flatten()}")
                            elif hasattr(norm, 'means') and hasattr(norm, 'stds'):
                                print(f"  normalizer means: {norm.means.flatten()}")
                                print(f"  normalizer stds: {norm.stds.flatten()}")

                action_idx += 1
                ep_ret += float(r)

                if save_videos and envw.last_raw_obs is not None:
                    fr = _frame_from_raw_obs(envw.last_raw_obs, video_cams)
                    if fr is not None:
                        frames.append(fr)

                t += 1
                # Check for episode termination (done) or success
                episode_success = info.get("success", False) if isinstance(info, dict) else False
                if done or episode_success:
                    if episode_success:
                        print(f"[EVAL] Episode succeeded at t={t}, stopping early")
                    break

            # ---- new: write video ----
            if save_videos and len(frames) > 0:
                out_path = os.path.join(video_dir, f"ep_{ep:03d}.mp4")
                # macro_block_size=None avoids ffmpeg issues with non-multiple-of-16 sizes
                imageio.mimsave(out_path, frames, fps=int(video_fps), macro_block_size=None)
                print(f"[mimicgen eval] wrote video: {out_path}", flush=True)
            # success flag (wrapper tries common info keys; may be None)
            success = bool(info.get("success", False)) if isinstance(info, dict) else False
            successes.append(success)
            returns.append(ep_ret)
            lengths.append(t + 1)

        try:
            env.close()
        except Exception:
            pass

        out = {
            "sim/success_rate": float(np.mean(successes)) if len(successes) else 0.0,
            "sim/avg_return": float(np.mean(returns)) if len(returns) else 0.0,
            "sim/avg_len": float(np.mean(lengths)) if len(lengths) else 0.0,
        }
        return out

    # ------------------------------------------------------------------
    # RLBench live eval (3D voxel DLP). Runs N episodes in the live wrapper,
    # records front_rgb frames into wandb.Video instead of writing to disk.
    # ------------------------------------------------------------------
    def eval_rlbench_rollouts(
        self,
        make_env_fn,
        *,
        make_policy_fn=None,
        n_episodes: int = 2,
        max_steps: int = 200,
        exe_steps: int = 1,
        video_fps: int = 10,
        wandb_step: int = None,
        log_voxel_viz: bool = True,
        video_dir_override: str = None,
    ):
        """Run `n_episodes` rollouts in the RLBenchDLPEnv wrapper, log each
        episode's front_rgb video to wandb.

        - make_env_fn() must return an RLBenchDLPEnv-compatible env. The env is
          NOT shut down here so it can be reused across eval cycles (the
          headless CoppeliaSim cross-launch texture-loss bug otherwise wipes
          textures on re-launch).
        - Action chunking mirrors eval_mimicgen_rollouts: each plan emits a
          horizon-length trajectory, the first `exe_steps` actions are executed
          before re-planning.
        """
        import torch as _torch
        import numpy as _np
        device = _torch.device(getattr(self, "device", "cuda:0"))

        env = make_env_fn()
        # LanguageConditionedPolicy owns a frozen CLIP encoder; when provided,
        # we pass lang + action_cond into self.ema_model so rollout matches
        # the training-time conditioning. Without it, we fall back to
        # unconditional sampling (pre-existing 3D behavior).
        policy = make_policy_fn() if make_policy_fn is not None else None

        a_dim = int(self.ema_model.action_dim)
        gripper_dim = int(getattr(self.ema_model.model, "gripper_dim", 0))
        bg_dim = int(getattr(self.ema_model.model, "bg_dim", 0))
        z_scale = float(getattr(self.dataset, "action_z_scale", 1.0))

        # Optional voxel viz (gt + rec_full + rec_fg + rec_bg + kp overlay).
        # Same logger used by train_dlp_voxel.py val visuals.
        _log_rgb_voxels = None
        if log_voxel_viz:
            try:
                from eval.eval_vox import log_rgb_voxels as _log_rgb_voxels
            except Exception as e:
                print(f"[eval_rlbench] voxel viz disabled: {e}", flush=True)

        def _voxel_panel(env_obj, prefix, step_for_log):
            """Decode env_obj's most-recent voxel + kp through the DLP and log
            gt/rec_full/rec_fg/rec_bg under '<prefix>/...' to wandb."""
            if _log_rgb_voxels is None:
                return
            vox = getattr(env_obj, "_last_voxel", None)
            kp  = getattr(env_obj, "_last_kp",    None)
            cov = getattr(env_obj, "_last_cov",   None)
            dlp = getattr(env_obj, "dlp_model",   None)
            if vox is None or dlp is None:
                return
            with _torch.no_grad():
                vob = vox.unsqueeze(0).to(next(dlp.parameters()).device)
                rgb_only = vob[:, :int(env_obj.dlp_cfg.get("ch", 3))].contiguous() \
                    if env_obj.dlp_cfg else vob[:, :3].contiguous()
                kp_b  = kp.unsqueeze(0).to(rgb_only.device)  if kp  is not None else None
                cov_b = cov.unsqueeze(0).to(rgb_only.device) if cov is not None else None
                meta = {"kmeans_kp": kp_b, "kmeans_cov": cov_b} if kp_b is not None else None
                full = dlp(rgb_only, deterministic=True, warmup=False,
                           with_loss=False, meta=meta)
            gt   = full["x"][0].cpu()
            rec  = (full["rec"][0] if full.get("rec") is not None else full["rec_rgb"][0]).cpu()
            fg   = full["dec_objects"][0].cpu() if full.get("dec_objects") is not None else None
            bg_v = None
            if full.get("bg_mask") is not None and full.get("bg") is not None:
                bg_v = (full["bg_mask"] * full["bg"][:, :3])[0].cpu()
            mu = full["mu_tot"][0]
            if mu.dim() == 3: mu = mu[0]
            mu = mu.cpu()
            try:
                _log_rgb_voxels(name=f"{prefix}/gt",       rgb_vol=gt,  KPx=mu, step=step_for_log)
                _log_rgb_voxels(name=f"{prefix}/rec_full", rgb_vol=rec, KPx=mu, step=step_for_log)
                if fg is not None:
                    _log_rgb_voxels(name=f"{prefix}/rec_fg", rgb_vol=fg, KPx=mu, step=step_for_log)
                if bg_v is not None:
                    _log_rgb_voxels(name=f"{prefix}/rec_bg", rgb_vol=bg_v, KPx=None, step=step_for_log)
            except Exception as e:
                print(f"[eval_rlbench] voxel viz failed for {prefix}: {e}", flush=True)

        use_gripper_obs = bool(getattr(self.dataset, "use_gripper_obs", False))
        use_bg_obs = bool(getattr(self.dataset, "use_bg_obs", False))

        # Keypose mode: each predicted pose is the next *keypose* target. Force
        # exe_steps=1 so we apply one keypose per replan -- matches 3DDA /
        # ParticleSplat receding-horizon control. The env wrapper's
        # EndEffectorPoseViaPlanning action mode handles the multi-step
        # trajectory to reach each target. Caller should drop max_steps
        # accordingly (~15 keypose attempts is plenty per episode).
        keypose_mode = bool(getattr(self.dataset, 'keypose_mode', False))
        if keypose_mode and exe_steps != 1:
            print(f"[eval_rlbench] keypose_mode=True: forcing exe_steps {exe_steps} -> 1", flush=True)
            exe_steps = 1

        successes, lengths = [], []
        print(f"[eval_rlbench_rollouts] Starting {n_episodes} episodes, "
              f"max_steps={max_steps}, exe_steps={exe_steps} keypose_mode={keypose_mode}", flush=True)

        for ep in range(n_episodes):
            try:
                obs_dict = env.reset(variation=ep)
            except Exception as e:
                print(f"[eval_rlbench] ep={ep} reset failed: {type(e).__name__}: {e}", flush=True)
                successes.append(0.0); lengths.append(0); continue

            obs_vec = _np.asarray(obs_dict["obs"], dtype=_np.float32).reshape(-1)
            language = obs_dict.get("language", "")
            print(f"\n[eval_rlbench] ep {ep+1}/{n_episodes}  variation={obs_dict.get('variation_number')}  "
                  f"lang={language!r}", flush=True)

            # Encode the instruction once per episode; reuse cached tokens for
            # every denoising step. Without a policy, lang_tok stays None and
            # the model runs unconditional.
            lang_tok = None
            lang_mask_tok = None
            if policy is not None:
                policy.set_instruction(language if language else "")
                lang_tok = policy._lang_tokens.to(device)[:1]
                lang_mask_tok = (
                    policy._lang_mask.to(device)[:1] if policy._lang_mask is not None else None
                )

            # Voxel viz from the reset frame (one panel per episode).
            step_for_log = wandb_step if wandb_step is not None else self.step
            _voxel_panel(env, prefix=f"sim/voxel/ep_{ep:02d}/reset",
                         step_for_log=step_for_log)

            action_buffer = None
            action_idx = 0
            t = 0
            done = False
            success = False
            last_reward = 0.0

            while t < max_steps and not done:
                # Replan when buffer is empty / exhausted
                need_replan = (action_buffer is None) or (action_idx >= exe_steps)
                if need_replan:
                    cond_parts = []
                    if use_gripper_obs and gripper_dim > 0:
                        gs = _np.asarray(obs_dict["gripper_state"], dtype=_np.float32).reshape(1, -1)
                        gs_norm = self.dataset.normalizer.normalize(gs, "gripper_state")[0]
                        cond_parts.append(_torch.from_numpy(gs_norm).float().to(device))
                    if use_bg_obs and bg_dim > 0:
                        bg = _np.asarray(obs_dict["bg_features"], dtype=_np.float32).reshape(1, -1)
                        bg_norm = self.dataset.normalizer.normalize(bg, "bg_features")[0]
                        cond_parts.append(_torch.from_numpy(bg_norm).float().to(device))
                    obs_norm = self.dataset.normalizer.normalize(obs_vec[None], "observations")[0]
                    cond_parts.append(_torch.from_numpy(obs_norm).float().to(device))
                    cond_0 = _torch.cat(cond_parts, dim=-1)[None, :]
                    cond = {0: cond_0}

                    # Proprioception conditioning: pin action[0] to the current
                    # gripper pose, normalized in actions-space. Requires the
                    # env's gripper_state to have the same dim as action
                    # (true for rot6d-based absolute-action configs).
                    # In keypose mode we skip the pin -- pinning collapses
                    # traj[1] toward action[0] -> jitter. Current gripper still
                    # flows in via cond[0]'s gripper_state component. Matches
                    # 3DDA's design (diffuser_actor.py:215-216).
                    action_cond = None
                    if keypose_mode:
                        action_cond = {}
                    elif policy is not None and "gripper_state" in obs_dict:
                        gs_raw = _np.asarray(obs_dict["gripper_state"],
                                             dtype=_np.float32).reshape(1, -1)
                        if gs_raw.shape[-1] == a_dim:
                            gs_as_act = self.dataset.normalizer.normalize(gs_raw, "actions")
                            action_cond = {0: _torch.from_numpy(gs_as_act).float().to(device)}

                    if policy is not None:
                        sample = self.ema_model(
                            cond, lang=lang_tok, lang_mask=lang_mask_tok,
                            action_cond=action_cond, verbose=False,
                        )
                    else:
                        sample = self.ema_model(cond, verbose=False)
                    traj = sample.trajectories[0]  # (H, transition_dim)
                    # Buffer slot selection:
                    #  - keypose_mode (v2): trajectory contains only future keyposes,
                    #    slot 0 IS the next keypose to execute -> use traj[:].
                    #  - legacy visuomotor pin: action_cond={0: a0} pins slot 0 to
                    #    the current pose -> skip slot 0, play out traj[1..H-1].
                    #  - no pin (action_cond=None): use traj[:].
                    if keypose_mode:
                        action_buffer = traj[:, :a_dim].detach().cpu().numpy().astype(_np.float32)
                    elif action_cond is not None and len(action_cond) > 0:
                        action_buffer = traj[1:, :a_dim].detach().cpu().numpy().astype(_np.float32)
                    else:
                        action_buffer = traj[:, :a_dim].detach().cpu().numpy().astype(_np.float32)
                    action_idx = 0

                a_norm = action_buffer[action_idx]
                a = self.dataset.normalizer.unnormalize(a_norm[None], "actions")[0]
                if z_scale != 1.0:
                    a[2] /= z_scale

                obs_dict, r, done, info = env.step(a)
                if obs_dict is None:
                    print(f"[eval_rlbench] step failed at t={t}: {info}", flush=True)
                    break
                obs_vec = _np.asarray(obs_dict["obs"], dtype=_np.float32).reshape(-1)
                last_reward = float(r)
                action_idx += 1
                t += 1
                # RLBench emits reward=1.0 on task completion (no info["success"]).
                # Mirrors the 2D wrapper's check (last_reward >= 0.5).
                if last_reward >= 0.5:
                    success = True
                    print(f"[eval_rlbench]   succeeded at t={t} (reward={last_reward:.2f})", flush=True)
                    break
                if isinstance(info, dict) and info.get("success", False):
                    success = True
                    print(f"[eval_rlbench]   succeeded at t={t} (info.success)", flush=True)
                    break

            frames = env.pop_recorded_frames()
            # Final safety net: if the last step's reward crossed threshold but we
            # didn't break (shouldn't happen, but covers max_steps-exit races).
            if not success and last_reward >= 0.5:
                success = True
            successes.append(1.0 if success else 0.0)
            lengths.append(t)

            if frames and len(frames) > 0:
                # Save mp4 to <savepath>/eval_videos/, then upload that path to
                # wandb (avoids wandb's raw-data encoder which requires moviepy).
                import imageio.v2 as imageio
                step_for_log = wandb_step if wandb_step is not None else self.step
                status = "success" if success else "fail"
                if video_dir_override:
                    video_dir = video_dir_override
                    os.makedirs(video_dir, exist_ok=True)
                    out_path = os.path.join(video_dir, f"ep{ep:02d}_{status}.mp4")
                else:
                    video_dir = os.path.join(self.logdir, "eval_videos")
                    os.makedirs(video_dir, exist_ok=True)
                    out_path = os.path.join(video_dir, f"step{step_for_log}_ep{ep:02d}_{status}.mp4")
                # macro_block_size=None avoids ffmpeg failing on non-multiple-of-16 sizes
                imageio.mimsave(out_path, frames, fps=int(video_fps), macro_block_size=None)
                tag = f"sim/video/ep_{ep:02d}"
                wandb.log({
                    tag: wandb.Video(out_path, fps=int(video_fps), format="mp4"),
                    f"sim/ep_{ep:02d}/success": float(success),
                    f"sim/ep_{ep:02d}/length": int(t),
                    f"sim/ep_{ep:02d}/language": str(language),
                }, step=step_for_log)
                print(f"[eval_rlbench]   wrote {len(frames)} frames -> {out_path}  "
                      f"(also wandb '{tag}')", flush=True)

        # Intentionally NOT calling env.shutdown(): keep the CoppeliaSim
        # process alive across eval cycles to avoid the headless OpenGL
        # cross-launch texture-loss bug (project_coppeliasim_crosslaunch).
        out = {
            "sim/success_rate": float(_np.mean(successes)) if len(successes) else 0.0,
            "sim/avg_len":      float(_np.mean(lengths))   if len(lengths)   else 0.0,
            "sim/n_episodes":   int(len(successes)),
        }
        return out

