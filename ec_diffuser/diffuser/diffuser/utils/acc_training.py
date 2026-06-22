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
from accelerate import Accelerator

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

class AccTrainer(object):
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

        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        )
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference


        self.accelerator = accelerator
        if self.accelerator is not None:
            self.device = self.accelerator.device
            set_global_device(self.device)
            self.model, self.optimizer, self.dataloader = self.accelerator.prepare(
                self.model, self.optimizer, self.dataloader)

        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(self.model)
        else:
            model = self.model
        self.ema_model.load_state_dict(model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.accelerator.unwrap_model(self.model))

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        timer = Timer()
        model_loss = self.accelerator.unwrap_model(self.model).loss
        for batch in self.dataloader:
        # for step in range(n_train_steps):
            with self.accelerator.accumulate(self.model):
                # batch = next(self.dataloader)
                batch = batch_to_device(batch)
                loss, infos = model_loss(*batch)
                self.accelerator.backward(loss)

                self.optimizer.step()
                self.optimizer.zero_grad()

            if self.accelerator.is_local_main_process:
                if self.step % self.update_ema_every == 0:
                    self.step_ema()

                if self.step % self.save_freq == 0:
                    label = self.step // self.label_freq * self.label_freq
                    self.save(label)

                if self.step % self.log_freq == 0:
                    infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                    print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}', flush=True)
                    wandb.log({'step':self.step, 'loss': loss, **infos})

                if self.step == 0 and self.sample_freq:
                    self.render_reference(self.n_reference)

                if self.sample_freq and self.step % self.sample_freq == 0:
                    self.render_samples()

                self.step += 1

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        if self.accelerator is not None:
            data = {
                'step': self.step,
                'model': self.accelerator.unwrap_model(self.model).state_dict(),
                'ema': self.ema_model.state_dict()
            }
        else:
            data = {
                'step': self.step,
                'model': self.model.state_dict(),
                'ema': self.ema_model.state_dict()
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

        if self.accelerator is not None:
            data = torch.load(loadpath, map_location=self.accelerator.device)
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.load_state_dict(data['model'])
        else:
            data = torch.load(loadpath)
            self.model.load_state_dict(data['model'])
        print(f'[ utils/training ] Loaded model from {loadpath}', flush=True)
        self.step = data['step']
        self.ema_model.load_state_dict(data['ema'])

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
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
        # Account for gripper_dim and bg_dim if present
        gripper_dim = getattr(self.dataset, 'gripper_dim', 0)
        bg_dim = getattr(self.dataset, 'bg_dim', 0)
        obs_start_idx = self.dataset.action_dim + gripper_dim + bg_dim
        normed_observations = trajectories[:, :, obs_start_idx:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        savepath = os.path.join(self.logdir, f'_sample-reference.png')
        self.renderer.composite(savepath, observations)

    def render_samples(self, batch_size=2, n_samples=2):
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

            ## [ n_samples x horizon x (action_dim + gripper_dim + observation_dim) ]
            samples = self.ema_model(conditions)
            trajectories = to_np(samples.trajectories)

            ## [ n_samples x horizon x observation_dim ]
            # Account for gripper_dim and bg_dim if present
            gripper_dim = getattr(self.dataset, 'gripper_dim', 0)
            bg_dim = getattr(self.dataset, 'bg_dim', 0)
            obs_start_idx = self.dataset.action_dim + gripper_dim + bg_dim
            normed_observations = trajectories[:, :, obs_start_idx:]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.png')
            self.renderer.composite(savepath, observations)
