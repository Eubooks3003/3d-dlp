import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
from ttkwidgets import TickScale
from PIL import Image, ImageTk
import threading
import numpy as np
import os
import json
from tqdm import tqdm

import torch
from torchvision.transforms import ToTensor, ToPILImage
from modules.diffusion_modules import PINTDenoiser, GaussianDiffusionPINT, TrainerDiffuseDDLP
from train_diffuse_ddlp import ParticleNormalization

# keypoint
from .keypoint import KeyPoint

# models (lives at project root next to gui/)
from models import ObjectDynamicsDLP, ObjectDLP
from datasets.blender import BlenderRGBD

class GUILoad:
    def __init__(self):
        super().__init__()
        # self.model_type = model_type
        # self.model_name = model_name
        # self.use_depth = use_depth
        # self.device_name = device_name

    def load_model(self):
        if self.model_type == 'diffuse_ddlp':
            conf_path = os.path.join('./checkpoints', f'{self.model_name}', 'ddlp_hparams.json')
        else:
            conf_path = os.path.join('./checkpoints', f'{self.model_name}', 'hparams.json')
        with open(conf_path, 'r') as f:
            config = json.load(f)
        ds = config['ds']
        root = config['root']
        image_size = config['image_size']
        ch = config['ch']
        enc_channels = config['enc_channels']
        prior_channels = config['prior_channels']
        use_correlation_heatmaps = config['use_correlation_heatmaps']
        enable_enc_attn = config['enable_enc_attn']
        filtering_heuristic = config['filtering_heuristic']

        if ch == 4:
            self.use_depth = True
            

        self.ds_name = ds
        self.ds_root = root
        self.image_size = image_size

        if self.model_type == 'ddlp' or self.model_type == 'diffuse_ddlp':
            model_type = 'ddlp'
            self.model = ObjectDynamicsDLP(cdim=ch, enc_channels=enc_channels, prior_channels=prior_channels,
                                           image_size=image_size, n_kp=config['n_kp'],
                                           learned_feature_dim=config['learned_feature_dim'],
                                           pad_mode=config['pad_mode'],
                                           sigma=config['sigma'],
                                           dropout=config['dropout'], patch_size=config['patch_size'],
                                           n_kp_enc=config['n_kp_enc'],
                                           n_kp_prior=config['n_kp_prior'], kp_range=config['kp_range'],
                                           kp_activation=config['kp_activation'],
                                           anchor_s=config['anchor_s'],
                                           use_resblock=config['use_resblock'],
                                           timestep_horizon=config['timestep_horizon'],
                                           predict_delta=config['predict_delta'],
                                           scale_std=config['scale_std'],
                                           offset_std=config['offset_std'], obj_on_alpha=config['obj_on_alpha'],
                                           obj_on_beta=config['obj_on_beta'], pint_heads=config['pint_heads'],
                                           pint_layers=config['pint_layers'], pint_dim=config['pint_dim'],
                                           use_correlation_heatmaps=use_correlation_heatmaps,
                                           enable_enc_attn=enable_enc_attn, filtering_heuristic=filtering_heuristic).to(
                torch.device(f'{self.device_name}'))
        else:
            model_type = 'dlp'
            self.model = ObjectDLP(cdim=ch, enc_channels=enc_channels, prior_channels=prior_channels,
                                   image_size=image_size, n_kp=config['n_kp'],
                                   learned_feature_dim=config['learned_feature_dim'],
                                   pad_mode=config['pad_mode'],
                                   sigma=config['sigma'],
                                   dropout=config['dropout'], patch_size=config['patch_size'],
                                   n_kp_enc=config['n_kp_enc'],
                                   n_kp_prior=config['n_kp_prior'], kp_range=config['kp_range'],
                                   kp_activation=config['kp_activation'],
                                   anchor_s=config['anchor_s'],
                                   use_resblock=config['use_resblock'],
                                   scale_std=config['scale_std'],
                                   offset_std=config['offset_std'], obj_on_alpha=config['obj_on_alpha'],
                                   obj_on_beta=config['obj_on_beta'], use_tracking=config['use_tracking'],
                                   use_correlation_heatmaps=use_correlation_heatmaps,
                                   enable_enc_attn=enable_enc_attn, filtering_heuristic=filtering_heuristic,
                                   separate_depth_features=config['separate_depth_features'], depth_feature_dim=config['depth_feature_dim'],
                                   split_loss=config['split_loss'], depth_loss_ratio=config['depth_loss_ratio']).to(
                torch.device(f'{self.device_name}'))
        model_ckpt_name = os.path.join('./checkpoints', f'{self.model_name}', f'{ds}_{self.model_type}.pth')
        
        self.model.load_state_dict(torch.load(model_ckpt_name, map_location=torch.device(f'{self.device_name}')))
        self.model.eval()
        self.model.requires_grad_(False)
        print(f"loaded model from {model_ckpt_name}")

        if self.model_type == 'diffuse_ddlp':
            diff_conf_path = os.path.join('./checkpoints', f'{self.model_name}', 'diffusion_hparams.json')
            with open(diff_conf_path, 'r') as f:
                diffusion_config = json.load(f)
            diffuse_frames = diffusion_config['diffuse_frames']  # number of particle frames to generate
            lr = diffusion_config['lr']
            train_num_steps = diffusion_config['train_num_steps']
            diffusion_num_steps = diffusion_config['diffusion_num_steps']
            loss_type = diffusion_config['loss_type']
            particle_norm = diffusion_config['particle_norm']
            device = torch.device(f'{self.device_name}')
            result_dir = os.path.join('./checkpoints', f'{self.model_name}')
            diffusion_config['result_dir'] = result_dir

            features_dim = 2 + 2 + 1 + 1 + config['learned_feature_dim']
            # features: xy, scale_xy, depth, obj_on, particle features
            # total particles: n_kp + 1 for bg

            denoiser_model = PINTDenoiser(features_dim, hidden_dim=config['pint_dim'],
                                          projection_dim=config['pint_dim'],
                                          n_head=config['pint_heads'], n_layer=config['pint_layers'],
                                          block_size=diffuse_frames, dropout=0.1,
                                          predict_delta=False, positional_bias=True,
                                          max_particles=config['n_kp_enc'] + 1,
                                          self_condition=False,
                                          learned_sinusoidal_cond=False, random_fourier_features=False,
                                          learned_sinusoidal_dim=16).to(device)

            diffusion = GaussianDiffusionPINT(
                denoiser_model,
                seq_length=diffuse_frames,
                timesteps=diffusion_num_steps,  # number of steps
                sampling_timesteps=diffusion_num_steps,
                loss_type=loss_type,  # L1 or L2
                objective='pred_x0',
            ).to(device)

            particle_normalizer = ParticleNormalization(diffusion_config, mode=particle_norm).to(device)

            # expects input: [batch_size, feature_dim, seq_len]

            self.diffusion_model = TrainerDiffuseDDLP(
                diffusion,
                ddlp_model=self.model,
                diffusion_config=diffusion_config,
                particle_norm=particle_normalizer,
                train_batch_size=1,
                train_lr=lr,
                train_num_steps=train_num_steps,  # total training steps
                gradient_accumulate_every=1,  # gradient accumulation steps
                ema_decay=0.995,  # exponential moving average decay
                amp=False,  # turn on mixed precision
                seq_len=diffuse_frames,
                save_and_sample_every=1000,
                results_folder=result_dir
            )

            self.diffusion_model.load()


    def load_image(self):
        if self.img is None:
            if self.diffusion_img is None:
                self.img = Image.open(self.image_path)
                if self.ds_name == 'phyre':
                    self.img = Image.fromarray(255 - np.array(self.img))
            else:
                self.img = self.diffusion_img

        # If you have a depth PIL image, show side-by-side; else show RGB only.
        depth = getattr(self, "depth_img", None)

        if depth is not None:
            # normalize/convert if needed
            depth = depth.convert("L")
            w = self.canvas_size
            h = self.canvas_size

            rgb_resized   = self.img.resize((w, h), Image.LANCZOS)
            depth_resized = depth.resize((w, h), Image.NEAREST)

            combo = Image.new("RGB", (2 * w, h))
            combo.paste(rgb_resized, (0, 0))
            combo.paste(depth_resized.convert("RGB"), (w, 0))

            # widen the canvas to fit both
            self.canvas.config(width=2 * w, height=h)
            pil_to_show = combo
        else:
            # single image as before
            self.canvas.config(width=self.canvas_size, height=self.canvas_size)
            pil_to_show = self.img.resize((self.canvas_size, self.canvas_size), Image.LANCZOS)

        self.img_tk = ImageTk.PhotoImage(pil_to_show)
        # (Re)draw
        if hasattr(self, "img_container"):
            self.canvas.delete(self.img_container)
        self.img_container = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)


        
    
    def get_model(self):
        return self.model

    def get_diffuse_model(self):
        return self.diffuse_model