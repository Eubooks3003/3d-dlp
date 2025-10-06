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
# from train_diffuse_ddlp import ParticleNormalization

# keypoint
from .keypoint import KeyPoint

# models (lives at project root next to gui/)
from models import DLP
from datasets.blender_ds import BlenderRGBD

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
        
        # data and general
        ds = config['ds']
        ch = config['ch']  # image channels
        image_size = config['image_size']
        root = config['root']  # dataset root

        run_prefix = config['run_prefix']
        load_model = config['load_model']
        pretrained_path = config['pretrained_path']  # path of pretrained model to load, if None, train from scratch

        device = config['device']
        if 'cuda' in device:
            device = torch.device(f'{device}' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')
        # model
        pad_mode = config['pad_mode']
        n_kp_per_patch = config['n_kp_per_patch']  # kp per patch in prior, best to leave at 1
        n_kp_prior = config['n_kp_prior']  # number of prior kp to filter for the kl
        n_kp_enc = config['n_kp_enc']  # total posterior kp
        patch_size = config['patch_size']  # prior patch size
        anchor_s = config['anchor_s']  # posterior patch/glimpse ratio of image size

        features_dist = config.get('features_dist', 'gauss')
        learned_feature_dim = config['learned_feature_dim']
        learned_bg_feature_dim = config.get('learned_bg_feature_dim', learned_feature_dim)
        n_fg_categories = config.get('n_fg_categories', 8)  # Number of foreground feature categories (if categorical)
        n_fg_classes = config.get('n_fg_classes', 4)  # Number of foreground feature classes per category
        n_bg_categories = config.get('n_bg_categories', 4)  # Number of background feature categories
        n_bg_classes = config.get('n_bg_classes', 4)

        dropout = config['dropout']
        use_resblock = config['use_resblock']

        pint_enc_layers = config['pint_enc_layers']
        pint_enc_heads = config['pint_enc_heads']

        normalize_rgb = config['normalize_rgb']
        obj_res_from_fc = config["obj_res_from_fc"]
        obj_ch_mult = config["obj_ch_mult"]
        obj_ch_mult_prior = config.get("obj_ch_mult_prior", obj_ch_mult)
        obj_base_ch = config["obj_base_ch"]
        obj_final_cnn_ch = config["obj_final_cnn_ch"]
        bg_res_from_fc = config["bg_res_from_fc"]
        bg_ch_mult = config["bg_ch_mult"]
        bg_base_ch = config["bg_base_ch"]
        bg_final_cnn_ch = config["bg_final_cnn_ch"]
        num_res_blocks = config["num_res_blocks"]
        cnn_mid_blocks = config.get('cnn_mid_blocks', False)
        mlp_hidden_dim = config.get('mlp_hidden_dim', 256)

        # optimization
        batch_size = config['batch_size']
        lr = config['lr']
        num_epochs = config['num_epochs']
        start_epoch = config.get('start_epoch', 0)
        weight_decay = config['weight_decay']
        adam_betas = config['adam_betas']
        adam_eps = config['adam_eps']
        use_scheduler = config['use_scheduler']
        scheduler_gamma = config['scheduler_gamma']
        warmup_epoch = config['warmup_epoch']
        recon_loss_type = config['recon_loss_type']
        beta_kl = config['beta_kl']
        beta_rec = config['beta_rec']
        beta_obj = config.get('beta_obj', 0.0)
        kl_balance = config['kl_balance']  # balance between visual features and the other particle attributes

        # priors
        scale_std = config['scale_std']
        offset_std = config['offset_std']
        obj_on_alpha = config['obj_on_alpha']  # transparency beta distribution "a"
        obj_on_beta = config['obj_on_beta']  # transparency beta distribution "b"

        # evaluation
        eval_epoch_freq = config['eval_epoch_freq']
        eval_im_metrics = config['eval_im_metrics']

        # visualization
        topk = min(config['topk'], config['n_kp_enc'])  # top-k particles to plot
        iou_thresh = config['iou_thresh']  # threshold for NMS for plotting bounding boxes

        #RGBD Stuff
        separate_depth_features = config["separate_depth_features"]  # use separate depth feature encoding
        depth_feature_dim = config["depth_feature_dim"]  # depth feature dimension if separate encoding
        split_loss = config["split_loss"]  # split loss into components for logging
        depth_loss_ratio = config["depth_loss_ratio"]  # weight of depth loss if split_loss is True


        if ch == 4:
            self.use_depth = True
            
        self.ds_name = ds
        self.ds_root = root
        self.image_size = image_size

        if self.ds_name == "mimicgen":
            self.cams = config['cams']

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
            model = DLP(
                cdim=ch,  # Number of input image channels
                image_size=image_size,  # Input image size (assumed square)
                normalize_rgb=normalize_rgb,  # If True, normalize RGB to [-1, 1], else keep [0, 1]

                # Keypoint and patch configuration
                n_kp_per_patch=n_kp_per_patch,  # Number of proposal/prior keypoints to extract per patch
                patch_size=patch_size,  # Size of patches for keypoint proposal network
                anchor_s=anchor_s,  # Glimpse size ratio relative to image size
                n_kp_enc=n_kp_enc,  # Number of posterior keypoints to learn
                n_kp_prior=n_kp_prior,  # Number of keypoints to filter from prior proposals

                # Network configuration
                pad_mode=pad_mode,  # Padding mode for CNNs ('zeros' or 'replicate')
                dropout=dropout,  # Dropout rate for transformers

                # Feature representation
                features_dist=features_dist,  # Distribution type for features ('gauss' or 'categorical')
                learned_feature_dim=learned_feature_dim,  # Dimension of learned visual features
                learned_bg_feature_dim=learned_bg_feature_dim,
                # Background feature dimension (if None, equals learned_feature_dim)
                n_fg_categories=n_fg_categories,  # Number of foreground feature categories (if categorical)
                n_fg_classes=n_fg_classes,  # Number of foreground feature classes per category
                n_bg_categories=n_bg_categories,  # Number of background feature categories
                n_bg_classes=n_bg_classes,  # Number of background feature classes per category

                # Prior distributions parameters
                scale_std=scale_std,  # Prior standard deviation for scale
                offset_std=offset_std,  # Prior standard deviation for offset
                obj_on_alpha=obj_on_alpha,  # Alpha parameter for transparency Beta distribution
                obj_on_beta=obj_on_beta,  # Beta parameter for transparency Beta distribution

                # Object decoder architecture
                obj_res_from_fc=obj_res_from_fc,  # Initial resolution for object encoder-decoder
                obj_ch_mult_prior=obj_ch_mult_prior,  # Channel multipliers for prior patch encoder (kp proposals)
                obj_ch_mult=obj_ch_mult,  # Channel multipliers for object encoder-decoder
                obj_base_ch=obj_base_ch,  # Base channels for object encoder-decoder
                obj_final_cnn_ch=obj_final_cnn_ch,  # Final CNN channels for object encoder-decoder

                # Background decoder architecture
                bg_res_from_fc=bg_res_from_fc,  # Initial resolution for background encoder-decoder
                bg_ch_mult=bg_ch_mult,  # Channel multipliers for background encoder-decoder
                bg_base_ch=bg_base_ch,  # Base channels for background encoder-decoder
                bg_final_cnn_ch=bg_final_cnn_ch,  # Final CNN channels for background encoder-decoder

                # Network architecture options
                use_resblock=use_resblock,  # Use residual blocks in encoders-decoders
                num_res_blocks=num_res_blocks,  # Number of residual blocks per resolution
                cnn_mid_blocks=cnn_mid_blocks,  # Use middle blocks in CNN
                mlp_hidden_dim=mlp_hidden_dim,  # Hidden dimension for MLPs

                # Particle interaction transformer (PINT) configuration
                pint_enc_layers=pint_enc_layers,  # Number of PINT encoder layers
                pint_enc_heads=pint_enc_heads,  # Number of PINT encoder attention heads

                # Dynamics configuration
                timestep_horizon=1,
                
                #RGBD Stuff
                separate_depth_features=separate_depth_features, 
                depth_feature_dim=depth_feature_dim,
                split_loss=split_loss, 
                depth_loss_ratio=depth_loss_ratio).to(
                torch.device(f'{self.device_name}'))
            self.model = model
            print("DLP model created")
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

            # particle_normalizer = ParticleNormalization(diffusion_config, mode=particle_norm).to(device)

            # expects input: [batch_size, feature_dim, seq_len]

            # self.diffusion_model = TrainerDiffuseDDLP(
            #     diffusion,
            #     ddlp_model=self.model,
            #     diffusion_config=diffusion_config,
            #     particle_norm=particle_normalizer,
            #     train_batch_size=1,
            #     train_lr=lr,
            #     train_num_steps=train_num_steps,  # total training steps
            #     gradient_accumulate_every=1,  # gradient accumulation steps
            #     ema_decay=0.995,  # exponential moving average decay
            #     amp=False,  # turn on mixed precision
            #     seq_len=diffuse_frames,
            #     save_and_sample_every=1000,
            #     results_folder=result_dir
            # )

            # self.diffusion_model.load()


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