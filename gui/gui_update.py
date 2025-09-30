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

from .keypoint import KeyPoint
from gui.gui_load import GUILoad

class GUIUpdate(GUILoad):
    def __init__(self):
        super().__init__()

    def update_image(self):
        # Get updated keypoints coordinates and scales
        updated_coordinates = [kp.get_coordinates() for kp in self.keypoints]
        updated_scales = [kp.get_scale() for kp in self.keypoints]
        updated_scale_multipliers = [kp.get_scale_multiplier() for kp in self.keypoints]
        updated_features = [kp.get_features() for kp in self.keypoints]
        updated_features_indices = [kp.get_features_index() for kp in self.keypoints]
        updated_obj_ons = [kp.get_obj_on() for kp in self.keypoints]
        updated_depth_features = [kp.get_depth_features() for kp in self.keypoints]

        # Convert coordinates and scales to NumPy arrays
        updated_coordinates = np.array(updated_coordinates)  # [n_kp * n_frames, 2]
        updated_scales = np.array(updated_scales)
        updated_features = np.array(updated_features)
        updated_features_indices = np.array(updated_features_indices)
        updated_scale_multipliers = np.array(updated_scale_multipliers)
        updated_obj_ons = np.array(updated_obj_ons)
        updated_depth_features = np.array(updated_depth_features)

        # decode particles
        # TODO: Change Keypoint class to allow to use updated_depth_features and context
        decoder_dict = self.decode_particles(updated_coordinates, updated_scales, self.original_depths, updated_obj_ons,
                                             updated_features, updated_depth_features, self.original_context, self.original_bg)
        rec = decoder_dict['rec'][0]                 # usually [C,H,W]
        x  = rec.unsqueeze(0) if rec.dim() == 3 else rec  # -> [1,C,H,W]

        H, W = x.shape[-2], x.shape[-1]

        # build a displayable RGB image (use first 3 channels)
        rgb_uint8 = (x[0, :3].clamp(0, 1) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
        self.img   = Image.fromarray(rgb_uint8)


        d_vis = x[0, 3].detach().cpu().numpy()        # HxW in [0,1]
        d_vis = np.clip(d_vis, 0.0, 1.0)
        d_u8  = (d_vis * 255.0).astype(np.uint8)
        self.depth_img = Image.fromarray(d_u8, mode="L")

        # if 'ddlp' in self.model_type:
        #     images = decoder_dict['rec']
        #     self.seq_img = images.reshape(1, self.n_frames, *images.shape[1:]).clamp(0, 1).cpu()

        self.load_image()
        rgb, depth, intr, depth_is_norm, near, far = self._rgb_depth_for_o3d(x)
        if depth is not None:
            self.o3d_viewer.set_data(
                rgb_hw3=rgb,
                depth_hw=depth,
                intr=intr,
                depth_is_normalized=depth_is_norm,
                near=near, far=far,
                sample_step=1,          # increase to 2/3 if clouds are huge
                max_points=300_000      # optional cap
            )


        # Plot the updated keypoints
        for kp in self.keypoints:
            if kp.kp_label is not None:
                kp.kp_label.destroy()
            if kp.slider is not None:
                kp.slider.destroy()
                kp.slider_label.destroy()
            if kp.slider_obj is not None:
                kp.slider_obj.destroy()
                kp.slider_obj_label.destroy()
            if kp.scroller is not None:
                kp.scroller.destroy()
                kp.scroller_label.destroy()
            if kp.depth_slider is not None:
                kp.depth_slider.destroy()
                kp.depth_slider_label.destroy()
            if kp.gcanvas is not None:
                kp.gcanvas.destroy()
                kp.img_tk = None

        self.keypoints = []

        # ORIGINAL
        if self.model_type == 'dlp':
            to_pil = ToPILImage()

            gl = decoder_dict['dec_objects']            # [B,N,C,H,W] or [N,C,H,W]
            if gl.dim() == 4:
                gl = gl.unsqueeze(0)                    # -> [1,N,C,H,W]
            gl = gl.clamp(0, 1)

            B, N, C, H, W = gl.shape
            alpha = gl[:, :, :1]
            rgb   = gl[:, :, 1:4]

            # RGB (alpha-premultiplied), one PIL per kp
            rgba0 = (alpha * rgb)[0].clamp(0, 1).cpu()  # [N,3,H,W]
            glimpse_imgs_rgb = [to_pil(rgba0[i]) for i in range(N)]

            # Optional: depth previews kept separate (locals)
            depth_imgs = None
            if C == 5:
                d = gl[0, :, 4:5]                       # [N,1,H,W]
                dmin = d.amin(dim=(2, 3), keepdim=True)
                dmax = d.amax(dim=(2, 3), keepdim=True)
                d = (d - dmin) / (dmax - dmin + 1e-8)
                depth_imgs = [to_pil(d[i].cpu()) for i in range(N)]

            # Use RGB PILs for the GUI
            # TODO: Fix depth features and context
            self.add_keypoints(
                keypoints=updated_coordinates,
                scales=self.original_scales,
                scale_multipliers=updated_scale_multipliers,
                obj_ons=updated_obj_ons,
                features=self.original_features,
                features_depth=updated_depth_features,
                feature_indices=updated_features_indices,
                contexts=None,
                glimpses=glimpse_imgs_rgb,
            )
        else:
            glimpses = decoder_dict['dec_objects']
            alpha, rgb = torch.split(glimpses, [1, 3], dim=2)
            rgba = alpha * rgb  # [T, n_particles, 3, h, w]
            rgba = rgba.clamp(0, 1).permute(1, 0, 2, 3, 4).cpu()  # [n_particles, T, 3, h, w]
            glimpses = []
            for i in range(rgba.shape[0]):
                kp_glimpses = [ToPILImage()(rgba[i, j]) for j in range(rgba.shape[1])]
                glimpses.append(kp_glimpses)
            self.add_keypoints_trajectory(keypoints=updated_coordinates.reshape(-1, self.n_frames,
                                                                                updated_coordinates.shape[-1]),
                                          scales=self.original_scales,
                                          scale_multipliers=updated_scale_multipliers.reshape(-1, self.n_frames),
                                          obj_ons=updated_obj_ons.reshape(-1, self.n_frames),
                                          features=self.original_features,
                                          feature_indices=updated_features_indices.reshape(-1, self.n_frames),
                                          glimpses=glimpses)
        n_kp = len(self.keypoints)
        self.coordinates = updated_coordinates
        self.scales = updated_scales
        self.features = updated_features
        self.obj_ons = updated_obj_ons
        self.depth_features = updated_depth_features

        if self.hide_particles.get():
            self.load_image()
    
    def decode_particles(self, kp=None, scales=None, depths=None, obj_ons=None,
                        features=None, features_depth=None, context=None, bg=None):
        import numpy as np
        import torch

        T = self.n_frames
        dev = torch.device(self.device_name)

        # Defaults
        if kp is None:               kp = self.original_keypoints
        if scales is None:           scales = self.original_scales
        if depths is None:           depths = self.original_depths
        if obj_ons is None:          obj_ons = self.original_obj_ons
        if features is None:         features = self.original_features
        if features_depth is None:   features_depth = self.original_depth_features
        if bg is None:               bg = self.original_bg

        # ---- Keypoints -> [1, T, num_kp, 2]
        z_kp = self.normalize_kp(kp, normalize=True).reshape(-1, T, 2)      # [num_kp, T, 2]
        z_kp = torch.tensor(z_kp, device=dev, dtype=torch.float).permute(1, 0, 2).unsqueeze(0).contiguous()

        # ---- Scales -> [1, T, num_kp, 2]
        z_scales_np = np.asarray(scales, dtype=float).reshape(-1, T, 2)      # [num_kp, T, 2]
        z_scales = torch.tensor(z_scales_np, device=dev, dtype=torch.float).permute(1, 0, 2).unsqueeze(0).contiguous()

        # ---- Depths (scalar) -> [1, T, num_kp, 1]
        z_depths_np = np.asarray(depths, dtype=float).reshape(-1, T, 1)      # [num_kp, T, 1]
        z_depths = torch.tensor(z_depths_np, device=dev, dtype=torch.float).permute(1, 0, 2).unsqueeze(0).contiguous()

        # ---- Obj_ons (scalar) -> [1, T, num_kp, 1]  (add channel dim)
        z_obj_ons_np = np.asarray(obj_ons, dtype=float).reshape(-1, T)       # [num_kp, T]
        z_obj_ons = torch.tensor(z_obj_ons_np, device=dev, dtype=torch.float).permute(1, 0).unsqueeze(0).unsqueeze(-1).contiguous()

        # ---- RGB Features -> [1, T, num_kp, F]
        # Infer F robustly
        feats_np = np.asarray(features)
        feat_dim = feats_np.shape[-1] if feats_np.ndim >= 2 else int(self.original_features.shape[-1])
        z_features_np = feats_np.reshape(-1, T, feat_dim)                    # [num_kp, T, F]
        z_features = torch.tensor(z_features_np, device=dev, dtype=torch.float).permute(1, 0, 2).unsqueeze(0).contiguous()

        # ---- BG features per-frame -> [1, T, F_bg]
        bg_np = np.asarray(bg, dtype=float)
        if bg_np.ndim == 1:
            # tile per frame if only a single vector was provided
            bg_np = np.repeat(bg_np[None, :], T, axis=0)                     # [T, F_bg]
        else:
            bg_np = bg_np.reshape(T, -1)
        z_bg = torch.tensor(bg_np, device=dev, dtype=torch.float).unsqueeze(0).contiguous()

        # ---- Depth features (scalar per particle) -> [1, T, num_kp, 1]
        dfeats_np = np.asarray(features_depth, dtype=float).reshape(-1, T, 1)  # [num_kp, T, 1]
        z_depth_features = torch.tensor(dfeats_np, device=dev, dtype=torch.float).permute(1, 0, 2).unsqueeze(0).contiguous()

        # ---- Optional context per-frame -> [1, T, C]
        if context is not None:
            ctx_np = np.asarray(context, dtype=float)
            if ctx_np.ndim == 1:
                ctx_np = np.repeat(ctx_np[None, :], T, axis=0)               # [T, C]
            else:
                ctx_np = ctx_np.reshape(T, -1)
            z_context = torch.tensor(ctx_np, device=dev, dtype=torch.float).unsqueeze(0).contiguous()
        else:
            z_context = None

        # Diagnostics
        print("z_kp shape:", z_kp.shape)                         # [1, T, num_kp, 2]
        print("z_scales shape:", z_scales.shape)                 # [1, T, num_kp, 2]
        print("z_depths shape:", z_depths.shape)                 # [1, T, num_kp, 1]
        print("z_obj_ons shape:", z_obj_ons.shape)               # [1, T, num_kp, 1]
        print("z_features shape:", z_features.shape)             # [1, T, num_kp, F]
        print("z_depth_features shape:", z_depth_features.shape) # [1, T, num_kp, 1]
        print("z_bg_features shape:", z_bg.shape)                # [1, T, F_bg]
        print("z_context shape:", None if z_context is None else z_context.shape)

        decoder_dict = self.model.decode_all(
            z_kp,
            z_scale=z_scales,
            z_features=z_features,
            z_depth_features=z_depth_features,
            z_bg_features=z_bg,
            obj_on_sample=z_obj_ons,
            z_depth=z_depths,
            z_ctx=z_context
        )
        return decoder_dict

    
    def update_image_t(self):
        # Get updated keypoints coordinates and scales
        updated_coordinates = [kp.get_coordinates() for kp in self.keypoints]
        updated_scales = [kp.get_scale() for kp in self.keypoints]
        updated_features = [kp.get_features() for kp in self.keypoints]
        updated_obj_ons = [kp.get_obj_on() for kp in self.keypoints]

        # Convert coordinates and scales to NumPy arrays
        updated_coordinates = np.array(updated_coordinates)
        updated_scales = np.array(updated_scales)
        updated_features = np.array(updated_features)
        updated_obj_ons = np.array(updated_obj_ons)

        inter_steps = np.linspace(0, 1, num=self.num_interpolation_steps, endpoint=True)

        imgs = []

        for k in range(len(inter_steps)):
            t = inter_steps[k]
            inter_coordinates = t * np.array(updated_coordinates) + (1 - t) * self.coordinates.reshape(-1,
                                                                                                       self.coordinates.shape[
                                                                                                           -1])
            inter_scales = t * np.array(updated_scales) + (1 - t) * self.scales.reshape(-1, self.scales.shape[-1])
            inter_features = t * np.array(updated_features) + (1 - t) * self.features.reshape(-1,
                                                                                              self.features.shape[-1])
            inter_obj_ons = t * np.array(updated_obj_ons) + (1 - t) * self.obj_ons.reshape(-1, )
            decoder_dict = self.decode_particles(inter_coordinates, inter_scales, self.original_depths,
                                                 inter_obj_ons, inter_features, self.original_bg)
            imgs.append(ToPILImage()(decoder_dict['rec'][0].clamp(0, 1).cpu()))

        def func(index):
            if index < len(imgs):
                print(f'interpolation index: {index}')
                self.img = imgs[index]
                self.img_tk = ImageTk.PhotoImage(self.img.resize((self.canvas_size, self.canvas_size), Image.LANCZOS))
                # self.canvas.itemconfig(self.img_container, image=self.img_tk)
                # self.canvas.delete('all')
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)
                index = index + 1
                self.root.after(20, func, index)

        # for l in range(len(imgs)):
        #     self.img = imgs[l]
        #     self.img_tk = ImageTk.PhotoImage(self.img.resize((self.canvas_size, self.canvas_size), Image.ANTIALIAS))
        #     # self.canvas.itemconfig(self.img_container, image=self.img_tk)
        #     self.canvas.delete('all')
        #     self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)
        #     # self.load_image()
        #     self.root.after(4000, func, l)
        func(0)

        # Plot the updated keypoints
        # print(f'{updated_features_indices}')
        # if self.model_type == 'dlp':
        #     glimpses = decoder_dict['dec_objects'][0].clamp(0, 1).cpu()
        #     alpha, rgb = torch.split(glimpses, [1, 3], dim=1)
        #     rgba = alpha * rgb  # [n_particles, 3, h, w]
        #     rgba = rgba.clamp(0, 1).cpu()
        #     glimpses = [ToPILImage()(rgba[i]) for i in range(rgba.shape[0])]
        #     self.add_keypoints(keypoints=updated_coordinates,
        #                        scales=self.original_scales, scale_multipliers=updated_scale_multipliers,
        #                        obj_ons=updated_obj_ons,
        #                        features=self.original_features, feature_indices=updated_features_indices,
        #                        glimpses=glimpses)
        # else:
        #     glimpses = decoder_dict['dec_objects']
        #     alpha, rgb = torch.split(glimpses, [1, 3], dim=2)
        #     rgba = alpha * rgb  # [T, n_particles, 3, h, w]
        #     rgba = rgba.clamp(0, 1).permute(1, 0, 2, 3, 4).cpu()  # [n_particles, T, 3, h, w]
        #     glimpses = []
        #     for i in range(rgba.shape[0]):
        #         kp_glimpses = [ToPILImage()(rgba[i, j]) for j in range(rgba.shape[1])]
        #         glimpses.append(kp_glimpses)
        #     self.add_keypoints_trajectory(keypoints=updated_coordinates.reshape(-1, self.n_frames,
        #                                                                         updated_coordinates.shape[-1]),
        #                                   scales=self.original_scales,
        #                                   scale_multipliers=updated_scale_multipliers.reshape(-1, self.n_frames),
        #                                   obj_ons=updated_obj_ons.reshape(-1, self.n_frames),
        #                                   features=self.original_features,
        #                                   feature_indices=updated_features_indices.reshape(-1, self.n_frames),
        #                                   glimpses=glimpses)
        # n_kp = len(self.keypoints)
        # self.coordinates = updated_coordinates
        # self.scales = updated_scales
        # self.features = updated_features
        # self.obj_ons = updated_obj_ons

    def update_image_threaded(self):

        def check_if_ready(thread):
            if thread.is_alive():
                self.root.after(200, check_if_ready, thread)
            else:
                self.root.after(1000, self.update_image)

        trd = threading.Thread(target=self.update_image_t)
        trd.start()
        self.root.after(1, check_if_ready, trd)