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

        # Convert coordinates and scales to NumPy arrays
        updated_coordinates = np.array(updated_coordinates)  # [n_kp * n_frames, 2]
        updated_scales = np.array(updated_scales)
        updated_features = np.array(updated_features)
        updated_features_indices = np.array(updated_features_indices)
        updated_scale_multipliers = np.array(updated_scale_multipliers)
        updated_obj_ons = np.array(updated_obj_ons)

        if self.model_type != 'dlp':
            # assume (diffuse-)ddlp
            # copy features to all timesteps, modify coordinates to stay on the line
            updated_scale_multipliers = updated_scale_multipliers.reshape(-1, self.n_frames, 1)
            updated_scale_multipliers[:, 1:] = updated_scale_multipliers[:, :1]
            updated_scales = self.original_scales * (1 / updated_scale_multipliers)
            updated_scale_multipliers = updated_scale_multipliers.reshape(-1, )

            updated_obj_ons = updated_obj_ons.reshape(-1, self.n_frames)
            updated_obj_ons[:, 1:] = updated_obj_ons[:, :1]
            updated_obj_ons = updated_obj_ons.reshape(-1)

            updated_features_indices = updated_features_indices.reshape(-1, self.n_frames)
            updated_features = self.original_features[updated_features_indices[:, 0]]
            updated_features_indices = updated_features_indices.reshape(-1, )
            updated_features = updated_features.reshape(-1, updated_features.shape[-1])

            updated_coordinates = updated_coordinates.reshape(-1, self.n_frames, updated_coordinates.shape[-1])
            updated_coordinates = self.transform_coordinates(self.coordinates, updated_coordinates)
            # updated_coordinates = new_coor.reshape(-1, new_coor.shape[-1])

        print(updated_coordinates)
        # decode particles
        # TODO: Change Keypoint class to allow to use updated_depth_features
        decoder_dict = self.decode_particles(updated_coordinates, updated_scales, self.original_depths, updated_obj_ons,
                                             updated_features, self.original_depth_features, self.original_bg)
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
            self.add_keypoints(
                keypoints=updated_coordinates,
                scales=self.original_scales,
                scale_multipliers=updated_scale_multipliers,
                obj_ons=updated_obj_ons,
                features=self.original_features,
                features_depth=self.original_depth_features,
                feature_indices=updated_features_indices,
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

        # NEW - re-encode
        # get particles
        # particle_dict = self.get_particles()
        # if self.model_type == 'dlp':
        #     kp = particle_dict['z'][0]  # [n_kp, 2], [-1, 1]
        #     kp = self.normalize_kp(kp, normalize=False)
        #     kp = kp.cpu().numpy()
        #
        #     scales = particle_dict['z_scale'][0].cpu().numpy()
        #     depths = particle_dict['z_depth'][0].cpu().numpy()
        #     features = particle_dict['z_features'][0].cpu().numpy()
        #     obj_ons = particle_dict['obj_on'][0].cpu().numpy()
        #     bg = particle_dict['z_bg'][0].cpu().numpy()
        #
        #     glimpses = particle_dict['dec_objects_original']
        #     alpha, rgb = torch.split(glimpses, [1, 3], dim=2)
        #     rgba = alpha * rgb  # [1, n_particles, 3, h, w]
        #     rgba = rgba[0].clamp(0, 1).cpu()
        #     glimpses = [ToPILImage()(rgba[i]) for i in range(rgba.shape[0])]
        # else:
        #     # ddlp: [T, n_kp, features]
        #     kp = particle_dict['z']  # [T, n_kp, 2], [-1, 1]
        #     kp = self.normalize_kp(kp, normalize=False)
        #     kp = kp.permute(1, 0, 2).cpu().numpy()  # [n_kp, T, features]
        #
        #     scales = particle_dict['z_scale'].permute(1, 0, 2).cpu().numpy()
        #     depths = particle_dict['z_depth'].permute(1, 0, 2).cpu().numpy()
        #     features = particle_dict['z_features'].permute(1, 0, 2).cpu().numpy()
        #     obj_ons = particle_dict['obj_on'].permute(1, 0).cpu().numpy()
        #     bg = particle_dict['z_bg'].cpu().numpy()  # [T, f]
        #
        #     glimpses = particle_dict['dec_objects_original']
        #     alpha, rgb = torch.split(glimpses, [1, 3], dim=2)
        #     rgba = alpha * rgb  # [T, n_particles, 3, h, w]
        #     rgba = rgba.clamp(0, 1).permute(1, 0, 2, 3, 4).cpu()  # [n_particles, T, 3, h, w]
        #     glimpses = []
        #     for i in range(rgba.shape[0]):
        #         kp_glimpses = [ToPILImage()(rgba[i, j]) for j in range(rgba.shape[1])]
        #         glimpses.append(kp_glimpses)
        #
        # self.coordinates = kp
        # self.scales = scales
        # self.features = features
        # self.obj_ons = obj_ons
        # self.depths = self.original_depths = depths
        # self.original_bg = bg
        # # Add keypoints
        # if self.model_type == 'dlp':
        #     self.add_keypoints(keypoints=kp, scales=scales, scale_multipliers=updated_scale_multipliers,
        #                        obj_ons=obj_ons,
        #                        features=features, feature_indices=updated_features_indices,
        #                        glimpses=glimpses)
        # else:
        #     feature_indices = np.array(list(range(len(kp))))[:, None].repeat(self.n_frames, axis=1)
        #     self.add_keypoints_trajectory(keypoints=kp, scales=scales,
        #                                   scale_multipliers=self.original_scale_multiplires,
        #                                   obj_ons=obj_ons,
        #                                   features=features,
        #                                   feature_indices=feature_indices,
        #                                   glimpses=glimpses)

        if self.hide_particles.get():
            self.load_image()
    
    def decode_particles(self, kp=None, scales=None, depths=None, obj_ons=None, features=None, features_depth = None, bg=None):
        if kp is None:
            kp = self.original_keypoints
        if scales is None:
            scales = self.original_scales
        if depths is None:
            depths = self.original_depths
        if obj_ons is None:
            obj_ons = self.original_obj_ons
        if features is None:
            features = self.original_features
        if bg is None:
            bg = self.original_bg
        z_kp = self.normalize_kp(kp, normalize=True).reshape(-1, self.n_frames, 2)  # [n_kp, T, 2]
        z_kp = z_kp.permute(1, 0, 2).contiguous()  # [T, n_kp, 2]
        z_scales = torch.tensor(scales, device=torch.device(self.device_name), dtype=torch.float).reshape(-1,
                                                                                                          self.n_frames,
                                                                                                          2)
        z_scales = z_scales.permute(1, 0, 2).contiguous()  # # [T, n_kp, 2]
        z_depths = torch.tensor(depths, device=torch.device(self.device_name), dtype=torch.float).reshape(-1,
                                                                                                          self.n_frames,
                                                                                                          1)
        # [n_kp, T, 1]
        z_depths = z_depths.permute(1, 0, 2).contiguous()  # [T, n_kp, 1]
        z_obj_ons = torch.tensor(obj_ons, device=torch.device(self.device_name), dtype=torch.float).reshape(-1,
                                                                                                            self.n_frames)  # [n_kp, T]
        z_obj_ons = z_obj_ons.permute(1, 0).contiguous()  # [T, n_kp]
        z_features = torch.tensor(features,
                                  device=torch.device(self.device_name), dtype=torch.float).reshape(-1, self.n_frames,
                                                                                                    self.original_features.shape[
                                                                                                        -1])
        z_features = z_features.permute(1, 0, 2).contiguous()  # [T, n_kp, F]
        z_bg = torch.tensor(bg, device=torch.device(self.device_name), dtype=torch.float).reshape(self.n_frames,
                                                                                                  self.original_bg.shape[
                                                                                                      -1])  # [T, F]
        z_features_depth = torch.tensor(features_depth,
                                  device=torch.device(self.device_name), dtype=torch.float).reshape(-1, self.n_frames,
                                                                                                    self.original_depth_features.shape[
                                                                                                        -1])
        z_features_depth = z_features_depth.permute(1, 0, 2).contiguous()
        decoder_dict = self.model.decode_all(z_kp, z_features, z_bg, z_obj_ons, z_depth=z_depths, z_scale=z_scales, z_features_depth=z_features_depth)
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