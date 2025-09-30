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

from models import DLP
from datasets.blender_ds import BlenderRGBD

from .keypoint import KeyPoint
from gui.gui_load import GUILoad

class GUISelect(GUILoad):
    def __init__(self):
        super().__init__()
    
    def on_model_select(self, dir_name):
        # clear
        self.clear_screen()
        self.model_name = dir_name
        if dir_name != '':
            # self.model_type = 'ddlp' if 'ddlp' in dir_name else 'dlp'
            if 'diffuse' in dir_name:
                self.model_type = 'diffuse_ddlp'
            elif 'ddlp' in dir_name:
                self.model_type = 'ddlp'
            else:
                self.model_type = 'dlp'

            if self.model_type == 'ddlp' or self.model_type == 'diffuse_ddlp':
                self.n_frames = 4
            else:
                self.n_frames = 1
            self.load_model()

            if self.model_type != 'diffuse_ddlp':
                # locate example
                # self.example_dir = f'./assets/{self.ds_name}'
                # # self.example_dir = f'/media/newhd/data/obj3d/train'
                # if not os.path.exists(self.example_dir):
                #     raise SystemExit(f'Examples for dataset {self.ds_name} not found.'
                #                      f' Please make sure to put each example in its own dir under {self.example_dir}.'
                #                      f' For example: root -> assets > {self.example_dir} -> 1 -> *.png'
                #                      f' The directory should include image files.')
                # self.available_examples = os.listdir(self.example_dir)
                # if len(self.available_examples) == 0:
                #     raise SystemExit(f'Examples for dataset {self.ds_name} not found.'
                #                      f' Please make sure to put each example in its own dir under {self.example_dir}.'
                #                      f' For example: root -> assets > {self.example_dir} -> 1 -> *.png'
                #                      f' The directory should include image files.')
                # print(f'available examples: {self.available_examples}')

                if not self.use_depth:
                    # example scroller

                    self.scroller_example_label = ttk.Label(
                        self.root,
                        text='example:', font=('Arial', 10), background='#818485', foreground='black'
                    )
                    # self.scroller_label.pack(side=tk.TOP)
                    self.scroller_example_label.grid(row=0, column=1)

                    example_var = tk.StringVar()
                    example_var.set(f'{self.available_examples[0]}')  # Set the default value
                    self.scroller_example = ttk.OptionMenu(
                        self.root,
                        example_var,
                        f'{self.available_examples[0]}',  # Set the default values
                        *self.available_examples, command=self.on_example_select, style='my.TMenubutton'
                    )
                    self.scroller_example['menu'].configure(font=('Arial', 10))
                    # self.scroller.pack(side=tk.TOP, pady=10)
                    self.scroller_example.grid(row=1, column=1, pady=10)
                else:
                    available_depth_choices = [str(x) for x in range(1, 11)]

                    self.scroller_example_label = ttk.Label(
                        self.root, text='Depth Datasets:', font=('Arial', 10),
                        background='#818485', foreground='black'
                    )
                    self.scroller_example_label.grid(row=0, column=1)

                    # keep the var as an attribute
                    self.example_var = tk.StringVar(value=available_depth_choices[0])

                    self.scroller_depth_example = ttk.OptionMenu(
                        self.root,
                        self.example_var,
                        available_depth_choices[0],      # default shown
                        *available_depth_choices,        # <-- UNPACK the list
                        command=self.on_example_select_depth,
                        style='my.TMenubutton'
                    )

                    self.scroller_depth_example['menu'].configure(font=('Arial', 10))
                    # self.scroller.pack(side=tk.TOP, pady=10)
                    self.scroller_depth_example.grid(row=1, column=1, pady=10)
            else:
                # diffuse-ddlp
                # generate button
                self.gen_btn = ttk.Button(self.root, text="Generate", command=self.on_generate_press, style='my.TButton')
                self.gen_btn.grid(row=1, column=1, padx=10, pady=1, ipadx=2, ipady=1)

            self.anim_cbox_label = ttk.Label(
                self.root,
                text='   animate:   ', font=('Arial', 10), background='#818485', foreground='black'
            )
            # self.scroller_label.pack(side=tk.TOP)
            self.animate_transitions = tk.BooleanVar(value=False)
            self.anim_cbox_label.grid(row=0, column=2)
            self.anim_cbox = ttk.Checkbutton(self.root, variable=self.animate_transitions, command=self.on_anim_checkbox)
            self.anim_cbox.grid(row=1, column=2)

    def _reset_ui(self):
        self.clear_buttons()
        self.selection_rect = None
        self.selection_start = None
        self.image_path = None
        self.img = None
        self.depth_img = None
        if self.canvas is not None:
            self.close_all()
            self.canvas.destroy()

    def _create_controls(self):
        self.update_btn = ttk.Button(self.root, text="Update", command=self.get_update_img_func, style='my.TButton')
        self.update_btn.grid(row=5, column=1, padx=10, pady=10, ipadx=10, ipady=10)

        self.reset_btn = ttk.Button(self.root, text="Reset", command=self.reset, style='my.TButton')
        self.reset_btn.grid(row=3, column=0, padx=10, pady=10, ipadx=10, ipady=10)

        if 'ddlp' in self.model_type:
            self.play_btn = ttk.Button(self.root, text="Play", command=self.play_video, style='my.TButton')
            self.play_btn.grid(row=3, column=2, padx=10, pady=10, ipadx=10, ipady=10)

        self.hide_particles = tk.BooleanVar(value=False)
        self.hide_kp_label = ttk.Label(self.root, text='  hide particles:  ', font=('Arial', 10),
                                    background='#818485', foreground='black')
        self.hide_kp_label.grid(row=4, column=0)
        self.hide_kp = ttk.Checkbutton(self.root, variable=self.hide_particles, command=self.on_hide_kp)
        self.hide_kp.grid(row=5, column=0)

    def _make_canvas(self):
        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size)
        self.canvas.grid(row=3, column=1)
        self.load_image()  # uses self.img / optional self.depth_img
        self.canvas.bind('<ButtonPress-1>', self.on_canvas_press)
        self.canvas.bind('<B1-Motion>', self.on_canvas_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_canvas_release)

    # --- 2) shared particle unpack + glimpse building ---
    def _unpack_particles_and_glimpses(self, particle_dict):
        if self.model_type == 'dlp':
            kp = self.normalize_kp(particle_dict['z'][0], normalize=False).cpu().numpy().squeeze(0)  # [N,2]
            scales   = particle_dict['z_scale'][0].cpu().numpy()
            depths   = particle_dict['z_depth'][0].cpu().numpy()
            features = particle_dict['z_features'][0].cpu().numpy()
            obj_ons  = particle_dict['obj_on'][0].cpu().numpy()
            bg       = particle_dict['z_bg_features'][0].cpu().numpy()
            depth_features = particle_dict['z_depth_features'][0].cpu().numpy()
            context = particle_dict['z_context']

            # print the size of all the particle_dict entries above without 0 indexing
            print("Particle Dict z_scale shape:", particle_dict['z_scale'].shape)
            print("Particle Dict z_depth shape:", particle_dict['z_depth'].shape)
            print("Particle Dict z_features shape:", particle_dict['z_features'].shape)
            print("Particle Dict obj_on shape:", particle_dict['obj_on'].shape)
            print("Particle Dict z_bg_features shape:", particle_dict['z_bg_features'].shape)
            print("Particle Dict z_depth_features shape:", particle_dict['z_depth_features'].shape)
            print("Particle Dict z_context shape:", None if context is None else context.shape)

            if context is not None:
                context = context[0].cpu().numpy()
            # depth_features = None if depth_features is None else depth_features.cpu().numpy()

            gl = particle_dict['dec_objects_original']       # [B=1,N,C,H,W]
            B, N, C, H, W = gl.shape
            alpha, rgb = gl[:, :, :1], gl[:, :, 1:4]
            rgba0 = (alpha * rgb)[0].clamp(0, 1).cpu()       # [N,3,H,W]
            rgb_glimpses = [ToPILImage()(rgba0[i]) for i in range(N)]
            self.original_glimpses = rgb_glimpses
            depth_glimpses = None
            if C == 5:
                d = gl[0, :, 4:5]
                dmin = d.amin(dim=(2,3), keepdim=True); dmax = d.amax(dim=(2,3), keepdim=True)
                d = (d - dmin) / (dmax - dmin + 1e-8)
                depth_glimpses = [ToPILImage()(d[i].cpu()) for i in range(N)]

            return dict(
                kp=kp, scales=scales, depths=depths, features=features, depth_features=depth_features,
                obj_ons=obj_ons, bg=bg, rgb_glimpses=rgb_glimpses, depth_glimpses=depth_glimpses,
                feature_indices=list(range(len(kp))), context=context
            )
        else:
            # DDLP temporal
            kp = self.normalize_kp(particle_dict['z'], normalize=False).permute(1,0,2).cpu().numpy()  # [N,T,2]
            scales   = particle_dict['z_scale'].permute(1,0,2).cpu().numpy()
            depths   = particle_dict['z_depth'].permute(1,0,2).cpu().numpy()
            features = particle_dict['z_features'].permute(1,0,2).cpu().numpy()
            depth_features = particle_dict.get('z_depth_features')
            depth_features = None if depth_features is None else depth_features.permute(1,0,2).cpu().numpy()
            obj_ons  = particle_dict['obj_on'].permute(1,0).cpu().numpy()
            bg       = particle_dict['z_bg'].cpu().numpy()

            gl = particle_dict['dec_objects_original']       # [T,N,C,H,W]
            alpha, rgb = gl[:, :, :1], gl[:, :, 1:4]
            rgba = (alpha * rgb).clamp(0,1).permute(1,0,2,3,4).cpu()   # [N,T,3,H,W]
            rgb_glimpses = [[ToPILImage()(rgba[i,j]) for j in range(rgba.shape[1])] for i in range(rgba.shape[0])]
            # (depth glimpses optional—add if you decide to use them in DDLP)
            depth_glimpses = None

            return dict(
                kp=kp, scales=scales, depths=depths, features=features, depth_features=depth_features,
                obj_ons=obj_ons, bg=bg, rgb_glimpses=rgb_glimpses, depth_glimpses=depth_glimpses,
                feature_indices=np.array(list(range(len(kp))))[:,None].repeat(self.n_frames, axis=1)
            )

    # --- 3) shared “apply to GUI” step ---
    def _apply_particle_state(self, state):
        self.keypoints = []
        self.selected_keypoints = []
        # Squeeze out batch dim
        self.coordinates = self.original_keypoints = state['kp']
        self.scales     = self.original_scales    = state['scales'][0]
        self.depths     = self.original_depths    = state['depths']
        self.features   = self.original_features  = state['features'][0]
        self.obj_ons  = self.original_obj_ons  = np.asarray(state['obj_ons'][0]).squeeze().reshape(-1).astype(float)
        self.features_depth = self.original_depth_features = np.asarray(state['depth_features'][0]).squeeze().reshape(-1).astype(float)
        self.original_bg = state['bg']
        self.context = self.original_context = state['context']
        if self.context is not None:
            self.context = self.original_context = state['context'][0]
        N = self.obj_ons.shape[0]
        self.original_scale_multipliers = np.ones(N, dtype=float)



        # Print coordinates shape
        print("Coordinates shape:", self.coordinates.shape)
        print("Shapes :", self.coordinates.shape, self.scales.shape, self.obj_ons.shape, self.features.shape)
        # Print shape for everything
        # TODO: make the squeezing better

        print("Features shape:", self.features.shape)
        print("Depth Features shape:", self.features_depth.shape)
        print("Scales shape:", self.scales.shape)
        print("Obj_ons shape:", self.obj_ons.shape)
        print("Scale multipliers shape:", self.original_scale_multipliers.shape)
        if self.model_type == 'dlp':
            self.original_feature_indices = state['feature_indices']
            print("Feature indices shape:", self.original_feature_indices)
            print("Glimpses length:", len(state['rgb_glimpses']))
            self.add_keypoints(
                keypoints=self.coordinates,
                scales=self.scales,
                scale_multipliers=self.original_scale_multipliers,
                obj_ons=self.obj_ons,
                features=self.features,
                features_depth=self.features_depth,
                contexts=self.context,
                feature_indices=self.original_feature_indices,
                glimpses=state['rgb_glimpses'],   # single PIL per kp
            )
        else:
            self.original_feature_indices = state['feature_indices']
            self.add_keypoints_trajectory(
                keypoints=self.coordinates,
                scales=self.scales,
                scale_multipliers=self.original_scale_multipliers,
                obj_ons=self.obj_ons,
                features=self.features,
                feature_indices=self.original_feature_indices,
                glimpses=state['rgb_glimpses'],   # list of [T] PILs per kp
            )

        self.n_kp = len(self.keypoints)

    def on_example_select(self, example_dir):
        self._reset_ui()
        self.chosen_example = example_dir

        # resolve image sequence from folder
        files = sorted(os.listdir(os.path.join(self.example_dir, example_dir)),
                    key=lambda x: int(x.split('.')[-2].split('_')[-1]))
        if not files: raise SystemExit(f'Examples for {self.ds_name} not found…')
        if len(files) < self.n_frames: raise SystemExit(f'Not enough frames for DDLP…')

        self.image_path = os.path.join(self.example_dir, example_dir, files[0])
        self.seq_path   = [os.path.join(self.example_dir, example_dir, files[t]) for t in range(self.n_frames)]

        self._create_controls()
        self._make_canvas()                 # calls load_image() which uses self.image_path / self.img

        particle_dict = self.get_particles()
        state = self._unpack_particles_and_glimpses(particle_dict)
        self._apply_particle_state(state)

    def on_example_select_depth(self, choice):
        self._reset_ui()

        # lazy-init ds
        if not hasattr(self, "_blender_ds"):
            self._blender_ds = BlenderRGBD(root=self.ds_root, mode="train", image_size=self.image_size)
        ds = self._blender_ds

        # choose item
        s = str(choice)
        if s.isdigit(): pos = max(0, min(int(s)-1, len(ds)-1))
        else:
            try: gid = int(os.path.splitext(s)[0].split("_")[-1]); pos = ds.id2pos.get(gid, 0)
            except Exception: pos = 0

        # fetch + normalize exactly like training
        x4chw, gid_t = ds[pos]
        gid = int(gid_t.item()) if hasattr(gid_t, "item") else int(gid_t)
        x = x4chw.unsqueeze(0).float()                                   # [1,4,H,W]
        near, far = ds.depth_range
        if x[:, :3].max().item() > 1.5: x[:, :3] /= 255.0
        if far > near:
            x[:, 3:] = (x[:, 3:] - near) / (far - near); x[:, 3:].clamp_(0,1)
        else:
            d = x[:, 3:]; d = d - d.amin(dim=(2,3), keepdim=True); d = d / (d.amax(dim=(2,3), keepdim=True)+1e-6); x[:,3:] = d

        # TODO: Set up an internal Dataset class and save all these params to that
        # stash GUI bits (rgb + depth PIL)
        self.x_prepped = x.to(torch.device(self.device_name))
        self.x_original = self.x_prepped # For resetting PLYs
        rgb_uint8 = (x[0, :3].clamp(0,1) * 255.0).byte().permute(1,2,0).cpu().numpy()
        self.img = Image.fromarray(rgb_uint8)
        d_vis = (x[0, 3].detach().cpu().numpy() * 255.0).astype(np.uint8)
        self.depth_range = ds.depth_range
        self.depth_img = Image.fromarray(d_vis, mode="L")
        self.K_intr = ds.get_intrinsics(pos) # Get Intrinsics
        self.current_pos = pos; self.current_gid = gid
        self.image_path = os.path.join(self.ds_root, "rgb", f"rgb_{gid:05d}.png")
        self.seq_path = [self.image_path for _ in range(getattr(self, "n_frames", 1))]
        self.chosen_example = f"gid={gid} (pos {pos})"

        self._create_controls()
        self._make_canvas()                 # load_image() will show side-by-side if depth_img is set

        particle_dict = self.get_particles()
        state = self._unpack_particles_and_glimpses(particle_dict)
        self._apply_particle_state(state)

        #Open3D Viewer
        rgb, depth, intr, depth_is_norm, near, far = self._rgb_depth_for_o3d(self.x_prepped)
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

    
    def _rgb_depth_for_o3d(self, x):
        """
        Returns (rgb_hw3_float, depth_hw_float, intr, depth_is_normalized, near, far)
        - rgb in [0,1]
        - depth_hw in meters if you have metric; else normalized [0,1] with (near, far)
        """
        H, W = x.shape[-2], x.shape[-1]
        intr = self.K_intr  # {"fx","fy","cx","cy"} floats

        # RGB from x_prepped (already scaled to [0,1] earlier)
        rgb = x[0, :3].clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
        depth = x[0, 3].detach().cpu().numpy()

        # Depth: if channel exists, use it; otherwise None
        depth_is_normalized = True
        near, far = getattr(self, "depth_range", (None, None))

        # if self.x.shape[1] > 3:
        #     # you normalized depth earlier to [0,1] using near/far if you had them
        #     depth = self.x[0, 3].detach().cpu().numpy()
        #     depth_is_normalized = (near is not None and far is not None)
        # else:
        #     depth = None

        return rgb, depth, intr, depth_is_normalized, near, far

