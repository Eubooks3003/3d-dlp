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

class GUIReset:
    def __init__(self):
        super().__init__()

    def reset_buttons(self):
        self.clear_buttons()
        n_kp = len(self.keypoints)
        # Create buttons
        self.update_btn = ttk.Button(self.root, text="Update", command=self.get_update_img_func, style='my.TButton')
        # self.update_btn.pack(side=tk.BOTTOM, padx=10, pady=10, ipadx=10, ipady=10)
        self.update_btn.grid(row=5, column=1, padx=10, pady=10, ipadx=10, ipady=10)

        self.reset_btn = ttk.Button(self.root, text="Reset", command=self.reset, style='my.TButton')
        # self.reset_btn.pack(side=tk.TOP, padx=10, pady=10, ipadx=10, ipady=10)
        self.reset_btn.grid(row=3, column=0, padx=10, pady=10, ipadx=10, ipady=10)

        if 'ddlp' in self.model_type:
            self.play_btn = ttk.Button(self.root, text="Play", command=self.play_video, style='my.TButton')
            # self.play_btn.pack(side=tk.BOTTOM, padx=10, pady=10, ipadx=10, ipady=10)
            self.play_btn.grid(row=3, column=2, padx=10, pady=10, ipadx=10, ipady=10)
    def reset(self):
        print("RESETTING")
        self.reset_image()

        if self.use_depth:
            print("RESETTING PLY")
            self.reset_ply()
    def reset_image(self):
        print("RESETTING IMAGE")
        # Clear the canvas and keypoints list
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

        # self.update_btn.destroy()
        # self.reset_btn.destroy()
        self.canvas.delete('all')
        self.canvas.destroy()
        self.keypoints = []
        self.selected_keypoints = []
        self.selection_rect = None
        self.selection_start = None
        self.reset_buttons()
        self.hide_particles = tk.BooleanVar(value=False)
        self.hide_kp_label = ttk.Label(
            self.root,
            text='  hide particles:  ', font=('Arial', 10), background='#818485', foreground='black'
        )
        self.hide_kp_label.grid(row=4, column=0)
        self.hide_kp = ttk.Checkbutton(self.root, variable=self.hide_particles, command=self.on_hide_kp)
        self.hide_kp.grid(row=5, column=0)
        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size)
        # self.canvas.pack()
        self.canvas.grid(row=3, column=1)
        # Reload and display the original image
        self.img = None
        self.depth_img = None
        self.load_image()
        # Add keypoints again if needed
        if self.model_type == 'dlp':
            self.add_keypoints(keypoints=self.original_keypoints, scales=self.original_scales,
                               scale_multipliers=self.original_scale_multiplires, obj_ons=self.original_obj_ons,
                               features=self.original_features, features_depth=self.original_depth_features, feature_indices=self.original_feature_indices,
                               glimpses=self.original_glimpses)
        else:
            self.add_keypoints_trajectory(keypoints=self.original_keypoints, scales=self.original_scales,
                                          scale_multipliers=self.original_scale_multiplires,
                                          obj_ons=self.original_obj_ons,
                                          features=self.original_features,
                                          feature_indices=self.original_feature_indices,
                                          glimpses=self.original_glimpses)
        self.coordinates = self.original_keypoints
        self.scales = self.original_scales
        self.features = self.original_features
        self.obj_ons = self.original_obj_ons
        self.original_scale_multiplires = np.ones_like(self.obj_ons)
        self.depths = self.original_depths
        self.canvas.bind('<ButtonPress-1>', self.on_canvas_press)
        self.canvas.bind('<B1-Motion>', self.on_canvas_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_canvas_release)
    
    def reset_ply(self):
        rgb, depth, intr, depth_is_norm, near, far = self._rgb_depth_for_o3d(self.x_original)
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
