
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

COLORS = ['#FF0000', '#0000FF', '#FFFF00', '#00FF00', '#800080', '#FFA500',
          '#FFC0CB', '#008000', '#FF69B4', '#800000', '#FFD700', '#000080',
          '#808080', '#FF7F50', '#FFA07A', '#FF4500', '#00FFFF', '#00008B',
          '#FF1493', '#008080', '#8B0000', '#7B68EE', '#228B22', '#20B2AA',
          '#DAA520', '#4B0082', '#FF00FF', '#00FF7F', '#1E90FF', '#B22222',
          '#C71585', '#ADFF2F', '#F0E68C', '#FF8C00', '#FFDAB9', '#FA8072',
          '#FFFFE0', '#00BFFF', '#2E8B57', '#FAEBD7', '#FF4500', '#D8BFD8']


class KeyPoint:
    def __init__(self, canvas, x, y, scale, index, feature_index, features=(None,), features_depth = (None, ), context=None, scale_multiplier=1.0, obj_on=1.0,
                 glimpse=None, timestep=0, to_kp=None, gui=None):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.scale = scale  # [2, ]
        self.index = index
        self.feature_index = feature_index
        self.features = features[feature_index]
        self.features_dict = {f'{i}': features[i] for i in range(len(features))}
        self.features_depth = features_depth[feature_index]
        self.depth_features_dict = {f'{i}': features_depth[i] for i in range(len(features_depth))}
        self.context = context
        self.timestep = timestep
        self.to_kp = to_kp  # (x, y)
        self.gui = gui
        self.radius = 5 if (timestep == 0 or timestep == self.gui.n_frames - 1) else 2
        self.scale_multiplier = scale_multiplier
        self.obj_on = obj_on
        self.glimpse = glimpse
        self.glimpse_size = 100

        text = f"{self.index}[t={self.timestep}]" if self.timestep == 0 else f''
        self.text = canvas.create_text(
            x + 10,
            y + 10,
            text=text,
            anchor=tk.NW,
            fill='black',
            tags='keypoint'
        )
        self.item = canvas.create_oval(
            x - self.radius,
            y - self.radius,
            x + self.radius,
            y + self.radius,
            fill=COLORS[index % len(COLORS)],
            tags='keypoint'
        )

        if self.to_kp is not None:
            self.arrow = self.canvas.create_line(x, y, self.to_kp[0], self.to_kp[1], arrow=tk.LAST,
                                                 fill=COLORS[index % len(COLORS)],
                                                 tags='keypoint')

        self.canvas.tag_bind(self.item, '<ButtonPress-1>', self.on_press)
        self.canvas.tag_bind(self.item, '<B1-Motion>', self.on_drag)
        self.canvas.tag_bind(self.item, '<ButtonRelease-1>', self.on_release)

        self.kp_label = None
        self.slider, self.slider_label = None, None
        self.scroller, self.scroller_label = None, None
        self.slider_obj, self.slider_obj_label = None, None
        self.gcanvas = None
        self.img_tk = None
        self.selected = False

    def select_kp(self):
        self.canvas.itemconfigure(self.item, width=3, outline='black')
        self.selected = True
        self.gui.selected_keypoints.append(self)

    def deselect_kp(self):
        self.selected = False
        self.canvas.itemconfigure(self.item, width=1, outline='')

    def on_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        for kp in self.gui.selected_keypoints:
            kp.start_x = event.x
            kp.start_y = event.y
        if self.gui is not None:
            self.gui.close_all()
        if self.kp_label is not None:
            self.kp_label.destroy()
        if self.slider is not None:
            self.slider.destroy()
            self.slider_label.destroy()
        if self.slider_obj is not None:
            self.slider_obj.destroy()
            self.slider_obj_label.destroy()
        if self.scroller is not None:
            self.scroller.destroy()
            self.scroller_label.destroy()
        if self.gcanvas is not None:
            self.gcanvas.destroy()
            self.img_tk = None
        if not self.selected:
            self.select_kp()
            # Create canvas to display image
            self.gcanvas = tk.Canvas(self.gui.root, width=self.glimpse_size, height=self.glimpse_size)
            # self.canvas.pack()
            self.gcanvas.grid(row=6, column=1)
            self.img_tk = ImageTk.PhotoImage(
                self.glimpse.resize((self.glimpse_size, self.glimpse_size), Image.LANCZOS))
            self.gcanvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

            self.kp_label = ttk.Label(
                self.gui.root,
                text=f'particle #{self.index} [t={self.timestep}]', font=('Arial', 10), background='#818485',
                foreground='black'
            )
            # self.kp_label.pack(side=tk.TOP)
            self.kp_label.grid(row=6, column=0)
            if self.timestep == 0:
                self.slider_label = ttk.Label(
                    self.gui.root,
                    text='scale:', font=('Arial', 10), background='#818485', foreground='black'
                )
                # self.slider_label.pack(side=tk.TOP)
                self.slider_label.grid(row=7, column=1)
                self.slider = TickScale(
                    self.gui.root,
                    from_=0.1,
                    to=3.0,
                    orient=tk.HORIZONTAL,
                    length=150,
                    # resolution=0.1,
                    # showvalue=True,
                    command=self.on_scale_change
                )

                self.slider.set(self.scale_multiplier)
                # self.slider.pack(side=tk.TOP)
                self.slider.grid(row=8, column=1)

                self.slider_obj_label = ttk.Label(
                    self.gui.root,
                    text='transparency:', font=('Arial', 10), background='#818485', foreground='black'
                )
                # self.slider_obj_label.pack(side=tk.TOP)
                self.slider_obj_label.grid(row=9, column=1)

                self.slider_obj = TickScale(
                    self.gui.root,
                    from_=0.0,
                    to=1.0,
                    orient=tk.HORIZONTAL,
                    length=150,
                    # resolution=0.1,
                    # showvalue=True,
                    command=self.on_obj_on_change
                )
                self.slider_obj.set(self.obj_on)
                # self.slider_obj.pack(side=tk.TOP)
                self.slider_obj.grid(row=10, column=1)

                # Create and place the features menu
                self.scroller_label = ttk.Label(
                    self.gui.root,
                    text='visual appearance:', font=('Arial', 10), background='#818485', foreground='black'
                )
                # self.scroller_label.pack(side=tk.TOP)
                self.scroller_label.grid(row=11, column=1)

                feat_var = tk.StringVar()
                feat_var.set(f'{self.feature_index}')  # Set the default value
                self.scroller = ttk.OptionMenu(
                    self.gui.root,
                    feat_var,
                    f'{self.feature_index}',  # Set the default values
                    *list(self.features_dict.keys()), command=self.on_features_change, style='my.TMenubutton'
                )
                self.scroller['menu'].configure(font=('Arial', 10))
                # self.scroller.pack(side=tk.TOP, pady=10)
                self.scroller.grid(row=12, column=1)


                # Depth Features 
                self.scroller_label = ttk.Label(
                    self.gui.root,
                    text='Depth Features:', font=('Arial', 10), background='#818485', foreground='black'
                )
                # self.scroller_label.pack(side=tk.TOP)
                self.scroller_label.grid(row=11, column=1)

                feat_var = tk.StringVar()
                feat_var.set(f'{self.feature_index}')  # Set the default value
                self.scroller = ttk.OptionMenu(
                    self.gui.root,
                    feat_var,
                    f'{self.feature_index}',  # Set the default values
                    *list(self.depth_features_dict.keys()), command=self.on_depth_features_change, style='my.TMenubutton'
                )
                self.scroller['menu'].configure(font=('Arial', 10))
                # self.scroller.pack(side=tk.TOP, pady=10)
                self.scroller.grid(row=12, column=1)

    def on_drag(self, event):
        if self.kp_label is not None:
            self.kp_label.destroy()
        if self.slider is not None:
            self.slider.destroy()
            self.slider_label.destroy()
        if self.slider_obj is not None:
            self.slider_obj.destroy()
            self.slider_obj_label.destroy()
        if self.scroller is not None:
            self.scroller.destroy()
            self.scroller_label.destroy()
        if self.gcanvas is not None:
            self.gcanvas.destroy()
            self.img_tk = None
        # if self.selected:

        for kp in self.gui.selected_keypoints:
            if kp.timestep == 0 or kp.timestep == self.gui.n_frames - 1:
                dx = event.x - kp.start_x
                dy = event.y - kp.start_y
                self.canvas.move(kp.item, dx, dy)
                self.canvas.move(kp.text, dx, dy)
                if kp.to_kp is not None:
                    self.canvas.move(kp.arrow, dx, dy)
                kp.start_x = event.x
                kp.start_y = event.y

        # dx = event.x - self.start_x
        # dy = event.y - self.start_y
        # self.canvas.move(self.item, dx, dy)
        # self.canvas.move(self.text, dx, dy)
        # self.start_x = event.x
        # self.start_y = event.y

    def on_release(self, event):
        self.deselect_kp()

    def get_coordinates(self):
        coords = self.canvas.coords(self.item)  # [4,]
        x = coords[0] + self.radius
        y = coords[1] + self.radius
        kp = np.array([x, y])
        return kp

    def on_scale_change(self, scale):
        self.scale_multiplier = float(scale)
        # self.canvas.itemconfigure(self.text, text=f"{self.index}[t={self.timestep}]")
        for kp in self.gui.selected_keypoints:
            kp.scale_multiplier = float(scale)
            # self.canvas.itemconfigure(kp.text, text=f"{kp.index}[t={self.timestep}]")
            print(f'{kp.index}: scale-multiplier: {kp.scale_multiplier}')

    def on_obj_on_change(self, obj_on):
        self.obj_on = float(obj_on)
        for kp in self.gui.selected_keypoints:
            kp.obj_on = float(obj_on)
            print(f'{kp.index}: obj_on: {kp.obj_on}')

    def on_features_change(self, index):
        self.feature_index = int(index)
        self.features = self.features_dict[f'{index}']
        for kp in self.gui.selected_keypoints:
            kp.feature_index = int(index)
            kp.features = kp.features_dict[f'{index}']
            print(f'{kp.index}: features: {kp.features}')
    
    def on_depth_features_change(self, index):
        self.feature_index = int(index)
        self.depth_features = self.depth_features_dict[f'{index}']
        for kp in self.gui.selected_keypoints:
            kp.feature_index = int(index)
            kp.depth_features = kp.depth_features_dict[f'{index}']
            print(f'{kp.index}: features: {kp.depth_features_dict}')


    def get_scale(self):
        return self.scale * (1 / self.scale_multiplier)

    def get_scale_multiplier(self):
        return self.scale_multiplier

    def get_obj_on(self):
        return self.obj_on

    def get_features(self):
        return self.features

    def get_features_index(self):
        return self.feature_index
