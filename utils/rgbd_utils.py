
import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils

def _unwrap_base_dataset(ds):
    """Peel off Subset/Concat wrappers to reach the underlying dataset object."""
    while hasattr(ds, 'dataset'):
        ds = ds.dataset
    return ds
def get_depth_range(dataloader):
    base_ds = _unwrap_base_dataset(dataloader.dataset)
    try:
        depth_range = getattr(base_ds, "depth_range", None)
        if depth_range is None:
            depth_range = (0.2, 2.0)
        else:
            if torch.is_tensor(depth_range):
                depth_range = (float(depth_range[0].item()), float(depth_range[1].item()))
            else:
                depth_range = (float(depth_range[0]), float(depth_range[1]))
    except Exception:
        depth_range = (0.2, 2.0)
    near, far = depth_range

    return near, far

def normalize_rgbd(x, *, near=None, far=None, rgb_mode="unit", depth_clip=True):
    """
    x: [B, T, C, H, W] or [B, 1, C, H, W] (your code already ensures a time dim)
    - RGB in ch 0..2 -> [0,1] if 0..255
    - Depth in ch 3   -> [0,1] using near/far if provided, else per-sample min-max
    """
    assert x.dim() == 5, f"expected [B,T,C,H,W], got {tuple(x.shape)}"
    B, T, C, H, W = x.shape
    x = x.float()

    # RGB → [0,1]
    if C >= 3 and x[:, :, :3].amax().item() > 1.5:
        x[:, :, :3] = x[:, :, :3] / 255.0
    if rgb_mode == "imagenet":
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1,1,3,1,1)
        std  = x.new_tensor([0.229, 0.224, 0.225]).view(1,1,3,1,1)
        x[:, :, :3] = (x[:, :, :3] - mean) / std  # only if your losses/models expect it

    # Depth → [0,1]
    if C > 3:
        d = x[:, :, 3:4]
        if (near is not None) and (far is not None) and (far > near):
            d = (d - near) / (far - near)
            if depth_clip: d.clamp_(0.0, 1.0)
        else:
            d_min = d.amin(dim=(-2,-1), keepdim=True)
            d_max = d.amax(dim=(-2,-1), keepdim=True)
            d = (d - d_min) / (d_max - d_min + 1e-6)
        x[:, :, 3:4] = d
    return x