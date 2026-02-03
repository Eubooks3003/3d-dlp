"""
Utility functions for logging and plotting.
+ Spatial Transformer Network (STN) ~ JIT
+ Correlation maps ~ JIT
"""
# imports
import inspect
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.style as style
import cv2
import datetime
import os
import json
import imageio
import fnmatch
import zipfile
# torch
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.ops as ops
from typing import Tuple

matplotlib.use("Agg")


def color_map(num=100):
    colormap = ["FF355E",
                "8ffe09",
                "1d5dec",
                "FF9933",
                "FFFF66",
                "CCFF00",
                "AAF0D1",
                "FF6EFF",
                "FF00CC",
                "299617",
                "AF6E4D"] * num
    s = ''
    for color in colormap:
        s += color
    b = bytes.fromhex(s)
    cm = np.frombuffer(b, np.uint8)
    cm = cm.reshape(len(colormap), 3)
    return cm


def plot_keypoints_on_image(k, image_tensor, radius=1, thickness=1, kp_range=(0, 1), plot_numbers=True):
    # https://github.com/DuaneNielsen/keypoints
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    height, width = image_tensor.size(1), image_tensor.size(2)
    num_keypoints = k.size(0)

    if len(k.shape) != 2:
        raise Exception('Individual images and keypoints, not batches')

    k = k.clone()
    k[:, 0] = ((k[:, 0] - kp_range[0]) / (kp_range[1] - kp_range[0])) * (height - 1)
    k[:, 1] = ((k[:, 1] - kp_range[0]) / (kp_range[1] - kp_range[0])) * (width - 1)
    k.round_()
    k = k.detach().cpu().numpy()

    img = transforms.ToPILImage()(image_tensor.cpu())

    img = np.array(img)
    cmap = color_map()
    cm = cmap[:num_keypoints].astype(int)
    count = 0
    eps = 8
    for co_ord, color in zip(k, cm):
        c = color.item(0), color.item(1), color.item(2)
        co_ord = co_ord.squeeze()
        cv2.circle(img, (int(co_ord[1]), int(co_ord[0])), radius, c, thickness)
        if plot_numbers:
            cv2.putText(img, f'{count}', (int(co_ord[1] - eps), int(co_ord[0] - eps)), font, fontScale, c, 2,
                        cv2.LINE_AA)
        count += 1

    return img

def plot_keypoints_on_image_batch(kp_batch_tensor, img_batch_tensor, radius=1, thickness=1, max_imgs=8,
                                  kp_range=(0, 1), plot_numbers=False):
    num_plot = min(max_imgs, img_batch_tensor.shape[0])
    img_with_kp = []
    for i in range(num_plot):
        img_np = plot_keypoints_on_image(kp_batch_tensor[i], img_batch_tensor[i], radius=radius, thickness=thickness,
                                         kp_range=kp_range, plot_numbers=plot_numbers)
        img_tensor = torch.tensor(img_np).float() / 255.0
        img_with_kp.append(img_tensor.permute(2, 0, 1))
    img_with_kp = torch.stack(img_with_kp, dim=0)
    return img_with_kp


def plot_batch_kp(img_batch_tensor, kp_batch_tensor, rec_batch_tensor, max_imgs=8):
    batch_size, _, _, im_size = img_batch_tensor.shape
    max_index = (im_size - 1)
    num_plot = min(max_imgs, img_batch_tensor.shape[0])
    img_with_kp_np = []
    for i in range(num_plot):
        img_with_kp_np.append(plot_keypoints_on_image(kp_batch_tensor[i], img_batch_tensor[i], radius=2, thickness=1))
    img_np = img_batch_tensor.permute(0, 2, 3, 1).clamp(0, 1).data.cpu().numpy()
    rec_np = rec_batch_tensor.permute(0, 2, 3, 1).clamp(0, 1).data.cpu().numpy()
    fig = plt.figure()
    for i in range(num_plot):
        # image
        ax = fig.add_subplot(3, num_plot, i + 1)
        ax.imshow(img_np[i])
        ax.axis('equal')
        ax.set_axis_off()
        # kp
        ax = fig.add_subplot(3, num_plot, i + 1 + num_plot)
        ax.imshow(img_with_kp_np[i])
        ax.axis('equal')
        ax.set_axis_off()
        # rec
        ax = fig.add_subplot(3, num_plot, i + 1 + 2 * num_plot)
        ax.imshow(rec_np[i])
        ax.axis('equal')
        ax.set_axis_off()
    return fig


def plot_glimpse_obj_on(dec_object_glimpses, obj_on, save_dir):
    # plots glimpses with their obj_on value
    # author: Dan Haramati
    _, dec_object_glimpses = torch.split(dec_object_glimpses, [1, 3], dim=2)
    B, N, C, H, W = dec_object_glimpses.shape
    n_row, n_col = 1, B

    fig = plt.figure(figsize=(2 * n_col, 7 * n_row))
    fig.suptitle(f"Particle Glimpses Object-On", fontsize=20)

    for i in range(B):
        ax = fig.add_subplot(n_row, n_col, i + 1)
        glimpses = dec_object_glimpses[i]
        glimpses = torch.cat([glimpses[i] for i in range(len(glimpses))], dim=1)
        glimpses = glimpses.detach().cpu().numpy()
        glimpses = np.moveaxis(glimpses, 0, -1)
        ax.imshow(glimpses)
        ax.set_xticks([], [])
        ax.set_yticks(range(W // 2 - 1, W // 2 + W * N - 1, W), [f"{obj_on[i][m]:1.2f}" for m in range(N)])
        for j in range(1, N):
            ax.axhline(y=j * W, color='black')

    fig.tight_layout()
    plt.savefig(save_dir, bbox_inches='tight')


def reparameterize(mu, logvar, eps=None, return_eps=False):
    """
    This function applies the reparameterization trick:
    z = mu(X) + sigma(X)^0.5 * epsilon, where epsilon ~ N(0,I)
    :param mu: mean of x
    :param logvar: log variance of x
    :return z: the sampled latent variable
    """
    device = mu.device
    std = torch.exp(0.5 * logvar)
    if eps is None:
        eps = torch.randn_like(mu, device=device)
    z = mu + eps * std
    if return_eps:
        return z, eps
    else:
        return z


# --- helper: build 3D affine grid + sample (translation only for the mask utilities) ---
def _affine_grid_sample_3d(x, theta, out_shape, mode="nearest", padding_mode="zeros"):
    """
    x:        [N, C, D, H, W]
    theta:    [N, 3, 4]  (3D affine)
    out_shape (N, C, D, H, W)
    """
    N, C, D, H, W = out_shape
    grid = F.affine_grid(theta, size=out_shape, align_corners=False)     # [N, D, H, W, 3]
    return F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode, align_corners=False)


def _as_triple(val):
    # accept int or (D,H,W)
    if isinstance(val, (tuple, list)):
        assert len(val) == 3, "feature_dim/mask_size must be int or a 3-tuple (D,H,W)"
        return int(val[0]), int(val[1]), int(val[2])
    v = int(val)
    return v, v, v


# =========================
#        3D versions
# =========================
def create_masks_fast(center, anchor_s, feature_dim=16, patch_size=None):
    """
    3D analogue of your create_masks_fast:
      - center: [B, K, 3] in [-1,1], interpreted as (x, y, z)
      - feature_dim: int or (D, H, W) of the target mask volume
      - anchor_s: fraction that controls the cube side length (per-axis via (dim-1))
    Returns:
      [B, K, 1, D, H, W] with a single translated 1-cube per keypoint.
    """
    B, K = center.shape[0], center.shape[1]
    D, H, W = _as_triple(feature_dim)

    # per-axis cube size (like 2D: round(anchor_s * (dim-1)))
    if patch_size is None:
        pd = int(np.round(anchor_s * (D - 1)))
        ph = int(np.round(anchor_s * (H - 1)))
        pw = int(np.round(anchor_s * (W - 1)))
    else:
        pd, ph, pw = _as_triple(patch_size)

    pd = max(pd, 1); ph = max(ph, 1); pw = max(pw, 1)

    # small cube of ones, then pad to full D,H,W
    masks = torch.ones(B * K, 1, pd, ph, pw, device=center.device, dtype=torch.float)

    # padding order for 5D pad is (W_l, W_r, H_l, H_r, D_l, D_r)
    pad_w = (W - pw) // 2
    pad_h = (H - ph) // 2
    pad_d = (D - pd) // 2
    masks = F.pad(masks, pad=(pad_w, W - pw - pad_w, pad_h, H - ph - pad_h, pad_d, D - pd - pad_d))

    # translation only: place cube at kp location
    # center is in [-1,1] for (x, y, z). We translate by Delta = 0 - center (same as your 2D code).
    delta = (0.0 - center).reshape(-1, 3)  # [B*K, 3]
    theta = torch.zeros(B * K, 3, 4, device=center.device, dtype=masks.dtype)
    theta[:, 0, 0] = 1.0; theta[:, 1, 1] = 1.0; theta[:, 2, 2] = 1.0  # identity
    theta[:, 0, 3] = delta[:, 0]  # x translation
    theta[:, 1, 3] = delta[:, 1]  # y translation
    theta[:, 2, 3] = delta[:, 2]  # z translation

    transformed = _affine_grid_sample_3d(
        masks, theta, (B * K, 1, D, H, W), mode="nearest", padding_mode="zeros"
    )
    return transformed.view(B, K, 1, D, H, W)


def create_masks_with_scale(kp_batch, anchor_s, image_size, scale=None, scale_normalized=False):
    """
    3D analogue of create_masks_with_scale:
      - kp_batch: [B, K, 3] in [-1,1] (x, y, z)
      - image_size: int or (D, H, W) of the target volume
      - if scale is None -> z_scale = (patch / image) per-axis from anchor_s
      - else use 'scale' (sigmoid if not normalized)
    Returns:
      [B, K, 1, D, H, W]
    """
    B, K = kp_batch.shape[:2]
    D, H, W = _as_triple(image_size)

    # cube size driven by anchor_s per axis
    pd = int(np.round(anchor_s * (D - 1)))
    ph = int(np.round(anchor_s * (H - 1)))
    pw = int(np.round(anchor_s * (W - 1)))
    pd = max(pd, 1); ph = max(ph, 1); pw = max(pw, 1)

    patches = torch.ones(B, K, 1, pd, ph, pw, device=kp_batch.device, dtype=torch.float)
    N = B * K

    if scale is None:
        z_scale = torch.tensor(
            [pd / D, ph / H, pw / W], device=kp_batch.device, dtype=kp_batch.dtype
        ).view(1, 1, 3).expand(B, K, 3)
    else:
        z_scale = scale if scale_normalized else torch.sigmoid(scale)  # [B, K, 3] in (0,1)

    z_pos   = kp_batch.reshape(-1, 3)        # [N, 3]
    z_scale = z_scale.view(-1, 3)            # [N, 3]
    patches = patches.view(N, 1, pd, ph, pw) # [N, 1, pd, ph, pw]

    # build affine theta (scale+translate), then use inverse=True behavior:
    # Instead of duplicating your spatial_transform_3d, construct theta^{-1} explicitly.
    # For inverse grid: diag = 1/scale, trans = -pos/scale  (same as your 2D inverse branch)
    eps = 1e-9
    inv_theta = torch.zeros(N, 3, 4, device=kp_batch.device, dtype=patches.dtype)
    inv_theta[:, 0, 0] = 1.0 / (z_scale[:, 0] + eps)
    inv_theta[:, 1, 1] = 1.0 / (z_scale[:, 1] + eps)
    inv_theta[:, 2, 2] = 1.0 / (z_scale[:, 2] + eps)
    inv_theta[:, 0, 3] = - z_pos[:, 0] / (z_scale[:, 0] + eps)
    inv_theta[:, 1, 3] = - z_pos[:, 1] / (z_scale[:, 1] + eps)
    inv_theta[:, 2, 3] = - z_pos[:, 2] / (z_scale[:, 2] + eps)

    out = _affine_grid_sample_3d(
        patches, inv_theta, (N, 1, D, H, W), mode="nearest", padding_mode="zeros"
    )
    return out.view(B, K, 1, D, H, W)




def get_bb_from_masks(masks, width, height):
    # extracts bounding boxes (bb) from masks.
    # batch version
    # https://discuss.pytorch.org/t/find-bounding-box-around-ones-in-batch-of-masks/141266
    # masks: [n_masks, 1, feature_dim, feature_dim]
    masks = masks.bool().squeeze(1)
    b, h, w = masks.shape
    coor = torch.zeros(size=(b, 4), dtype=torch.int, device=masks.device)
    scales = torch.zeros(size=(b, 2), dtype=torch.float, device=masks.device)  # normalized scales
    centers = torch.zeros(size=(b, 2), dtype=torch.float, device=masks.device)  # normalized [-1, 1] centers of bbs

    rows = torch.any(masks, axis=2)
    cols = torch.any(masks, axis=1)

    rmins = torch.argmax(rows.float(), dim=1)
    rmaxs = h - torch.argmax(rows.float().flip(dims=[1]), dim=1) - 1
    cmins = torch.argmax(cols.float(), dim=1)
    cmaxs = w - torch.argmax(cols.float().flip(dims=[1]), dim=1) - 1

    ws = (cmins * (width / w)).clamp(0, width).int()
    wt = (cmaxs * (width / w)).clamp(0, width).int()
    hs = (rmins * (height / h)).clamp(0, height).int()
    ht = (rmaxs * (height / h)).clamp(0, height).int()

    coor[:, 0] = ws  # ws
    coor[:, 1] = hs  # hs
    coor[:, 2] = wt  # wt
    coor[:, 3] = ht  # ht

    # normalized scales
    scales[:, 1] = (wt - ws) / width
    scales[:, 0] = (ht - hs) / height
    # normalized centers
    centers[:, 1] = 2 * (((ws + wt) / 2) / width - 0.5)
    centers[:, 0] = 2 * (((hs + ht) / 2) / height - 0.5)

    output_dict = {'coor': coor, 'scales': scales, 'centers': centers}
    return output_dict


def get_bb_from_z_scale(kp, z_scale, width, height, scale_normalized=False):
    # extracts bounding boxes (bb) from keypoints and scales.
    # kp: [n_kp, 2], range: (-1, 1)
    # z_scale: [n_kp, 2], range: (0, 1)
    # scale_normalized: False if scale is not in [0, 1]
    n_kp = kp.shape[0]
    coor = torch.zeros(size=(n_kp, 4), dtype=torch.int, device=kp.device)
    kp_norm = 0.5 + kp / 2  # [0, 1]
    if scale_normalized:
        scale_norm = z_scale
    else:
        # scale_norm = 0.5 + z_scale / 2
        scale_norm = torch.sigmoid(z_scale)
    for i in range(n_kp):
        x_kp = kp_norm[i, 1] * width
        x_scale = scale_norm[i, 1] * width
        y_kp = kp_norm[i, 0] * height
        y_scale = scale_norm[i, 0] * height
        ws = (x_kp - x_scale / 2).clamp(0, width).int()
        wt = (x_kp + x_scale / 2).clamp(0, width).int()
        hs = (y_kp - y_scale / 2).clamp(0, height).int()
        ht = (y_kp + y_scale / 2).clamp(0, height).int()
        coor[i, 0] = ws
        coor[i, 1] = hs
        coor[i, 2] = wt
        coor[i, 3] = ht
    return coor


def get_bb_from_masks_batch(masks, width, height):
    # extracts bounding boxes (bb) from a batch of masks.
    # masks: [batch_size, n_masks, 1, feature_dim, feature_dim]
    coor = torch.zeros(size=(masks.shape[0], masks.shape[1], 4), dtype=torch.int, device=masks.device)
    for i in range(masks.shape[0]):
        coor[i, :, :] = get_bb_from_masks(masks[i], width, height)
    return coor


def nms_single(boxes, scores, iou_thresh=0.5, return_scores=False, remove_ind=None):
    # non-maximal suppression on bb and scores from one image.
    # boxes: [n_bb, 4], scores: [n_boxes]
    nms_indices = ops.nms(boxes.float(), scores, iou_thresh)
    # remove low scoring indices from nms output
    if remove_ind is not None:
        # final_indices = [ind for ind in nms_indices if ind not in remove_ind]
        final_indices = list(set(nms_indices.data.cpu().numpy()) - set(remove_ind))
        # print(f'removed indices: {remove_ind}')
    else:
        final_indices = nms_indices
    nms_boxes = boxes[final_indices]  # [n_bb_nms, 4]
    if return_scores:
        return nms_boxes, final_indices, scores[final_indices]
    else:
        return nms_boxes, final_indices


def remove_low_score_bb_single(boxes, scores, return_scores=False, mode='mean', thresh=0.4, hard_thresh=None):
    # filters out low-scoring bounding boxes. The score is usually the variance of the particle.
    # boxes: [n_bb, 4], scores: [n_boxes]
    if hard_thresh is None:
        if mode == 'mean':
            mean_score = scores.mean()
            # indices = (scores > mean_score)
            indices = torch.nonzero(scores > thresh, as_tuple=True)[0].data.cpu().numpy()
        else:
            normalzied_scores = (scores - scores.min()) / (scores.max() - scores.min())
            # indices = (normalzied_scores > thresh)
            indices = torch.nonzero(normalzied_scores > thresh, as_tuple=True)[0].data.cpu().numpy()
    else:
        # indices = (scores > hard_thresh)
        indices = torch.nonzero(scores > hard_thresh, as_tuple=True)[0].data.cpu().numpy()
    boxes_t = boxes[indices]
    scores_t = scores[indices]
    if return_scores:
        return indices, boxes_t, scores_t
    else:
        return indices, boxes_t


def get_low_score_bb_single(scores, mode='mean', thresh=0.4, hard_thresh=None):
    # get indices of low-scoring bounding boxes.
    # boxes: [n_bb, 4], scores: [n_boxes]
    if hard_thresh is None:
        if mode == 'mean':
            indices = torch.nonzero(scores < thresh, as_tuple=True)[0].data.cpu().numpy()
        else:
            normalzied_scores = (scores - scores.min()) / (scores.max() - scores.min())
            indices = torch.nonzero(normalzied_scores < thresh, as_tuple=True)[0].data.cpu().numpy()
    else:
        indices = torch.nonzero(scores < hard_thresh, as_tuple=True)[0].data.cpu().numpy()
    return indices


def plot_bb_on_image_from_masks_nms(masks, image_tensor, scores, iou_thresh=0.5, thickness=1, hard_thresh=None):
    # plot bounding boxes on a single image, use non-maximal suppression to filter low-scoring bbs.
    # masks: [n_masks, 1, feature_dim, feature_dim]
    n_masks = masks.shape[0]
    mask_h, mask_w = masks.shape[2], masks.shape[3]
    height, width = image_tensor.size(1), image_tensor.size(2)
    img = transforms.ToPILImage()(image_tensor.cpu())
    img = np.array(img)
    cmap = color_map()
    cm = cmap[:n_masks].astype(int)
    count = 0
    # get bb coor
    bb_from_masks = get_bb_from_masks(masks, width, height)
    coors = bb_from_masks['coor']  # [n_masks, 4]
    # remove low-score bb
    if hard_thresh is None:
        low_score_ind = get_low_score_bb_single(scores, mode='mean', hard_thresh=2.0)
    else:
        low_score_ind = get_low_score_bb_single(scores, mode='mean', hard_thresh=hard_thresh)
    # nms
    # remove_ind = low_score_ind if hard_thresh is not None else None
    remove_ind = low_score_ind
    # nms
    # remove_ind = low_score_ind if hard_thresh is not None else None
    coors_nms, nms_indices, scores_nms = nms_single(coors, scores, iou_thresh, return_scores=True,
                                                    remove_ind=remove_ind)
    # [n_masks_nms, 4]
    for coor, color in zip(coors_nms, cm):
        c = color.item(0), color.item(1), color.item(2)
        ws = (coor[0] - thickness).clamp(0, width)
        hs = (coor[1] - thickness).clamp(0, height)
        wt = (coor[2] + thickness).clamp(0, width)
        ht = (coor[3] + thickness).clamp(0, height)
        bb_s = (int(ws), int(hs))
        bb_t = (int(wt), int(ht))
        cv2.rectangle(img, bb_s, bb_t, c, thickness, 1)
        score_text = f'{scores_nms[count]:.2f}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.3
        thickness = 1
        box_w = bb_t[0] - bb_s[0]
        box_h = bb_t[1] - bb_s[1]
        org = (int(bb_s[0] + box_w / 4), int(bb_s[1] + box_h / 2))
        cv2.putText(img, score_text, org, font, fontScale, thickness=thickness, color=c, lineType=cv2.LINE_AA)
        count += 1

    return img, nms_indices


def plot_bb_on_image_batch_from_masks_nms(mask_batch_tensor, img_batch_tensor, scores, iou_thresh=0.5, thickness=1,
                                          max_imgs=8, hard_thresh=None):
    # plot bounding boxes on a batch of images, use non-maximal suppression to filter low-scoring bbs.
    # mask_batch_tensor: [batch_size, n_kp, 1, feature_dim, feature_dim]
    num_plot = min(max_imgs, img_batch_tensor.shape[0])
    img_with_bb = []
    indices = []
    for i in range(num_plot):
        img_np, nms_indices = plot_bb_on_image_from_masks_nms(mask_batch_tensor[i], img_batch_tensor[i], scores[i],
                                                              iou_thresh, thickness=thickness, hard_thresh=hard_thresh)
        img_tensor = torch.tensor(img_np).float() / 255.0
        img_with_bb.append(img_tensor.permute(2, 0, 1))
        indices.append(nms_indices)
    img_with_bb = torch.stack(img_with_bb, dim=0)
    return img_with_bb, indices


def plot_bb_on_image_from_z_scale_nms(kp, z_scale, image_tensor, scores, iou_thresh=0.5, thickness=1, hard_thresh=None,
                                      scale_normalized=False):
    # plot bounding boxes on a single image, use non-maximal suppression to filter low-scoring bbs.
    # kp: [n_kp, 2], range: (-1, 1)
    # z_scale: [n_kp, 2], range: (0, 1)
    n_kp = kp.shape[0]
    height, width = image_tensor.size(1), image_tensor.size(2)
    img = transforms.ToPILImage()(image_tensor.cpu())
    img = np.array(img)
    cmap = color_map()
    cm = cmap[:n_kp].astype(int)
    count = 0
    # get bb coor
    coors = get_bb_from_z_scale(kp, z_scale, width, height, scale_normalized=scale_normalized)  # [n_masks, 4]
    # remove low-score bb
    if hard_thresh is None:
        low_score_ind = get_low_score_bb_single(scores, mode='mean', hard_thresh=2.0)
    else:
        low_score_ind = get_low_score_bb_single(scores, mode='mean', hard_thresh=hard_thresh)
    # nms
    # remove_ind = low_score_ind if hard_thresh is not None else None
    remove_ind = low_score_ind
    coors_nms, nms_indices, scores_nms = nms_single(coors, scores, iou_thresh, return_scores=True,
                                                    remove_ind=remove_ind)
    # [n_masks_nms, 4]
    for coor, color in zip(coors_nms, cm):
        c = color.item(0), color.item(1), color.item(2)
        ws = (coor[0] - thickness).clamp(0, width)
        hs = (coor[1] - thickness).clamp(0, height)
        wt = (coor[2] + thickness).clamp(0, width)
        ht = (coor[3] + thickness).clamp(0, height)
        bb_s = (int(ws), int(hs))
        bb_t = (int(wt), int(ht))
        cv2.rectangle(img, bb_s, bb_t, c, thickness, 1)
        score_text = f'{scores_nms[count]:.2f}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.3
        thickness = 1
        box_w = bb_t[0] - bb_s[0]
        box_h = bb_t[1] - bb_s[1]
        org = (int(bb_s[0] + box_w / 4), int(bb_s[1] + box_h / 2))
        cv2.putText(img, score_text, org, font, fontScale, thickness=thickness, color=c, lineType=cv2.LINE_AA)
        count += 1

    return img, nms_indices


def plot_bb_on_image_batch_from_z_scale_nms(kp_batch_tensor, z_scale_batch_tensor, img_batch_tensor, scores,
                                            iou_thresh=0.5, thickness=1, max_imgs=8, hard_thresh=None,
                                            scale_normalized=False):
    # plot bounding boxes on a batch of images, use non-maximal suppression to filter low-scoring bbs.
    # kp_batch_tensor: [batch_size, n_kp, 2]
    # z_scale_batch_tensor: [batch_size, n_kp, 2]
    num_plot = min(max_imgs, img_batch_tensor.shape[0])
    img_with_bb = []
    indices = []
    for i in range(num_plot):
        img_np, nms_indices = plot_bb_on_image_from_z_scale_nms(kp_batch_tensor[i], z_scale_batch_tensor[i],
                                                                img_batch_tensor[i], scores[i], iou_thresh,
                                                                thickness=thickness, hard_thresh=hard_thresh,
                                                                scale_normalized=scale_normalized)
        img_tensor = torch.tensor(img_np).float() / 255.0
        img_with_bb.append(img_tensor.permute(2, 0, 1))
        indices.append(nms_indices)
    img_with_bb = torch.stack(img_with_bb, dim=0)
    return img_with_bb, indices


def plot_bb_on_image_from_masks(masks, image_tensor, thickness=1):
    # vanilla plotting of bbs from masks.
    # masks: [n_masks, 1, feature_dim, feature_dim]
    n_masks = masks.shape[0]
    mask_h, mask_w = masks.shape[2], masks.shape[3]
    height, width = image_tensor.size(1), image_tensor.size(2)

    img = transforms.ToPILImage()(image_tensor.cpu())

    img = np.array(img)
    cmap = color_map()
    cm = cmap[:n_masks].astype(int)
    count = 0
    for mask, color in zip(masks, cm):
        c = color.item(0), color.item(1), color.item(2)
        mask = mask.int().squeeze()  # [feature_dim, feature_dim]
        #         print(mask.shape)
        indices = (mask == 1).nonzero(as_tuple=False)
        #         print(indices.shape)
        if indices.shape[0] > 0:
            ws = (indices[0][1] * (width / mask_w) - thickness).clamp(0, width).int()
            wt = (indices[-1][1] * (width / mask_w) + thickness).clamp(0, width).int()
            hs = (indices[0][0] * (height / mask_h) - thickness).clamp(0, height).int()
            ht = (indices[-1][0] * (height / mask_h) + thickness).clamp(0, height).int()
            bb_s = (int(ws), int(hs))
            bb_t = (int(wt), int(ht))
            cv2.rectangle(img, bb_s, bb_t, c, thickness, 1)
            count += 1
    return img


def plot_bb_on_image_batch_from_masks(mask_batch_tensor, img_batch_tensor, thickness=1, max_imgs=8):
    # vanilla plotting of bbs from a batch of masks.
    # mask_batch_tensor: [batch_size, n_kp, 1, feature_dim, feature_dim]
    num_plot = min(max_imgs, img_batch_tensor.shape[0])
    img_with_bb = []
    for i in range(num_plot):
        img_np = plot_bb_on_image_from_masks(mask_batch_tensor[i], img_batch_tensor[i], thickness=thickness)
        img_tensor = torch.tensor(img_np).float() / 255.0
        img_with_bb.append(img_tensor.permute(2, 0, 1))
    img_with_bb = torch.stack(img_with_bb, dim=0)
    return img_with_bb


def create_segmentation_map(
    x: torch.Tensor,
    masks: torch.Tensor,
    scores: torch.Tensor,
    alpha: float = 0.7,
    score_threshold: float = 1e-2,
    cmap_name: str = "rainbow",
) -> torch.Tensor:
    """
    Overlays colored masks onto the RGB channels.
    If x has 4 channels (RGBD), the depth channel is left unchanged.

    x:      [B, C, H, W], C in {3,4}
    masks:  [B, K, H, W]
    scores: [B, K]
    returns seg_map with same shape as x
    """
    assert x.dim() == 4 and x.size(1) in (3, 4), f"expected [B,3/4,H,W], got {tuple(x.shape)}"
    B, C, H, W = x.shape
    device = x.device

    seg_map = x.clone()  # keep depth if present
    rgb = seg_map[:, :3]  # operate on RGB only

    cmap = plt.get_cmap(cmap_name)

    for b in range(B):
        valid = torch.where(scores[b] > score_threshold)[0]
        if valid.numel() == 0:
            continue

        overlay = torch.zeros(3, H, W, device=device)

        # assign distinct colors across valid masks
        denom = max(1, valid.numel() - 1)
        for i, mi in enumerate(valid.tolist()):
            color_idx = i / denom
            color = torch.tensor(cmap(color_idx)[:3], device=device, dtype=rgb.dtype).view(3, 1, 1)
            mask = masks[b, mi].unsqueeze(0).to(rgb.dtype)  # [1,H,W]
            overlay = torch.max(overlay, mask * color)

        rgb[b] = rgb[b] * (1 - alpha) + overlay * alpha

    # seg_map now has RGB blended; depth (if any) unchanged
    return seg_map


def depth_to_rgb(depth: torch.Tensor, cmap_name="viridis"):
    """
    depth: [B,1,H,W] in meters or normalized
    Returns: [B,3,H,W] uint in [0,1] for visualization (does NOT modify input)
    """
    assert depth.dim() == 4 and depth.size(1) == 1
    d = depth.clone()

    d = d.clamp_(0, 1)

    # map to RGB via matplotlib colormap
    cmap = plt.get_cmap(cmap_name)
    # d: [B,1,H,W] -> [B,H,W]
    d_np = d.squeeze(1).detach().cpu().numpy()
    rgb_list = []
    for i in range(d_np.shape[0]):
        rgba = cmap(d_np[i])  # [H,W,4]
        rgb = torch.from_numpy(rgba[..., :3]).permute(2,0,1)  # [3,H,W]
        rgb_list.append(rgb)
    rgb = torch.stack(rgb_list, dim=0).to(depth.device, dtype=depth.dtype)
    return rgb


def prepare_logdir(runname, src_dir='./', accelerator=None):
    td_prefix = datetime.datetime.now().strftime("%d%m%y_%H%M%S")
    dir_name = f'{td_prefix}_{runname}'
    path_to_dir = os.path.join(src_dir, dir_name)
    path_to_fig_dir = os.path.join(path_to_dir, 'figures')
    path_to_save_dir = os.path.join(path_to_dir, 'saves')
    if accelerator is not None and accelerator.is_main_process:
        os.makedirs(path_to_dir, exist_ok=True)
        os.makedirs(path_to_fig_dir, exist_ok=True)
        os.makedirs(path_to_save_dir, exist_ok=True)
    elif accelerator is None:
        os.makedirs(path_to_dir, exist_ok=True)
        os.makedirs(path_to_fig_dir, exist_ok=True)
        os.makedirs(path_to_save_dir, exist_ok=True)
    else:
        pass
    return path_to_dir


def save_config(src_dir, hparams, fname='hparams'):
    path_to_conf = os.path.join(src_dir, f'{fname}.json')
    with open(path_to_conf, "w") as outfile:
        json.dump(hparams, outfile, indent=2)


def get_config(fpath):
    with open(fpath, 'r') as f:
        config = json.load(f)
    return config


def log_line(src_dir, line):
    log_file = os.path.join(src_dir, 'log.txt')
    with open(log_file, 'a') as fp:
        fp.writelines(line)


def animate_trajectories(orig_trajectory, pred_trajectory, pred_trajectory_2=None, path='./traj_anim.gif',
                         duration=4 / 50, rec_to_pred_t=10, title=None, t1='', t2='', goal_img=None,
                         orig_trajectory2=None, pred_trajectory_12=None, pred_trajectory_22=None, goal_img2=None):
    # rec_to_pred_t: the timestep from which prediction transitions from reconstruction to generation
    # goal_img: np array: [h, w, ch]
    # prepare images
    font = cv2.FONT_HERSHEY_SIMPLEX
    origin = (5, 15)
    fontScale = 0.4
    color = (255, 255, 255)
    gt_border_color = (255, 0, 0)
    rec_border_color = (0, 0, 255)
    gen_border_color = (0, 255, 0)
    border_size = 2
    thickness = 1
    gt_traj_prep = []
    pred_traj_prep = []
    pred_traj_prep_2 = []

    # second view
    gt_traj_prep2 = []
    pred_traj_prep12 = []
    pred_traj_prep_22 = []

    goal_img_prep = None
    if goal_img is not None:
        goal_img_prep = (goal_img.clip(0, 1) * 255).astype(np.uint8).copy()
        # Add border to goal image
        goal_img_prep = cv2.copyMakeBorder(goal_img_prep, border_size, border_size, border_size, border_size,
                                           cv2.BORDER_CONSTANT, value=(128, 128, 128))  # Gray border for goal
        if goal_img2 is not None:
            goal_img_prep2 = (goal_img2.clip(0, 1) * 255).astype(np.uint8).copy()
            # Add border to goal image
            goal_img_prep2 = cv2.copyMakeBorder(goal_img_prep2, border_size, border_size, border_size, border_size,
                                                cv2.BORDER_CONSTANT, value=(128, 128, 128))  # Gray border for goal

        # Create "Goal" text plate
        goal_text_color = (0, 0, 0)
        goal_fontScale = 0.4
        goal_thickness = 1
        goal_font = cv2.FONT_HERSHEY_SIMPLEX
        goal_text_h = 20
        goal_text_w = 50
        goal_text_plate = (np.ones((goal_text_h, goal_text_w, 3)) * 255).astype(np.uint8)
        goal_text_origin = (5, goal_text_h // 2 + 5)
        goal_text_plate = cv2.putText(goal_text_plate, 'Goal', goal_text_origin, goal_font,
                                      goal_fontScale, goal_text_color, goal_thickness, cv2.LINE_AA)

    for i in range(orig_trajectory.shape[0]):
        image = (orig_trajectory[i] * 255).astype(np.uint8).copy()
        image = cv2.putText(image, f'GT:{i}', origin, font, fontScale, color, thickness, cv2.LINE_AA)
        # add border
        image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT,
                                   value=gt_border_color)
        gt_traj_prep.append(image)

        text = f'REC:{i}' if i < rec_to_pred_t else f'P{t1}:{i}'
        image = (pred_trajectory[i].clip(0, 1) * 255).astype(np.uint8).copy()
        image = cv2.putText(image, text, origin, font, fontScale, color, thickness, cv2.LINE_AA)
        # add border
        border_color = rec_border_color if i < rec_to_pred_t else gen_border_color
        image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT,
                                   value=border_color)
        pred_traj_prep.append(image)

        if pred_trajectory_2 is not None:
            text = f'REC:{i}' if i < rec_to_pred_t else f'P{t2}:{i}'
            image = (pred_trajectory_2[i].clip(0, 1) * 255).astype(np.uint8).copy()
            image = cv2.putText(image, text, origin, font, fontScale, color, thickness, cv2.LINE_AA)
            # add border
            border_color = rec_border_color if i < rec_to_pred_t else gen_border_color
            image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT,
                                       value=border_color)
            pred_traj_prep_2.append(image)

        if orig_trajectory2 is not None:
            image = (orig_trajectory2[i] * 255).astype(np.uint8).copy()
            image = cv2.putText(image, f'GT:{i}', origin, font, fontScale, color, thickness, cv2.LINE_AA)
            # add border
            image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT,
                                       value=gt_border_color)
            gt_traj_prep2.append(image)

        if pred_trajectory_12 is not None:
            text = f'REC:{i}' if i < rec_to_pred_t else f'P{t1}:{i}'
            image = (pred_trajectory_12[i].clip(0, 1) * 255).astype(np.uint8).copy()
            image = cv2.putText(image, text, origin, font, fontScale, color, thickness, cv2.LINE_AA)
            # add border
            border_color = rec_border_color if i < rec_to_pred_t else gen_border_color
            image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT,
                                       value=border_color)
            pred_traj_prep12.append(image)

        if pred_trajectory_22 is not None:
            text = f'REC:{i}' if i < rec_to_pred_t else f'P{t2}:{i}'
            image = (pred_trajectory_22[i].clip(0, 1) * 255).astype(np.uint8).copy()
            image = cv2.putText(image, text, origin, font, fontScale, color, thickness, cv2.LINE_AA)
            # add border
            border_color = rec_border_color if i < rec_to_pred_t else gen_border_color
            image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT,
                                       value=border_color)
            pred_traj_prep_22.append(image)

    total_images = []
    for i in range(len(orig_trajectory)):
        white_border = (np.ones((gt_traj_prep[i].shape[0], 4, gt_traj_prep[i].shape[-1])) * 255).astype(np.uint8)
        if pred_trajectory_2 is not None:
            concat_img = np.concatenate([gt_traj_prep[i],
                                         white_border,
                                         pred_traj_prep[i],
                                         white_border,
                                         pred_traj_prep_2[i]], axis=1)
        else:
            concat_img = np.concatenate([gt_traj_prep[i],
                                         white_border,
                                         pred_traj_prep[i]], axis=1)
        if orig_trajectory2 is not None:
            white_separator = (np.ones((8, concat_img.shape[1], 3)) * 255).astype(np.uint8)
            white_border = (np.ones((gt_traj_prep2[i].shape[0], 4, gt_traj_prep2[i].shape[-1])) * 255).astype(np.uint8)
            if pred_trajectory_22 is not None:
                concat_img2 = np.concatenate([gt_traj_prep2[i],
                                              white_border,
                                              pred_traj_prep12[i],
                                              white_border,
                                              pred_traj_prep_22[i]], axis=1)
            else:
                concat_img2 = np.concatenate([gt_traj_prep2[i],
                                              white_border,
                                              pred_traj_prep12[i]], axis=1)
            concat_img = np.concatenate([concat_img, white_separator, concat_img2], axis=0)

        # Add title if provided
        if title is not None:
            text_color = (0, 0, 0)
            fontScale = 0.25
            thickness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            h = 25
            w = concat_img.shape[1]
            text_plate = (np.ones((h, w, 3)) * 255).astype(np.uint8)
            w_orig = orig_trajectory.shape[1] // 2
            origin = (w_orig // 6, h // 2)
            text_plate = cv2.putText(text_plate, title, origin, font, fontScale, text_color, thickness,
                                     cv2.LINE_AA)
            concat_img = np.concatenate([text_plate, concat_img], axis=0)

        # Add goal image if provided
        if goal_img is not None:
            mult_factor = 1 if goal_img2 is None else 2
            # Create white separator
            white_separator = (np.ones((8, concat_img.shape[1], 3)) * 255).astype(np.uint8)

            # Create goal section with text and image
            goal_section_width = concat_img.shape[1]
            goal_img_center_x = goal_section_width // 2 - (goal_img_prep.shape[1] * mult_factor) // 2

            # Create goal section background
            goal_section_height = goal_img_prep.shape[0]
            goal_section = (np.ones((goal_section_height, goal_section_width, 3)) * 255).astype(np.uint8)

            # Place goal text on the left
            text_x_pos = max(0, goal_img_center_x - goal_text_plate.shape[1] - 10)
            text_y_start = (goal_section_height - goal_text_plate.shape[0]) // 2
            text_y_end = text_y_start + goal_text_plate.shape[0]
            goal_section[text_y_start:text_y_end, text_x_pos:text_x_pos + goal_text_plate.shape[1]] = goal_text_plate

            # Place goal image in the center
            goal_section[:, goal_img_center_x:goal_img_center_x + goal_img_prep.shape[1]] = goal_img_prep

            if goal_img2 is not None:
                goal_section[:, goal_img_center_x + goal_img_prep.shape[1]:goal_img_center_x + 2 * goal_img_prep.shape[
                    1]] = goal_img_prep2

            # Concatenate everything
            concat_img = np.concatenate([concat_img, white_separator, goal_section], axis=0)

        total_images.append(concat_img)

    imageio.mimsave(path, total_images, duration=(1000 / duration), loop=0)  # 1/50


import torch
import torch.nn.functional as F

def spatial_transform(image, z_pos, z_scale, out_dims, inverse=False, eps=1e-9, padding_mode="zeros"):
    """
    3D spatial transformer.
    image:   [batch_size * n_kp, ch, D, H, W]
    z_pos:   [batch_size * n_kp, 3]   (x, y, z) in [-1, 1]
    z_scale: [batch_size * n_kp, 3]   (sx, sy, sz) in (0, 1] if forward; unnormalized if inverse is False -> same as 2D code
    out_dims: tuple (batch_size * n_kp, ch, D*, H*, W*)
    """
    N = image.shape[0]
    theta = torch.zeros(N, 3, 4, device=image.device, dtype=image.dtype)

    if not inverse:
        sx = z_scale[:, 0]; sy = z_scale[:, 1]; sz = z_scale[:, 2]
        tx = z_pos[:, 0];   ty = z_pos[:, 1];   tz = z_pos[:, 2]
    else:
        sx = 1.0 / (z_scale[:, 0] + eps)
        sy = 1.0 / (z_scale[:, 1] + eps)
        sz = 1.0 / (z_scale[:, 2] + eps)
        tx = - z_pos[:, 0] / (z_scale[:, 0] + eps)
        ty = - z_pos[:, 1] / (z_scale[:, 1] + eps)
        tz = - z_pos[:, 2] / (z_scale[:, 2] + eps)

    # diag scales
    theta[:, 0, 0] = sx
    theta[:, 1, 1] = sy
    theta[:, 2, 2] = sz
    # translations
    theta[:, 0, 3] = tx
    theta[:, 1, 3] = ty
    theta[:, 2, 3] = tz

    grid = F.affine_grid(theta, out_dims, align_corners=False)     # [N, D*, H*, W*, 3]
    return F.grid_sample(image, grid, mode='bilinear', padding_mode=padding_mode, align_corners=False)


def generate_dlp_logo():
    logo = """
    ██████╗ ██╗     ██████╗ 
    ██╔══██╗██║     ██╔══██╗
    ██║  ██║██║     ██████╔╝
    ██║  ██║██║     ██╔═══╝ 
    ██████╔╝███████╗██║     
    ╚═════╝ ╚══════╝╚═╝     

         ██╗   ██╗██████╗ 
         ██║   ██║╚════██╗
         ██║   ██║ █████╔╝
         ╚██╗ ██╔╝ ╚═══██╗
          ╚████╔╝ ██████╔╝
           ╚═══╝  ╚═════╝ 
    """

    # Add some decorative borders
    width = max(len(line) for line in logo.split('\n'))
    border = '═' * (width + 4)

    # Build the final string with borders
    result = ['╔' + border + '╗']
    for line in logo.split('\n'):
        if line.strip():
            result.append('║ ' + line + ' ' * (width - len(line)) + '   ║')
    result.append('╚' + border + '╝')

    return '\n'.join(result)


def format_epoch_summary(
        epoch, loss, loss_rec, loss_kl, kl_balance,
        loss_kl_kp, loss_kl_feat, loss_kl_scale, loss_kl_depth,
        loss_kl_obj_on, mu_tot, mu_offset, valid_loss, best_valid_loss,
        best_valid_epoch, obj_on, mu_scale, mu_depth,
        eval_epoch_freq, val_lpips=None, best_val_lpips=None,
        best_val_lpips_epoch=None, loss_kl_context=None,
        loss_kl_dyn=None, psnr=None
):
    """Format epoch training summary in an organized way."""

    sections = {
        "Epoch": f"Epoch {epoch}",
        "Loss Metrics": [
            f"Loss: {loss:.3f}",
            f"Reconstruction: {loss_rec:.3f}",
            f"KL: {loss_kl:.3f}",
            f"KL-Balance: {kl_balance:.3f}",
            f"KL KP: {loss_kl_kp:.3f}",
            f"KL Features: {loss_kl_feat:.3f}",
            f"KL Scale: {loss_kl_scale:.3f}",
            f"KL Depth: {loss_kl_depth:.3f}",
            f"KL Transparency (obj_on): {loss_kl_obj_on:.3f}"
        ],
        "Attribute Statistics": [
            f"Total Mu: min={mu_tot.min():.3f}, max={mu_tot.max():.3f}",
            f"Mu Offset: min={mu_offset.min():.3f}, max={mu_offset.max():.3f}",
            f"Object On: min={obj_on.min():.3f}, max={obj_on.max():.3f}",
            f"Scale: min={mu_scale.sigmoid().min():.3f}, max={mu_scale.sigmoid().max():.3f}",
            f"Depth: min={mu_depth.min():.3f}, max={mu_depth.max():.3f}"
        ],
        "Validation": [
            f"Loss (freq: {eval_epoch_freq}): {valid_loss:.3f}",
            f"Best Loss: {best_valid_loss:.3f} @ epoch {best_valid_epoch}"
        ]
    }

    # Add optional KL metrics
    if loss_kl_context is not None:
        sections["Loss Metrics"].append(f"KL Context: {loss_kl_context:.3f}")
    if loss_kl_dyn is not None:
        sections["Loss Metrics"].append(f"KL Dynamics: {loss_kl_dyn:.3f}")

    # Add optional validation metrics
    if psnr is not None:
        sections["Validation"].append(f"Mean PSNR: {psnr:.3f}")
    if val_lpips is not None:
        sections["Validation"].extend([
            f"LPIPS (freq: {eval_epoch_freq}): {val_lpips:.3f}",
            f"Best LPIPS: {best_val_lpips:.3f} @ epoch {best_val_lpips_epoch}"
        ])

    # Build the formatted string
    summary = []
    for section, content in sections.items():
        summary.append(f"\n=== {section} ===")
        if isinstance(content, list):
            summary.extend(content)
        else:
            summary.append(content)

    return "\n".join(summary)


def format_epoch_summary_dvae(
        epoch, loss, loss_rec, loss_kl, kl_balance,
        loss_kl_feat,
        valid_loss, best_valid_loss,
        best_valid_epoch,
        eval_epoch_freq, val_lpips=None, best_val_lpips=None,
        best_val_lpips_epoch=None, loss_kl_context=None,
        loss_kl_dyn=None, psnr=None
):
    """Format epoch training summary in an organized way."""

    sections = {
        "Epoch": f"Epoch {epoch}",
        "Loss Metrics": [
            f"Loss: {loss:.3f}",
            f"Reconstruction: {loss_rec:.3f}",
            f"KL: {loss_kl:.3f}",
            f"KL-Balance: {kl_balance:.3f}",
            f"KL Features: {loss_kl_feat:.3f}",
        ],
        "Validation": [
            f"Loss (freq: {eval_epoch_freq}): {valid_loss:.3f}",
            f"Best Loss: {best_valid_loss:.3f} @ epoch {best_valid_epoch}"
        ]
    }

    # Add optional KL metrics
    if loss_kl_context is not None:
        sections["Loss Metrics"].append(f"KL Context: {loss_kl_context:.3f}")
    if loss_kl_dyn is not None:
        sections["Loss Metrics"].append(f"KL Dynamics: {loss_kl_dyn:.3f}")

    # Add optional validation metrics
    if psnr is not None:
        sections["Validation"].append(f"Mean PSNR: {psnr:.3f}")
    if val_lpips is not None:
        sections["Validation"].extend([
            f"LPIPS (freq: {eval_epoch_freq}): {val_lpips:.3f}",
            f"Best LPIPS: {best_val_lpips:.3f} @ epoch {best_val_lpips_epoch}"
        ])

    # Build the formatted string
    summary = []
    for section, content in sections.items():
        summary.append(f"\n=== {section} ===")
        if isinstance(content, list):
            summary.extend(content)
        else:
            summary.append(content)

    return "\n".join(summary)


def plot_training_metrics(metrics_data, run_name, fig_dir, max_plots_per_figure=6, figsize=(12, 15),
                          style_name='seaborn-v0_8-darkgrid'):
    """
    Create professional-looking training metric plots with adaptive layout based on number of metrics.

    Args:
        metrics_data: List of tuples (data, label, color, include_minmax)
        run_name: Name of the run
        fig_dir: Directory to save figures
        max_plots_per_figure: Maximum number of subplots in a single figure
        figsize: Base figure size (width, height) - will be adjusted for fewer plots
        style_name: Matplotlib style to use
    """
    style.use(style_name)
    # plt.rcParams['font.family'] = 'sans-serif'
    # plt.rcParams['font.sans-serif'] = ['Arial']

    num_metrics = len(metrics_data)

    # Decide whether to create multiple figures
    if num_metrics > max_plots_per_figure:
        # Multiple figures approach
        num_figures = math.ceil(num_metrics / max_plots_per_figure)
        metrics_per_fig = math.ceil(num_metrics / num_figures)

        for fig_num in range(num_figures):
            start_idx = fig_num * metrics_per_fig
            end_idx = min((fig_num + 1) * metrics_per_fig, num_metrics)
            current_metrics = metrics_data[start_idx:end_idx]

            # Adjust figure size based on number of subplots
            adjusted_height = figsize[1] * (len(current_metrics) / max_plots_per_figure)
            fig = plt.figure(figsize=(figsize[0], adjusted_height))

            _create_subplots(fig, current_metrics, f"{run_name} (Group {fig_num + 1}/{num_figures})")

            # Save each figure
            plt.savefig(f'{fig_dir}/{run_name}_metrics_group{fig_num + 1}.png',
                        dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

    else:
        # Single figure approach
        # Adjust figure size based on number of plots
        adjusted_height = figsize[1] * (num_metrics / max_plots_per_figure)
        fig = plt.figure(figsize=(figsize[0], adjusted_height))

        _create_subplots(fig, metrics_data, run_name)

        plt.savefig(f'{fig_dir}/{run_name}_metrics.png',
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()


def _create_subplots(fig, metrics_data, title):
    """Helper function to create subplots for a given figure."""
    num_plots = len(metrics_data)

    for i, (data, label, color, include_minmax) in enumerate(metrics_data, start=1):
        ax = fig.add_subplot(num_plots, 1, i)

        # Plot main line
        line = ax.plot(np.arange(len(data)), data,
                       label=label,
                       color=color,
                       linewidth=2,
                       alpha=0.8)[0]

        # Add markers
        ax.plot(np.arange(len(data)), data,
                'o',
                color=color,
                markersize=3,
                alpha=0.5)

        # Styling
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        ax.set_xlabel("Epochs", fontsize=10, fontweight='bold')
        ax.set_ylabel("Value", fontsize=10, fontweight='bold')
        ax.set_title(label, pad=10, fontsize=12, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc="upper right", fontsize=9, facecolor='white',
                  edgecolor='none', framealpha=0.8)

        if include_minmax:
            min_val = min(data)
            max_val = max(data)
            min_idx = np.argmin(data)
            max_idx = np.argmax(data)
            y_range = max_val - min_val

            # Add min/max annotations
            ax.annotate(f'Min: {min_val:.3f}',
                        xy=(min_idx, min_val),
                        xytext=(10, -10),
                        textcoords='offset points',
                        fontsize=8,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.8),
                        arrowprops=dict(arrowstyle='->',
                                        connectionstyle='arc3,rad=0.2',
                                        color=color,
                                        alpha=0.6))

            if abs(max_val - min_val) > y_range * 0.1:
                ax.annotate(f'Max: {max_val:.3f}',
                            xy=(max_idx, max_val),
                            xytext=(10, 10),
                            textcoords='offset points',
                            fontsize=8,
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8),
                            arrowprops=dict(arrowstyle='->',
                                            connectionstyle='arc3,rad=-0.2',
                                            color=color,
                                            alpha=0.6))

        ax.set_facecolor('#f8f9fa')

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def save_metrics_data(metrics_data, run_name, save_dir):
    """
    Save metrics data to disk in JSON format.

    Args:
        metrics_data: List of tuples (data, label, color, include_minmax)
        run_name: Name of the run
        save_dir: Directory to save the data

    Example usage:

    metrics_data = [
        (losses[1:], "Total Loss", "#2d72bc", True),
        (losses_kl[1:], "KL Loss", "#c92a2a", True),
        (losses_rec[1:], "Reconstruction Loss", "#087f5b", True),
        (valid_losses[1:], "Validation Loss", "#862e9c", True)
    ]

    # Save the metrics
    save_metrics_data(metrics_data, run_name, "saved_metrics")
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{run_name}_metrics.json")

    # Convert data to serializable format
    serializable_data = []
    for data, label, color, include_minmax in metrics_data:
        # Convert numpy arrays/values to lists/native Python types
        if isinstance(data, np.ndarray):
            data = data.tolist()
        elif isinstance(data, (np.generic, np.number)):
            data = data.item()

            # Convert any remaining numpy types within the list
        if isinstance(data, list):
            data = [float(x) if isinstance(x, (np.generic, np.number)) else x for x in data]

        serializable_data.append({
            "data": data,
            "label": label,
            "color": color,
            "include_minmax": include_minmax
        })

    with open(save_path, 'w') as f:
        json.dump(serializable_data, f, indent=4)


def load_metrics_data(run_name, save_dir):
    """
    Load metrics data from disk.

    Args:
        run_name: Name of the run
        save_dir: Directory where the data is saved

    Returns:
        List of tuples (data, label, color, include_minmax)

    Example usage:
    loaded_metrics = load_metrics_data(run_name, "saved_metrics")
    plot_training_metrics(loaded_metrics, run_name, "figures")
    """
    load_path = os.path.join(save_dir, f"{run_name}_metrics.json")

    with open(load_path, 'r') as f:
        loaded_data = json.load(f)

    # Convert back to the format expected by plot_training_metrics
    metrics_data = [
        (np.array(item["data"]) if isinstance(item["data"], list) else item["data"],
         item["label"],
         item["color"],
         item["include_minmax"])
        for item in loaded_data
    ]

    return metrics_data


def save_code_backup(source_dir='.', backup_dir='backups', verbose=False):
    """
    Creates a compressed backup of code files while maintaining directory structure.
    Only includes .py, README, requirements.txt, .ipynb, and environment.yml files.

    Args:
        source_dir (str): Source directory to backup (defaults to current directory)
        backup_dir (str): Directory where backups will be stored (defaults to 'backups')
        verbose (bool): If True, print each file added (defaults to False)

    Returns:
        str: Path to the created backup file
    """
    # Create backup directory if it doesn't exist
    os.makedirs(backup_dir, exist_ok=True)

    # Generate backup filename with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_filename = f'code_backup_{timestamp}.zip'
    backup_path = os.path.join(backup_dir, backup_filename)

    # Patterns to match
    patterns = ['*.py', 'README*', 'requirements.txt', '*.ipynb', 'environment.yml']

    # Directories to skip
    skip_dirs = {'wandb', '.git', '__pycache__', 'logs', '.venv', 'venv', 'env',
                 'node_modules', '.idea', '.vscode', 'build', 'dist', '*.egg-info'}

    file_count = 0
    # Create zip file
    with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through directory
        for root, dirs, files in os.walk(source_dir):
            # Skip excluded directories (modify dirs in-place to prevent descent)
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.endswith('.egg-info')]

            # Skip the backup directory itself
            if os.path.abspath(root).startswith(os.path.abspath(backup_dir)):
                continue

            # Filter files based on patterns
            for pattern in patterns:
                for filename in fnmatch.filter(files, pattern):
                    # Get the full file path
                    file_path = os.path.join(root, filename)

                    # Calculate path relative to source_dir for maintaining structure
                    rel_path = os.path.relpath(file_path, source_dir)

                    # Add file to zip
                    zipf.write(file_path, rel_path)
                    file_count += 1
                    if verbose:
                        print(f"Added: {rel_path}")

    info = f"Backup created: {backup_path} ({file_count} files, {os.path.getsize(backup_path) / (1024 * 1024):.2f} MB)"
    return info


def printarr(*arrs, float_width=6):
    """
    Print a pretty table giving name, shape, dtype, type, and content information for input tensors or scalars.

    Call like: printarr(my_arr, some_other_arr, maybe_a_scalar). Accepts a variable number of arguments.

    Inputs can be:
        - Numpy tensor arrays
        - Pytorch tensor arrays
        - Jax tensor arrays
        - Python ints / floats
        - None

    It may also work with other array-like types, but they have not been tested.

    Use the `float_width` option specify the precision to which floating point types are printed.

    Author: Nicholas Sharp (nmwsharp.com)
    Canonical source: https://gist.github.com/nmwsharp/54d04af87872a4988809f128e1a1d233
    License: This snippet may be used under an MIT license, and it is also released into the public domain.
             Please retain this docstring as a reference.
    """

    frame = inspect.currentframe().f_back
    default_name = "[temporary]"

    ## helpers to gather data about each array
    def name_from_outer_scope(a):
        if a is None:
            return '[None]'
        name = default_name
        for k, v in frame.f_locals.items():
            if v is a:
                name = k
                break
        return name

    def dtype_str(a):
        if a is None:
            return 'None'
        if isinstance(a, int):
            return 'int'
        if isinstance(a, float):
            return 'float'
        return str(a.dtype)

    def shape_str(a):
        if a is None:
            return 'N/A'
        if isinstance(a, int):
            return 'scalar'
        if isinstance(a, float):
            return 'scalar'
        return str(list(a.shape))

    def type_str(a):
        return str(type(a))[8:-2]  # TODO this is is weird... what's the better way?

    def device_str(a):
        if hasattr(a, 'device'):
            device_str = str(a.device)
            if len(device_str) < 10:
                # heuristic: jax returns some goofy long string we don't want, ignore it
                return device_str
        return ""

    def format_float(x):
        return f"{x:{float_width}g}"

    def minmaxmean_str(a):
        if a is None:
            return ('N/A', 'N/A', 'N/A')
        if isinstance(a, int) or isinstance(a, float):
            return (format_float(a), format_float(a), format_float(a))

        # compute min/max/mean. if anything goes wrong, just print 'N/A'
        min_str = "N/A"
        try:
            min_str = format_float(a.min())
        except:
            pass
        max_str = "N/A"
        try:
            max_str = format_float(a.max())
        except:
            pass
        mean_str = "N/A"
        try:
            mean_str = format_float(a.mean())
        except:
            pass

        return (min_str, max_str, mean_str)

    try:

        props = ['name', 'dtype', 'shape', 'type', 'device', 'min', 'max', 'mean']

        # precompute all of the properties for each input
        str_props = []
        for a in arrs:
            minmaxmean = minmaxmean_str(a)
            str_props.append({
                'name': name_from_outer_scope(a),
                'dtype': dtype_str(a),
                'shape': shape_str(a),
                'type': type_str(a),
                'device': device_str(a),
                'min': minmaxmean[0],
                'max': minmaxmean[1],
                'mean': minmaxmean[2],
            })

        # for each property, compute its length
        maxlen = {}
        for p in props: maxlen[p] = 0
        for sp in str_props:
            for p in props:
                maxlen[p] = max(maxlen[p], len(sp[p]))

        # if any property got all empty strings, don't bother printing it, remove if from the list
        props = [p for p in props if maxlen[p] > 0]

        # print a header
        header_str = ""
        for p in props:
            prefix = "" if p == 'name' else " | "
            fmt_key = ">" if p == 'name' else "<"
            header_str += f"{prefix}{p:{fmt_key}{maxlen[p]}}"
        print(header_str)
        print("-" * len(header_str))

        # now print the acual arrays
        for strp in str_props:
            for p in props:
                prefix = "" if p == 'name' else " | "
                fmt_key = ">" if p == 'name' else "<"
                print(f"{prefix}{strp[p]:{fmt_key}{maxlen[p]}}", end='')
            print("")

    finally:
        del frame


def calc_model_size(model):
    num_trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    # estimate model size on disk: https://discuss.pytorch.org/t/finding-model-size/130275/2
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return {'n_params': num_trainable_params, 'size_mb': size_all_mb}


class LinearWithWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, gamma=0.95, steps=(1, 2), factors=(1.0, 0.1, 1.0), verbose=False):
        self.steps = steps
        self.factors = factors
        self.gamma = gamma
        super().__init__(optimizer, verbose=verbose)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        if epoch < self.steps[0]:
            # warmup
            lr_factor = 1.0
        elif self.steps[0] <= epoch < self.steps[1]:
            # noisy
            lr_factor = self.factors[1]
        else:
            # standard scheduling
            lr_factor = self.factors[-1] * self.gamma ** (epoch - self.steps[-1])
        return lr_factor


def modulate(x, scale, shift, residual=False):
    if residual:
        return (scale + 1.0) * x + shift
    else:
        return scale * x + shift


"""
JIT scripts
"""


@torch.jit.script
def affine_grid_sample(x, theta, out_dims: Tuple[int, int, int, int], mode: str, align_corners: bool = False,
                       padding_mode: str = 'zeros'):
    # construct sampling grid
    grid = F.affine_grid(theta, torch.Size(out_dims), align_corners=align_corners)
    # sample image from grid
    return F.grid_sample(x, grid, align_corners=align_corners, mode=mode, padding_mode=padding_mode)
