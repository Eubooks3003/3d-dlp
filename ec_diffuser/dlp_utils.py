import os
import json
import pickle
from argparse import Namespace

import numpy as np
import matplotlib.pyplot as plt
import torch

from dlp2.models import ObjectDLP
from dlp2.utils.util_func import plot_keypoints_on_image
from torchvision import utils as vutils
from latent_classifier import MLPClassifier
import cv2


"""
Misc
"""

def set_ipdb_debugger():
    import sys
    import ipdb
    import traceback

    def info(t, value, tb):
        if t == KeyboardInterrupt:
            traceback.print_exception(t, value, tb)
            return
        else:
            traceback.print_exception(t, value, tb)
            ipdb.pm()

    sys.excepthook = info


def check_config(config, isaac_env_cfg=None, policy_config=None):
    method = config['Model']['method']
    obs_type = config['Model']['obsType']
    obs_mode = config['Model']['obsMode']

    assert method in ['ECRL', 'SMORL', 'Unstructured']
    assert obs_type in ['State', 'Image']

    if obs_type == 'State':
        if method in ['ECRL', 'SMORL']:
            assert obs_mode == 'state'
        if method == 'Unstructured':
            assert obs_mode == 'state_unstruct'

    if obs_type == 'Image':
        if method == 'ECRL':
            assert obs_mode in ['dlp', 'slot']
        if method == 'SMORL':
            assert obs_mode == 'dlp' and config['Model']['ChamferReward']
        if method == 'Unstructured':
            assert obs_mode == 'vae' or obs_mode == 'vqvae'

    if config['Model']['ChamferReward']:
        assert method in ['ECRL', 'SMORL'] and obs_type == 'Image' and obs_mode == 'dlp'

    # policy related
    if policy_config is not None:
        if obs_mode in ['state', 'slot']:
            assert not policy_config[method][obs_type]['actor_kwargs'].get('masking', False)

    # environment related
    # if isaac_env_cfg is not None:
    #     if isaac_env_cfg["env"].get("PushT", False):
    #         assert isaac_env_cfg["env"].get("numColors", isaac_env_cfg["env"]["numObjects"]) == 1


def get_run_name(config, isaac_env_cfg, seed):
    name = f"{isaac_env_cfg['env']['numObjects']}C_{config['Model']['method']}_{config['Model']['obsType']}"
    if config['Model']['obsMode'] == 'slot':
        name += "_Slot"
    if config['Model']['ChamferReward']:
        name += "_ChamferReward"
    if isaac_env_cfg['env']['tableDims'][0] < 0.5:
        name += "_SmallTable"
    for key in ["AdjacentGoals", "OrderedPush", "PushT", "RandColor", "RandNumObj"]:
        if key in isaac_env_cfg['env'] and isaac_env_cfg['env'][key]:
            name += f"_{key}"
    return name


"""
Logging
"""


def compute_gradients(parameters):
    total_gradient_norm = None
    for p in parameters:
        # if p.grad is None:
        #     continue
        current = p.grad.data.norm(2) ** 2
        if total_gradient_norm is None:
            total_gradient_norm = current
        else:
            total_gradient_norm += current
    return total_gradient_norm ** 0.5


def compute_params(parameters):
    total_param_norm = None
    for p in parameters:
        current = p.data.norm(2) ** 2
        if total_param_norm is None:
            total_param_norm = current
        else:
            total_param_norm += current
    return total_param_norm ** 0.5


def get_max_param(parameters):
    max_p = 0
    for p in parameters:
        current = p.data.abs().max()
        if current > max_p:
            max_p = current
    return max_p


"""
Pretrained Representation
"""

def load_pretrained_rep_model(dir_path, model_type='dlp', pandapush=True, langtable=False):

    if model_type not in ['dlp', 'vqvae', 'vae', 'slot']:
        return None
    if pandapush:
        if model_type == 'dlp':
            ckpt_path = os.path.join(dir_path, f'{model_type}_panda_push.pth')
        elif model_type == 'slot':
            ckpt_path = os.path.join(dir_path, f'{model_type}_panda_push.ckpt')
    elif langtable:
        if model_type == 'dlp':
            ckpt_path = os.path.join(dir_path, f'{model_type}_langtable.pth')
    else:
        if model_type == 'dlp':
            ckpt_path = os.path.join(dir_path, 'saves', f'{model_type}_kitchen.pth')
        else:
            ckpt_path = os.path.join(dir_path, f'{model_type}_kitchen.ckpt')
    if model_type == 'dlp':
        print("\nLoading pretrained DLP...")
        # load config
        conf_path = os.path.join(dir_path, 'hparams.json')
        with open(conf_path, 'r') as f:
            config = json.load(f)
        # initialize model
        model = ObjectDLP(cdim=config['cdim'], enc_channels=config['enc_channels'],
                          prior_channels=config['prior_channels'],
                          image_size=config['image_size'], n_kp=config['n_kp'],
                          learned_feature_dim=config['learned_feature_dim'],
                          bg_learned_feature_dim=config['bg_learned_feature_dim'],
                          pad_mode=config['pad_mode'],
                          sigma=config['sigma'],
                          dropout=False, patch_size=config['patch_size'], n_kp_enc=config['n_kp_enc'],
                          n_kp_prior=config['n_kp_prior'], kp_range=config['kp_range'],
                          kp_activation=config['kp_activation'],
                          anchor_s=config['anchor_s'],
                          use_resblock=False,
                          scale_std=config['scale_std'],
                          offset_std=config['offset_std'], obj_on_alpha=config['obj_on_alpha'],
                          obj_on_beta=config['obj_on_beta'])
        # load model from checkpoint
        model.load_state_dict(torch.load(ckpt_path))
    elif model_type == 'vqvae':
        ### VQ-VAE: Need to get relevant files from the ECRL codebase: https://github.com/DanHrmti/ECRL/tree/master/vae
        # from vae.models.vqvae import VQModel
        # print("\nLoading pretrained VAE...")
        # # load config
        # conf_path = os.path.join(dir_path, 'hparams.json')
        # with open(conf_path, 'r') as f:
        #     config = json.load(f)
        # model = VQModel(embed_dim=config['embed_dim'],
        #                 n_embed=config['n_embed'],
        #                 double_z=False,
        #                 z_channels=config['z_channels'],
        #                 resolution=config['image_size'],
        #                 in_channels=config['ch'],
        #                 out_ch=config['ch'],
        #                 ch=config['base_ch'],
        #                 ch_mult=config['ch_mult'],  # num_down = len(ch_mult)-1
        #                 num_res_blocks=config['num_res_blocks'],
        #                 attn_resolutions=config['attn_resolutions'],
        #                 dropout=config['dropout'],
        #                 device=torch.device(config['device']),
        #                 ckpt_path=config['pretrained_path'],
        #                 ignore_keys=[],
        #                 remap=None,
        #                 sane_index_shape=False)
        # # load model from checkpoint
        # model.load_state_dict(torch.load(ckpt_path))
        # del model.loss
        # model.loss = None
        raise NotImplementedError("VQ-VAE requires additional files from the ECRL codebase")
    elif model_type == 'vae':
        ### VAE: Need to get relevant files from the ECRL codebase: https://github.com/DanHrmti/ECRL/tree/master/vae
        # from vae.models.vae import VAEModel
        # print("\nLoading pretrained VAE...")
        # # load config
        # conf_path = os.path.join(dir_path, 'hparams.json')
        # with open(conf_path, 'r') as f:
        #     config = json.load(f)
        # # initialize model
        # model = VAEModel(double_z=False,
        #                  z_channels=config['z_channels'],
        #                  resolution=config['image_size'],
        #                  in_channels=config['ch'],
        #                  out_ch=config['ch'],
        #                  ch=config['base_ch'],
        #                  ch_mult=config['ch_mult'],  # num_down = len(ch_mult)-1
        #                  num_res_blocks=config['num_res_blocks'],
        #                  attn_resolutions=config['attn_resolutions'],
        #                  dropout=config['dropout'],
        #                  latent_dim=config['latent_dim'],
        #                  kl_weight=config['beta_kl'],
        #                  device=torch.device(config['device']),
        #                  ckpt_path=config['pretrained_path'],
        #                  ignore_keys=[],
        #                  remap=None,
        #                  sane_index_shape=False)
        # # load model from checkpoint
        # model.load_state_dict(torch.load(ckpt_path))
        # del model.loss
        # model.loss = None
        raise NotImplementedError("VAE requires additional files from the ECRL codebase")
    elif model_type == 'slot':
        ### Slot Attention: Need to get relevant files from the ECRL codebase: https://github.com/DanHrmti/ECRL/tree/master/slot_attention
        # from slot_attention.method import SlotAttentionMethod
        # from slot_attention.slot_attention_model import SlotAttentionModel
        # from slot_attention.utils import to_rgb_from_tensor, sa_segment
        # print("\nLoading pretrained Slot-Attention...")
        # # load config
        # ckpt = torch.load(ckpt_path)
        # params = Namespace(**ckpt["hyper_parameters"])
        # # initialize model
        # sa = SlotAttentionModel(
        #     resolution=params.resolution,
        #     num_slots=params.num_slots,
        #     num_iterations=params.num_iterations,
        #     slot_size=params.slot_size,
        # )
        # # load model from checkpoint
        # model = SlotAttentionMethod.load_from_checkpoint(ckpt_path, model=sa, datamodule=None)
        raise NotImplementedError("Slot Attention requires additional files from the ECRL codebase")
    else:
        raise NotImplementedError(f"Pretrained model type '{model_type}' is not supported")

    model.eval()
    model.requires_grad_(False)

    print(f"Loaded pretrained representation model from {ckpt_path}\n")

    return model


def get_dlp_rep(dlp_output, topk=None):
    pixel_xy = dlp_output['z']
    scale_xy = dlp_output['mu_scale']
    depth = dlp_output['mu_depth']
    visual_features = dlp_output['mu_features']
    transp = dlp_output['obj_on'].unsqueeze(dim=-1)
    rep = torch.cat((pixel_xy, scale_xy, depth, visual_features, transp), dim=-1)

    if topk is not None:
        _, indices = torch.topk(transp.squeeze(-1), k=topk, dim=-1, largest=True)
        batch_indices = torch.arange(rep.shape[0]).view(-1, 1).to(rep.device)
        rep = rep[batch_indices, indices]
    return rep

def extract_dlp_features_with_bg(images, dlp_model, device, topk=None):
    orig_image_shape = images.shape
    if len(orig_image_shape) == 3:
        images = np.expand_dims(images, axis=0)
    if isinstance(images, torch.Tensor):
        normalized_images = images.float() / 255
    else:
        normalized_images = images.astype('float32') / 255
        normalized_images = torch.tensor(normalized_images, device=device)

    with torch.no_grad():
        encoded_output = dlp_model.encode_all(normalized_images, deterministic=True)
        particles = get_dlp_rep(encoded_output, topk=topk)

    return particles, encoded_output['z_bg']

def get_recon_from_dlps(particles, z_bg, latent_rep_model, device, ret_glimpse=False):
    if not isinstance(particles, torch.Tensor):
        particles = torch.tensor(particles, device=device)
    if not isinstance(z_bg, torch.Tensor):
        z_bg = torch.tensor(z_bg, device=device)
    last_dim = particles.shape[-1]
    pixel_xy = particles[..., :2]
    scale_xy = particles[..., 2:4]
    depth = particles[..., 4:5]
    visual_features = particles[..., 5:last_dim-1]
    transp = particles[..., last_dim-1]
    with torch.no_grad():
        decoder_out = latent_rep_model.decode_all(pixel_xy, visual_features, z_bg, transp, z_depth=depth, z_scale=scale_xy)
    recon = decoder_out['rec']
    recon = recon.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    recon = (recon * 255).astype(np.uint8)
    if ret_glimpse:
        dec_object_glimpses = decoder_out['dec_objects'].squeeze()
        _, dec_object_glimpses = torch.split(dec_object_glimpses, [1, 3], dim=1)
        dec_object_glimpses = torch.cat([dec_object_glimpses[i] for i in range(len(dec_object_glimpses))], dim=1)
        dec_object_glimpses = dec_object_glimpses.cpu().numpy()
        dec_object_glimpses = np.moveaxis(dec_object_glimpses, 0, -1)
        H, W, C = dec_object_glimpses.shape
        n_particles = 24
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(dec_object_glimpses)
        ax.set_title(f"Particle Glimpses", fontsize=12)
        ax.set_xticks([], [])
        ax.set_yticks(range(W // 2 - 1, W // 2 + W * n_particles - 1, W), [f"{i}" for i in range(n_particles)])
        for i in range(1, n_particles):
            ax.axhline(y=i * W, color='black')
        fig.canvas.draw()
        glimpse_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        # resize to be same shapre as recon
        glimpse_plot = glimpse_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # close the plot
        plt.close(fig)
        # glimpse_plot = cv2.resize(glimpse_plot, (recon.shape[1], recon.shape[0]))
        recon = cv2.resize(recon, (glimpse_plot.shape[1], glimpse_plot.shape[1]))
        return recon, glimpse_plot
    return recon

def extract_dlp_image(images, latent_rep_model, device, topk=None):
    orig_image_shape = images.shape
    if len(orig_image_shape) == 3:
        images = np.expand_dims(images, axis=0)
    normalized_images = images.astype('float32') / 255
    normalized_images = torch.tensor(normalized_images, device=device)

    with torch.no_grad():
        encoded_output = latent_rep_model.encode_all(normalized_images, deterministic=True)
        pixel_xy = encoded_output['z']
        # # filter particle to 10 based on transparency
        if topk is not None:
            transp = encoded_output['obj_on'].unsqueeze(dim=-1)
            _, indices = torch.topk(transp.squeeze(-1), k=topk, dim=-1, largest=True)
            batch_indices = torch.arange(pixel_xy.shape[0]).view(-1, 1).to(pixel_xy.device)
            pixel_xy = pixel_xy[batch_indices, indices]

    dlp_images = []
    for kp_xy, image in zip(pixel_xy, normalized_images):
        dlp_images.append(
            plot_keypoints_on_image(kp_xy, image, radius=2, thickness=1, kp_range=(-1, 1), plot_numbers=False))

    if len(dlp_images) == 1:
        dlp_images = dlp_images[0]

    return dlp_images

"""
Reward
"""


def batch_pairwise_dist(x, y, metric='l2_simple'):
    assert metric in ['l2', 'l2_simple', 'l1', 'cosine'], f'metric {metric} unrecognized'
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    if metric == 'cosine':
        dist_func = torch.nn.functional.cosine_similarity
        P = -dist_func(x.unsqueeze(2), y.unsqueeze(1), dim=-1, eps=1e-8)
    elif metric == 'l1':
        P = torch.abs(x.unsqueeze(2) - y.unsqueeze(1)).sum(-1)
    elif metric == 'l2_simple':
        P = ((x.unsqueeze(2) - y.unsqueeze(1)) ** 2).sum(-1)
    else:
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind_x = torch.arange(0, num_points_x, device=x.device)
        diag_ind_y = torch.arange(0, num_points_y, device=y.device)
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = rx.transpose(2, 1) + ry - 2 * zz
    return P

def load_latent_classifier(config, num_objects):
    if config['Model']['obsMode'] == 'dlp' and (config['Model']['ChamferReward'] or config['Model']['method'] == 'SMORL'):
        dir_path = config['Reward']['LatentClassifier']['path']
        latent_classifier_chkpt_path = f'{dir_path}/latent_classifier_{num_objects}C_dlp_push_5C'
        latent_classifier = MLPClassifier(**config['Reward']['LatentClassifier']['params'])
        latent_classifier.mlp.load_state_dict(torch.load(latent_classifier_chkpt_path))
        print(f"Loaded latent_classifier model from {latent_classifier_chkpt_path}")
        return latent_classifier
    

################ 3D DLP ##############



import os
import math
import numpy as np
import torch

# ---------------------------
# AARCH-safe conversions
# ---------------------------

import os
import numpy as np
import torch

def _torch_from_any_no_numpy_bridge(x, *, device=None, dtype=torch.float32):
    if x is None:
        return None
    if torch.is_tensor(x):
        t = x
        if device is not None:
            t = t.to(device=device)
        return t.to(dtype=dtype)
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, np.ndarray):
        x = x.tolist()
    return torch.tensor(x, dtype=dtype, device=device)

def _to_np_no_torch_numpy(t: torch.Tensor, dtype=np.float32) -> np.ndarray:
    return np.asarray(t.detach().cpu().tolist(), dtype=dtype)

def _write_ply_ascii_xyzrgb(out_path, xyz: np.ndarray, rgb: np.ndarray = None):
    """
    xyz: [N,3] float32
    rgb: [N,3] uint8 or None
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if rgb is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        if rgb is None:
            for x, y, z in xyz:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        else:
            for (x, y, z), (r, g, b) in zip(xyz, rgb):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

@torch.no_grad()
def log_rgb_voxels(
    name: str,
    rgb_vol,
    alpha_vol=None,
    KPx=None,
    *,
    step=None,
    mode="splat",
    topk=60000,
    alpha_thresh=0.05,   # kept for API compat
    mesh_iso=0.2,        # unused (splat only)
    pad=2.0,
    show_axes=True,
):
    """
    AARCH-safe Plotly voxel logger:
      - NEVER calls Tensor.numpy() (PyTorch NumPy bridge)
      - ALSO avoids torch.as_tensor(numpy_*) which can fail when torch lacks NumPy support
      - Accepts torch tensors, python lists, numpy arrays/scalars (via .tolist() / .item()).
      - Draws KPx crosses internally.
    """
    import numpy as np
    import torch
    import plotly.graph_objects as go
    import wandb

    # ---------- conversions that do NOT rely on torch<->numpy bridge ----------
    def _torch_from_any(x, *, device=None, dtype=torch.float32):
        """
        Convert x to torch.Tensor WITHOUT torch.as_tensor on numpy objects.
        """
        if x is None:
            return None

        if torch.is_tensor(x):
            t = x
            if device is not None:
                t = t.to(device=device)
            return t.to(dtype=dtype)

        # numpy scalar -> python scalar
        if isinstance(x, np.generic):
            x = x.item()

        # numpy array -> python nested lists
        if isinstance(x, np.ndarray):
            x = x.tolist()

        # list/tuple/scalar -> torch.tensor (safe)
        t = torch.tensor(x, dtype=dtype, device=device)
        return t

    def _as_cd_hw_torch(x):
        # If you have _as_CDHW, use it; but still convert robustly after.
        try:
            t = _as_CDHW(x)
        except NameError:
            t = x

        # robust convert (works even if t is numpy array / numpy scalar / list)
        t = _torch_from_any(t, dtype=torch.float32)

        # shape normalization
        if t.ndim == 5:
            t = t[0]
        if not (t.ndim == 4 and int(t.shape[0]) == 3):
            raise ValueError(f"rgb_vol must be [3,D,H,W] or [1,3,D,H,W], got shape {tuple(t.shape)}")
        return t

    def _as_dhw_torch(x, *, device):
        try:
            t = _as_DHW(x)
        except NameError:
            t = x
        t = _torch_from_any(t, device=device, dtype=torch.float32)
        if t.ndim == 4:
            t = t[0]
        if t.ndim != 3:
            raise ValueError(f"alpha_vol must be [D,H,W] or [1,D,H,W], got shape {tuple(t.shape)}")
        return t

    def _to_np_no_torch_numpy(t: torch.Tensor) -> np.ndarray:
        # Avoids Tensor.numpy(); converts via Python lists.
        return np.asarray(t.detach().cpu().tolist(), dtype=np.float32)

    def _add_kp_crosses(fig, kpx_any, half=2.0):
        if kpx_any is None:
            return
        if torch.is_tensor(kpx_any):
            kpx = _to_np_no_torch_numpy(kpx_any)
        else:
            if isinstance(kpx_any, np.generic):
                kpx_any = kpx_any.item()
            if isinstance(kpx_any, np.ndarray):
                kpx_any = kpx_any.tolist()
            kpx = np.asarray(kpx_any, dtype=np.float32)

        kpx = kpx.reshape(-1, 3)
        if kpx.size == 0:
            return

        xs, ys, zs = [], [], []
        for x, y, z in kpx:
            x0, x1 = x - half, x + half
            y0, y1 = y - half, y + half
            z0, z1 = z - half, z + half
            # X
            xs += [x0, x1, None]; ys += [y,  y,  None]; zs += [z,  z,  None]
            # Y
            xs += [x,  x,  None]; ys += [y0, y1, None]; zs += [z,  z,  None]
            # Z
            xs += [x,  x,  None]; ys += [y,  y,  None]; zs += [z0, z1, None]

        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name="KPs", line=dict(width=6)))

    # ---------- main ----------
    if mode != "splat":
        raise ValueError(f"Unknown mode='{mode}' (only 'splat' implemented).")

    RGB = _as_cd_hw_torch(rgb_vol)  # torch [3,D,H,W]
    D, H, W = map(int, RGB.shape[1:])
    device = RGB.device

    ALP = None
    if alpha_vol is not None:
        ALP = _as_dhw_torch(alpha_vol, device=device)
        if tuple(ALP.shape) != (D, H, W):
            raise ValueError(f"alpha_vol shape {tuple(ALP.shape)} != rgb spatial {(D, H, W)}")

    fig = go.Figure()

    # thresholds (same spirit as your old code)
    alpha_thresh_local = 0.25
    rgb_mag_thresh = 0.10

    weights = ALP.clamp(0, 1) if ALP is not None else None

    # IMPORTANT: use torch.sum(dim=0) only after RGB is guaranteed torch
    mag = torch.sqrt((RGB ** 2).sum(dim=0))  # [D,H,W]

    if weights is None:
        mask = mag >= rgb_mag_thresh
    else:
        mask = (weights >= alpha_thresh_local) & (mag >= rgb_mag_thresh)

    idx = mask.nonzero(as_tuple=False)  # [N,3] z,y,x
    if idx.numel() == 0:
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="markers", name="RGB voxels"))
        if step is not None:
            wandb.log({name: fig}, step=step)
        return fig

    if idx.shape[0] > topk:
        score = mag[idx[:, 0], idx[:, 1], idx[:, 2]] if weights is None else weights[idx[:, 0], idx[:, 1], idx[:, 2]]
        sel = torch.topk(score, k=topk, largest=True).indices
        idx = idx[sel]

    z_i, y_i, x_i = idx[:, 0], idx[:, 1], idx[:, 2]

    r = RGB[0, z_i, y_i, x_i]
    g = RGB[1, z_i, y_i, x_i]
    b = RGB[2, z_i, y_i, x_i]

    # map [-1,1] -> [0,1] if needed
    if (r.min() < 0) or (g.min() < 0) or (b.min() < 0):
        r = (r + 1) * 0.5
        g = (g + 1) * 0.5
        b = (b + 1) * 0.5

    Rv = (r.clamp(0, 1) * 255).to(torch.uint8)
    Gv = (g.clamp(0, 1) * 255).to(torch.uint8)
    Bv = (b.clamp(0, 1) * 255).to(torch.uint8)

    if weights is None:
        opac = torch.ones_like(r, dtype=torch.float32)
    else:
        opac = weights[z_i, y_i, x_i].to(torch.float32).clamp(0, 1)

    # plotly wants python lists/strings
    x_f = x_i.to(torch.float32).detach().cpu().tolist()
    y_f = y_i.to(torch.float32).detach().cpu().tolist()
    z_f = z_i.to(torch.float32).detach().cpu().tolist()

    Rl = Rv.detach().cpu().tolist()
    Gl = Gv.detach().cpu().tolist()
    Bl = Bv.detach().cpu().tolist()
    Ol = opac.detach().cpu().tolist()

    color_rgba = [f"rgba({Rl[k]},{Gl[k]},{Bl[k]},{float(Ol[k])})" for k in range(len(Rl))]

    fig.add_trace(
        go.Scatter3d(
            x=x_f, y=y_f, z=z_f,
            mode="markers",
            marker=dict(size=2, color=color_rgba),
            name="RGB voxels",
        )
    )

    if KPx is not None:
        _add_kp_crosses(fig, KPx, half=2.0)

    if show_axes:
        fig.update_scenes(
            xaxis=dict(range=[-pad, W - 1 + pad]),
            yaxis=dict(range=[-pad, H - 1 + pad]),
            zaxis=dict(range=[-pad, D - 1 + pad]),
            aspectmode="data",
        )

    fig.update_layout(
        title=f"RGB voxels ({mode})",
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=True,
        legend=dict(orientation="h"),
        scene=dict(xaxis_title="X (W)", yaxis_title="Y (H)", zaxis_title="Z (D)"),
    )

    if step is not None:
        wandb.log({name: fig}, step=step)
    return fig


def save_npz_volume(vol_any, out_npz_path: str):
    """
    Save raw volume locally (npz). Uses list conversion (AARCH-safe).
    """
    v = _torch_from_any_no_numpy_bridge(vol_any, dtype=torch.float32)
    v_np = _to_np_no_torch_numpy(v, dtype=np.float32)
    os.makedirs(os.path.dirname(out_npz_path), exist_ok=True)
    np.savez_compressed(out_npz_path, vol=v_np)


# ---------------------------
# 3D "recon from dlps"
# ---------------------------

@torch.no_grad()
def get_recon_from_dlps_3d(
    particles_flat,
    latent_rep_model,
    device,
    *,
    particle_dim: int,
    # how to unpack each particle vector
    pos_slice=(0, 3),   # xyz
    scale_slice=(3, 6),    # sx,sy,sz
    depth_slice=(6, 7),    # optional
    obj_on_slice=(7, 8),   # optional (transp/obj_on)
    feat_slice=(8, None),  # remaining dims
    # optional extra latents (if you have them)
    z_bg_features=None,
    z_context=None,
):
    """
    particles_flat:
      - flat array (K*particle_dim,) or (K, particle_dim) or (1,K,particle_dim)

    Returns:
      dec_dict from latent_rep_model.decode_all(...)
    """
    p = _torch_from_any_no_numpy_bridge(particles_flat, device=device, dtype=torch.float32)

    # normalize shape to [1,K,Dp]
    if p.ndim == 1:
        assert p.numel() % particle_dim == 0, f"flat particles size {p.numel()} not divisible by particle_dim={particle_dim}"
        K = p.numel() // particle_dim
        p = p.view(1, K, particle_dim)
    elif p.ndim == 2:
        p = p.unsqueeze(0)
    elif p.ndim == 3:
        pass
    else:
        raise ValueError(f"particles_flat must be 1D/2D/3D, got {tuple(p.shape)}")

    z = p[..., pos_slice[0]:pos_slice[1]]
    z_scale  = p[..., scale_slice[0]:scale_slice[1]]

    z_depth = None
    if depth_slice is not None:
        z_depth = p[..., depth_slice[0]:depth_slice[1]]

    z_obj_on = None
    if obj_on_slice is not None:
        z_obj_on = p[..., obj_on_slice[0]:obj_on_slice[1]]

    if feat_slice[1] is None:
        z_features = p[..., feat_slice[0]:]
    else:
        z_features = p[..., feat_slice[0]:feat_slice[1]]

    # bg/context can be tensors or None
    z_bg_features = _torch_from_any_no_numpy_bridge(z_bg_features, device=device, dtype=torch.float32) if z_bg_features is not None else None
    z_context     = _torch_from_any_no_numpy_bridge(z_context, device=device, dtype=torch.float32) if z_context is not None else None


    print("z shape; ", z.shape)
    print("z_scale shape; ", z_scale.shape)
    print("z_features shape; ", z_features.shape)
    print("z_obj_on shape; ", z_obj_on.shape)
    dec_dict = latent_rep_model.decode_all(
        z,
        z_scale,
        z_features,
        z_obj_on,
        z_depth,
        z_bg_features,
        z_context,
        warmup=False,
    )
    return dec_dict
