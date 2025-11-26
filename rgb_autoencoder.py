# voxel_autoencoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class VoxelRGBEncoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=32, latent_dim=256):
        super().__init__()
        # [B,3,64,64,64] -> [B,32,32,32,32]
        self.conv1 = nn.Conv3d(in_ch, base_ch, kernel_size=3, stride=2, padding=1)
        self.bn1   = nn.BatchNorm3d(base_ch)

        # [B,32,32,32,32] -> [B,64,16,16,16]
        self.conv2 = nn.Conv3d(base_ch, base_ch*2, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm3d(base_ch*2)

        # [B,64,16,16,16] -> [B,128,8,8,8]
        self.conv3 = nn.Conv3d(base_ch*2, base_ch*4, kernel_size=3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm3d(base_ch*4)

        self.flatten_dim = base_ch*4 * 8 * 8 * 8
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        z = self.fc_mu(x)
        return z


class VoxelRGBDecoder(nn.Module):
    def __init__(self, out_ch=3, base_ch=32, latent_dim=256):
        super().__init__()
        # mirror encoder dims
        self.start_D = self.start_H = self.start_W = 8
        self.start_ch = base_ch * 4
        self.fc = nn.Linear(latent_dim, self.start_ch * self.start_D * self.start_H * self.start_W)

        self.deconv1 = nn.ConvTranspose3d(self.start_ch, base_ch*2, kernel_size=4, stride=2, padding=1)
        self.bn1     = nn.BatchNorm3d(base_ch*2)

        self.deconv2 = nn.ConvTranspose3d(base_ch*2, base_ch, kernel_size=4, stride=2, padding=1)
        self.bn2     = nn.BatchNorm3d(base_ch)

        self.deconv3 = nn.ConvTranspose3d(base_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), self.start_ch, self.start_D, self.start_H, self.start_W)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = self.deconv3(x)  # [B,3,D,H,W]
        x = torch.sigmoid(x)  # assume voxel RGB in [0,1]
        return x


class VoxelRGBAutoencoder(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=32, latent_dim=256):
        super().__init__()
        self.encoder = VoxelRGBEncoder(in_ch=in_ch, base_ch=base_ch, latent_dim=latent_dim)
        self.decoder = VoxelRGBDecoder(out_ch=out_ch, base_ch=base_ch, latent_dim=latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

# losses_voxel_rgb.py

import torch
import torch.nn.functional as F


def voxel_rgb_recon_loss(
    x,                 # [B, C, D, H, W], C=3
    rec_x,             # [B, C, D, H, W]
    loss_type="mse",   # {"mse", "l1"}
    occ=None,          # [B,1,D,H,W] or None (optional occupancy mask)
    fg_weight=1.0,     # weight on foreground voxels if occ is not None
    bg_weight=1.0,     # weight on background voxels if occ is not None
):
    assert x.shape == rec_x.shape, f"shape mismatch: x {x.shape}, rec_x {rec_x.shape}"
    if loss_type == "mse":
        err = (rec_x - x) ** 2
    elif loss_type == "l1":
        err = (rec_x - x).abs()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    if occ is not None:
        # occ: [B,1,D,H,W] in {0,1} (or [0,1])
        if occ.shape[1] != 1:
            raise ValueError(f"occ must be [B,1,D,H,W], got {occ.shape}")
        fg = (occ > 0.5).float()
        bg = 1.0 - fg
        w = fg_weight * fg + bg_weight * bg  # [B,1,D,H,W]
        w = w.expand_as(err)                 # broadcast to [B,C,D,H,W]
        err = err * w
        loss = err.sum() / w.sum().clamp_min(1.0)
    else:
        loss = err.mean()

    # PSNR (mean over batch)
    with torch.no_grad():
        mse_per_batch = ((rec_x - x) ** 2).view(x.shape[0], -1).mean(dim=1)  # [B]
        psnr_per_batch = -10.0 * torch.log10(mse_per_batch + 1e-8)
        psnr = psnr_per_batch.mean()

    loss_dict = {
        "loss": loss,
        "loss_rec": loss,
        "psnr": psnr,
    }
    return loss, loss_dict
