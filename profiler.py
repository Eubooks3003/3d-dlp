import torch
from hy_models import DLP
# datasets
from datasets.get_dataset import get_video_dataset
# utils
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.autograd.profiler as profiler

if __name__ == '__main__':
    ds = 'balls'
    root = '/media/newhd/data/gswm_balls/BALLS_OCCLUSION'
    timestep_horizon = 20
    image_size = 64
    batch_size = 6
    dataset = get_video_dataset(ds, root, seq_len=timestep_horizon + 1, mode='train', image_size=image_size)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True,
                            drop_last=True)
    # example hyper-parameters
    # batch_size = 1
    beta_kl = 0.1
    beta_rec = 1.0
    kl_balance = 0.001  # balance between spatial attributes (x, y, scale, depth) and visual features
    n_kp_enc = 8
    n_kp_prior = 64
    patch_size = 8  # patch size for the prior to generate prior proposals
    # patch_size = 64  # patch size for the prior to generate prior proposals
    learned_feature_dim = 3  # visual features
    anchor_s = 0.25  # effective patch size for the posterior: anchor_s * image_size
    image_size = 64
    # image_size = 256
    # pad_mode = 'replicate'
    pad_mode = 'zeros'
    ch = 3
    enc_channels = [32, 64, 128]
    patch_channels = (32, 32, 64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    pint_layers = 6  # transformer-based dynamics module number of layers
    pint_heads = 8  # transformer-based dynamics module attention heads
    pint_dim = 256  # transformer-based dynamics module inner dimension (+projection dim)
    beta_dyn = 0.1  # beta-kl for the dynamics loss
    num_static_frames = 4  # "burn-in frames", number of initial frames with kl w.r.t. constant prior (as in DLPv2)
    context_dim = 7
    deterministic = False
    noisy = warmup = False
    predict_delta = True
    # attn_norm_type = 'ln'
    # attn_norm_type = 'pn'
    attn_norm_type = 'rms'
    model = DLP(cdim=ch, enc_channels=enc_channels, patch_channels=patch_channels,
                image_size=image_size, n_kp=n_kp_enc, learned_feature_dim=learned_feature_dim,
                learned_bg_feature_dim=learned_feature_dim,
                context_dim=context_dim,
                patch_size=patch_size, n_kp_enc=n_kp_enc,
                n_kp_prior=n_kp_prior,
                anchor_s=anchor_s, pad_mode=pad_mode,
                timestep_horizon=timestep_horizon, predict_delta=predict_delta,
                pint_layers=pint_layers, pint_heads=pint_heads,
                pint_dim=pint_dim, filtering_heuristic="none", attn_norm_type='rms').to(device)
    print(f'model.info():')
    print(model.info())
    print("----------------------------------")

    with profiler.profile(with_stack=True, profile_memory=True) as prof:
        x_ts = (timestep_horizon + 1) if timestep_horizon > 1 else 1
        x = torch.rand(batch_size, x_ts, ch, image_size, image_size, device=device)
        model_output = model(x, deterministic, warmup, noisy)
    print(prof.key_averages(group_by_stack_n=15).table(sort_by='self_cpu_time_total', row_limit=15))
    print(prof.key_averages(group_by_stack_n=15).table(sort_by='self_cpu_memory_usage', row_limit=15))

    for batch in tqdm(dataloader):
        batch[0].to(device)
        continue
