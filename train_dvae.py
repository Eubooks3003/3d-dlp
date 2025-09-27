"""
Single-GPU training of DVae
"""
# imports
import numpy as np
import os
from tqdm import tqdm
import matplotlib
import argparse
# torch
import torch
from utils.loss_functions import calc_reconstruction_loss, LossLPIPS
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.optim as optim
# modules
from vae_models import DVae
# datasets
from datasets.get_dataset import get_video_dataset
# util functions
from utils.util_func import prepare_logdir, save_config, log_line, get_config, LinearWithWarmupScheduler, \
    format_epoch_summary_dvae, plot_training_metrics, save_metrics_data, save_code_backup
from eval.eval_model import evaluate_validation_dvae_elbo_dyn, animate_trajectory_ddlp
from eval.eval_gen_metrics import eval_ddlp_im_metric

matplotlib.use("Agg")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def train_dvae(config_path='./configs/balls.json'):
    # load config
    try:
        config = get_config(config_path)
    except FileNotFoundError:
        raise SystemExit("config file not found")
    hparams = config  # to save a copy of the hyper-parameters
    device = config['device']
    if 'cuda' in device:
        device = torch.device(f'{device}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    # data and general
    ds = config['ds']
    ch = config['ch']  # image channels
    image_size = config['image_size']
    n_views = config.get('n_views', 1)
    root = config['root']  # dataset root
    run_prefix = config['run_prefix']
    load_model = config['load_model']
    pretrained_path = config['pretrained_path']  # path of pretrained model to load, if None, train from scratch

    # model
    timestep_horizon = config['timestep_horizon']
    pad_mode = config['pad_mode']

    # visual latent features
    features_dist = config.get('features_dist', 'gauss')
    learned_feature_dim = config['learned_feature_dim']
    n_fg_categories = config.get('n_fg_categories', 8)  # Number of foreground feature categories (if categorical)
    n_fg_classes = config.get('n_fg_classes', 4)  # Number of foreground feature classes per category

    # latent context
    context_dist = config.get('context_dist', 'gauss')
    context_dim = config['context_dim']
    ctx_pool_mode = config.get("ctx_pool_mode", "none")
    n_ctx_categories = config.get('n_ctx_categories', 8)  # Number of context feature categories (if categorical)
    n_ctx_classes = config.get('n_ctx_classes', 4)  # Number of context feature classes per category

    dropout = config['dropout']
    use_resblock = config['use_resblock']

    # optimization
    batch_size = config['batch_size']
    lr = config['lr']
    num_epochs = config['num_epochs']
    start_epoch = config.get('start_epoch', 0)
    weight_decay = config['weight_decay']
    adam_betas = config['adam_betas']
    adam_eps = config['adam_eps']
    use_scheduler = config['use_scheduler']
    scheduler_gamma = config['scheduler_gamma']
    warmup_epoch = config['warmup_epoch']
    recon_loss_type = config['recon_loss_type']
    beta_kl = config['beta_kl']
    beta_dyn = config['beta_dyn']
    beta_rec = config['beta_rec']
    beta_dyn_rec = config['beta_dyn_rec']
    kl_balance = config['kl_balance']  # balance between visual features and the other particle attributes
    num_static_frames = config['num_static_frames']  # frames for which kl is calculated w.r.t constant prior params

    # evaluation
    eval_epoch_freq = config['eval_epoch_freq']
    eval_im_metrics = config['eval_im_metrics']
    cond_steps = config['cond_steps']  # conditional frames for the dynamics module during inference
    ctx_for_eval = config.get('ctx_for_eval', False)

    # visualization
    animation_horizon = config['animation_horizon']

    # transformer - PINT
    pint_enc_layers = config['pint_enc_layers']
    pint_enc_heads = config['pint_enc_heads']
    pint_ctx_layers = config['pint_ctx_layers']
    pint_ctx_heads = config['pint_ctx_heads']
    pint_dyn_layers = config['pint_dyn_layers']
    pint_dyn_heads = config['pint_dyn_heads']
    pint_dim = config['pint_dim']

    predict_delta = config['predict_delta']  # dynamics module predicts the delta from previous step

    normalize_rgb = config['normalize_rgb']
    res_from_fc = config["res_from_fc"]
    ch_mult = config["ch_mult"]
    base_ch = config["base_ch"]
    final_cnn_ch = config["final_cnn_ch"]
    num_res_blocks = config["num_res_blocks"]
    cnn_mid_blocks = config.get('cnn_mid_blocks', False)
    mlp_hidden_dim = config.get('mlp_hidden_dim', 256)
    use_ep_done_mask = config.get('ep_done_mask', False)

    # actions
    action_condition = config.get('action_condition', False)
    action_dim = config.get('action_dim', 0)
    null_action_embed = config.get('null_action_embed', False)

    random_action_condition = config.get('random_action_condition', False)
    random_action_dim = config.get('random_action_dim', 0)

    # language
    language_condition = config.get('language_condition', False)
    language_embed_dim = config.get('language_embed_dim', 0)
    language_max_len = config.get('language_max_len', 32)

    # image goal condition
    img_goal_condition = config.get('image_goal_condition', False)

    # load data
    dataset = get_video_dataset(ds, root, seq_len=timestep_horizon + 1, mode='train', image_size=image_size)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True,
                            drop_last=True)
    # model
    model = DVae(cdim=ch,  # Number of input image channels
                image_size=image_size,  # Input image size (assumed square)
                normalize_rgb=normalize_rgb,  # If True, normalize RGB to [-1, 1], else keep [0, 1]
                n_views=n_views,  # number of input views (e.g., multiple cameras)

                # Network configuration
                pad_mode=pad_mode,  # Padding mode for CNNs ('zeros' or 'replicate')
                dropout=dropout,  # Dropout rate for transformers

                # Feature representation
                features_dist=features_dist,  # Distribution type for features ('gauss' or 'categorical')
                learned_feature_dim=learned_feature_dim,  # Dimension of learned visual features
                n_fg_categories=n_fg_categories,  # Number of foreground feature categories (if categorical)
                n_fg_classes=n_fg_classes,  # Number of foreground feature classes per category

                # decoder architecture
                res_from_fc=res_from_fc,  # Initial resolution for background encoder-decoder
                ch_mult=ch_mult,  # Channel multipliers for background encoder-decoder
                base_ch=base_ch,  # Base channels for background encoder-decoder
                final_cnn_ch=final_cnn_ch,  # Final CNN channels for background encoder-decoder

                # Network architecture options
                use_resblock=use_resblock,  # Use residual blocks in encoders-decoders
                num_res_blocks=num_res_blocks,  # Number of residual blocks per resolution
                cnn_mid_blocks=cnn_mid_blocks,  # Use middle blocks in CNN
                mlp_hidden_dim=mlp_hidden_dim,  # Hidden dimension for MLPs

                # Particle interaction transformer (PINT) configuration
                pint_enc_layers=pint_enc_layers,  # Number of PINT encoder layers
                pint_enc_heads=pint_enc_heads,  # Number of PINT encoder attention heads

                # Dynamics configuration
                timestep_horizon=timestep_horizon,  # Number of timesteps to predict ahead
                n_static_frames=num_static_frames,  # Number of initial frames for static KL optimization
                predict_delta=predict_delta,  # Predict position deltas instead of absolute positions
                context_dim=context_dim,  # Context latent dimension (if None, equals learned_feature_dim)
                ctx_dist=context_dist,  # Context distribution type ('gauss' or 'categorical')
                n_ctx_categories=n_ctx_categories,  # Number of context categories (if categorical)
                n_ctx_classes=n_ctx_classes,  # Number of context classes per category
                ctx_pool_mode=ctx_pool_mode,  # Context pooling mode ('none' = per-particle context)

                # Context and dynamics transformer configuration
                pint_dyn_layers=pint_dyn_layers,  # Number of dynamics transformer layers
                pint_dyn_heads=pint_dyn_heads,  # Number of dynamics transformer heads
                pint_dim=pint_dim,  # Hidden dimension for PINT
                pint_ctx_layers=pint_ctx_layers,  # Number of context transformer layers
                pint_ctx_heads=pint_ctx_heads,

                # external conditioning
                action_condition=action_condition,  # condition on actions
                action_dim=action_dim,  # dimension of input actions
                null_action_embed=null_action_embed,
                random_action_condition=random_action_condition,
                random_action_dim=random_action_dim,
                # learn a "no-input-action" embedding, to learn on action-free videos as well
                language_condition=language_condition,  # condition on language embedding
                language_embed_dim=language_embed_dim,  # embedding dimension for each token
                language_max_len=language_max_len,  # maximum tokens per prompt
                img_goal_condition=img_goal_condition,  # condition the future on image goal
                ).to(device)
    model_info = model.info()
    print(model.info())

    # prepare saving location
    run_name = f'{ds}_dvae' + run_prefix
    log_dir = prepare_logdir(runname=run_name, src_dir='./')
    fig_dir = os.path.join(log_dir, 'figures')
    save_dir = os.path.join(log_dir, 'saves')
    save_config(log_dir, hparams)
    log_line(log_dir, model_info)
    # save a backup of the code for this run
    backup_info = save_code_backup('.', backup_dir=os.path.join(log_dir, 'saves', 'code_backup'))
    log_line(log_dir, backup_info)
    print(backup_info)

    # prepare loss functions
    if recon_loss_type == "vgg":
        recon_loss_func = LossLPIPS(normalized_rgb=normalize_rgb).to(device)
    else:
        recon_loss_func = calc_reconstruction_loss

    # optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=adam_betas, eps=adam_eps, weight_decay=weight_decay)
    # accelerate baking
    if use_scheduler:
        scheduler = LinearWithWarmupScheduler(optimizer, gamma=scheduler_gamma, verbose=False,
                                              steps=(max(warmup_epoch, 1), max(warmup_epoch, 1) + 1),
                                              factors=(1.0, 1.0, 1.0 * scheduler_gamma))
    else:
        scheduler = None

    if load_model and pretrained_path is not None:
        try:
            model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=False))
            print("loaded model from checkpoint")
        except:
            print("model checkpoint not found")

    # log statistics
    losses = []
    losses_rec = []
    losses_kl = []
    losses_kl_feat = []
    losses_kl_dyn = []
    losses_kl_context = []

    # initialize validation statistics
    valid_loss = best_valid_loss = 1e8
    valid_losses = []
    best_valid_epoch = 0

    # save PSNR values of the reconstruction
    psnrs = []

    # image metrics
    if eval_im_metrics:
        val_lpipss = []
        best_val_lpips_epoch = 0
        val_lpips = best_val_lpips = 1e8

    # iteration counter for discounting, optional
    iter_per_epoch = 1 * len(dataloader)
    # dynamics_warmup_iters = max(warmup_epoch, 1) * max(10_000, int(0.8 * iter_per_epoch))
    # iter_per_step = dynamics_warmup_iters // timestep_horizon
    # max_iterations_per_step = [iter_per_step * (i + 1) for i in range(timestep_horizon)]
    iteration = 0  # initialize iterations counter
    warmup_iteration = 0
    max_warmup_iterations = int(0.8 * iter_per_epoch)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        batch_losses = []
        batch_losses_rec = []
        batch_losses_kl = []
        batch_losses_kl_feat = []
        batch_losses_kl_dyn = []
        batch_losses_kl_context = []
        batch_psnrs = []

        pbar = tqdm(iterable=dataloader)
        for batch in pbar:
            x = batch[0].to(device)
            actions = None if not action_condition else batch[1].to(device)
            lang_str = None if not language_condition else batch[2]
            lang_embed = None if not language_condition else batch[3].to(device)
            ep_done_mask = None if not use_ep_done_mask else batch[-1].to(device)
            x_goal = None if not img_goal_condition else batch[3].to(device)
            warmup = (epoch < warmup_epoch)
            discount = None
            if n_views > 1:
                # expect: [bs, T, n_views, ...]
                x = x.permute(0, 2, 1, 3, 4, 5)
                x = x.reshape(-1, *x.shape[2:])  # [bs * n_views, T, ...]
                if x_goal is not None:
                    x_goal = x_goal.reshape(-1, *x_goal.shape[2:])  # [bs * n_views, ...]
                if actions is not None:
                    actions = actions.permute(0, 2, 1, 3)
                    actions = actions.reshape(-1, *actions.shape[2:])
                if ep_done_mask is not None:
                    ep_done_mask = ep_done_mask.permute(0, 2, 1)
                    ep_done_mask - ep_done_mask.reshape(-1, *ep_done_mask.shape[2:])
            model_output = model(x, actions=actions, lang_embed=lang_embed, warmup=warmup, with_loss=True,
                                 beta_kl=beta_kl,
                                 beta_dyn=beta_dyn, beta_rec=beta_rec, kl_balance=kl_balance,
                                 dynamic_discount=discount, recon_loss_type=recon_loss_type,
                                 recon_loss_func=recon_loss_func, beta_dyn_rec=beta_dyn_rec,
                                 done_mask=ep_done_mask, x_goal=x_goal)
            # calculate loss
            all_losses = model_output['loss_dict']
            iteration += 1

            loss = all_losses['loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # output for logging and plotting
            rec_x = model_output['rec_rgb']
            psnr = all_losses['psnr']

            loss_kl = all_losses['kl']
            loss_kl_dyn = all_losses['kl_dyn']
            loss_rec = all_losses['loss_rec']
            loss_kl_feat = all_losses['loss_kl_feat']
            loss_kl_context = all_losses['loss_kl_context']

            # log
            batch_psnrs.append(psnr.data.cpu().item())
            batch_losses.append(loss.data.cpu().item())
            batch_losses_rec.append(loss_rec.data.cpu().item())
            batch_losses_kl.append(loss_kl.data.cpu().item())
            batch_losses_kl_feat.append(loss_kl_feat.data.cpu().item())
            batch_losses_kl_dyn.append(loss_kl_dyn.data.cpu().item())
            batch_losses_kl_context.append(loss_kl_context.data.cpu().item())
            # progress bar
            if epoch < warmup_epoch:
                pbar.set_description_str(f'epoch #{epoch} (warmup)')
            else:
                pbar.set_description_str(f'epoch #{epoch}')
            pbar.set_postfix(loss=loss.data.cpu().item(), rec=loss_rec.data.cpu().item(),
                             kl=loss_kl.data.cpu().item(),
                             kl_dyn=loss_kl_dyn.data.cpu().item())
            if warmup:
                warmup_iteration += 1
                if warmup_iteration > max_warmup_iterations:
                    warmup_iteration = 0
                    break
            # break  # for debug
        pbar.close()
        losses.append(np.mean(batch_losses))
        losses_rec.append(np.mean(batch_losses_rec))
        losses_kl.append(np.mean(batch_losses_kl))
        losses_kl_feat.append(np.mean(batch_losses_kl_feat))
        losses_kl_dyn.append(np.mean(batch_losses_kl_dyn))
        losses_kl_context.append(np.mean(batch_losses_kl_context))
        if len(batch_psnrs) > 0:
            psnrs.append(np.mean(batch_psnrs))
        # scheduler
        if use_scheduler:
            scheduler.step()
            curr_lr = scheduler.get_lr()
            lr_str = f'learning rate: {curr_lr}'
            print(lr_str)
            log_line(log_dir, lr_str)

        # epoch summary
        log_str = format_epoch_summary_dvae(
            epoch=epoch,
            loss=losses[-1],
            loss_rec=losses_rec[-1],
            loss_kl=losses_kl[-1],
            kl_balance=kl_balance,
            loss_kl_feat=losses_kl_feat[-1],
            valid_loss=valid_loss,
            best_valid_loss=best_valid_loss,
            best_valid_epoch=best_valid_epoch,
            eval_epoch_freq=eval_epoch_freq,
            val_lpips=val_lpips if eval_im_metrics else None,
            best_val_lpips=best_val_lpips if eval_im_metrics else None,
            best_val_lpips_epoch=best_val_lpips_epoch if eval_im_metrics else None,
            psnr=psnrs[-1] if len(psnrs) > 0 else None,
            loss_kl_dyn=losses_kl_dyn[-1],
            loss_kl_context=losses_kl_context[-1]
        )
        print(log_str)
        log_line(log_dir, log_str)

        if epoch % eval_epoch_freq == 0 or epoch == num_epochs - 1:
            x = x.view(-1, *x.shape[2:])
            # for plotting purposes
            max_imgs = 8
            vutils.save_image(torch.cat([x[:max_imgs, -3:],
                                         rec_x[:max_imgs, -3:]],
                                        dim=0).data.cpu(), '{}/image_{}.jpg'.format(fig_dir, epoch),
                              nrow=8, pad_value=1)

            torch.save(model.state_dict(), os.path.join(save_dir, f'{ds}_dvae{run_prefix}.pth'))
            animate_trajectory_ddlp(model, config, epoch, device=device, fig_dir=fig_dir,
                                    timestep_horizon=animation_horizon, num_trajetories=1,
                                    train=True, cond_steps=cond_steps)
            print("validation step...")
            valid_loss = evaluate_validation_dvae_elbo_dyn(model, config, epoch, batch_size=batch_size,
                                                      recon_loss_type=recon_loss_type, device=device,
                                                      save_image=True, fig_dir=fig_dir,
                                                      recon_loss_func=recon_loss_func, beta_rec=beta_rec,
                                                      beta_dyn=beta_dyn,
                                                      timestep_horizon=timestep_horizon, beta_dyn_rec=beta_dyn_rec,
                                                      beta_kl=beta_kl, kl_balance=kl_balance,
                                                      animation_horizon=animation_horizon)
            log_str = f'validation loss: {valid_loss:.3f}\n'
            print(log_str)
            log_line(log_dir, log_str)
            if best_valid_loss > valid_loss:
                log_str = f'validation loss updated: {best_valid_loss:.3f} -> {valid_loss:.3f}\n'
                print(log_str)
                log_line(log_dir, log_str)
                best_valid_loss = valid_loss
                best_valid_epoch = epoch
                torch.save(model.state_dict(),
                           os.path.join(save_dir,
                                        f'{ds}_dvae{run_prefix}_best.pth'))
            torch.cuda.empty_cache()
            if eval_im_metrics and epoch > 0:
                valid_imm_results = eval_ddlp_im_metric(model, device, config,
                                                        timestep_horizon=animation_horizon, val_mode='val',
                                                        eval_dir=log_dir, use_all_ctx=ctx_for_eval,
                                                        cond_steps=cond_steps, batch_size=batch_size)
                log_str = f'validation: lpips: {valid_imm_results["lpips"]:.3f}, '
                log_str += f'psnr: {valid_imm_results["psnr"]:.3f}, ssim: {valid_imm_results["ssim"]:.3f}\n'
                val_lpips = valid_imm_results['lpips']
                print(log_str)
                log_line(log_dir, log_str)
                if (not torch.isinf(torch.tensor(val_lpips))) and (best_val_lpips > val_lpips):
                    log_str = f'validation lpips updated: {best_val_lpips:.3f} -> {val_lpips:.3f}\n'
                    print(log_str)
                    log_line(log_dir, log_str)
                    best_val_lpips = val_lpips
                    best_val_lpips_epoch = epoch
                    torch.save(model.state_dict(),
                               os.path.join(save_dir, f'{ds}_dvae{run_prefix}_best_lpips.pth'))
                torch.cuda.empty_cache()
        valid_losses.append(valid_loss)
        if eval_im_metrics:
            val_lpipss.append(val_lpips)
        # plot graphs
        if epoch > start_epoch:
            metrics_data = [
                (losses[1:], "Total Loss", "#2d72bc", True),
                (losses_kl[1:], "KL Loss", "#c92a2a", True),
                (losses_rec[1:], "Reconstruction Loss", "#087f5b", True),
                (valid_losses[1:], "Validation Loss", "#862e9c", True),
            ]
            save_metrics_data(metrics_data, run_name, save_dir=os.path.join(save_dir, 'metrics'))
            plot_training_metrics(metrics_data, run_name, fig_dir, max_plots_per_figure=4)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DVae Single-GPU Training")
    parser.add_argument("-d", "--dataset", type=str, default='balls_occlusion',
                        help="dataset of to train the model on: ['traffic', 'clevrer', 'obj3d128', 'phyre']")
    args = parser.parse_args()
    ds = args.dataset
    if ds.endswith('json'):
        conf_path = ds
    else:
        conf_path = os.path.join('./configs', f'{ds}.json')

    train_dvae(conf_path)
