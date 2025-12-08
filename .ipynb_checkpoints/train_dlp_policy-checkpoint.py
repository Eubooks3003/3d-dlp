"""
Single-GPU training of DLP-Policy
"""
# imports
import numpy as np
import os
from tqdm import tqdm
import matplotlib
import argparse
# torch
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
# modules
from dlp_policy import DLPPolicy
# datasets
from datasets.get_dataset import get_video_dataset
# util functions
from utils.util_func import prepare_logdir, save_config, log_line, \
    get_config, LinearWithWarmupScheduler, plot_training_metrics, save_metrics_data, save_code_backup

matplotlib.use("Agg")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def train_dlp_policy(config_path='./configs/panda_policy.json'):
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
    n_views = config.get('n_views', 1)
    root = config['root']  # dataset root
    run_prefix = config['run_prefix']
    timestep_horizon = config['timestep_horizon']
    image_size = config['image_size']

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
    loss_type = config['recon_loss_type']

    # evaluation
    eval_epoch_freq = config['eval_epoch_freq']

    # actions
    action_dim = config.get('action_dim', 0)

    # load data
    dataset = get_video_dataset(ds, root, seq_len=timestep_horizon + 1, mode='train', image_size=image_size)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True,
                            drop_last=True)
    # model
    model = DLPPolicy(config_path).to(device)
    dlp_model_info = model.dlp_model.info()
    # print(dlp_model_info)

    # prepare saving location
    run_name = f'{ds}_gdlppolicy' + run_prefix
    log_dir = prepare_logdir(runname=run_name, src_dir='./')
    save_dir = os.path.join(log_dir, 'saves')
    save_config(log_dir, hparams)
    save_config(log_dir, model.dlp_config, fname='dlp_hparams')
    log_line(log_dir, dlp_model_info)
    # save a backup of the code for this run
    backup_info = save_code_backup('.', backup_dir=os.path.join(log_dir, 'saves', 'code_backup'))
    log_line(log_dir, backup_info)
    print(backup_info)

    # optimizer and scheduler
    optimizer = optim.Adam(model.policy.parameters(), lr=lr, betas=adam_betas, eps=adam_eps, weight_decay=weight_decay)
    # accelerate baking
    if use_scheduler:
        scheduler = LinearWithWarmupScheduler(optimizer, gamma=scheduler_gamma, verbose=True,
                                              steps=(1, 2), factors=(1.0, 0.75, 0.75 * scheduler_gamma))
    else:
        scheduler = None

    # log statistics
    losses = []

    # initialize validation statistics
    valid_loss = best_valid_loss = 1e8
    valid_losses = []
    best_valid_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        model.policy.train()
        batch_losses = []
        pbar = tqdm(iterable=dataloader)
        for batch in pbar:
            x = batch[0].to(device)
            actions = batch[1][:, :timestep_horizon].to(device)
            x_goal = batch[3].to(device)
            if n_views > 1:
                # expect: [bs, T, n_views, ...]
                x = x.permute(0, 2, 1, 3, 4, 5)
                x = x.reshape(-1, *x.shape[2:])  # [bs * n_views, T, ...]
                if x_goal is not None:
                    x_goal = x_goal.reshape(-1, *x_goal.shape[2:]) # [bs * n_views, ...]
                if actions is not None:
                    # take action from one view, they should be the same for all views
                    actions = actions[:, :, 0].contiguous()
            model_output = model(x, x_goal=x_goal)
            # calculate loss
            loss = model.calc_loss(actions, model_output, loss_type=loss_type)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # log
            batch_losses.append(loss.data.cpu().item())

            # progress bar
            pbar.set_description_str(f'epoch #{epoch}')
            pbar.set_postfix(loss=loss.data.cpu().item())
            # break  # for debug
        pbar.close()
        losses.append(np.mean(batch_losses))
        # scheduler
        if use_scheduler:
            scheduler.step()

        # epoch summary
        log_str = (f'epoch: {epoch}, loss: {loss.data.cpu().item()},'
                   f' valid_loss: {valid_loss}, best_valid_loss: {best_valid_loss} @ epoch {best_valid_epoch}')
        print(log_str)
        log_line(log_dir, log_str)

        if epoch % eval_epoch_freq == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'{ds}_gdlppolicy{run_prefix}.pth'))
            print("validation step...")
            valid_loss = eval_model(model, config, epoch, device)
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
                                        f'{ds}_gdlppolicy{run_prefix}_best.pth'))
            torch.cuda.empty_cache()
        valid_losses.append(valid_loss)
        # plot graphs
        if epoch > start_epoch:
            metrics_data = [
                (losses[1:], "Total Loss", "#2d72bc", True),
                (valid_losses[1:], "Validation Loss", "#862e9c", True),
            ]
            save_metrics_data(metrics_data, run_name, save_dir=os.path.join(save_dir, 'metrics'))
            plot_training_metrics(metrics_data, run_name, log_dir, max_plots_per_figure=4)
    return model

def eval_model(model, config, epoch, device):
    # set model to eval mode
    model.policy.eval()
    # data and general
    ds = config['ds']
    n_views = config.get('n_views', 1)
    root = config['root']  # dataset root
    run_prefix = config['run_prefix']
    timestep_horizon = config['timestep_horizon']
    image_size = config['image_size']

    # optimization
    batch_size = config['batch_size']
    loss_type = config['recon_loss_type']

    # load data
    dataset = get_video_dataset(ds, root, seq_len=timestep_horizon + 1, mode='valid', image_size=image_size)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True,
                            drop_last=False)

    losses = []
    pbar = tqdm(iterable=dataloader)
    for batch in pbar:
        x = batch[0][:, :timestep_horizon + 1].to(device)
        # actions = batch[1][:, :timestep_horizon + 1].to(device)
        actions = batch[1][:, :timestep_horizon].to(device)
        x_goal = batch[3].to(device)
        if n_views > 1:
            # expect: [bs, T, n_views, ...]
            x = x.permute(0, 2, 1, 3, 4, 5)
            x = x.reshape(-1, *x.shape[2:])  # [bs * n_views, T, ...]
            if x_goal is not None:
                x_goal = x_goal.reshape(-1, *x_goal.shape[2:])  # [bs * n_views, ...]
            if actions is not None:
                # take action from one view, they should be the same for all views
                actions = actions[:, :, 0].contiguous()
        with torch.no_grad():
            model_output = model(x, x_goal=x_goal)
            # calculate loss
            loss = model.calc_loss(actions, model_output, loss_type=loss_type)

        # progress bar
        pbar.set_description_str(f'valid epoch #{epoch}')
        pbar.set_postfix(loss=loss.data.cpu().item())
        # log
        losses.append(loss.data.cpu().numpy())
    pbar.close()

    return np.mean(losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DLP-Policy Single-GPU Training")
    parser.add_argument("-d", "--dataset", type=str, default='panda_policy',
                        help="dataset of to train the model on: ['panda_policy']")
    args = parser.parse_args()
    ds = args.dataset
    if ds.endswith('json'):
        conf_path = ds
    else:
        conf_path = os.path.join('./configs', f'{ds}.json')

    train_dlp_policy(conf_path)
