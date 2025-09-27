import os
import numpy as np
# from tqdm import tqdm
import torchvision
import torch
from torch.utils.data import DataLoader, Dataset


# import matplotlib.pyplot as plt

def load_episode(path):
    episode = np.load(path, allow_pickle=True).item()
    episode = {k: episode[k] for k in episode.keys()}
    episode['observations'] = episode['observations'].astype(np.uint8)
    episode['rewards'] = episode['rewards'].astype(float)
    return episode


class ProcgenDataset(Dataset):
    def __init__(self, root, mode, ep_len=100, sample_length=20, image_size=64):
        assert mode in ['train', 'val', 'valid', 'test']
        if mode == 'val':
            mode = 'valid'
        self.root = root
        self.image_size = image_size
        self.mode = mode
        self.sample_length = sample_length
        self.ep_len = ep_len

        valid_ratio = 0.1

        episode_filenames = sorted(os.listdir(self.root), key=lambda x: int(x.split('_')[-2]), reverse=False)
        seeds = np.unique([int(x.split('_')[-2]) for x in episode_filenames])
        valid_size = int(valid_ratio * len(seeds))
        valid_seeds = seeds[-valid_size:]
        train_seeds = seeds[:-valid_size]
        check_cond = lambda x, f: int(x.split('_')[-2]) in f and self.sample_length <= int(
            x.split('_')[-3]) <= self.ep_len

        self.train_episodes = [f for f in episode_filenames if check_cond(f, train_seeds)]
        self.valid_episodes = [f for f in episode_filenames if check_cond(f, valid_seeds)]
        self.episodes = self.train_episodes if self.mode == 'train' else self.valid_episodes
        self.episodes_len = [int(x.split('_')[-3]) for x in self.episodes]
        self.seq_per_episode = self.ep_len - self.sample_length + 1
        print(
            f'total episodes: {len(self.episodes)}, min len: {min(self.episodes_len)}, max len: {max(self.episodes_len)}')

    def __getitem__(self, index):
        if self.mode == 'train':
            # Implement continuous indexing
            ep = index // self.seq_per_episode
            # ep = np.argmax((index < self.episodes_len_cumsum))
            offset = index % self.seq_per_episode
            # offset = index % self.seq_per_episode[ep]
            end = offset + self.sample_length
            # if `end` is after the episode ended, move backwards
            ep_len = self.episodes_len[ep]
            if end > ep_len:
                # print(f'before: offset: {offset}, end: {end}, ep_len: {ep_len}')
                if self.sample_length > ep_len:
                    offset = 0
                    end = offset + self.sample_length
                else:
                    offset = ep_len - self.sample_length
                    end = ep_len
                # print(f'after: offset: {offset}, end: {end}, ep_len: {ep_len}')

            e = self.episodes[ep]
            episode = load_episode(os.path.join(self.root, e))
            obs = torch.tensor(episode['observations'][offset:end], dtype=torch.float) / 255.0
            imgs = torchvision.transforms.Resize(size=(self.image_size, self.image_size), antialias=True)(obs)
            actions = torch.tensor(episode['actions'][offset:end], dtype=torch.float)
            rewards = torch.tensor(episode['rewards'][offset:end], dtype=torch.float)
        else:
            e = self.episodes[index]
            episode = load_episode(os.path.join(self.root, e))
            obs = torch.tensor(episode['observations'][:-1], dtype=torch.float) / 255.0
            imgs = torchvision.transforms.Resize(size=(self.image_size, self.image_size), antialias=True)(obs)
            actions = torch.tensor(episode['actions'], dtype=torch.float)
            rewards = torch.tensor(episode['rewards'], dtype=torch.float)
            # pad
            pad_size = self.ep_len - imgs.shape[0]
            img_padding = imgs[-1:].repeat(pad_size, 1, 1, 1)
            imgs = torch.cat([imgs, img_padding], dim=0)
            action_padding = actions[-1:].repeat(pad_size, 1)
            actions = torch.cat([actions, action_padding], dim=0)
            reward_padding = actions[-1:].repeat(pad_size, 1)
            rewards = torch.cat([rewards, reward_padding], dim=0)

        ids = torch.zeros(0)
        aux = torch.zeros(0)

        return imgs, actions, rewards, ids, aux

    def __len__(self):
        length = len(self.episodes)
        if self.mode == 'train':
            return length * self.seq_per_episode
        else:
            return length

class ProcgenDatasetImage(Dataset):
    def __init__(self, root, mode, ep_len=100, sample_length=20, image_size=64):
        assert mode in ['train', 'val', 'valid', 'test']
        if mode == 'val':
            mode = 'valid'
        self.root = root
        self.image_size = image_size
        self.mode = mode
        self.sample_length = sample_length
        self.ep_len = ep_len

        valid_ratio = 0.1

        episode_filenames = sorted(os.listdir(self.root), key=lambda x: int(x.split('_')[-2]), reverse=False)
        seeds = np.unique([int(x.split('_')[-2]) for x in episode_filenames])
        valid_size = int(valid_ratio * len(seeds))
        valid_seeds = seeds[-valid_size:]
        train_seeds = seeds[:-valid_size]
        check_cond = lambda x, f: int(x.split('_')[-2]) in f and self.sample_length <= int(
            x.split('_')[-3]) <= self.ep_len

        self.train_episodes = [f for f in episode_filenames if check_cond(f, train_seeds)]
        self.valid_episodes = [f for f in episode_filenames if check_cond(f, valid_seeds)]
        self.episodes = self.train_episodes if self.mode == 'train' else self.valid_episodes
        self.episodes_len = [int(x.split('_')[-3]) for x in self.episodes]
        self.seq_per_episode = self.ep_len - self.sample_length + 1
        print(
            f'total episodes: {len(self.episodes)}, min len: {min(self.episodes_len)}, max len: {max(self.episodes_len)}')

    def __getitem__(self, index):
        if self.mode == 'train':
            # Implement continuous indexing
            ep = index // self.seq_per_episode
            # ep = np.argmax((index < self.episodes_len_cumsum))
            offset = index % self.seq_per_episode
            # offset = index % self.seq_per_episode[ep]
            end = offset + self.sample_length
            # if `end` is after the episode ended, move backwards
            ep_len = self.episodes_len[ep]
            if end > ep_len:
                # print(f'before: offset: {offset}, end: {end}, ep_len: {ep_len}')
                if self.sample_length > ep_len:
                    offset = 0
                    end = offset + self.sample_length
                else:
                    offset = ep_len - self.sample_length
                    end = ep_len
                # print(f'after: offset: {offset}, end: {end}, ep_len: {ep_len}')

            e = self.episodes[ep]
            episode = load_episode(os.path.join(self.root, e))
            obs = torch.tensor(episode['observations'][offset:end], dtype=torch.float) / 255.0
            imgs = torchvision.transforms.Resize(size=(self.image_size, self.image_size), antialias=True)(obs)
            actions = torch.tensor(episode['actions'][offset:end], dtype=torch.float)
            rewards = torch.tensor(episode['rewards'][offset:end], dtype=torch.float)
        else:
            e = self.episodes[index]
            episode = load_episode(os.path.join(self.root, e))
            obs = torch.tensor(episode['observations'][:-1], dtype=torch.float) / 255.0
            imgs = torchvision.transforms.Resize(size=(self.image_size, self.image_size), antialias=True)(obs)
            actions = torch.tensor(episode['actions'], dtype=torch.float)
            rewards = torch.tensor(episode['rewards'], dtype=torch.float)
            # pad
            pad_size = self.ep_len - imgs.shape[0]
            img_padding = imgs[-1:].repeat(pad_size, 1, 1, 1)
            imgs = torch.cat([imgs, img_padding], dim=0)
            action_padding = actions[-1:].repeat(pad_size, 1)
            actions = torch.cat([actions, action_padding], dim=0)
            reward_padding = actions[-1:].repeat(pad_size, 1)
            rewards = torch.cat([rewards, reward_padding], dim=0)

        ids = torch.zeros(0)
        aux = torch.zeros(0)

        return imgs, actions, rewards, ids, aux

    def __len__(self):
        length = len(self.episodes)
        return length * self.seq_per_episode


if __name__ == '__main__':
    root = '/media/newhd/data/procgen/maze'
    test_epochs = True
    ds = ProcgenDataset(root=root, ep_len=100, sample_length=11, mode='valid', image_size=64)
    dl = DataLoader(ds, shuffle=True, pin_memory=False, batch_size=10, num_workers=4)
    batch = next(iter(dl))
    im = batch[0]
    actions = batch[1]
    rewards = batch[2]
    print(im.shape)
    print(actions.shape)
    print(rewards.shape)
    # img_np = im[0, 0].permute(1, 2, 0).data.cpu().numpy()
    # fig = plt.figure(figsize=(5, 5))
    # ax = fig.add_subplot(111)
    # ax.imshow(img_np)
    # plt.show()

    # if test_epochs:
    #     pbar = tqdm(iterable=dl)
    #     for batch in pbar:
    #         pass
    #     pbar.close()
