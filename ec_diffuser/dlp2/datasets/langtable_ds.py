import os
import os.path as osp
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import glob
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def list_images_in_dir(path):
    valid_images = [".jpg", ".gif", ".png"]
    img_list = []
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        img_list.append(os.path.join(path, f))
    return img_list


class LanguageTableDataset(Dataset):
    def __init__(self, root, mode, ep_len=45, sample_length=15, image_size=128, normalize_embeddings=True):
        assert mode in ['train', 'val', 'valid', 'test']
        if mode == 'val':
            mode = 'valid'
        self.root = os.path.join(root, mode)
        self.image_size = image_size
        self.normalize_embeddings = normalize_embeddings
        self.mode = mode
        self.sample_length = sample_length

        # Get all numbers
        get_dir_num = lambda x: int(x)

        self.folders = [d for d in os.listdir(self.root) if osp.isdir(osp.join(self.root, d))]
        self.folders.sort(key=get_dir_num)

        self.episodes = []
        self.episodes_metadata = []
        self.episodes_len = []
        self.ep_len = ep_len
        self.seq_per_episode = self.ep_len - self.sample_length + 1
        # self.seq_per_episode = []

        for f in self.folders:
            dir_name = os.path.join(self.root, str(f))
            paths = list(glob.glob(osp.join(dir_name, '*.png')))
            get_num = lambda x: int(osp.splitext(osp.basename(x))[0])
            paths.sort(key=get_num)
            if len(paths) < self.ep_len:
                # continue
                self.episodes_len.append(len(paths))
            else:
                self.episodes_len.append(self.ep_len)
            while len(paths) < self.ep_len:
                paths.append(paths[-1])
            self.episodes.append(paths[:self.ep_len])
            self.episodes_metadata.append(os.path.join(dir_name, 'metadata.json'))
        max_episodes = 10_000
        self.episodes = self.episodes[:max_episodes]
        self.episodes_metadata = self.episodes_metadata[:max_episodes]
        self.episodes_len = self.episodes_len[:max_episodes]

    def __getitem__(self, index):
        imgs = []
        actions = []
        rewards = []
        # instructions = []
        text_embeddings = []
        if self.mode == 'train':
            # Implement continuous indexing
            ep = index // self.seq_per_episode
            offset = index % self.seq_per_episode
            end = offset + self.sample_length
            # if `end` is after the episode ended, move backwards
            ep_len = self.episodes_len[ep]
            with open(self.episodes_metadata[ep], 'r') as f:
                metadata = json.load(f)
            # print(np.array(metadata['text_emb']).shape)
            # print(f"ep: {ep} - {index}: {metadata['instruction']}")
            instructions = metadata['instruction']
            if end > ep_len:
                if self.sample_length > ep_len:
                    offset = 0
                    end = offset + self.sample_length
                else:
                    offset = ep_len - self.sample_length
                    end = ep_len

            e = self.episodes[ep]
            ep_actions = torch.tensor(metadata['action'], dtype=torch.float)
            ep_rewards = torch.tensor(metadata['reward'], dtype=torch.float)
            for image_index in range(offset, end):
                img = Image.open(e[image_index])
                img = img.resize((self.image_size, self.image_size))
                img = transforms.ToTensor()(img)[:3]
                imgs.append(img)
                # instructions.append(metadata['instruction'])
                # text_embeddings.append(torch.tensor(metadata['text_emb'], dtype=torch.float))
                a_i = image_index if image_index < ep_actions.shape[0] else -1
                r_i = image_index if image_index < ep_rewards.shape[0] else -1
                actions.append(ep_actions[a_i])
                rewards.append(ep_rewards[r_i])
            actions = torch.stack(actions, dim=0)
            rewards = torch.stack(rewards, dim=0)
        else:
            with open(self.episodes_metadata[index], 'r') as f:
                metadata = json.load(f)
            instructions = metadata['instruction']
            for path in self.episodes[index]:
                img = Image.open(path)
                img = img.resize((self.image_size, self.image_size))
                img = transforms.ToTensor()(img)[:3]
                imgs.append(img)
                # instructions.append(metadata['instruction'])
                # text_embeddings.append(torch.tensor(metadata['text_emb'], dtype=torch.float))
            actions = torch.tensor(metadata['action'], dtype=torch.float)
            rewards = torch.tensor(metadata['reward'], dtype=torch.float)

        img = torch.stack(imgs, dim=0).float()
        # text_embeddings = torch.stack(text_embeddings, dim=0)

        num_pad = img.shape[0] - actions.shape[0]
        if num_pad > 0:
            action_pad = actions[-1:]
            actions = torch.cat([actions, action_pad.repeat(num_pad, 1)], dim=0)
            reward_pad = rewards[-1:]
            rewards = torch.cat([rewards, reward_pad.repeat(num_pad)], dim=0)
        ind = index
        # if self.normalize_embeddings:
            # text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        # return img, text_embeddings, instructions, actions, rewards, ind
        return img, instructions, actions, rewards, ind

    def __len__(self):
        length = len(self.episodes)
        if self.mode == 'train':
            return length * self.seq_per_episode
        else:
            return length


class LanguageTableDatasetImage(Dataset):
    def __init__(self, root, mode, ep_len=45, sample_length=1, res=128):
        # path = os.path.join(root, mode)
        # assume 10 frames-per-second
        # the data is generated such that a task is completed if the completion condition is met for 3 seconds or more.
        # that means that we can cut off 3 seconds (=30 frames) from the end of the episode)
        assert mode in ['train', 'val', 'valid', 'test']
        if mode == 'val':
            mode = 'valid'
        self.root = os.path.join(root, mode)
        self.image_size = res
        self.mode = mode
        # self.sample_length = sample_length

        # Get all numbers
        get_dir_num = lambda x: int(x)

        self.folders = [d for d in os.listdir(self.root) if osp.isdir(osp.join(self.root, d))]
        self.folders.sort(key=get_dir_num)

        self.episodes = []
        self.episodes_metadata = []
        self.episodes_len = []
        self.ep_len = ep_len
        self.seq_per_episode = self.ep_len - sample_length + 1
        # self.seq_per_episode = []

        for f in self.folders:
            dir_name = os.path.join(self.root, str(f))
            paths = list(glob.glob(osp.join(dir_name, '*.png')))
            get_num = lambda x: int(osp.splitext(osp.basename(x))[0])
            paths.sort(key=get_num)
            if len(paths) < self.ep_len:
                # continue
                self.episodes_len.append(len(paths))
            else:
                self.episodes_len.append(self.ep_len)
            while len(paths) < self.ep_len:
                paths.append(paths[-1])
            self.episodes.append(paths[:self.ep_len])
            self.episodes_metadata.append(os.path.join(dir_name, 'metadata.json'))

    def __getitem__(self, index):
        # Implement continuous indexing
        ep = index // self.seq_per_episode
        offset = index % self.seq_per_episode
        end = offset + 1
        # if `end` is after the episode ended, move backwards
        ep_len = self.episodes_len[ep]
        with open(self.episodes_metadata[ep], 'r') as f:
            metadata = json.load(f)
        # print(np.array(metadata['text_emb']).shape)
        # print(f"ep: {ep} - {index}: {metadata['instruction']}")
        instructions = metadata['instruction']
        if end > ep_len:
            offset = ep_len - 1
            end = ep_len

        e = self.episodes[ep]
        imgs = []
        for image_index in range(offset, end):
            img = Image.open(e[image_index])
            img = img.resize((self.image_size, self.image_size))
            img = transforms.ToTensor()(img)[:3]
            imgs.append(img)
        img = torch.stack(imgs, dim=0).float().squeeze(0)
        return img

    def __len__(self):
        length = len(self.episodes)
        return length * self.seq_per_episode


if __name__ == '__main__':
    test_epochs = True
    norm_embeddings = True
    # --- episodic setting --- #
    root='/data/carlq/project_data/ECRL/language-table'
    # ds = LanguageTableDatasetImage(root=root, ep_len=45, sample_length=1, mode='valid', res=128)
    ds = LanguageTableDataset(root=root, ep_len=45, sample_length=45, mode='train', image_size=128, normalize_embeddings=norm_embeddings)
    print(len(ds))
    dl = DataLoader(ds, shuffle=False, pin_memory=False, batch_size=5, num_workers=4)
    batch = next(iter(dl))
    # print(f'batch: {batch.shape}')
    im, instructions, actions, rewards, ind = batch
    print(f'im: {im.shape}')  # , text_embeddings: {text_embeddings.shape}')
    # print(f'text_embeddings[0]: {text_embeddings[0]}')
    # print(f'text_embeddings[1]: {text_embeddings[1]}')
    print(f'actions: {actions.shape}, rewards: {rewards.shape}')
    print(f'instructions: {len(instructions)}, instructions[0]: {instructions[0]}')
    print(f'instructions: {len(instructions)}, instructions[1]: {instructions[1]}')
    # print(im.shape)
    # img_np = im.permute(1, 2, 0).data.cpu().numpy()
    # fig = plt.figure(figsize=(5, 5))
    # ax = fig.add_subplot(111)
    # ax.imshow(img_np)
    # plt.show()

    if test_epochs:
        from tqdm import tqdm

        pbar = tqdm(iterable=dl)
        for batch in pbar:
            pass
        pbar.close()
