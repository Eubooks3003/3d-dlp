"""
Dataset class for the Kubric-MOVi Dataset
"""

import os
import os.path as osp
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import glob
from PIL import Image, ImageFile
import json
import random
import numpy as np

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


# --- new preprocessing functions for the episodic setting --- #
class KubricDataset(Dataset):
    def __init__(self, root, mode, ep_len=24, sample_length=20, image_size=128, dense=True, clips_per_video=1):
        assert mode in ['train', 'val', 'valid', 'validation', 'test']
        if mode in ["val", "valid"]:
            mode = 'validation'
        self.mode = mode
        self.dense = dense
        self.clips_per_video = clips_per_video if not dense else 1

        self.n_max_train = 50_000
        self.n_max_valid = 1000
        self.root = os.path.join(root, self.mode, 'rgb')
        self.image_size = image_size
        self.sample_length = sample_length

        get_dir_num = lambda x: int(x)
        self.get_num = lambda x: int(osp.splitext(osp.basename(x))[0])

        # Get all numbers
        self.folders = []
        for file in os.listdir(self.root):
            try:
                self.folders.append(file)
            except ValueError:
                continue
        self.folders.sort(key=lambda x: int(x))

        if self.mode == 'train' and self.n_max_train > 0:
            self.folders = self.folders[:self.n_max_train]
        elif self.mode == 'valid' and self.n_max_valid > 0:
            self.folders = self.folders[:self.n_max_valid]

        self.episodes = []
        self.episodes_len = []
        self.EP_LEN = ep_len
        self.seq_per_episode = self.EP_LEN - self.sample_length + 1

        for f in self.folders:
            dir_name = os.path.join(self.root, str(f))
            paths = list(glob.glob(osp.join(dir_name, '*.jpg')))
            # if len(paths) != self.EP_LEN:
            #     continue
            # assert len(paths) == self.EP_LEN, 'len(paths): {}'.format(len(paths))
            get_num = lambda x: int(osp.splitext(osp.basename(x))[0])
            paths.sort(key=get_num)
            self.episodes_len.append(len(paths))
            while len(paths) < self.EP_LEN:
                paths.append(paths[-1])
            self.episodes.append(paths)

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        imgs = []

        if self.mode == 'train':
            if self.dense:
                # DENSE MODE
                ep = index // self.seq_per_episode
                offset = index % self.seq_per_episode
                end = offset + self.sample_length

                ep_len = self.episodes_len[ep]
                ep_path = self.episodes[ep]

                if end > ep_len:
                    if self.sample_length > ep_len:
                        offset = 0
                        end = offset + self.sample_length
                    else:
                        offset = ep_len - self.sample_length
                        end = ep_len

                for image_index in range(offset, end):
                    img = Image.open(ep_path[image_index])
                    img = self.transform(img)[:3]
                    imgs.append(img)

            else:
                # SPARSE MODE
                ep = index // self.clips_per_video
                clip_num = index % self.clips_per_video
                ep_path = self.episodes[ep]
                ep_len = self.episodes_len[ep]

                max_offset = max(1, ep_len - self.sample_length // 2)
                # To make clip selection more deterministic per clip_num, optionally:
                #   seed = (ep * self.clips_per_video + clip_num)
                #   rng = random.Random(seed)
                #   offset = rng.randint(0, max_offset)
                offset = random.randint(0, max_offset)
                end = offset + self.sample_length

                for image_index in range(offset, end):
                    img = Image.open(ep_path[image_index])
                    img = self.transform(img)[:3]
                    imgs.append(img)

        else:
            # EVAL MODE — always return full video (padded if needed)
            paths = self.episodes[index]
            for path in paths:
                img = Image.open(path)
                img = self.transform(img)[:3]
                imgs.append(img)

        img = torch.stack(imgs, dim=0).float()

        # Placeholder meta fields
        pos = torch.zeros(0)
        size = torch.zeros(0)
        id = torch.zeros(0)
        in_camera = torch.zeros(0)

        return img, pos, size, id, in_camera

    def __len__(self):
        length = len(self.folders)
        if self.mode == 'train':
            if self.dense:
                return length * self.seq_per_episode
            else:
                return length * self.clips_per_video
        else:
            return length


class KubricDatasetImage(Dataset):
    def __init__(self, root, mode, ep_len=24, sample_length=1, image_size=128):
        assert mode in ['train', 'val', 'valid', 'validation', 'test']
        if mode in ["val", "valid"]:
            mode = 'validation'
        self.mode = mode

        self.n_max_train = 50_000
        self.n_max_valid = 1000
        self.root = os.path.join(root, self.mode, 'rgb')
        self.image_size = image_size
        self.sample_length = sample_length

        get_dir_num = lambda x: int(x)
        self.get_num = lambda x: int(osp.splitext(osp.basename(x))[0])

        # Get all numbers
        self.folders = []
        for file in os.listdir(self.root):
            try:
                self.folders.append(file)
            except ValueError:
                continue
        self.folders.sort(key=lambda x: int(x))

        if self.mode == 'train' and self.n_max_train > 0:
            self.folders = self.folders[:self.n_max_train]
        elif self.mode == 'valid' and self.n_max_valid > 0:
            self.folders = self.folders[:self.n_max_valid]

        self.episodes = []
        self.episodes_len = []
        self.EP_LEN = ep_len
        self.seq_per_episode = self.EP_LEN - self.sample_length + 1

        for f in self.folders:
            dir_name = os.path.join(self.root, str(f))
            paths = list(glob.glob(osp.join(dir_name, '*.jpg')))
            # if len(paths) != self.EP_LEN:
            #     continue
            # assert len(paths) == self.EP_LEN, 'len(paths): {}'.format(len(paths))
            get_num = lambda x: int(osp.splitext(osp.basename(x))[0])
            paths.sort(key=get_num)
            self.episodes_len.append(len(paths))
            while len(paths) < self.EP_LEN:
                paths.append(paths[-1])
            self.episodes.append(paths)

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        imgs = []
        # DENSE MODE
        ep = index // self.seq_per_episode
        offset = index % self.seq_per_episode
        end = offset + self.sample_length

        ep_len = self.episodes_len[ep]
        ep_path = self.episodes[ep]

        if end > ep_len:
            if self.sample_length > ep_len:
                offset = 0
                end = offset + self.sample_length
            else:
                offset = ep_len - self.sample_length
                end = ep_len

        for image_index in range(offset, end):
            img = Image.open(ep_path[image_index])
            img = self.transform(img)[:3]
            imgs.append(img)

        img = torch.stack(imgs, dim=0).float()
        pos = torch.zeros(0)
        size = torch.zeros(0)
        id = torch.zeros(0)
        in_camera = torch.zeros(0)

        return img, pos, size, id, in_camera

    def __len__(self):
        length = len(self.folders)
        return length * self.seq_per_episode


if __name__ == '__main__':
    test_epochs = True
    plot = False
    # --- episodic setting --- #
    root = '/data/kubric/movi_c'
    ds = KubricDataset(root=root, ep_len=24, sample_length=10, mode='train', image_size=128, dense=True)
    dl = DataLoader(ds, shuffle=True, pin_memory=False, batch_size=4, num_workers=0)
    batch = next(iter(dl))
    im = batch[0]
    print(im.shape)

    if plot:
        import matplotlib.pyplot as plt

        img_np = im[0, 0].permute(1, 2, 0).data.cpu().numpy()
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        ax.imshow(img_np)
        plt.show()

    if test_epochs:
        from tqdm import tqdm

        pbar = tqdm(iterable=dl)
        for batch in pbar:
            pass
        pbar.close()
