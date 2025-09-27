"""
Dataset class for the UCF-101 Dataset
"""

import os
import os.path as osp
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import glob
import random
import csv
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


class UCF101Video(Dataset):
    def __init__(self, root, mode, ep_len=100, sample_length=10, image_size=128, dense=False, clips_per_video=2):
        # path = os.path.join(root, mode)
        if mode == 'valid':
            mode = 'val'
        assert mode in ['train', 'val', 'test']
        if mode != 'train':
            mode = 'test'
        self.root = root
        self.image_size = image_size
        self.mode = mode
        self.sample_length = sample_length
        self.max_ep_len = ep_len
        self.dense = dense
        self.clips_per_video = clips_per_video if not dense else 1

        self.csv_path = os.path.join(self.root, 'frames', 'data_file.csv')
        self.frames_path = os.path.join(self.root, 'frames', self.mode)
        self.ep_paths = []
        with open(self.csv_path, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if row[0] == mode:
                    ep_len = int(row[-1])
                    im_paths = [os.path.join(self.frames_path, f'{row[1]}', f'{row[2]}-{i + 1:04}.jpg') for i in
                                range(ep_len)]
                    self.ep_paths.append(im_paths)
        self.n_episodes = len(self.ep_paths)
        #         print(f'n_episodes: {n_episodes}')
        self.ep_lens = [len(e) for e in self.ep_paths]
        self.n_frames = sum(self.ep_lens)
        #         print(f'n_frames: {n_frames}')
        self.index_to_ep = {}
        self.index_to_frame_num = {}
        curr = 0
        for e_i, e in enumerate(self.ep_paths):
            ep_len = len(e)
            for j in range(ep_len):
                self.index_to_ep[curr] = e_i
                self.index_to_frame_num[curr] = j
                curr += 1

    def __getitem__(self, index):
        # print(index)
        imgs = []
        if self.mode == 'train':
            if self.dense:
                # Implement continuous indexing
                ep = self.index_to_ep[index]
                ep_len = self.ep_lens[ep]
                offset = self.index_to_frame_num[index]
                end = offset + self.sample_length
                if end > ep_len:
                    # print(f'before: offset: {offset}, end: {end}, ep_len: {ep_len}')
                    if self.sample_length > ep_len:
                        offset = 0
                        end = offset + self.sample_length
                    else:
                        offset = ep_len - self.sample_length
                        end = ep_len

                e = self.ep_paths[ep]
                for image_index in range(offset, end):
                    img = Image.open(osp.join(e[image_index]))
                    img = img.resize((self.image_size, self.image_size))
                    img = transforms.ToTensor()(img)[:3]
                    imgs.append(img)
            else:
                # SPARSE MODE with multiple clips per video
                ep = index // self.clips_per_video
                clip_num = index % self.clips_per_video

                ep_len = self.ep_lens[ep]
                ep_path = self.ep_paths[ep]

                max_offset = max(1, ep_len - self.sample_length)
                # To make clip selection more deterministic per clip_num, optionally:
                #   seed = (ep * self.clips_per_video + clip_num)
                #   rng = random.Random(seed)
                #   offset = rng.randint(0, max_offset)
                offset = random.randint(0, max_offset)

                for image_index in range(offset, offset + self.sample_length):
                    img = Image.open(ep_path[image_index])
                    img = img.resize((self.image_size, self.image_size))
                    img = transforms.ToTensor()(img)[:3]
                    imgs.append(img)
        else:
            paths = self.ep_paths[index]
            while len(paths) < self.max_ep_len:
                paths.append(self.ep_paths[index][-1])
            for im_i, path in enumerate(paths):
                img = Image.open(path)
                img = img.resize((self.image_size, self.image_size))
                img = transforms.ToTensor()(img)[:3]
                imgs.append(img)
                if im_i == self.max_ep_len - 1:
                    break

        img = torch.stack(imgs, dim=0).float()
        pos = torch.zeros(0)
        size = torch.zeros(0)
        id = torch.zeros(0)
        in_camera = torch.zeros(0)

        return img, pos, size, id, in_camera

    def __len__(self):
        if self.mode == 'train':
            if self.dense:
                return self.n_frames
            else:
                return self.n_episodes * self.clips_per_video
        else:
            return self.n_episodes


class UCF101Image(Dataset):
    def __init__(self, root, mode, ep_len=100, sample_length=1, image_size=128):
        # path = os.path.join(root, mode)
        if mode == 'valid':
            mode = 'val'
        assert mode in ['train', 'val', 'test']
        if mode != 'train':
            mode = 'test'
        self.root = root
        self.image_size = image_size
        self.mode = mode
        self.sample_length = sample_length
        self.max_ep_len = ep_len

        self.csv_path = os.path.join(self.root, 'frames', 'data_file.csv')
        self.frames_path = os.path.join(self.root, 'frames', self.mode)
        self.ep_paths = []
        with open(self.csv_path, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if row[0] == mode:
                    ep_len = int(row[-1])
                    im_paths = [os.path.join(self.frames_path, f'{row[1]}', f'{row[2]}-{i + 1:04}.jpg') for i in
                                range(ep_len)]
                    self.ep_paths.append(im_paths)
        self.n_episodes = len(self.ep_paths)
        #         print(f'n_episodes: {n_episodes}')
        self.ep_lens = [len(e) for e in self.ep_paths]
        self.n_frames = sum(self.ep_lens)
        #         print(f'n_frames: {n_frames}')
        self.index_to_ep = {}
        self.index_to_frame_num = {}
        curr = 0
        for e_i, e in enumerate(self.ep_paths):
            ep_len = len(e)
            for j in range(ep_len):
                self.index_to_ep[curr] = e_i
                self.index_to_frame_num[curr] = j
                curr += 1

    def __getitem__(self, index):
        # print(index)
        imgs = []
        # Implement continuous indexing
        ep = self.index_to_ep[index]
        ep_len = self.ep_lens[ep]
        offset = self.index_to_frame_num[index]
        end = offset + self.sample_length
        if end > ep_len:
            # print(f'before: offset: {offset}, end: {end}, ep_len: {ep_len}')
            if self.sample_length > ep_len:
                offset = 0
                end = offset + self.sample_length
            else:
                offset = ep_len - self.sample_length
                end = ep_len

        e = self.ep_paths[ep]
        for image_index in range(offset, end):
            img = Image.open(osp.join(e[image_index]))
            img = img.resize((self.image_size, self.image_size))
            img = transforms.ToTensor()(img)[:3]
            imgs.append(img)

        img = torch.stack(imgs, dim=0).float()
        pos = torch.zeros(0)
        size = torch.zeros(0)
        id = torch.zeros(0)
        in_camera = torch.zeros(0)

        return img, pos, size, id, in_camera

    def __len__(self):
        return self.n_frames


if __name__ == '__main__':
    test_epochs = True
    plot = True
    # --- episodic setting --- #
    # root = '/mnt/data/tal/smthv2'
    root = '/media/newhd/data/ucf101'
    ds = UCF101Video(root=root, ep_len=100, sample_length=10, mode='valid', image_size=128)
    dl = DataLoader(ds, shuffle=True, pin_memory=False, batch_size=10, num_workers=0)
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
