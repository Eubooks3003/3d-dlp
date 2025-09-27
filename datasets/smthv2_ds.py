"""
Dataset class for the Something-Something-V2 Dataset
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

# Function to load the index from a directory
def load_index(directory: str):
    with open(os.path.join(directory, "index.json"), "r",  encoding='utf-8') as f:
        return json.load(f)

# Function to load an embedding for a specific id
def load_embedding(index, root: str, sample_id: str):
    # Get the path to the .pt file from the index
    # index = load_index(directory)
    sample_path = index.get(sample_id)
    sample_path = os.path.join(root, sample_path.split('/')[-1])

    if sample_path:
        # Load the .pt file
        data = torch.load(sample_path)
        return data
    else:
        print(f"ID {sample_id} not found")
        return None


# --- new preprocessing functions for the episodic setting --- #
class SmthDatasetV1(Dataset):
    def __init__(self, root, mode, ep_len=100, sample_length=20, image_size=128, prep_path=False):
        assert mode in ['train', 'val', 'valid', 'validation', 'test']
        if mode == 'val' or mode == 'valid':
            mode = 'validation'
        self.n_max_train = 50_000
        self.n_max_valid = 1000
        self.prep_path = prep_path
        self.root = root
        self.frames_root = os.path.join(self.root, 'frames')
        self.labels_root = os.path.join(self.root, 'labels')
        self.image_size = image_size

        self.mode = mode
        with open(os.path.join(self.labels_root, f'{mode}.json'), 'r', encoding='utf-8') as file:
            label_data = json.load(file)
        self.ep_idx = [label_data[i]['id'] for i in range(len(label_data))]
        self.sample_length = sample_length

        # Get all numbers
        get_dir_num = lambda x: int(x)
        self.get_num = lambda x: int(osp.splitext(osp.basename(x))[0])

        #         self.folders = [d for d in os.listdir(self.frames_root) if osp.isdir(osp.join(self.frames_root, d))]
        self.folders = self.ep_idx
        self.folders.sort(key=get_dir_num)

        self.episodes = []
        self.episodes_len = []
        self.EP_LEN = ep_len
        self.seq_per_episode = self.EP_LEN - self.sample_length + 1
        if self.mode == 'train' and self.n_max_train > 0:
            self.folders = self.folders[:self.n_max_train]
        elif self.mode == 'validation' and self.n_max_valid > 0:
            self.folders = self.folders[:self.n_max_valid]

        if self.prep_path:
            for f in self.folders:
                dir_name = os.path.join(self.frames_root, str(f))
                paths = list(glob.glob(osp.join(dir_name, '*.jpg')))
                get_num = lambda x: int(osp.splitext(osp.basename(x))[0])
                paths.sort(key=get_num)
                if len(paths) < self.EP_LEN:
                    # continue
                    self.episodes_len.append(len(paths))
                else:
                    self.episodes_len.append(self.EP_LEN)
                while len(paths) < self.EP_LEN:
                    paths.append(paths[-1])
                self.episodes.append(paths[:self.EP_LEN])
        # print(f'episodes: {len(self.episodes)}, min: {min(self.episodes_len)}, max: {max(self.episodes_len)}')

    def get_path(self, idx):
        ep_idx = self.folders[idx]
        dir_name = os.path.join(self.frames_root, str(ep_idx))
        path = list(glob.glob(osp.join(dir_name, '*.jpg')))
        path.sort(key=self.get_num)
        orig_len = len(path)
        while len(path) < self.EP_LEN:
            path.append(path[-1])
        return path, orig_len

    def __getitem__(self, index):

        imgs = []
        if self.mode == 'train':
            # Implement continuous indexing
            ep = index // self.seq_per_episode
            offset = index % self.seq_per_episode
            end = offset + self.sample_length
            # if `end` is after the episode ended, move backwards
            if self.prep_path:
                ep_len = self.episodes_len[ep]
                ep_path = self.episodes[ep]
            else:
                ep_path, ep_len = self.get_path(ep)
            #             print(f'{ep_pahth}: {ep_len}')
            #             ep_len = self.episodes_len[ep]
            if end > ep_len:
                # print(f'before: offset: {offset}, end: {end}, ep_len: {ep_len}')
                if self.sample_length > ep_len:
                    offset = 0
                    end = offset + self.sample_length
                else:
                    offset = ep_len - self.sample_length
                    end = ep_len
                # print(f'after: offset: {offset}, end: {end}, ep_len: {ep_len}')

            #             e = self.episodes[ep]
            e = ep_path
            for image_index in range(offset, end):
                img = Image.open(osp.join(e[image_index]))
                img = img.resize((self.image_size, self.image_size))
                img = transforms.ToTensor()(img)[:3]
                imgs.append(img)
        else:
            if self.prep_path:
                paths = self.episodes[index]
            else:
                paths, ep_len = self.get_path(index)
            for path in paths:
                img = Image.open(path)
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
        length = len(self.folders)
        if self.mode == 'train':
            return length * self.seq_per_episode
        else:
            return length


class SmthDataset(Dataset):
    def __init__(self, root, mode, ep_len=100, sample_length=20, image_size=128, prep_path=False, dense=False,
                 clips_per_video=2):
        assert mode in ['train', 'val', 'valid', 'validation', 'test']
        if mode in ['val', 'valid']:
            mode = 'validation'
        self.mode = mode
        self.dense = dense
        self.clips_per_video = clips_per_video if not dense else 1

        self.n_max_train = 180_000
        self.n_max_valid = 100
        self.prep_path = prep_path
        self.root = root
        self.frames_root = os.path.join(self.root, 'frames')
        self.labels_root = os.path.join(self.root, 'labels')
        self.text_root = os.path.join(self.root, 'text_labels', f'{mode}')
        if os.path.exists(os.path.join(self.text_root, 'index.json')):
            self.text_paths = load_index(self.text_root)
        else:
            self.text_paths = None
        self.image_size = image_size

        with open(os.path.join(self.labels_root, f'{mode}.json'), 'r', encoding='utf-8') as file:
            label_data = json.load(file)

        self.ep_idx = [label_data[i]['id'] for i in range(len(label_data))]
        self.sample_length = sample_length

        get_dir_num = lambda x: int(x)
        self.get_num = lambda x: int(osp.splitext(osp.basename(x))[0])

        self.folders = self.ep_idx
        self.folders.sort(key=get_dir_num)

        self.episodes = []
        self.episodes_len = []
        self.EP_LEN = ep_len
        self.seq_per_episode = self.EP_LEN - self.sample_length + 1

        if self.mode == 'train' and self.n_max_train > 0:
            self.folders = self.folders[:self.n_max_train]
        elif self.mode == 'validation' and self.n_max_valid > 0:
            self.folders = self.folders[:self.n_max_valid]

        if self.prep_path:
            for f in self.folders:
                dir_name = os.path.join(self.frames_root, str(f))
                paths = list(glob.glob(osp.join(dir_name, '*.jpg')))
                paths.sort(key=self.get_num)
                if len(paths) < self.EP_LEN:
                    self.episodes_len.append(len(paths))
                else:
                    self.episodes_len.append(self.EP_LEN)
                while len(paths) < self.EP_LEN:
                    paths.append(paths[-1])
                self.episodes.append(paths[:self.EP_LEN])

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor()
        ])

    def get_path(self, idx):
        ep_idx = self.folders[idx]
        dir_name = os.path.join(self.frames_root, str(ep_idx))
        path = list(glob.glob(osp.join(dir_name, '*.jpg')))
        path.sort(key=self.get_num)
        orig_len = len(path)
        while len(path) < self.EP_LEN:
            path.append(path[-1])
        return path, orig_len

    def __getitem__(self, index):
        imgs = []
        dones = []

        if self.mode == 'train':
            if self.dense:
                # DENSE MODE
                ep = index // self.seq_per_episode
                offset = index % self.seq_per_episode
                end = offset + self.sample_length

                if self.prep_path:
                    ep_len = self.episodes_len[ep]
                    ep_path = self.episodes[ep]
                else:
                    ep_path, ep_len = self.get_path(ep)

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

                    done_i = torch.tensor((image_index < ep_len),
                                          dtype=torch.int)  # episode_mask: 1 if valid else 0, after end of episode
                    dones.append(done_i)

                # instructions
                if self.text_paths is not None:
                    ep_idx = self.folders[ep]
                    inst = load_embedding(self.text_paths, self.text_root, ep_idx)
                    instruction = inst['label']
                    # instructions embeddings
                    instruction_embedding = inst['embedding'].detach()
                else:
                    instruction = instruction_embedding = torch.zeros(0)

            else:
                # SPARSE MODE with multiple clips per video
                ep = index // self.clips_per_video
                clip_num = index % self.clips_per_video

                if self.prep_path:
                    ep_path = self.episodes[ep]
                    ep_len = self.episodes_len[ep]
                else:
                    ep_path, ep_len = self.get_path(ep)

                # max_offset = max(1, ep_len - self.sample_length)
                max_offset = max(1, ep_len - self.sample_length // 2)
                # To make clip selection more deterministic per clip_num, optionally:
                #   seed = (ep * self.clips_per_video + clip_num)
                #   rng = random.Random(seed)
                #   offset = rng.randint(0, max_offset)
                offset = random.randint(0, max_offset)

                for image_index in range(offset, offset + self.sample_length):
                    img = Image.open(ep_path[image_index])
                    img = self.transform(img)[:3]
                    imgs.append(img)

                    done_i = torch.tensor((image_index < ep_len),
                                          dtype=torch.int)  # episode_mask: 1 if valid else 0, after end of episode
                    dones.append(done_i)

                # instructions
                if self.text_paths is not None:
                    ep_idx = self.folders[ep]
                    inst = load_embedding(self.text_paths, self.text_root, ep_idx)
                    instruction = inst['label']
                    # instructions embeddings
                    instruction_embedding = inst['embedding'].detach()
                else:
                    instruction = instruction_embedding = torch.zeros(0)

        else:
            # EVAL MODE — always return full video (padded if needed)
            if self.prep_path:
                paths = self.episodes[index]
                ep_len = self.episodes_len[index]
            else:
                paths, ep_len = self.get_path(index)
            for pi, path in enumerate(paths):
                img = Image.open(path)
                img = self.transform(img)[:3]
                imgs.append(img)

                done_i = torch.tensor((pi < ep_len),
                                      dtype=torch.int)  # episode_mask: 1 if valid else 0, after end of episode
                dones.append(done_i)

            # instructions
            if self.text_paths is not None:
                ep_idx = self.folders[index]
                inst = load_embedding(self.text_paths, self.text_root, ep_idx)
                instruction = inst['label']
                # instructions embeddings
                instruction_embedding = inst['embedding'].detach()
            else:
                instruction = instruction_embedding = torch.zeros(0)

        img = torch.stack(imgs, dim=0).float()
        dones = torch.stack(dones, dim=0).int()

        # Placeholder meta fields
        pos = torch.zeros(0)
        # size = torch.zeros(0)
        # id = torch.zeros(0)
        # in_camera = torch.zeros(0)

        return img, pos, instruction, instruction_embedding, dones

    def __len__(self):
        length = len(self.folders)
        if self.mode == 'train':
            if self.dense:
                return length * self.seq_per_episode
            else:
                return length * self.clips_per_video
        else:
            return length


class SmthDatasetImage(Dataset):
    def __init__(self, root, mode, ep_len=100, sample_length=1, image_size=128, prep_path=False):
        assert mode in ['train', 'val', 'valid', 'validation', 'test']
        if mode == 'val' or mode == 'valid':
            mode = 'validation'
        self.n_max_train = -1
        self.n_max_valid = 1000
        self.prep_path = prep_path
        self.root = root
        self.frames_root = os.path.join(self.root, 'frames')
        self.labels_root = os.path.join(self.root, 'labels')
        self.text_root = os.path.join(self.root, 'text_labels', f'{mode}')
        if os.path.exists(os.path.join(self.text_root, 'index.json')):
            self.text_paths = load_index(self.text_root)
        else:
            self.text_paths = None
        self.image_size = image_size

        self.mode = mode
        with open(os.path.join(self.labels_root, f'{mode}.json'), 'r', encoding='utf-8') as file:
            label_data = json.load(file)
        self.ep_idx = [label_data[i]['id'] for i in range(len(label_data))]
        self.sample_length = sample_length

        # Get all numbers
        get_dir_num = lambda x: int(x)
        self.get_num = lambda x: int(osp.splitext(osp.basename(x))[0])

        #         self.folders = [d for d in os.listdir(self.frames_root) if osp.isdir(osp.join(self.frames_root, d))]
        self.folders = self.ep_idx
        self.folders.sort(key=get_dir_num)

        self.episodes = []
        self.episodes_len = []
        self.EP_LEN = ep_len
        self.seq_per_episode = self.EP_LEN - self.sample_length + 1
        if self.mode == 'train' and self.n_max_train > 0:
            self.folders = self.folders[:self.n_max_train]
        elif self.mode == 'validation' and self.n_max_valid > 0:
            self.folders = self.folders[:self.n_max_valid]

        if self.prep_path:
            for f in self.folders:
                dir_name = os.path.join(self.frames_root, str(f))
                paths = list(glob.glob(osp.join(dir_name, '*.jpg')))
                get_num = lambda x: int(osp.splitext(osp.basename(x))[0])
                paths.sort(key=get_num)
                if len(paths) < self.EP_LEN:
                    # continue
                    self.episodes_len.append(len(paths))
                else:
                    self.episodes_len.append(self.EP_LEN)
                while len(paths) < self.EP_LEN:
                    paths.append(paths[-1])
                self.episodes.append(paths[:self.EP_LEN])

    def get_path(self, idx):
        ep_idx = self.folders[idx]
        dir_name = os.path.join(self.frames_root, str(ep_idx))
        path = list(glob.glob(osp.join(dir_name, '*.jpg')))
        path.sort(key=self.get_num)
        orig_len = len(path)
        while len(path) < self.EP_LEN:
            path.append(path[-1])
        return path, orig_len

    def __getitem__(self, index):
        imgs = []
        dones = []
        # if self.mode == 'train':
        # Implement continuous indexing
        ep = index // self.seq_per_episode
        offset = index % self.seq_per_episode
        end = offset + self.sample_length
        # if `end` is after the episode ended, move backwards
        if self.prep_path:
            ep_len = self.episodes_len[ep]
            ep_path = self.episodes[ep]
        else:
            ep_path, ep_len = self.get_path(ep)
        #             print(f'{ep_pahth}: {ep_len}')
        #             ep_len = self.episodes_len[ep]
        if end > ep_len:
            # print(f'before: offset: {offset}, end: {end}, ep_len: {ep_len}')
            if self.sample_length > ep_len:
                offset = 0
                end = offset + self.sample_length
            else:
                offset = ep_len - self.sample_length
                end = ep_len
            # print(f'after: offset: {offset}, end: {end}, ep_len: {ep_len}')

        #             e = self.episodes[ep]
        e = ep_path
        for image_index in range(offset, end):
            img = Image.open(osp.join(e[image_index]))
            img = img.resize((self.image_size, self.image_size))
            img = transforms.ToTensor()(img)[:3]
            imgs.append(img)

            done_i = torch.tensor((image_index < ep_len),
                                  dtype=torch.int)  # episode_mask: 1 if valid else 0, after end of episode
            dones.append(done_i)
        # else:
        #     if self.prep_path:
        #         paths = self.episodes[index]
        #     else:
        #         paths, ep_len = self.get_path(index)
        #     for path in paths:
        #         img = Image.open(path)
        #         img = img.resize((self.image_size, self.image_size))
        #         img = transforms.ToTensor()(img)[:3]
        #         imgs.append(img)

        # instructions
        if self.text_paths is not None:
            ep_idx = self.folders[ep]
            inst = load_embedding(self.text_paths, self.text_root, ep_idx)
            instruction = inst['label']
            # instructions embeddings
            instruction_embedding = inst['embedding'].detach()
        else:
            instruction = instruction_embedding = torch.zeros(0)

        img = torch.stack(imgs, dim=0).float()
        dones = torch.stack(dones, dim=0).int()
        pos = torch.zeros(0)
        size = torch.zeros(0)
        id = torch.zeros(0)
        in_camera = torch.zeros(0)

        return img, pos, instruction, instruction_embedding, dones

    def __len__(self):
        length = len(self.folders)
        return length * self.seq_per_episode


if __name__ == '__main__':
    test_epochs = True
    plot = False
    # --- episodic setting --- #
    # root = '/mnt/data/tal/smthv2'
    root = '/media/newhd/data/sthv2'
    smth_ds = SmthDataset(root=root, ep_len=100, sample_length=10, mode='train', image_size=128)
    smth_dl = DataLoader(smth_ds, shuffle=True, pin_memory=False, batch_size=4, num_workers=0)
    batch = next(iter(smth_dl))
    im = batch[0]
    instruction = batch[2]
    instruction_embedding = batch[3]
    print(im.shape)
    print(f'instructions: {len(instruction)}, instruction[0]: {instruction[0]}')
    print(f'instructions embed: {instruction_embedding.shape}')

    if plot:
        import matplotlib.pyplot as plt

        img_np = im[0, 0].permute(1, 2, 0).data.cpu().numpy()
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        ax.imshow(img_np)
        plt.show()

    if test_epochs:
        from tqdm import tqdm

        pbar = tqdm(iterable=smth_dl)
        for batch in pbar:
            pass
        pbar.close()
