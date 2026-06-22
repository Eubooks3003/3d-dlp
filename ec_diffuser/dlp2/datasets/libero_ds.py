from torch.utils.data import Dataset
from torchvision import transforms
import glob
import os
import os.path as osp
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class LiberoSpatial(Dataset):
    def __init__(self, root, mode, ep_len=50, sample_length=1, res=128, episodic=False):
        dir_name = os.path.join(root, mode)
        assert mode in ['train', 'val', 'valid']
        if mode == 'val':
            mode = 'valid'
        self.root = os.path.join(root, mode)
        self.res = res
        self.episodic = episodic

        self.mode = mode
        self.sample_length = sample_length

        # Get all numbers
        self.paths = list(glob.glob(osp.join(dir_name, '*.png')))
        self.paths.sort()
        print(self.paths[:5])

    def __getitem__(self, index):

        img = Image.open(self.paths[index])
        img = img.resize((self.res, self.res))
        img = transforms.ToTensor()(img)[:3]

        img = img.unsqueeze(0)
        pos = torch.zeros(0)
        size = torch.zeros(0)
        id = torch.zeros(0)
        in_camera = torch.zeros(0)

        return img, pos, size, id, in_camera

    def __len__(self):
        length = len(self.paths)
        return length


if __name__ == '__main__':
    root = '/scratch/cluster/carlq/research/ECRL/data_dlp/libero_spatial'
    # mode = 'train'
    mode = 'valid'
    ds = LiberoSpatial(root, mode, ep_len=50, sample_length=1, res=64)
    print(ds[0])
