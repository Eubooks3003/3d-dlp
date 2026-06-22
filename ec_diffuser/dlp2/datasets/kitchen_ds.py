from torch.utils.data import Dataset
from torchvision import transforms
import glob
import os
import os.path as osp
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class KitchenDS(Dataset):
    def __init__(self, root, mode, sample_length=1, res=128):
        # path = os.path.join(root, mode)
        assert mode in ['train', 'val', 'valid']
        if mode == 'val':
            mode = 'valid'
        self.root = os.path.join(root, mode)
        self.res = res

        self.mode = mode
        self.sample_length = sample_length

        # Get all numbers
        self.folders = []
        for file in os.listdir(self.root):
            try:
                self.folders.append(int(file))
            except ValueError:
                continue
        self.folders.sort()

        self.paths = []

        for f in self.folders:
            dir_name = os.path.join(self.root, str(f))
            paths = list(glob.glob(osp.join(dir_name, '*.png')))
            # if len(paths) != self.EP_LEN:
            #     continue
            # assert len(paths) == self.EP_LEN, 'len(paths): {}'.format(len(paths))
            get_num = lambda x: int(osp.splitext(osp.basename(x))[0])
            paths.sort(key=get_num)
            self.paths.extend(paths)

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
    root = '/scratch/cluster/carlq/research/ECRL/data_dlp/franka_kitchen'
    # mode = 'train'
    mode = 'valid'
    ds = KitchenDS(root, mode, sample_length=1, res=64)
    print(len(ds))