import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob
import random
data_load_dir = 'vqbet_datasets_for_release/relay_kitchen/image_obs'
data_save_dir = 'data_dlp/franka_kitchen/'

# load raw image data
paths = glob.glob(os.path.join(data_load_dir, '*.npy'))
random.shuffle(paths)
total_episodes = len(paths)
num_train_ep = int(0.85 * total_episodes)

# create dirs for saving dataset
train_dir = os.path.join(data_save_dir, 'train')
valid_dir = os.path.join(data_save_dir, 'valid')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)
print(f"Saving processed data in: {data_save_dir}")
print(f'num_train_ep: {num_train_ep}')

for ep in tqdm(range(total_episodes)):
    if ep < num_train_ep:
        ep_dir = os.path.join(train_dir, str(ep))
    else:
        ep_dir = os.path.join(valid_dir, str(ep))

    os.makedirs(ep_dir, exist_ok=True)
    loaded_file = np.load(paths[ep])
    print(loaded_file.shape)
    for i in range(loaded_file.shape[0]):
        im = loaded_file[i].transpose(1, 2, 0)
        Image.fromarray(im).save(os.path.join(ep_dir, f'{i}.png'))
