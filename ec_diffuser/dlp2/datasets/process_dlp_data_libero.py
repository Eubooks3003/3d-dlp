import os
import numpy as np
from PIL import Image
from tqdm import tqdm

path_to_npy = 'data_dlp/libero_spatial/all_np_image_front.npy'
data_save_dir = 'data_dlp/libero_spatial/'

# load raw image data
loaded_file = np.load(path_to_npy).squeeze(1)
if len(loaded_file.shape) == 6:
    n_episodes, horizon, n_views, c, h, w = loaded_file.shape
    loaded_file = loaded_file.reshape([n_episodes, -1, c, h, w])
print(f"Processing data from: {path_to_npy}")

# create dirs for saving dataset
train_dir = os.path.join(data_save_dir, 'train')
valid_dir = os.path.join(data_save_dir, 'valid')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)
print(f"Saving processed data in: {data_save_dir}")

# random permutation
loaded_file = np.random.permutation(loaded_file)

total_timesteps = loaded_file.shape[0]
num_train_timesteps = int(0.85 * total_timesteps)
print(f'num_train_timesteps: {num_train_timesteps}')


for timestep in tqdm(range(loaded_file.shape[0])):
    if timestep < num_train_timesteps:
        ep_dir = os.path.join(train_dir)
    else:
        ep_dir = os.path.join(valid_dir)

    im = loaded_file[timestep].transpose(1, 2, 0)
    im = im[::-1]
    Image.fromarray((im * 255).astype(np.uint8)).save(os.path.join(ep_dir, f'{timestep}.png'))