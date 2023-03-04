from os import path, listdir, makedirs
import sys
from utils import names, get_dataset_dirs, get_shanghai_gt_name, get_shanghai_image_name, get_density_name
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
from misc import gaussian_filter_density

# dataset = names.shanghaitech_B
dataset = names.shanghaitech_A
dirs_dict = get_dataset_dirs(dataset_name=dataset)

gt_dir = dirs_dict[names.train_data][1]
images_dir = dirs_dict[names.train_data][0]
density_dir = dirs_dict[names.train_data][2]
makedirs(density_dir, exist_ok=True)
n_train_samples = len(listdir(gt_dir))

for sample_idx in range(1, n_train_samples+1):
    gt_path = path.join(gt_dir, get_shanghai_gt_name(sample_idx))
    image_path = path.join(images_dir, get_shanghai_image_name(sample_idx))
    mat = io.loadmat(gt_path)["image_info"][0,0][0,0][0].astype(int)
    image = plt.imread(image_path)
    density = gaussian_filter_density(mat, image.shape[:2])
    with open(path.join(density_dir, get_density_name(sample_idx)), 'wb') as f:
        np.save(f, density)


gt_dir = dirs_dict[names.test_data][1]
images_dir = dirs_dict[names.test_data][0]
density_dir = dirs_dict[names.test_data][2]
makedirs(density_dir, exist_ok=True)
n_train_samples = len(listdir(gt_dir))

for sample_idx in range(1, n_train_samples+1):
    gt_path = path.join(gt_dir, get_shanghai_gt_name(sample_idx))
    image_path = path.join(images_dir, get_shanghai_image_name(sample_idx))
    mat = io.loadmat(gt_path)["image_info"][0,0][0,0][0].astype(int)
    image = plt.imread(image_path)
    density = gaussian_filter_density(mat, image.shape[:2])
    with open(path.join(density_dir, get_density_name(sample_idx)), 'wb') as f:
        np.save(f, density)