from torch.utils.data import Dataset
from utils import get_dataset_dirs
from utils import names, get_shanghai_image_name, get_density_name
from os import path, listdir
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize
import torch
import torchvision.transforms.functional as F


class DensityDataSet(Dataset):
    def __init__(self, dataset_dir, general_transform, image_transform, pooling_factor=16, device='cpu'):
        self.images_dir = path.join(dataset_dir, names.images)
        self.densities_dir = path.join(dataset_dir, names.densities)
        self.images_names = listdir(self.images_dir)
        self.n_samples = len(listdir(self.densities_dir))
        self.general_transform = general_transform
        self.image_transform = image_transform
        self.pooling_factor = pooling_factor
        self.device= device

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        image = Image.open(path.join(self.images_dir, self.images_names[idx]))
        if image.mode == 'L':
            image = image.convert('RGB')
        density_map_name = '{}.npy'.format(self.images_names[idx].split('.')[0])
        density_map = np.load(path.join(self.densities_dir, density_map_name))
        image = ToTensor()(image)
        density_map = ToTensor()(density_map)
        if self.general_transform:
            image, density_map = self.general_transform((image, density_map))
        if self.image_transform:
            image = self.image_transform(image)
        return {names.images: image.to(self.device), names.densities: density_map.to(self.device)}


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_density_map):
        img, density_map = img_density_map
        if torch.rand(1)[0] < self.p:
            return torch.flip(img, dims=(3,)), torch.flip(density_map, dims=(3,))
        else:
            return img, density_map


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_density_map):
        img, density_map = img_density_map
        if torch.rand(1)[0] < self.p:
            return torch.flip(img, dims=(2,)), torch.flip(density_map, dims=(2,))
        else:
            return img, density_map


class CropPatches(object):
    def __init__(self, crop_factor=16, n_random_patches=5):
        self.crop_factor = crop_factor
        self.n_random_patches = n_random_patches

    def __call__(self, img_density_map):
        img, density_map = img_density_map
        img_width, img_height = img.shape[2], img.shape[1]
        crop_width = img_width // 2 - np.mod((img_width // 2),  self.crop_factor)
        crop_height = (img_height // 2) - np.mod((img_height // 2), self.crop_factor)

        crop_1 = F.crop(img, top=0, left=0, height=crop_height, width=crop_width)
        crop_2 = F.crop(img, top=0, left=crop_width, height=crop_height, width=crop_width)
        crop_3 = F.crop(img, top=crop_height, left= 0, height=crop_height, width=crop_width)
        crop_4 = F.crop(img, top=crop_height, left= crop_width, height=crop_height, width=crop_width)

        density_crop_1 = F.crop(density_map, top=0, left=0, height=crop_height, width=crop_width)
        density_crop_2 = F.crop(density_map, top=0, left=crop_width, height=crop_height, width=crop_width)
        density_crop_3 = F.crop(density_map, top=crop_height, left= 0, height=crop_height, width=crop_width)
        density_crop_4 = F.crop(density_map, top=crop_height, left= crop_width, height=crop_height, width=crop_width)

        if self.n_random_patches:
            random_crops_top = torch.randint(0, img_height - crop_height, (5, 1))
            random_crops_left = torch.randint(0, img_width - crop_height, (5, 1))

            img_random_crops = [F.crop(img, top=random_crops_top[i], left=random_crops_left[i], height=crop_height, width=crop_width) for i in range(5)]
            density_random_crops = [F.crop(density_map, top=random_crops_top[i], left=random_crops_left[i], height=crop_height, width=crop_width) for i in range(5)]
            image_crops = torch.stack([crop_1, crop_2, crop_3, crop_4] + img_random_crops)
            density_crops = torch.stack([density_crop_1, density_crop_2, density_crop_3, density_crop_4] + density_random_crops)
        else:
            image_crops = torch.stack([crop_1, crop_2, crop_3, crop_4])
            density_crops = torch.stack([density_crop_1, density_crop_2, density_crop_3, density_crop_4])
        return image_crops, density_crops


# class CropByFactor(object):
#     def __init__(self, crop_factor):
#         self.crop_factor = crop_factor
#
#     def __call__(self, img_density_map):
#         img, density_map = img_density_map
#         img_width, img_height = img.size
#         cropped_width_size = img_width - np.mod(img_width, self.crop_factor)
#         cropped_height_size = img_height - np.mod(img_height, self.crop_factor)
#         cropped_img = F.crop(img, 0, 0, cropped_height_size, cropped_width_size)
#         cropped_density_map = F.crop(density_map, 0, 0, img_height, img_width)
#         return cropped_img, cropped_density_map
#

if __name__ == '__main__':
    # cp = CropPatches(16)
    # img = torch.randn((3, 400, 400))
    # a = cp(img, img)

    dataset_name = names.shanghaitech_A
    dataset_dir = path.join(path.dirname(__file__), names.datasets, dataset_name, names.train_data)
    transforms = Compose([CropPatches(16), RandomHorizontalFlip(), RandomVerticalFlip()])
    train_dataset = DensityDataSet(dataset_dir, transforms, None)
    a = train_dataset[0]
    b=1
    # def __getitem__(self, index):
    #     index = index % len(self.data_files)
    #     fname = self.data_files[index]
    #     img, dmap = self.read_image_and_dmap(fname)
    #     if self.main_transform is not None:
    #         img, dmap = self.main_transform((img, dmap))
    #     if self.img_transform is not None:
    #         img = self.img_transform(img)
    #     if self.dmap_transform is not None:
    #         dmap = self.dmap_transform(dmap)
    #     return {'image': img, 'densitymap': dmap}
    #
    # def read_image_and_dmap(self, fname):
    #     img = Image.open(path.join(self.img_path, fname))
    #     if img.mode == 'L':
    #         print('There is a grayscale image.')
    #         img = img.convert('RGB')
    #
    #     dmap = np.load(os.path.join(
    #         self.dmap_path, os.path.splitext(fname)[0] + '.npy'))
    #     dmap = dmap.astype(np.float32, copy=False)
    #     dmap = Image.fromarray(dmap)
    #     return img, dmap



