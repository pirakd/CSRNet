from presets import ShanghaiA
from dataset import CropPatches, RandomVerticalFlip, RandomHorizontalFlip, Normalize
from torchvision.transforms import Compose
from dataset import DensityDataSet
from trainer import Trainer
from torch.nn import MSELoss
from torch.optim import Adam
from model import CSRNet
import torch
from torch.utils.data import DataLoader
from os import path, makedirs
from utils import get_root_path, redirect_output, names, get_time, log_args

root_path = get_root_path()
output_folder = names.output_folder
output_file_path = path.join(root_path, output_folder, path.basename(__file__).split('.')[0], get_time())
makedirs(output_file_path, exist_ok=True)
redirect_output(path.join(output_file_path, names.log_file))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('device: {}'.format(device))
config = ShanghaiA()
log_args(output_file_path, config)
model = CSRNet().to(device)
torch.sav
loss_func = MSELoss(size_average=False)
optimizer = Adam(model.parameters(), lr=1e-3)
train_dataset_dir = config.train_set_dir
test_dataset_dir = config.train_set_dir

train_transforms = Compose([CropPatches(16, 5), RandomHorizontalFlip(), RandomVerticalFlip()])
val_transforms = Compose([CropPatches(16, 0)])

image_transforms = Compose([Normalize(mean=[0.5, 0.5, 0.5], std=[0.225, 0.225, 0.225])])

train_dataset = DensityDataSet(train_dataset_dir, train_transforms, image_transforms, device=device)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

val_dataset = DensityDataSet(train_dataset_dir, val_transforms, image_transforms,  device=device)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

trainer = Trainer(config.max_train_epochs, loss_func=loss_func, optimizer=optimizer,
                  eval_interval=config.eval_interval, device=device)


train_stats, best_model = trainer.train(train_loader, val_loader, model)
torch.save(best_model.state_dict(), path.join(output_file_path, names.model_file))  # save checkpoints

