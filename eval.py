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
from utils import get_root_path, redirect_output, names, get_time, log_args, load_checkpoint
import json

model_path = path.join('path', names.model_file)
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
loss_func = MSELoss(size_average=False)
optimizer = Adam(model.parameters(), lr=config.learning_rate)
model, optimizer, start_epoch = load_checkpoint(model, optimizer, model_path)
test_dataset_dir = config.test_set_dir
val_transforms = Compose([CropPatches(16, 0)])
image_transforms = Compose([Normalize(mean=[0.5, 0.5, 0.5], std=[0.225, 0.225, 0.225])])
test_dataset = DensityDataSet(test_dataset_dir, val_transforms, image_transforms,  device=device)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
trainer = Trainer(config.max_train_epochs, loss_func=loss_func, optimizer=optimizer,
                  eval_interval=config.eval_interval, device=device)
loss, mae, mse = train_stats, best_model = trainer.eval(model, test_loader)
mae, mse = mae.detach().cpu().item(), mse.detach().cpu().item(),
result_dict = {'model_path':model_path, 'loss':loss, 'mae':mae, 'mse':mse}
with open(path.join(output_file_path, 'results'), 'w') as f:
    json.dump(result_dict, f, indent=4, separators=(',', ': '))