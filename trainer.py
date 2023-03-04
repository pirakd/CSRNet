import torch
import numpy as np
from utils import names
import copy


class Trainer:
    def __init__(self, max_num_epochs, loss_func, optimizer, eval_interval, device, verbosity=3,
                 early_stop=None):
        self.max_num_epochs = max_num_epochs
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.device = device
        self.eval_interval = eval_interval
        self.verbosity = verbosity
        self.early_stop = early_stop

    def train(self, train_loader, val_loader, model, max_evals_no_improvement=8):
        best_model = None
        self.losses = {'train': [], 'validation': []}
        self.metrics = {'train': [], 'validation': []}
        best_loss, best_mae, best_mse, best_epoch = 0, 0, 0, 0
        best_val_loss = torch.inf
        no_improvement_counter = 0

        val_loss, mae, mse = self.eval(model, val_loader)
        print('epoch {}, validation loss:{:.2e}, mae: {:.2f}, mse: {:.2f}'.format(0, val_loss, mae, mse))

        for epoch in range(self.max_num_epochs):
            train_loss = self.train_single_epoch(model, train_loader)

            self.losses['train'].append(train_loss)
            # self.metrics['train'].append(epoch_hits/n_samples_per_epoch)
            print('epoch {}, train_loss: {:.2e}'.format(epoch, train_loss))

            if (np.mod(epoch, self.eval_interval) == 0 and epoch) or (epoch+1 == self.max_num_epochs):
                val_loss, mae, mse = self.eval(model, val_loader)

                self.losses['validation'].append(val_loss)
                self.metrics['validation'].append((mae, mse))
                print('epoch {}, validation loss:{:.2e}, mae: {:.2f}, mse: {:.2f}'.format(epoch, val_loss, mae, mse))

                if best_val_loss > val_loss:
                    no_improvement_counter = 0
                    best_epoch, best_train_loss, best_val_loss, best_mae, best_mse = epoch, train_loss, val_loss, mae, mse
                    best_model = copy.deepcopy(model)
                else:
                    no_improvement_counter += 1

                if no_improvement_counter == max_evals_no_improvement:
                    print('early stopping on epoch {}, best epoch {}'.format(epoch, best_epoch))
                    break
                if epoch + 1 == self.max_num_epochs:
                    print('reached maximum number of epochs, best epoch {}'.format(best_epoch))

        train_stats = {'best_val_loss': best_val_loss, 'best_mae': best_mae, 'best_mse':best_mse,
                       'best_epoch': best_epoch}
        return train_stats, best_model

    def train_single_epoch(self, model, train_loader):
        n_samples = 0
        epoch_loss = 0
        model.train()
        for i, data in enumerate(train_loader):
            images = torch.squeeze(data[names.images])
            density_maps = torch.squeeze(data[names.densities], dim=0)
            n_samples += 1
            density_pred = model(images)
            loss = self.loss_func(density_pred, density_maps)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            print('sample {}, loss: {}'.format(i, loss))
        epoch_loss /= n_samples
        return epoch_loss

    def eval(self, model, val_loader):
        mae, mse = 0, 0
        n_samples = 0
        epoch_loss = 0
        model.train()
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                images = torch.squeeze(data[names.images])
                density_maps = torch.squeeze(data[names.densities], dim=0)
                n_samples += 1
                density_pred = model(images)
                loss = self.loss_func(density_pred, density_maps)
                diff = torch.abs(torch.sum(density_pred) - torch.sum(density_maps))
                mae += diff
                mse += torch.sum(diff**2)
                epoch_loss += loss.item()

        mae /= n_samples
        mse = torch.sqrt(mse / n_samples)
        epoch_loss /= n_samples
        return epoch_loss, mae, mse