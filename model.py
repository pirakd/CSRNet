from torchvision.models import vgg16
from torch import nn
from torchvision import models
import collections
import torch.nn.functional as F
import torch


def gen_layers(layers, in_channels=512, dilation_rate=2):
    layers_list = []
    for layer in layers:
        if layer == 'M':
            layers_list += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, layer, kernel_size=3,
                               padding=dilation_rate, dilation=dilation_rate)

            layers_list += [conv2d, nn.ReLU(inplace=True)]
            in_channels = layer
    return nn.Sequential(*layers_list)


class CSRNet(nn.Module):
    def __init__(self, dilation_rate=2):
        super(CSRNet, self).__init__()
        pretrained_model = vgg16(weights='IMAGENET1K_V1')
        self.frontend = pretrained_model.features[:23] # as in the paper
        self.backend = gen_layers([512, 512, 512, 256, 128, 64], 512, dilation_rate)
        self.backend.apply(self.weights_init)
        self.output = nn.Conv2d(64, 1, kernel_size=1)
        self.output.apply(self.weights_init)


    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output(x)
        x = F.interpolate(x, scale_factor=8)
        return x


    @staticmethod
    def weights_init(m, std=0.01, bias=0):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=std)
            nn.init.constant_(m.bias, bias)