import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ops import *

class Discriminator(nn.Module):
    def __init__(self, spectral_normed=True, num_rotation=4, channel=1):
        super(Discriminator, self).__init__()
        self.num_rotation = num_rotation
        self.resnet1 = resnet_D(channel, 128, spectral_normed = spectral_normed,
                            down_sampling = True, first_layer = True)
        self.resnet2 = resnet_D(128, 128, spectral_normed = spectral_normed,
                            down_sampling = True)
        self.resnet3 = resnet_D(128, 128, spectral_normed = spectral_normed)
        self.resnet4 = resnet_D(128, 128, spectral_normed = spectral_normed)
        self.fcgan = nn.Linear(128, 1)
        self.fcrot = nn.Linear(128, self.num_rotation)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        resnet1 = self.resnet1(x)
        resnet2 = self.resnet2(resnet1)
        resnet3 = self.resnet3(resnet2)
        resnet4 = self.resnet4(resnet3)
        resnet4 = self.relu(resnet4)
        resnet4 = torch.sum(resnet4,dim = (2,3))
        gan_logits = self.fcgan(resnet4)
        gan_prob = self.sigmoid(gan_logits)
        rotation_logits = self.fcrot(resnet4)
        rotation_prob = self.softmax(rotation_logits)

        return gan_prob, gan_logits, rotation_logits, rotation_prob


class Generator(nn.Module):
    def __init__(self, fake_batch_size=128, channel=1, output = 32):
        super(Generator, self).__init__()
        self.output = output
        s = 4
        if self.output == 48:
            s = 6
        self.s = s
        self.fake_batch_size = fake_batch_size
        self.fc = nn.Linear(fake_batch_size, s*s*256)
        self.conv = conv2d(256,channel, padding = 1, kernel_size = 3, stride = 1)
        self.resnet1 = resnet_G(256, 256, up_sampling = True)
        self.resnet2 = resnet_G(256, 256, up_sampling = True)
        self.resnet3 = resnet_G(256, 256, up_sampling = True)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        layer1 = self.fc(x)
        layer1 = layer1.view(-1, 256, self.s, self.s)
        resnet1 = self.resnet1(layer1)
        resnet2 = self.resnet2(resnet1)
        resnet3 = self.resnet3(resnet2)
        layer2 = self.relu(self.bn(resnet3))
        conv = self.conv(layer2)
        generated = self.tanh(conv)
        return generated

    def noise_sampling(self, sample_size):
        noise = torch.randn((sample_size, self.fake_batch_size))
        return noise

