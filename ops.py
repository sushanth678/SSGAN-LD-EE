import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class resnet_G(nn.Module):
    
    def __init__(self, input_size, output, kernel_size = 3, stride = 1, 
                spectral_normed = False, up_sampling = False):
        super(resnet_G, self).__init__()
        self.up_sampling = up_sampling
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(input_size)
        self.bn2 = nn.BatchNorm2d(output)
        self.upsample = nn.Upsample(scale_factor = 2, mode = 'nearest')
        self.conv1 = conv2d(input_size, output, spectral_normed = spectral_normed,
                            kernel_size = kernel_size, stride = stride, padding = 1)
        self.conv2 = conv2d(output, output, spectral_normed= spectral_normed, 
                            kernel_size = kernel_size, stride = stride, padding = 1)

    def forward(self, x):
        input = x
        x = self.relu(self.bn1(x))
        if self.up_sampling:
            x = self.upsample(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.conv2(self.relu(x))
        if self.up_sampling:
            ret = self.upsample(input) + x
        else:
            ret = input + x

        return ret

class resnet_D(nn.Module):

    def __init__(self, input_size, output, kernel = 3, stride = 1,
                spectral_normed = False, first_layer = False, down_sampling = False):
        super(resnet_D, self).__init__()
        self.down_sampling = down_sampling
        self.avgpool_short = nn.AvgPool2d(2, 2, padding = 1)
        self.avgpool2 = nn.AvgPool2d(2, 2, padding = 1)
        self.conv1 = conv2d(input_size, output, spectral_normed = spectral_normed,
                            kernel_size = kernel, stride = stride, padding = 1)
        self.conv2 = conv2d(output, output, spectral_normed = spectral_normed,
                            kernel_size = kernel, stride = stride, padding = 1)
        self.conv3 = conv2d(input_size, output, kernel_size = 1, stride = 1, padding = 0,
                                spectral_normed = False)
        self.relu = nn.ReLU()
        self.first_layer = first_layer

    def forward(self, x):

        input = x
        if self.first_layer:
            conv1 = self.relu(self.conv1(x))
            conv2 = self.relu(self.conv2(conv1))

        else:
            conv1 = self.conv1(self.relu(x))
            conv2 = self.conv2(self.relu(conv1))

        if self.down_sampling:
            conv2 = self.avgpool2(conv2)
            input = self.avgpool_short(input)

        return self.conv3(input) + conv2

class spectralnorm:

    def __init__(self, which):
        self.which = which

    def __call__(self, layer, input):

        weight_sn, u = self.compute_weight(layer)
        setattr(layer, self.which, weight_sn)
        setattr(layer, self.which + '_u', u)

    def compute_weight(self, layer):

        parameters = getattr(layer, self.which + '_orig')
        getter = getattr(layer, self.which + '_u')
        size = parameters.size()
        parameterlist = parameters.contiguous()
        parameterlist = parameterlist.view(size[0], -1)
        if parameterlist.is_cuda:
            getter = getter.cuda()
        tertiary = parameterlist.t() @ getter
        tertiary = tertiary/tertiary.norm()
        getter = parameterlist@tertiary
        getter = getter/getter.norm()
        spectral_normed_weights = parameterlist/(getter.t()@parameterlist@tertiary)
        spectral_normed_weights = spectral_normed_weights.view(*size)
        tensor = Variable(getter.data)

        return spectral_normed_weights, tensor

    @staticmethod
    def compute(layer, which):

        parameter = getattr(layer, which)
        sn = spectralnorm(which)
        del layer._parameters[which]
        layer.register_parameter(which+'_orig', nn.Parameter(parameter.data))
        input_channel = parameter.size(0)
        tensor = Variable(torch.randn(input_channel, 1) * 0.1, requires_grad=False)
        setattr(layer, which+'_u', tensor)
        setattr(layer, which, sn.compute_weight(layer)[0])
        layer.register_forward_pre_hook(sn)

        return sn

def spectral_norm(layer, which='weight'):
    
    spectralnorm.compute(layer, which)

    return layer

class conv2d(nn.Module):

    def __init__(self, input_size, output, padding, kernel_size = 4, stride = 2,
                spectral_normed = False):
        super(conv2d, self).__init__()

        self.conv = nn.Conv2d(input_size, output, kernel_size, stride, 
                                padding = padding)
        if spectral_normed:
            self.conv = spectral_norm(self.conv)

    def forward(self, input):
        return self.conv(input)

class deconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, padding, kernel_size = (4,4), stride = (2,2),
                spectral_normed = False, iter = 1):
        super(deconv2d, self).__init__()

        self.devconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 
                        stride, padding = padding)
        if spectral_normed:
            self.devconv = spectral_norm(self.deconv)

    def forward(self, input):
        out = self.devconv(input)
        return out    

