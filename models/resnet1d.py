
"""
@author: Kohei Tofu
@contact: fukouhei00 [at] yahoo.co.jp 
"""

import torch
import torch.nn as nn
import math
import models.model_base

def get_conv(in_planes, out_planes, kernel_size=3, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                     padding=kernel_size//2, bias=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size, \
                 stride=1, downsample=None, activate=None):
        super(BasicBlock, self).__init__()
        self.conv1 = get_conv(inplanes, planes, kernel_size=kernel_size, stride=stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        #self.relu = nn.LeakyReLU(inplace=True)
        #self.relu = activate
        self.conv2 = get_conv(planes, planes, kernel_size=kernel_size, stride=stride)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, stride=1,
                               padding=kernel_size//2, bias=True)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * self.expansion, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        #self.relu = nn.LeakyReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

        
class ResNet1D(nn.Module):

    def __init__(self, inplanes, block, layers, n_output, groups=1, width_per_group=64):
        self.inplanes = inplanes
        super(ResNet1D, self).__init__()
        self.groups = groups
        self.base_width = width_per_group
        
        self.conv1 = nn.Conv1d(1, inplanes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 256, layers[0], kernel_size=1, stride=1)
        self.layer2 = self._make_layer(block, 256, layers[1], kernel_size=5, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], kernel_size=5, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], kernel_size=1, stride=1)


        self.conv_merge = nn.Conv1d(512 * block.expansion, n_output,
                                    kernel_size=1, stride=1
                                    )

        
        #self.avgpool = nn.AdaptiveAvgPool1d(n_output, stride=1)
        #self.fc = nn.Linear(256 * block.expansion, n_output)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, kernel_size=3, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, kernel_size=kernel_size,
                            stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv_merge(x)

        return x


if __name__ == '__main__':
    imgs = torch.randn(64, 1, 256)

    #resnet = ResNet1D(32, BasicBlock, [2, 2, 2, 2], 256)
    #resnet = ResNet1D(32, Bottleneck, [2, 2, 2, 2], 512)
    resnet = ResNet1D(32, Bottleneck, [3, 4, 6, 3], 1024)

    out = resnet(imgs)

    print(imgs.shape)
    print(out.shape)