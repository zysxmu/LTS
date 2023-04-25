import torch
import torch.nn as nn
import logging
import os
from torch.hub import load_state_dict_from_url
from model_fixed_network.quant_conv import QConv, QConv_8bit, QLinear, QLinear_8bit
__all__ = ['resnet20_quant']

class QBasicBlock4Cifar(nn.Module):
    def __init__(self, args, inplanes, planes, stride=1):
        super(QBasicBlock4Cifar, self).__init__()

        # self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1 = QConv(args=args, in_channels=inplanes,
                           out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = QConv(args=args, in_channels=planes,
                           out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                # nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                QConv(args=args, in_channels=inplanes,
                                   out_channels=planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out

class QResNet4Cifar(nn.Module):
    def __init__(self, args, block, layers, num_classes=10):
        super(QResNet4Cifar, self).__init__()
        self.inplanes = 16

        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = QConv(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False, args=args)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(args, block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(args, block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(args, block, 64, layers[2], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(64, num_classes)
        self.fc = QLinear(64, num_classes, args=args)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, QConv):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, args, block, planes, blocks, stride=1):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(args, self.inplanes, planes, stride))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


def resnet20_quant(args, **kwargs):
    return QResNet4Cifar(args, QBasicBlock4Cifar, [3, 3, 3], **kwargs)


