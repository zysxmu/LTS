import math
import numpy as np
import torch.nn as nn



import torch
from model_fixed_network.quant_conv import QConv, QConv_8bit, QLinear, QLinear_8bit

__all__ = ['mv1_quant']

class QDepthwiseSeparableConv(nn.Module):
    def __init__(self, args, inp=0, outp=0, stride=0):
        super(QDepthwiseSeparableConv, self).__init__()
        assert stride in [1, 2]
        self.weight_bit = 2
        self.act_bit = 2

        layers = [
            QConv(inp, inp, kernel_size=3, stride=stride,
                            padding=1, groups=inp, bias=False, args=args),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),
            # QConv2d(
            #     inp, inp, 3, stride, 1, groups=inp, bias=False, pact_fp=getattr(FLAGS, 'pact_fp', False)),
            # SwitchBN2d(inp, affine=not getattr(FLAGS, 'stats_sharing', False)),
            # nn.ReLU(inplace=True),

            QConv(inp, outp, kernel_size=1, stride=1, padding=0, bias=False, args=args),
            nn.BatchNorm2d(outp),
            nn.ReLU(inplace=True),

            # QConv2d(inp, outp, 1, 1, 0, bias=False, pact_fp=getattr(FLAGS, 'pact_fp', False)),
            # SwitchBN2d(outp, affine=not getattr(FLAGS, 'stats_sharing', False)),
            # nn.ReLU(inplace=True),
        ]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        out = self.body(x)
        return out


class Model(nn.Module):
    def __init__(self, args, num_classes=1000):
        super(Model, self).__init__()

        # setting of inverted residual blocks
        self.block_setting = [
            # c, n, s
            [64, 1, 1],
            [128, 2, 2],
            [256, 2, 2],
            [512, 6, 2],
            [1024, 2, 2],
        ]

        j = 0
        # head
        channels = 32
        first_stride = 2
        self.head = nn.Sequential(
            # QConv2d(
            #     3, channels, 3,
            #     first_stride, 1, bias=False,
            #     bitw_min=8, bita_min=8, weight_only=True),
            # SwitchBN2d(channels, affine=not getattr(FLAGS, 'stats_sharing', False)),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(
            #     3, channels, kernel_size=3,
            #     stride=first_stride, padding=1, bias=False),
            QConv(3, channels, kernel_size=3,
                  stride=first_stride, padding=1, bias=False, args=args),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        # body
        for idx, [c, n, s] in enumerate(self.block_setting):
            outp = c
            if idx == len(self.block_setting) - 1:
                for i in range(n):
                    if i == 0:
                        setattr(self, 'stage_{}_layer_{}'.format(idx, i),
                                QDepthwiseSeparableConv(args, inp=channels, outp=outp, stride=s))
                    elif i == n - 1:
                        setattr(self, 'stage_{}_layer_{}'.format(idx, i),
                                QDepthwiseSeparableConv(args, inp=channels, outp=outp, stride=1))
                    else:
                        setattr(self, 'stage_{}_layer_{}'.format(idx, i),
                                QDepthwiseSeparableConv(args, inp=channels, outp=outp, stride=1))
                    channels = outp
                    j += 1

            elif idx == 0:
                for i in range(n):
                    assert n == 1
                    setattr(self, 'stage_{}_layer_{}'.format(idx, i),
                            QDepthwiseSeparableConv(args, inp=channels, outp=outp, stride=s))
                    channels = outp
                    j += 1
            else:
                for i in range(n):
                    if i == 0:
                        setattr(self, 'stage_{}_layer_{}'.format(idx, i),
                            QDepthwiseSeparableConv(args, inp=channels, outp=outp, stride=s))
                    elif i == n - 1:
                        setattr(self, 'stage_{}_layer_{}'.format(idx, i),
                            QDepthwiseSeparableConv(args, inp=channels, outp=outp, stride=1))
                    else:
                        setattr(self, 'stage_{}_layer_{}'.format(idx, i),
                            QDepthwiseSeparableConv(args, inp=channels, outp=outp, stride=1))
                    channels = outp
                    j += 1

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # classifier
        # self.classifier = nn.Linear(1024, num_classes)  # the last layer uses fp weights
        self.classifier = QLinear(1024, num_classes, args=args)
        self.reset_parameters()

    def forward(self, x):
        x = self.head(x)
        for idx, [_, n, _] in enumerate(self.block_setting):
            for i in range(n):
                x = getattr(self, 'stage_{}_layer_{}'.format(idx, i))(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def reset_parameters(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                # print('BatchNorm2d', n)
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # print('Linear', n)
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mv1_quant(args, **kwargs):
    return Model(args)