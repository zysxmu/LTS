import math
import numpy as np
import torch.nn as nn
from model_fixed_network.quant_conv import QConv, QConv_8bit, QLinear, QLinear_8bit

__all__ = ['mv2_quant']

class InvertedResidual(nn.Module):
    def __init__(self, inp, outp, stride, expand_ratio, args=None):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.residual_connection = stride == 1 and inp == outp

        layers = []
        # expand
        expand_inp = inp*expand_ratio
        if expand_ratio != 1:
            layers += [
                # QConv2d(inp, expand_inp, 1, 1, 0, bias=False, pact_fp=getattr(FLAGS, 'pact_fp', False), double_side=double_side),
                # SwitchBN2d(expand_inp, affine=not getattr(FLAGS, 'stats_sharing', False)),
                # nn.ReLU(inplace=True),
                QConv(in_channels=inp, out_channels=expand_inp, kernel_size=1, stride=1, padding=0, bias=False,
                      args=args),
                nn.BatchNorm2d(expand_inp),
                nn.ReLU(inplace=True),
            ]
        # depthwise + project back
        layers += [
            # QConv2d(
            #     expand_inp, expand_inp, 3, stride, 1,
            #     groups=expand_inp, bias=False,
            #     pact_fp=getattr(FLAGS, 'pact_fp', False),
            #     double_side=double_side if expand_ratio == 1 else False),
            # SwitchBN2d(expand_inp, affine=not getattr(FLAGS, 'stats_sharing', False)),
            # nn.ReLU(inplace=True),
            # QConv2d(expand_inp, outp, 1, 1, 0, bias=False, pact_fp=getattr(FLAGS, 'pact_fp', False)),
            # SwitchBN2d(outp, affine=not getattr(FLAGS, 'stats_sharing', False)),
            QConv(
                in_channels=expand_inp, out_channels=expand_inp, kernel_size=3, stride=stride, padding=1,
                groups=expand_inp, bias=False, args=args),
            nn.BatchNorm2d(expand_inp),
            nn.ReLU(inplace=True),
            QConv(
                in_channels=expand_inp, out_channels=outp, kernel_size=1, stride=1, padding=0,
                bias=False, args=args),
            nn.BatchNorm2d(outp),
        ]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        if self.residual_connection:
            res = self.body(x)
            res += x
        else:
            res = self.body(x)
        return res


class Model(nn.Module):
    def __init__(self, num_classes=1000, args=None):
        super(Model, self).__init__()

        # setting of inverted residual blocks
        self.block_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # head
        channels = 32
        first_stride = 2
        self.head = nn.Sequential(
                        # QConv2d(
                        #     3, channels, 3,
                        #     first_stride, 1, bias=False,
                        #     bitw_min=8, bita_min=8,
                        #     weight_only=True),
                        # SwitchBN2d(channels, affine=not getattr(FLAGS, 'stats_sharing', False)),
                        # nn.ReLU(inplace=True),
                        # nn.Conv2d(
                        #     3, channels, kernel_size=3,
                        #     stride=first_stride, padding=1, bias=False),
                        QConv(in_channels=3, out_channels=channels, kernel_size=3, stride=first_stride, padding=1,
                            bias=False, args=args),
                        nn.BatchNorm2d(channels),
                        nn.ReLU(inplace=True),
                    )
        # body
        for idx, [t, c, n, s] in enumerate(self.block_setting):
            outp = c
            for i in range(n):
                if i == 0:
                    setattr(self, 'stage_{}_layer_{}'.format(idx, i),
                        InvertedResidual(channels, outp, s, t, args=args))
                else:
                    setattr(self, 'stage_{}_layer_{}'.format(idx, i),
                        InvertedResidual(channels, outp, 1, t, args=args))
                channels = outp

        # tail
        outp = 1280
        self.tail = nn.Sequential(
                        # QConv2d(
                        #     channels, outp,
                        #     1, 1, 0, bias=False,
                        #     pact_fp=getattr(FLAGS, 'pact_fp', False),
                        #     double_side=double_side),
                        # SwitchBN2d(outp, affine=not getattr(FLAGS, 'stats_sharing', False)),
                        # nn.ReLU(inplace=True),
                        QConv(in_channels=channels, out_channels=outp, kernel_size=1, stride=1, padding=0,
                              bias=False, args=args),
                        nn.BatchNorm2d(outp),
                        nn.ReLU(inplace=True),
                    )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # classifier
        self.classifier = nn.Sequential(
            # QLinear(
            #     outp,
            #     num_classes,
            #     bitw_min=8,
            #     pact_fp=getattr(FLAGS, 'pact_fp', False)
            # )
            # nn.Linear(outp, num_classes)
            QLinear(outp, num_classes, args=args)
        )
        self.reset_parameters()

    def forward(self, x):
        x = self.head(x)
        for idx, [_, _, n, _] in enumerate(self.block_setting):
            for i in range(n):
                x = getattr(self, 'stage_{}_layer_{}'.format(idx, i))(x)
        x = self.tail(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mv2_quant(args, **kwargs):
    return Model(args=args)