


import os
import torch.nn as nn
import torch.nn.init as init
from custom_modules import *
from inspect import isfunction

__all__ = ['mobilenetv1_w1_fp', 'mobilenetv1_w1_quant']


def get_activation_layer(activation):

    assert (activation is not None)
    if isfunction(activation):
        return activation()
    elif isinstance(activation, str):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "relu6":
            return nn.ReLU6(inplace=True)
        else:
            raise NotImplementedError()
    else:
        assert (isinstance(activation, nn.Module))
        return activation

class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 groups=1,
                 bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation=(lambda: nn.ReLU(inplace=True))):
        super(ConvBlock, self).__init__()
        self.activate = (activation is not None)
        self.use_bn = use_bn

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(
                num_features=out_channels,
                eps=bn_eps)
        if self.activate:
            self.activ = get_activation_layer(activation)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


class ConvBlock_frozen(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 groups=1,
                 bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation=(lambda: nn.ReLU(inplace=True)), args=None):
        super(ConvBlock_frozen, self).__init__()
        self.activate = (activation is not None)
        self.use_bn = use_bn

        self.conv = QConv_frozen(in_channels, out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding, dilation=dilation,
                  groups=groups, bias=bias, args=args)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(
                num_features=out_channels,
                eps=bn_eps)
        if self.activate:
            self.activ = get_activation_layer(activation)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def conv1x1_block(in_channels, out_channels, stride=1, padding=0, groups=1, bias=False, use_bn=True, bn_eps=1e-5,
                  activation=(lambda: nn.ReLU(inplace=True))):
    """
    1x1 version of the standard convolution block.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 0
        Padding value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation
    )

def conv1x1_block_frozen(in_channels, out_channels, stride=1, padding=0, groups=1, bias=False, use_bn=True, bn_eps=1e-5,
                      activation=(lambda: nn.ReLU(inplace=True)), args=None):

    return ConvBlock_frozen(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation,
        args=args
    )

def conv3x3_block(in_channels, out_channels, stride=1, padding=1, dilation=1, groups=1, bias=False, use_bn=True,
                  bn_eps=1e-5, activation=(lambda: nn.ReLU(inplace=True))):
    """
    3x3 version of the standard convolution block.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation)

def conv3x3_block_frozen(in_channels, out_channels, stride=1, padding=1, dilation=1, groups=1, bias=False, use_bn=True,
                  bn_eps=1e-5, activation=(lambda: nn.ReLU(inplace=True)), args=None):
    return ConvBlock_frozen(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation,
        args=args
    )

def dwconv3x3_block(in_channels, out_channels, stride=1, padding=1, dilation=1, bias=False, bn_eps=1e-5,
                    activation=(lambda: nn.ReLU(inplace=True))):
    return dwconv_block(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        bn_eps=bn_eps,
        activation=activation)

def dwconv3x3_block_frozen(in_channels, out_channels, stride=1, padding=1, dilation=1, bias=False, bn_eps=1e-5,
                        activation=(lambda: nn.ReLU(inplace=True)), args=None):

    return dwconv_block_frozen(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        bn_eps=bn_eps,
        activation=activation,
        args=args
    )

def dwconv_block(in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, bias=False, use_bn=True,
                 bn_eps=1e-5, activation=(lambda: nn.ReLU(inplace=True))):
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=out_channels,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation)

def dwconv_block_frozen(in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, bias=False, use_bn=True,
                     bn_eps=1e-5, activation=(lambda: nn.ReLU(inplace=True)), args=None):
    return ConvBlock_frozen(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=out_channels,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation,
        args=args
    )


class DwsConvBlock(nn.Module):
    """
    Depthwise separable convolution block with BatchNorms and activations at each convolution layers. It is used as
    a MobileNet unit.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride):
        super(DwsConvBlock, self).__init__()
        self.dw_conv = dwconv3x3_block(
            in_channels=in_channels,
            out_channels=in_channels,
            stride=stride)
        self.pw_conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x

class DwsConvBlock_frozen(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 args=None):
        super(DwsConvBlock_frozen, self).__init__()
        self.dw_conv = dwconv3x3_block_frozen(
            in_channels=in_channels,
            out_channels=in_channels,
            stride=stride,
            args=args
        )
        self.pw_conv = conv1x1_block_frozen(
            in_channels=in_channels,
            out_channels=out_channels,
            args=args
        )

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class MobileNet(nn.Module):
    def __init__(self,
                 channels,
                 first_stage_stride,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(MobileNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        init_block_channels = channels[0][0]
        self.features.add_module("init_block", conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            stride=2))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels[1:]):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and ((i != 0) or first_stage_stride) else 1
                stage.add_module("unit{}".format(j + 1), DwsConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module('final_pool', nn.AvgPool2d(
            kernel_size=7,
            stride=1))

        self.output = nn.Linear(
            in_features=in_channels,
            out_features=num_classes)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if 'dw_conv.conv' in name:
                init.kaiming_normal_(module.weight, mode='fan_in')
            elif name == 'init_block.conv' or 'pw_conv.conv' in name:
                init.kaiming_normal_(module.weight, mode='fan_out')
            elif 'bn' in name:
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)
            elif 'output' in name:
                init.kaiming_normal_(module.weight, mode='fan_out')
                init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x

class MobileNet_quant(nn.Module):
    def __init__(self, channels, first_stage_stride, args=None, in_channels=3, in_size=(224, 224), num_classes=1000):
        super(MobileNet_quant, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        init_block_channels = channels[0][0]
        self.features.add_module("init_block", conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            stride=2))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels[1:]):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and ((i != 0) or first_stage_stride) else 1
                stage.add_module("unit{}".format(j + 1), DwsConvBlock_frozen(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    args=args
                ))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)

        self.last_bn = nn.BatchNorm1d(in_channels)
        self.features.add_module('final_pool', nn.AvgPool2d(
            kernel_size=7,
            stride=1))

        self.output = nn.Linear(
            in_features=in_channels,
            out_features=num_classes)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if 'dw_conv.conv' in name:
                init.kaiming_normal_(module.weight, mode='fan_in')
            elif name == 'init_block.conv' or 'pw_conv.conv' in name:
                init.kaiming_normal_(module.weight, mode='fan_out')
            elif 'bn' in name:
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)
            elif 'output' in name:
                init.kaiming_normal_(module.weight, mode='fan_out')
                init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        ### last bn
        x = self.last_bn(x)


        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_mobilenet(version,
                  width_scale,
                  mode=None,
                  args=None):
    if version == 'orig':
        channels = [[32], [64], [128, 128], [256, 256], [512, 512, 512, 512, 512, 512], [1024, 1024]]
        first_stage_stride = False
    elif version == 'fd':
        channels = [[32], [64], [128, 128], [256, 256], [512, 512, 512, 512, 512, 1024]]
        first_stage_stride = True
    else:
        raise ValueError("Unsupported MobileNet version {}".format(version))

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]

    net = MobileNet(
        channels=channels,
        first_stage_stride=first_stage_stride)
    if mode == 'fp':
        net = MobileNet(
            channels=channels,
            first_stage_stride=first_stage_stride
        )
    elif mode == 'quant':
        net = MobileNet_quant(
            channels=channels,
            first_stage_stride=first_stage_stride,
            args=args
        )
    elif mode == 'allQ':
        net = MobileNet_allQ(
            channels=channels,
            first_stage_stride=first_stage_stride,
            args=args
        )
    return net


def mobilenetv1_w1_fp(args):
    return get_mobilenet(version="orig", width_scale=1.0, mode='fp', args=args)

def mobilenetv1_w1_quant(args):
    return get_mobilenet(version="orig", width_scale=1.0, mode='quant', args=args)

def mobilenetv1_w1_allQ(args):
    return get_mobilenet(version="orig", width_scale=1.0, mode='allQ', args=args)