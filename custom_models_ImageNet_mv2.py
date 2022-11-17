import torch.nn as nn
from custom_modules import *
from inspect import isfunction
import torch
import torch.nn as nn
import torch.nn.init as init

__all__ = ['mobilenetv2_w1_fp', 'mobilenetv2_w1_quant', 'mobilenetv2_w1_allQ']

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)






def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, QConv_frozen):
        nn.init.kaiming_normal_(m.weight)





def get_activation_layer(activation):
    """
    Create activation layer from string/function.
    Parameters:
    ----------
    activation : function, or str, or nn.Module
        Activation function or name of activation function.
    Returns
    -------
    nn.Module
        Activation layer.
    """
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

class ConvBlock(torch.nn.Sequential):
    """
    Standard convolution block with Batch normalization and activation.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
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

        layers =[ nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        ]
        if self.use_bn:
            layers.append(nn.BatchNorm2d(
                num_features=out_channels,
                eps=bn_eps))
        if self.activate:
            layers.append(get_activation_layer(activation))
        super().__init__(*layers)

class ConvBlock_frozen(torch.nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1,
                 groups=1, bias=False, use_bn=True, bn_eps=1e-5, activation=(lambda: nn.ReLU(inplace=True)), args=None):
        super(ConvBlock_frozen, self).__init__()
        self.activate = (activation is not None)
        self.use_bn = use_bn

        layers =[ QConv_frozen(in_channels, out_channels, kernel_size=kernel_size,
                            stride=stride, padding=padding, dilation=dilation,
                            groups=groups, bias=bias, args=args)
                  ]
        if self.use_bn:
            layers.append(nn.BatchNorm2d(
                num_features=out_channels,
                eps=bn_eps))
        if self.activate:
            layers.append(get_activation_layer(activation))
        super().__init__(*layers)


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

def dwconv_block(in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, bias=False, use_bn=True,
                 bn_eps=1e-5, activation=(lambda: nn.ReLU(inplace=True))):
    """
    Depthwise version of the standard convolution block.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
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

def dwconv3x3_block(in_channels, out_channels, stride=1, padding=1, dilation=1, bias=False, bn_eps=1e-5,
                    activation=(lambda: nn.ReLU(inplace=True))):
    """
    3x3 depthwise version of the standard convolution block.
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
    bias : bool, default False
        Whether the layer uses a bias vector.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
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

class LinearBottleneck(nn.Module):
    """
    So-called 'Linear Bottleneck' layer. It is used as a MobileNetV2 unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the second convolution layer.
    expansion : bool
        Whether do expansion of channels.
    """
    def __init__(self, in_channels, out_channels, stride, expansion, args=None):
        super(LinearBottleneck, self).__init__()
        self.residual = (in_channels == out_channels) and (stride == 1)
        mid_channels = in_channels * 6 if expansion else in_channels


        layers = []
        if expansion != 0:
            layers.append(conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                activation="relu6"))
        layers.append(dwconv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride,
            activation="relu6"))

        layers.append(nn.Conv2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False))
        layers.append(nn.BatchNorm2d(
            num_features=out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.residual:
            identity = x
        x = self.conv(x)
        if self.residual:
            x = x + identity
        return x

class LinearBottleneck_frozen(nn.Module):

    def __init__(self, in_channels, out_channels, stride, expansion, args=None):
        super(LinearBottleneck_frozen, self).__init__()
        self.residual = (in_channels == out_channels) and (stride == 1)
        mid_channels = in_channels * 6 if expansion else in_channels

        layers = []
        if expansion != 0:
            layers.append(conv1x1_block_frozen(
                in_channels=in_channels,
                out_channels=mid_channels,
                activation="relu6",
                args=args
            ))
        layers.append(dwconv3x3_block_frozen(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride,
            activation="relu6",
            args=args
        ))

        layers.append(QConv_frozen(
            mid_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False, args=args))
        layers.append(nn.BatchNorm2d(
            num_features=out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.residual:
            identity = x
        x = self.conv(x)
        if self.residual:
            x = x + identity
        return x


class MobileNetV2(nn.Module):

    def __init__(self, channels, init_block_channels, final_block_channels, block=None, in_channels=3,
                 in_size=(224, 224), num_classes=1000, args=None):
        super(MobileNetV2, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        features = []
        features.append(conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            stride=2,
            activation="relu6"))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                expansion = (i != 0) or (j != 0)
                features.append(block(
                    in_channels=in_channels, out_channels=out_channels, stride=stride, expansion=expansion, args=args
                ))
                in_channels = out_channels
        features.append(conv1x1_block_frozen(in_channels=in_channels, out_channels=final_block_channels,
                                          activation="relu6", args=args))
        in_channels = final_block_channels
        self.features = nn.Sequential(*features)
        self.last_bn = nn.BatchNorm1d(in_channels)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_channels, num_classes),
        )

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.last_bn(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        ### add last bn

        ###
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return x

class MobileNetV2_allQ(nn.Module):

    def __init__(self, channels, init_block_channels, final_block_channels, block=None, in_channels=3,
                 in_size=(224, 224), num_classes=1000, args=None):
        super(MobileNetV2_allQ, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        features = []
        features.append(conv3x3_block_frozen(
            in_channels=in_channels,
            out_channels=init_block_channels,
            stride=2,
            activation="relu6"))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                expansion = (i != 0) or (j != 0)
                features.append(block(
                    in_channels=in_channels, out_channels=out_channels, stride=stride, expansion=expansion, args=args
                ))
                in_channels = out_channels
        features.append(conv1x1_block(in_channels=in_channels, out_channels=final_block_channels, activation="relu6"))
        in_channels = final_block_channels
        self.features = nn.Sequential(*features)
        self.last_bn = nn.BatchNorm1d(in_channels)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            QLinear(in_channels, num_classes, args, bias=True)
        )

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.last_bn(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        ### add last bn

        ###
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return x

def get_mobilenetv2(width_scale, mode='fp', args=None):

    init_block_channels = 32
    final_block_channels = 1280
    layers = [1, 2, 3, 4, 3, 3, 1]
    downsample = [0, 1, 1, 1, 0, 1, 0]
    channels_per_layers = [16, 24, 32, 64, 96, 160, 320]

    from functools import reduce
    channels = reduce(
        lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
        zip(channels_per_layers, layers, downsample),
        [[]])

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]
        init_block_channels = int(init_block_channels * width_scale)
        if width_scale > 1.0:
            final_block_channels = int(final_block_channels * width_scale)

    if mode == 'fp':
        net = MobileNetV2(
            channels=channels,
            init_block_channels=init_block_channels,
            final_block_channels=final_block_channels,
            block=LinearBottleneck,
            args=args
        )
    elif mode == 'quant':
        net = MobileNetV2(
            channels=channels,
            init_block_channels=init_block_channels,
            final_block_channels=final_block_channels,
            block=LinearBottleneck_frozen,
            args=args
        )
    elif mode == 'allQ':
        net = MobileNetV2_allQ(
            channels=channels,
            init_block_channels=init_block_channels,
            final_block_channels=final_block_channels,
            block=LinearBottleneck_frozen,
            args=args
        )

    return net

def mobilenetv2_w1_fp(args):
    return get_mobilenetv2(width_scale=1.0, mode='fp', args=args)

def mobilenetv2_w1_quant(args):
    return get_mobilenetv2(width_scale=1.0, mode='quant', args=args)

def mobilenetv2_w1_allQ(args):
    return get_mobilenetv2(width_scale=1.0, mode='allQ', args=args)