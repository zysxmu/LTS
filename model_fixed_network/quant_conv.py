import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

__all__ = ['QConv', 'QConv_First', 'QLinear', '']


class STE_discretizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_in, num_levels):
        x = x_in * (num_levels - 1)
        x = torch.round(x)
        x_out = x / (num_levels - 1)
        return x_out

    @staticmethod
    def backward(ctx, g):
        return g, None


# ref. https://github.com/ricky40403/DSQ/blob/master/DSQConv.py#L18
class QConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, args=None):
        super(QConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.STE_discretizer = STE_discretizer.apply
        self.weight_bit = 2  # defalut
        self.act_bit = 2  # defalut
        self.quan_weight = args.QWeightFlag
        self.quan_act = args.QActFlag
        if self.quan_weight:
            self.weight_levels = -1
            self.uW = nn.Parameter(data=torch.tensor(0).float())
            self.lW = nn.Parameter(data=torch.tensor(0).float())

        if self.quan_act:
            self.act_levels = -1
            self.uA = nn.Parameter(data=torch.tensor(0).float())
            self.lA = nn.Parameter(data=torch.tensor(0).float())

        self.register_buffer('init', torch.tensor([0]))

        self.register_buffer('prev_Qweight', torch.zeros_like(self.weight))
        self.register_buffer('distance', torch.zeros_like(self.weight))
        self.register_buffer('unchange_step', torch.zeros_like(self.weight))
        self.register_buffer('saved_weight', torch.zeros_like(self.weight))
        self.fixed_para_num = 0

        self.batch_num = 0

        self.step = -1
        self.warmup_epoch = args.warmup_epoch
        self.fixed_rate = args.fixed_rate
        self.distance_ema = args.distance_ema
        self.fixed_mode = args.fixed_mode
        self.total_epochs = args.epochs

        print(self.warmup_epoch, self.fixed_rate, self.distance_ema, self.fixed_mode)

    def weight_quantization(self, weight):
        if not self.quan_weight or self.weight_bit == 32:
            return weight

        weight = weight * (self.unchange_step == 0) + self.saved_weight * (self.unchange_step != 0)
        self.saved_weight = weight.detach().clone()

        self.weight_levels = 2 ** self.weight_bit
        weight = (weight - self.lW) / (self.uW - self.lW) # [0, 1]
        pre_quant_weight = weight.detach()
        weight = weight.clamp(min=0, max=1)  # [0, 1]
        weight = self.STE_discretizer(weight, self.weight_levels) # [0, 1]
        aft_quant_weight = weight.detach()
        weight = (weight - 0.5) * 2  # [-1, 1]

        if self.training and self.init == 0:
            # compute distance
            transition = (aft_quant_weight != self.prev_Qweight).float()
            if self.step == 0:
                self.distance = torch.abs((pre_quant_weight - aft_quant_weight).detach())
            else:
                self.distance = (1 - self.distance_ema) * torch.abs((pre_quant_weight -
                                                                     aft_quant_weight).detach()) \
                                + self.distance_ema * self.distance
                self.distance = self.distance * (transition == 0) + transition * (1 / (2 ** self.weight_bit - 1))
            self.prev_Qweight = aft_quant_weight.detach().clone()
            self.step += 1

            if self.step > self.batch_num * self.warmup_epoch:
                if self.fixed_mode == 'fixing':
                    self.unchange_step = \
                        self.unchange_step + self.step * (
                                    self.distance <= (1 / (2 ** self.weight_bit - 1) * self.fixed_rate)) * (
                                self.unchange_step == 0)
                elif self.fixed_mode == 'linear-growth':
                    fixed_rate = 1 / (self.total_epochs * self.batch_num - self.warmup_epoch * self.batch_num) \
                                 * (self.step - self.warmup_epoch * self.batch_num)
                    self.unchange_step = \
                        self.unchange_step + self.step * (
                                self.distance <= (1 / (2 ** self.weight_bit - 1) * fixed_rate)) * (
                                self.unchange_step == 0)
                elif self.fixed_mode == 'sine-growth':
                    fixed_rate = math.sin((self.step - self.warmup_epoch * self.batch_num) /
                                          (self.total_epochs * self.batch_num - self.warmup_epoch * self.batch_num) * math.pi / 2)
                    self.unchange_step = \
                        self.unchange_step + self.step * (
                                self.distance <= (1 / (2 ** self.weight_bit - 1) * fixed_rate)) * (
                                self.unchange_step == 0)
                else:
                    pass
            self.fixed_para_num = torch.sum(self.unchange_step != 0)

        return weight

    def act_quantization(self, x):
        if not self.quan_act or self.act_bit == 32:
            return x
        self.act_levels = 2 ** self.act_bit
        x = (x - self.lA) / (self.uA - self.lA)
        x = x.clamp(min=0, max=1)  # [0, 1]
        x = self.STE_discretizer(x, self.act_levels)
        return x

    def initialize(self, x):

        FPweight, FPact = self.weight.detach(), x.detach()

        if self.quan_weight:
            self.uW.data.fill_(FPweight.std() * 3.0)
            self.lW.data.fill_(-FPweight.std() * 3.0)

        if self.quan_act:
            self.uA.data.fill_(FPact.std() / math.sqrt(1 - 2 / math.pi) * 3.0)
            self.lA.data.fill_(FPact.min())


    def forward(self, x):

        if self.init == 1:
            self.initialize(x)


        FPweight, FPact = self.weight, x
        Qweight = self.weight_quantization(FPweight)
        Qact = self.act_quantization(FPact)

        output = F.conv2d(Qact, Qweight, self.bias, self.stride, self.padding, self.dilation,
                          self.groups)

        return output


# ref. https://github.com/ricky40403/DSQ/blob/master/DSQConv.py#L18
class QConv_8bit(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, args=None):
        super(QConv_8bit, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.STE_discretizer = STE_discretizer.apply
        self.bit = 8  # defalut
        self.quan_weight = args.QWeightFlag
        self.quan_act = args.QActFlag
        if self.quan_weight:
            self.weight_levels = -1
            self.uW = nn.Parameter(data=torch.tensor(0).float())
            self.lW = nn.Parameter(data=torch.tensor(0).float())

        if self.quan_act:
            self.act_levels = -1
            self.uA = nn.Parameter(data=torch.tensor(0).float())
            self.lA = nn.Parameter(data=torch.tensor(0).float())

        self.register_buffer('init', torch.tensor([0]))
        self.register_buffer('prev_Qweight', torch.zeros_like(self.weight))
        self.register_buffer('distance', torch.zeros_like(self.weight))
        self.register_buffer('unchange_step', torch.zeros_like(self.weight))
        self.register_buffer('saved_weight', torch.zeros_like(self.weight))
        self.fixed_para_num = 0

        self.batch_num = 0

        self.step = -1
        self.warmup_epoch = args.warmup_epoch
        self.fixed_rate = args.fixed_rate
        self.distance_ema = args.distance_ema
        self.fixed_mode = args.fixed_mode
        self.total_epochs = args.epochs

        print(self.warmup_epoch, self.fixed_rate, self.distance_ema, self.fixed_mode)

    def weight_quantization(self, weight):
        if not self.quan_weight or self.bit == 32:
            return weight

        weight = weight * (self.unchange_step == 0) + self.saved_weight * (self.unchange_step != 0)
        self.saved_weight = weight.detach().clone()

        self.weight_levels = 2 ** self.bit
        weight = (weight - self.lW) / (self.uW - self.lW) # [0, 1]
        pre_quant_weight = weight.detach()
        weight = weight.clamp(min=0, max=1)  # [0, 1]
        weight = self.STE_discretizer(weight, self.weight_levels) # [0, 1]
        aft_quant_weight = weight.detach()
        weight = (weight - 0.5) * 2  # [-1, 1]

        if self.training and self.init == 0:
            # compute distance
            transition = (aft_quant_weight != self.prev_Qweight).float()
            if self.step == 0:
                self.distance = torch.abs((pre_quant_weight - aft_quant_weight).detach())
            else:
                self.distance = (1 - self.distance_ema) * torch.abs((pre_quant_weight -
                                                                     aft_quant_weight).detach()) \
                                + self.distance_ema * self.distance
                self.distance = self.distance * (transition == 0) + transition * (1 / (2 ** self.bit - 1))
            self.prev_Qweight = aft_quant_weight.detach().clone()
            self.step += 1
            if self.step > self.batch_num * self.warmup_epoch:
                if self.fixed_mode == 'fixing':
                    self.unchange_step = \
                        self.unchange_step + self.step * (
                                    self.distance <= (1 / (2 ** self.bit - 1) * self.fixed_rate)) * (
                                self.unchange_step == 0)
                elif self.fixed_mode == 'linear-growth':
                    fixed_rate = 1 / (self.total_epochs * self.batch_num - self.warmup_epoch * self.batch_num) \
                                 * (self.step - self.warmup_epoch * self.batch_num)
                    self.unchange_step = \
                        self.unchange_step + self.step * (
                                self.distance <= (1 / (2 ** self.bit - 1) * fixed_rate)) * (
                                self.unchange_step == 0)
                elif self.fixed_mode == 'sine-growth':
                    fixed_rate = math.sin((self.step - self.warmup_epoch * self.batch_num) /
                                          (self.total_epochs * self.batch_num - self.warmup_epoch * self.batch_num) * math.pi / 2)
                    self.unchange_step = \
                        self.unchange_step + self.step * (
                                self.distance <= (1 / (2 ** self.bit - 1) * fixed_rate)) * (
                                self.unchange_step == 0)
                else:
                    pass
            self.fixed_para_num = torch.sum(self.unchange_step != 0)

        return weight

    def act_quantization(self, x):
        if not self.quan_act or self.bit == 32:
            return x
        self.act_levels = 2 ** self.bit
        x = (x - self.lA) / (self.uA - self.lA)
        x = x.clamp(min=0, max=1)  # [0, 1]
        x = self.STE_discretizer(x, self.act_levels)
        return x

    def initialize(self, x):

        FPweight, FPact = self.weight.detach(), x.detach()

        if self.quan_weight:
            self.uW.data.fill_(FPweight.std() * 3.0)
            self.lW.data.fill_(-FPweight.std() * 3.0)

        if self.quan_act:
            self.uA.data.fill_(FPact.std() / math.sqrt(1 - 2 / math.pi) * 3.0)
            self.lA.data.fill_(FPact.min())


    def forward(self, x):

        if self.init == 1:
            self.initialize(x)


        FPweight, FPact = self.weight, x
        Qweight = self.weight_quantization(FPweight)
        Qact = self.act_quantization(FPact)

        output = F.conv2d(Qact, Qweight, self.bias, self.stride, self.padding, self.dilation,
                          self.groups)

        return output

class QLinear(nn.Linear):
    def __init__(self, in_features, out_features, args, bias=True):
        super(QLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)

        self.STE_discretizer = STE_discretizer.apply
        self.weight_bit = 8  # defalut
        self.act_bit = 8  # defalut

        self.quan_weight = args.QWeightFlag
        self.quan_act = args.QActFlag
        if self.quan_weight:
            self.weight_levels = -1
            self.uW = nn.Parameter(data=torch.tensor(0).float())
            self.lW = nn.Parameter(data=torch.tensor(0).float())

        if self.quan_act:
            self.act_levels = -1
            self.uA = nn.Parameter(data=torch.tensor(0).float())
            self.lA = nn.Parameter(data=torch.tensor(0).float())

        self.register_buffer('init', torch.tensor([0]))
        self.register_buffer('prev_Qweight', torch.zeros_like(self.weight))
        self.register_buffer('distance', torch.zeros_like(self.weight))
        self.register_buffer('unchange_step', torch.zeros_like(self.weight))
        self.register_buffer('saved_weight', torch.zeros_like(self.weight))
        self.fixed_para_num = 0

        self.batch_num = 0

        self.step = -1
        self.warmup_epoch = args.warmup_epoch
        self.fixed_rate = args.fixed_rate
        self.distance_ema = args.distance_ema
        self.fixed_mode = args.fixed_mode
        self.total_epochs = args.epochs

        print(self.warmup_epoch, self.fixed_rate, self.distance_ema, self.fixed_mode)

    def weight_quantization(self, weight):
        if not self.quan_weight or self.weight_bit == 32:
            return weight

        weight = weight * (self.unchange_step == 0) + self.saved_weight * (self.unchange_step != 0)
        self.saved_weight = weight.detach().clone()

        self.weight_levels = 2 ** self.weight_bit
        weight = (weight - self.lW) / (self.uW - self.lW)

        pre_quant_weight = weight.detach()
        weight = weight.clamp(min=0, max=1)  # [0, 1]
        weight = self.STE_discretizer(weight, self.weight_levels)
        aft_quant_weight = weight.detach()
        weight = (weight - 0.5) * 2  # [-1, 1]

        if self.training and self.init == 0:
            # compute distance
            transition = (aft_quant_weight != self.prev_Qweight).float()
            if self.step == 0:
                self.distance = torch.abs((pre_quant_weight - aft_quant_weight).detach())
            else:
                self.distance = (1 - self.distance_ema) * torch.abs((pre_quant_weight -
                                                                     aft_quant_weight).detach()) \
                                + self.distance_ema * self.distance
                self.distance = self.distance * (transition == 0) + transition * (1 / (2 ** self.weight_bit - 1))
            self.prev_Qweight = aft_quant_weight.detach().clone()
            self.step += 1

            if self.step > self.batch_num * self.warmup_epoch:
                if self.fixed_mode == 'fixing':
                    self.unchange_step = \
                        self.unchange_step + self.step * (
                                    self.distance <= (1 / (2 ** self.weight_bit - 1) * self.fixed_rate)) * (
                                self.unchange_step == 0)
                elif self.fixed_mode == 'linear-growth':
                    fixed_rate = 1 / (self.total_epochs * self.batch_num - self.warmup_epoch * self.batch_num) \
                                 * (self.step - self.warmup_epoch * self.batch_num)
                    self.unchange_step = \
                        self.unchange_step + self.step * (
                                self.distance <= (1 / (2 ** self.weight_bit - 1) * fixed_rate)) * (
                                self.unchange_step == 0)
                elif self.fixed_mode == 'sine-growth':
                    fixed_rate = math.sin((self.step - self.warmup_epoch * self.batch_num) /
                                          (self.total_epochs * self.batch_num - self.warmup_epoch * self.batch_num) * math.pi / 2)
                    self.unchange_step = \
                        self.unchange_step + self.step * (
                                self.distance <= (1 / (2 ** self.weight_bit - 1) * fixed_rate)) * (
                                self.unchange_step == 0)
                else:
                    pass
            self.fixed_para_num = torch.sum(self.unchange_step != 0)
        return weight

    def act_quantization(self, x):
        if not self.quan_act or self.act_bit == 32:
            return x
        self.act_levels = 2 ** self.act_bit
        x = (x - self.lA) / (self.uA - self.lA)
        x = x.clamp(min=0, max=1)  # [0, 1]
        x = self.STE_discretizer(x, self.act_levels)
        return x

    def initialize(self, x):

        FPweight, FPact = self.weight.detach(), x.detach()

        if self.quan_weight:
            self.uW.data.fill_(FPweight.std() * 3.0)
            self.lW.data.fill_(-FPweight.std() * 3.0)

        if self.quan_act:
            self.uA.data.fill_(FPact.std() / math.sqrt(1 - 2 / math.pi) * 3.0)
            self.lA.data.fill_(FPact.min())


    def forward(self, x):

        if self.quan_weight and self.init == 1:
            self.initialize(x)

        FPweight, FPact = self.weight, x

        Qweight = self.weight_quantization(FPweight)
        Qact = self.act_quantization(FPact)

        output = F.linear(Qact, weight=Qweight, bias=self.bias)

        return output


class QLinear_8bit(nn.Linear):
    def __init__(self, in_features, out_features, args, bias=True):
        super(QLinear_8bit, self).__init__(in_features=in_features, out_features=out_features, bias=bias)

        self.STE_discretizer = STE_discretizer.apply
        self.bit = 8  # defalut

        self.quan_weight = args.QWeightFlag
        self.quan_act = args.QActFlag
        if self.quan_weight:
            self.weight_levels = -1
            self.uW = nn.Parameter(data=torch.tensor(0).float())
            self.lW = nn.Parameter(data=torch.tensor(0).float())

        if self.quan_act:
            self.act_levels = -1
            self.uA = nn.Parameter(data=torch.tensor(0).float())
            self.lA = nn.Parameter(data=torch.tensor(0).float())

        self.register_buffer('init', torch.tensor([0]))
        self.register_buffer('prev_Qweight', torch.zeros_like(self.weight))
        self.register_buffer('distance', torch.zeros_like(self.weight))
        self.register_buffer('unchange_step', torch.zeros_like(self.weight))
        self.register_buffer('saved_weight', torch.zeros_like(self.weight))
        self.fixed_para_num = 0

        self.batch_num = 0

        self.step = -1
        self.warmup_epoch = args.warmup_epoch
        self.fixed_rate = args.fixed_rate
        self.distance_ema = args.distance_ema
        self.fixed_mode = args.fixed_mode
        self.total_epochs = args.epochs

        print(self.warmup_epoch, self.fixed_rate, self.distance_ema, self.fixed_mode)

    def weight_quantization(self, weight):
        if not self.quan_weight or self.bit == 32:
            return weight

        weight = weight * (self.unchange_step == 0) + self.saved_weight * (self.unchange_step != 0)
        self.saved_weight = weight.detach().clone()

        self.weight_levels = 2 ** self.bit
        weight = (weight - self.lW) / (self.uW - self.lW)

        pre_quant_weight = weight.detach()
        weight = weight.clamp(min=0, max=1)  # [0, 1]
        weight = self.STE_discretizer(weight, self.weight_levels)
        aft_quant_weight = weight.detach()
        weight = (weight - 0.5) * 2  # [-1, 1]

        if self.training and self.init == 0:
            # compute distance
            transition = (aft_quant_weight != self.prev_Qweight).float()
            if self.step == 0:
                self.distance = torch.abs((pre_quant_weight - aft_quant_weight).detach())
            else:
                self.distance = (1 - self.distance_ema) * torch.abs((pre_quant_weight -
                                                                     aft_quant_weight).detach()) \
                                + self.distance_ema * self.distance
                self.distance = self.distance * (transition == 0) + transition * (1 / (2 ** self.bit - 1))
            self.prev_Qweight = aft_quant_weight.detach().clone()
            self.step += 1

            if self.step > self.batch_num * self.warmup_epoch:
                if self.fixed_mode == 'fixing':
                    self.unchange_step = \
                        self.unchange_step + self.step * (
                                    self.distance <= (1 / (2 ** self.bit - 1) * self.fixed_rate)) * (
                                self.unchange_step == 0)
                elif self.fixed_mode == 'linear-growth':
                    fixed_rate = 1 / (self.total_epochs * self.batch_num - self.warmup_epoch * self.batch_num) \
                                 * (self.step - self.warmup_epoch * self.batch_num)
                    self.unchange_step = \
                        self.unchange_step + self.step * (
                                self.distance <= (1 / (2 ** self.bit - 1) * fixed_rate)) * (
                                self.unchange_step == 0)
                elif self.fixed_mode == 'sine-growth':
                    fixed_rate = math.sin((self.step - self.warmup_epoch * self.batch_num) /
                                          (self.total_epochs * self.batch_num - self.warmup_epoch * self.batch_num) * math.pi / 2)
                    self.unchange_step = \
                        self.unchange_step + self.step * (
                                self.distance <= (1 / (2 ** self.bit - 1) * fixed_rate)) * (
                                self.unchange_step == 0)
                else:
                    pass
            self.fixed_para_num = torch.sum(self.unchange_step != 0)
        return weight

    def act_quantization(self, x):
        if not self.quan_act or self.bit == 32:
            return x
        self.act_levels = 2 ** self.bit
        x = (x - self.lA) / (self.uA - self.lA)
        x = x.clamp(min=0, max=1)  # [0, 1]
        x = self.STE_discretizer(x, self.act_levels)
        return x

    def initialize(self, x):

        FPweight, FPact = self.weight.detach(), x.detach()

        if self.quan_weight:
            self.uW.data.fill_(FPweight.std() * 3.0)
            self.lW.data.fill_(-FPweight.std() * 3.0)

        if self.quan_act:
            self.uA.data.fill_(FPact.std() / math.sqrt(1 - 2 / math.pi) * 3.0)
            self.lA.data.fill_(FPact.min())


    def forward(self, x):

        if self.quan_weight and self.init == 1:
            self.initialize(x)

        FPweight, FPact = self.weight, x

        Qweight = self.weight_quantization(FPweight)
        Qact = self.act_quantization(FPact)

        output = F.linear(Qact, weight=Qweight, bias=self.bias)

        return output