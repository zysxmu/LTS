import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['QConv_frozen', 'QConv_frozen_first', 'QLinear']

class STE_round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_in):
        x_out = torch.round(x_in)
        return x_out
    @staticmethod
    def backward(ctx, g):
        return g


# ref. https://github.com/hustzxd/LSQuantization/blob/master/lsq.py
class QConv_frozen(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, args, stride=1, padding=0, dilation=1, groups=1,
                 bias=True):
        super(QConv_frozen, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.STE_round = STE_round.apply
        self.bit_weight = args.bit_weight
        self.bit_act = args.bit_act

        self.fixed_rate = args.fixed_rate
        self.fixed = args.fixed
        self.fixed_mode = args.fixed_mode
        self.warmup_epoch = args.warmup_epoch
        self.batch_num = args.batch_num
        self.distance_ema = args.distance_ema
        self.revive = args.revive
        self.total_epochs = args.epochs
        self.fixed_para_num = 0

        self.Qn_w = -2 ** (self.bit_weight - 1)  # does not used for 1-bit
        self.Qp_w = 2 ** (self.bit_weight - 1) - 1  # does not used for 1-bit
        self.Qn_a = 0
        self.Qp_a = 2 ** (self.bit_act) - 1


        self.register_buffer('sW', torch.tensor(1).float())  # fix sW for our method
        self.sA = nn.Parameter(data=torch.tensor(1).float())


        self.register_buffer('init', torch.tensor([1]))
        self.register_buffer('prev_Qweight', torch.zeros_like(self.weight))
        self.register_buffer('unchange_step', torch.zeros_like(self.weight))
        self.register_buffer('norm_weight', torch.zeros_like(self.weight))
        self.register_buffer('distance', torch.zeros_like(self.weight))
        self.step = 0

    def weight_quantization(self, weight):
        if self.bit_weight == 32:
            return weight
        elif self.bit_weight == 1:
            weight = weight / (torch.abs(self.sW) + 1e-6)
            weight = weight.clamp(-1, 1)  # [-1, 1]
            weight = (weight + 1) / 2  # [0, 1]
            weight_q = self.STE_round(weight)  # {0, 1}
            weight_q = weight_q * 2 - 1  # {-1, 1}
            return weight_q
        else:
            weight = weight / (torch.abs(self.sW) + 1e-6)  # normalized such that 99% of weights lie in [-1, 1]
            self.norm_weight = weight
            weight = weight * 2 ** (self.bit_weight - 1)
            weight = weight.clamp(self.Qn_w, self.Qp_w)
            weight_q = self.STE_round(weight)  # {-2^(b-1), ..., 2^(b-1)-1}
            weight_q = weight_q / 2 ** (self.bit_weight - 1)  # fixed point representation
            return weight_q

    def act_quantization(self, x):
        if self.bit_act == 32:
            return x
        elif self.bit_weight == 1:
            x = x / (torch.abs(self.sA) + 1e-6)
            x = x.clamp(0, 1)  # [0, 1]
            x = self.STE_round(x)  # {0, 1}
            return x
        else:
            x = x / (torch.abs(self.sA) + 1e-6)  # normalized such that 99% of activations lie in [0, 1]
            x = x * 2 ** self.bit_act
            x = x.clamp(self.Qn_a, self.Qp_a)  # [0, 2^b-1]
            x = self.STE_round(x)  # {0, ..., 2^b-1}
            x = x / 2 ** self.bit_act  # fixed point representation
            return x

    def initialize(self, x):
        self.sW.data.fill_(self.weight.std() * 3.0)
        self.sA.data.fill_(x.std() / math.sqrt(1 - 2 / math.pi) * 3.0)
        self.init.fill_(0)

    def forward(self, x):

        if self.training and self.init:
            self.initialize(x)

        Qweight = self.weight_quantization(self.weight)
        if self.training and self.bit_weight != 32:

            self.step += 1

            transition = (Qweight != self.prev_Qweight).float()
            setattr(self.weight, "transition", transition)


            if self.step == 0:
                self.distance = torch.abs((self.norm_weight - Qweight).detach())
            else:
                self.distance = (1 - self.distance_ema) * torch.abs((self.norm_weight - Qweight).detach()) \
                                + self.distance_ema * self.distance
                self.distance = self.distance * (transition == 0) + transition * (2 / (2**self.bit_weight))

            if self.step > self.batch_num * self.warmup_epoch and self.fixed:
                if self.fixed_mode == 'fixing':
                    self.unchange_step = \
                        self.unchange_step + self.step * (self.distance <= (2 / (2**self.bit_weight) * self.fixed_rate)) * (
                                    self.unchange_step == 0)
                elif self.fixed_mode == 'linear-growth':
                    fixed_rate = 1 / (self.total_epochs*self.batch_num - self.warmup_epoch*self.batch_num) \
                                 * (self.step - self.warmup_epoch*self.batch_num)

                    self.unchange_step = \
                        self.unchange_step + self.step * (
                                    self.distance <= (2 / (2 ** self.bit_weight) * fixed_rate)) * (
                                self.unchange_step == 0)
                elif self.fixed_mode == 'sine-growth':
                    fixed_rate = math.sin((self.step - self.warmup_epoch*self.batch_num) /
                                          (self.total_epochs*self.batch_num - self.warmup_epoch*self.batch_num) * math.pi / 2)
                    self.unchange_step = \
                        self.unchange_step + self.step * (
                                self.distance <= (2 / (2 ** self.bit_weight) * fixed_rate)) * (
                                self.unchange_step == 0)
                else:
                    pass

            # setattr(self.weight, "test", 1)
            setattr(self.weight, "Qweight", Qweight)
            # setattr(self.weight, "unchange", self.unchange)
            setattr(self.weight, "distance", self.distance)
            setattr(self.weight, "unchange_step", self.unchange_step)

            self.fixed_para_num = torch.sum(self.unchange_step != 0)
            self.prev_Qweight = Qweight.detach().clone()

        Qact = self.act_quantization(x)
        output = F.conv2d(Qact, Qweight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return output



class QConv_frozen_first(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, args, stride=1, padding=0, dilation=1, groups=1,
                 bias=True):
        super(QConv_frozen_first, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.STE_round = STE_round.apply
        self.bit_weight = 8

        self.fixed_rate = args.fixed_rate
        self.fixed = args.fixed
        self.fixed_mode = args.fixed_mode
        self.warmup_epoch = args.warmup_epoch
        self.batch_num = args.batch_num
        self.distance_ema = args.distance_ema
        self.revive = args.revive
        self.total_epochs = args.epochs
        self.fixed_para_num = 0


        self.Qn_w = -2 ** (self.bit_weight - 1)  # does not used for 1-bit
        self.Qp_w = 2 ** (self.bit_weight - 1) - 1  # does not used for 1-bit


        self.register_buffer('sW', torch.tensor(1).float())  # fix sW for our method
        self.sA = nn.Parameter(data=torch.tensor(1).float())


        self.register_buffer('init', torch.tensor([1]))
        self.register_buffer('prev_Qweight', torch.zeros_like(self.weight))
        self.register_buffer('unchange_step', torch.zeros_like(self.weight))
        self.register_buffer('norm_weight', torch.zeros_like(self.weight))
        self.register_buffer('distance', torch.zeros_like(self.weight))
        self.step = 0


    def weight_quantization(self, weight):

        if self.bit_weight == 32:
            return weight
        elif self.bit_weight == 1:
            weight = weight / (torch.abs(self.sW) + 1e-6)
            weight = weight.clamp(-1, 1)  # [-1, 1]
            weight = (weight + 1) / 2  # [0, 1]
            weight_q = self.STE_round(weight)  # {0, 1}
            weight_q = weight_q * 2 - 1  # {-1, 1}
            return weight_q
        else:
            weight = weight / (torch.abs(self.sW) + 1e-6)  # normalized such that 99% of weights lie in [-1, 1]
            self.norm_weight = weight
            weight = weight * 2 ** (self.bit_weight - 1)
            weight = weight.clamp(self.Qn_w, self.Qp_w)
            weight_q = self.STE_round(weight)  # {-2^(b-1), ..., 2^(b-1)-1}
            weight_q = weight_q / 2 ** (self.bit_weight - 1)  # fixed point representation
            return weight_q

    def initialize(self, x):
        self.sW.data.fill_(self.weight.std() * 3.0)
        self.sA.data.fill_(x.std() / math.sqrt(1 - 2 / math.pi) * 3.0)
        self.init.fill_(0)

    def forward(self, x):



        if self.training and self.init:
            self.initialize(x)

        Qweight = self.weight_quantization(self.weight)
        if self.training and self.bit_weight != 32:

            self.step += 1

            transition = (Qweight != self.prev_Qweight).float()
            setattr(self.weight, "transition", transition)


            if self.step == 0:
                self.distance = torch.abs((self.norm_weight - Qweight).detach())
            else:
                self.distance = (1 - self.distance_ema) * torch.abs((self.norm_weight - Qweight).detach()) \
                                + self.distance_ema * self.distance
                self.distance = self.distance * (transition == 0) + transition * (2 / (2 ** self.bit_weight))

            if self.step > self.batch_num * self.warmup_epoch and self.fixed:
                if self.fixed_mode == 'fixing':
                    self.unchange_step = \
                        self.unchange_step + self.step * (
                                    self.distance <= (2 / (2 ** self.bit_weight) * self.fixed_rate)) * (
                                self.unchange_step == 0)
                elif self.fixed_mode == 'linear-growth':
                    fixed_rate = 1 / (self.total_epochs * self.batch_num - self.warmup_epoch * self.batch_num) \
                                 * (self.step - self.warmup_epoch * self.batch_num)
                    self.unchange_step = \
                        self.unchange_step + self.step * (
                                self.distance <= (2 / (2 ** self.bit_weight) * fixed_rate)) * (
                                self.unchange_step == 0)
                elif self.fixed_mode == 'sine-growth':
                    fixed_rate = math.sin((self.step - self.warmup_epoch * self.batch_num) /
                                          (
                                                      self.total_epochs * self.batch_num - self.warmup_epoch * self.batch_num) * math.pi / 2)
                    self.unchange_step = \
                        self.unchange_step + self.step * (
                                self.distance <= (2 / (2 ** self.bit_weight) * fixed_rate)) * (
                                self.unchange_step == 0)
                else:
                    pass


            # setattr(self.weight, "test", 1)
            setattr(self.weight, "Qweight", Qweight)
            setattr(self.weight, "distance", self.distance)
            setattr(self.weight, "unchange_step", self.unchange_step)

            self.fixed_para_num = torch.sum(self.unchange_step != 0)
            self.prev_Qweight = Qweight.detach().clone()

        output = F.conv2d(x, Qweight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return output

class QLinear(nn.Linear):
    def __init__(self, in_features, out_features, args, bias=True):
        super(QLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.STE_round = STE_round.apply
        self.bit_weight = 8
        self.bit_act = 8

        self.fixed_rate = args.fixed_rate
        self.fixed = args.fixed
        self.warmup_epoch = args.warmup_epoch
        self.batch_num = args.batch_num
        self.distance_ema = args.distance_ema
        self.revive = args.revive
        self.fixed_para_num = 0

        self.Qn_w = -2 ** (self.bit_weight - 1)  # does not used for 1-bit
        self.Qp_w = 2 ** (self.bit_weight - 1) - 1  # does not used for 1-bit
        self.Qn_a = 0
        self.Qp_a = 2 ** (self.bit_act) - 1

        self.sW = nn.Parameter(data=torch.tensor(1).float())
        self.sA = nn.Parameter(data=torch.tensor(1).float())


        self.register_buffer('init', torch.tensor([1]))
        self.register_buffer('prev_Qweight', torch.zeros_like(self.weight))
        self.register_buffer('unchange_step', torch.zeros_like(self.weight))
        self.register_buffer('norm_weight', torch.zeros_like(self.weight))
        self.register_buffer('distance', torch.zeros_like(self.weight))
        self.register_buffer('prev_sW', torch.zeros_like(self.sW))

        self.step = 0


    def weight_quantization(self, weight):
        if self.bit_weight == 32:
            return weight
        elif self.bit_weight == 1:
            weight = weight / (torch.abs(self.sW) + 1e-6)
            weight = weight.clamp(-1, 1)  # [-1, 1]
            weight = (weight + 1) / 2  # [0, 1]
            weight_q = self.STE_round(weight)  # {0, 1}
            weight_q = weight_q * 2 - 1  # {-1, 1}
            return weight_q
        else:
            self.norm_weight = weight
            weight = weight / (torch.abs(self.sW) + 1e-6)  # normalized such that 99% of weights lie in [-1, 1]
            weight = weight * 2 ** (self.bit_weight - 1)
            weight = weight.clamp(self.Qn_w, self.Qp_w)
            weight_q = self.STE_round(weight)  # {-2^(b-1), ..., 2^(b-1)-1}
            weight_q = weight_q / 2 ** (self.bit_weight - 1)  # fixed point representation
            record_weight_q = weight_q.detach().clone()
            weight_q = weight_q * self.sW
            return weight_q, record_weight_q

    def act_quantization(self, x):
        if self.bit_act == 32:
            return x
        elif self.bit_weight == 1:
            x = x / (torch.abs(self.sA) + 1e-6)
            x = x.clamp(0, 1)  # [0, 1]
            x = self.STE_round(x)  # {0, 1}
            return x
        else:
            x = x / (torch.abs(self.sA) + 1e-6)  # normalized such that 99% of activations lie in [0, 1]
            x = x * 2 ** self.bit_act
            x = x.clamp(self.Qn_a, self.Qp_a)  # [0, 2^b-1]
            x = self.STE_round(x)  # {0, ..., 2^b-1}
            x = x / 2 ** self.bit_act  # fixed point representation
            return x

    def initialize(self, x):
        self.sW.data.fill_(self.weight.std() * 3.0)
        self.sA.data.fill_(x.std() / math.sqrt(1 - 2 / math.pi) * 3.0)
        self.init.fill_(0)

    def forward(self, x):



        if self.training and self.init:
            self.initialize(x)

        Qweight, record_weight_q = self.weight_quantization(self.weight)
        if self.training and self.bit_weight != 32:

            self.step += 1

            transition = (record_weight_q != self.prev_Qweight).float()
            # transition = (Qweight != self.prev_Qweight).float()
            setattr(self.weight, "transition", transition)

            # if self.step == 0:
            #     self.distance = torch.abs((self.norm_weight - Qweight).detach())
            # else:
            #     self.distance = (1 - self.distance_ema) * torch.abs((self.norm_weight - Qweight).detach()) \
            #                     + self.distance_ema * self.distance
            #     self.distance = self.distance * (transition == 0) + transition * (2 / (2 ** self.bit_weight))

            # if self.step > self.batch_num * self.warmup_epoch and self.fixed:
            #     self.unchange_step = \
            #         self.unchange_step + self.step * (self.distance <= (2 / (2**self.bit_weight) * self.fixed_rate)) * (
            #                 self.unchange_step == 0)
            # self.unchange_step = self.unchange_step + self.step * (self.unchange < 0.001) * (self.unchange_step == 0)


            setattr(self.weight, "Qweight", Qweight)
            setattr(self.weight, "distance", self.distance)
            setattr(self.weight, "unchange_step", self.unchange_step)



            self.prev_Qweight = record_weight_q.detach().clone()
            # self.prev_Qweight = Qweight.detach().clone()
            self.prev_sW = self.sW.detach().clone()

        Qact = self.act_quantization(x)
        output = F.linear(Qact, weight=Qweight, bias=self.bias)

        return output



def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad

# class Conv2dLSQ(_Conv2dQ):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, nbits_w=8, **kwargs):
#         super(Conv2dLSQ, self).__init__(
#             in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
#             stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
#             nbits=nbits_w)
#
#     def forward(self, x):
#         Qn = -2 ** (self.nbits - 1)
#         Qp = 2 ** (self.nbits - 1) - 1
#         if self.training and self.init_state == 0:
#             # self.alpha.data.copy_(self.weight.abs().max() / 2 ** (self.nbits - 1))
#             self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
#             # self.alpha.data.copy_(self.weight.abs().max() * 2)
#             self.init_state.fill_(1)
#
#         g = 1.0 / math.sqrt(self.weight.numel() * Qp)
#
#         # Method1: 31GB GPU memory (AlexNet w4a4 bs 2048) 17min/epoch
#         alpha = grad_scale(self.alpha, g)
#         w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha
#         # w = w.clamp(Qn, Qp)
#         # q_w = round_pass(w)
#         # w_q = q_w * alpha
#
#         return F.conv2d(x, w_q, self.bias, self.stride,
#                         self.padding, self.dilation, self.groups)


class QConv_frozen_LSQ(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, args, stride=1, padding=0, dilation=1, groups=1,
                 bias=True):
        super(QConv_frozen_LSQ, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.STE_round = STE_round.apply
        self.bit_weight = args.bit_weight
        self.bit_act = args.bit_act

        self.fixed_rate = args.fixed_rate
        self.fixed = args.fixed
        self.fixed_mode = args.fixed_mode
        self.warmup_epoch = args.warmup_epoch
        self.batch_num = args.batch_num
        self.distance_ema = args.distance_ema
        self.revive = args.revive
        self.total_epochs = args.epochs
        self.fixed_para_num = 0

        self.Qn_w = -2 ** (self.bit_weight - 1)  # does not used for 1-bit
        self.Qp_w = 2 ** (self.bit_weight - 1) - 1  # does not used for 1-bit
        self.Qn_a = 0
        self.Qp_a = 2 ** (self.bit_act) - 1

        self.register_buffer('w_init_state', torch.zeros(1))


        self.sA = nn.Parameter(data=torch.tensor(1).float())
        self.register_buffer('init', torch.tensor([1]))
        self.register_buffer('prev_Qweight', torch.zeros_like(self.weight))
        self.register_buffer('unchange_step', torch.zeros_like(self.weight))
        self.register_buffer('norm_weight', torch.zeros_like(self.weight))
        self.register_buffer('distance', torch.zeros_like(self.weight))
        self.step = 0

    def weight_quantization(self, weight):

        if self.training and self.w_init_state == 0:
            self.w_alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(self.Qp_w))
            self.init_state.fill_(1)

        if self.bit_weight == 32:
            return weight
        elif self.bit_weight == 1:
            weight = weight / (torch.abs(self.sW) + 1e-6)
            weight = weight.clamp(-1, 1)  # [-1, 1]
            weight = (weight + 1) / 2  # [0, 1]
            weight_q = self.STE_round(weight)  # {0, 1}
            weight_q = weight_q * 2 - 1  # {-1, 1}
            return weight_q
        else:
            weight = weight / (torch.abs(self.sW) + 1e-6)  # normalized such that 99% of weights lie in [-1, 1]
            self.norm_weight = weight
            weight = weight * 2 ** (self.bit_weight - 1)
            weight = weight.clamp(self.Qn_w, self.Qp_w)
            weight_q = self.STE_round(weight)  # {-2^(b-1), ..., 2^(b-1)-1}
            weight_q = weight_q / 2 ** (self.bit_weight - 1)  # fixed point representation
            return weight_q

    def act_quantization(self, x):
        if self.bit_act == 32:
            return x
        elif self.bit_weight == 1:
            x = x / (torch.abs(self.sA) + 1e-6)
            x = x.clamp(0, 1)  # [0, 1]
            x = self.STE_round(x)  # {0, 1}
            return x
        else:
            x = x / (torch.abs(self.sA) + 1e-6)  # normalized such that 99% of activations lie in [0, 1]
            x = x * 2 ** self.bit_act
            x = x.clamp(self.Qn_a, self.Qp_a)  # [0, 2^b-1]
            x = self.STE_round(x)  # {0, ..., 2^b-1}
            x = x / 2 ** self.bit_act  # fixed point representation
            return x

    def initialize(self, x):
        self.sW.data.fill_(self.weight.std() * 3.0)
        self.sA.data.fill_(x.std() / math.sqrt(1 - 2 / math.pi) * 3.0)
        self.init.fill_(0)

    def forward(self, x):

        if self.training and self.init:
            self.initialize(x)

        Qweight = self.weight_quantization(self.weight)
        if self.training and self.bit_weight != 32:

            self.step += 1

            transition = (Qweight != self.prev_Qweight).float()
            setattr(self.weight, "transition", transition)


            if self.step == 0:
                self.distance = torch.abs((self.norm_weight - Qweight).detach())
            else:
                self.distance = (1 - self.distance_ema) * torch.abs((self.norm_weight - Qweight).detach()) \
                                + self.distance_ema * self.distance
                self.distance = self.distance * (transition == 0) + transition * (2 / (2**self.bit_weight))

            if self.step > self.batch_num * self.warmup_epoch and self.fixed:
                if self.fixed_mode == 'fixing':
                    self.unchange_step = \
                        self.unchange_step + self.step * (self.distance <= (2 / (2**self.bit_weight) * self.fixed_rate)) * (
                                    self.unchange_step == 0)
                elif self.fixed_mode == 'linear-growth':
                    fixed_rate = 1 / (self.total_epochs*self.batch_num - self.warmup_epoch*self.batch_num) \
                                 * (self.step - self.warmup_epoch*self.batch_num)

                    self.unchange_step = \
                        self.unchange_step + self.step * (
                                    self.distance <= (2 / (2 ** self.bit_weight) * fixed_rate)) * (
                                self.unchange_step == 0)
                elif self.fixed_mode == 'sine-growth':
                    fixed_rate = math.sin((self.step - self.warmup_epoch*self.batch_num) /
                                          (self.total_epochs*self.batch_num - self.warmup_epoch*self.batch_num) * math.pi / 2)
                    self.unchange_step = \
                        self.unchange_step + self.step * (
                                self.distance <= (2 / (2 ** self.bit_weight) * fixed_rate)) * (
                                self.unchange_step == 0)
                else:
                    pass

            setattr(self.weight, "Qweight", Qweight)
            setattr(self.weight, "distance", self.distance)
            setattr(self.weight, "unchange_step", self.unchange_step)

            self.fixed_para_num = torch.sum(self.unchange_step != 0)
            self.prev_Qweight = Qweight.detach().clone()

        Qact = self.act_quantization(x)
        output = F.conv2d(Qact, Qweight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return output