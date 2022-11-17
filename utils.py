from custom_modules import *
import torch
import torch.nn as nn
import numpy as np

__all__ = ['get_trainable_params']

def get_trainable_params(model):
    quant_weights = []
    fp_weights = []
    scale_params = []
    other_params = []
    for m in model.modules():
        if isinstance(m, QConv_frozen) or isinstance(m, QLinear) or isinstance(m, QConv_frozen_first):
            quant_weights.append(m.weight)
            if m.bias is not None:
                fp_weights.append(m.bias)
            if isinstance(m.sW, nn.Parameter):
                scale_params.append(m.sW)
            scale_params.append(m.sA)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            fp_weights.append(m.weight)
            if m.bias is not None:
                fp_weights.append(m.bias)
        elif isinstance(m, nn.SyncBatchNorm) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            if m.affine:
                other_params.append(m.weight)
                other_params.append(m.bias)

    trainable_params = list(model.parameters())
    print("# total params:", sum(p.numel() for p in trainable_params))
    print("# quantized weights:", sum(p.numel() for p in quant_weights))
    print("# fp weights:", sum(p.numel() for p in fp_weights))
    print("# scale params:", sum(p.numel() for p in scale_params))
    print("# other params:", sum(p.numel() for p in other_params))
    num_total_params = sum(p.numel() for p in quant_weights) + sum(p.numel() for p in fp_weights) + \
                       sum(p.numel() for p in scale_params) + sum(p.numel() for p in other_params)

    if sum(p.numel() for p in trainable_params) != num_total_params:
        raise Exception('Mismatched number of trainable parmas')

    return quant_weights, fp_weights, scale_params, other_params