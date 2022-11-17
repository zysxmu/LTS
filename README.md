# SpQAT
Code for the reviewing process of CVPR 2023 submission 12240.

It contains the related code of experiments on  CIFAR100/10 and ImageNet datasets. 

The training code of the detection task will be released publicly after publication.


## Environments
* Python >= 3.8
* PyTorch >= 1.7.1
* pytorchcv == 0.0.51
* TensorBoard

## run

### CIFAR

First, run ``run_cifar_fp.sh`` to get the full-precision ResNet-20 on CIFAR.

Second, run ``run_cifar.sh``.

The ``fixed_rate`` only works when the ``fixed_mode`` is ``fixing``.

### ImageNet

The pre-trained model will be download from torchvision or pytorchcv automatically.

First, run ``run_imagenet.sh``.
