import argparse
import logging
import os
import random

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from custom_models import *
from custom_modules import *
from custom_optims_frozen import *
from utils import *

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="Supplementary material for NeurIPS reviewing process")
# data and model
parser.add_argument('--dataset', type=str, default='cifar100', choices=('cifar10','cifar100'), 
                                 help='dataset to use CIFAR10|CIFAR100')
parser.add_argument('--arch', type=str, default='resnet20_quant', choices=('resnet20_quant_allQ'),
                    help='model architecture')
parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--seed', type=int, default=None, help='seed for initialization')

# training settings
parser.add_argument('--batch_size', type=int, default=256, help='mini-batch size for training')
parser.add_argument('--epochs', type=int, default=400, help='number of epochs for training')
parser.add_argument('--optimizer', type=str, default='SGD', choices=('SGD'), help='type of an optimizer')
parser.add_argument('--scheduler', type=str, default='cosine', choices=('step','cosine'), help='')
parser.add_argument('--decay_schedule', type=str, default="100-200-300", help='')
parser.add_argument('--gamma', type=float, default=0.1, help='')
parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')

# custom settings
parser.add_argument('--bit_weight', type=int, default=1, help='')
parser.add_argument('--bit_act', type=int, default=1, help='')
parser.add_argument('--scale_lr', type=float, default=0.01, help='')
parser.add_argument('--baseline', type=str2bool, default=False, help='')

parser.add_argument('--fixed', type=str2bool, default=False, help='')
parser.add_argument('--fixed_rate', type=float, default=0.1, help='only activate when fixed_mode=fixing')
parser.add_argument('--fixed_mode', type=str, default='linear-growth', choices=('fixing',
                                                                              'linear-growth',
                                                                              'sine-growth'), help='')
parser.add_argument('--distance_ema', type=float, default=0.99, help='')
parser.add_argument('--warmup_epoch', type=int, default=80, help='')
parser.add_argument('--revive', type=str2bool, default=False, help='')





# logging and misc
parser.add_argument('--log_dir', type=str, default='test')
parser.add_argument('--load_pretrain', type=str2bool, default=True, help='load pretrained full-precision model')
parser.add_argument('--pretrain_path', type=str, default='../results/ResNet20_CIFAR100/fp/checkpoint/last_checkpoint.pth', 
                                       help='path for pretrained full-preicion model')
args = parser.parse_args()
arg_dict = vars(args)


### make log directory
if not os.path.exists(args.log_dir):
    os.makedirs(os.path.join(args.log_dir, 'checkpoint'))

logging.basicConfig(filename=os.path.join(args.log_dir, "log.txt"),
                    level=logging.INFO,
                    format='')
log_string = 'configs\n'
for k, v in arg_dict.items():
    log_string += "{}: {}\t".format(k,v)
    print("{}: {}".format(k,v), end='\t')
logging.info(log_string+'\n')
print('')

### GPU setting
# os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### set the seed number
if args.seed is not None:
    print("The seed number is set to", args.seed)
    logging.info("The seed number is set to {}".format(args.seed))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic=True

def _init_fn(worker_id):
    seed = args.seed + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return

### train/test datasets
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.dataset == 'cifar10':
    args.num_classes = 10
    train_dataset = dsets.CIFAR10(root='./results/data/CIFAR10/',
                                train=True, 
                                transform=transform_train,
                                download=True)
    test_dataset = dsets.CIFAR10(root='./results/data/CIFAR10/',
                            train=False, 
                            transform=transform_test)
elif args.dataset == 'cifar100':
    args.num_classes = 100
    train_dataset = dsets.CIFAR100(root='./results/data/CIFAR100/',
                                train=True, 
                                transform=transform_train,
                                download=True)
    test_dataset = dsets.CIFAR100(root='./results/data/CIFAR100/',
                            train=False, 
                            transform=transform_test)
else:
    raise NotImplementedError

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=args.num_workers,
                                           worker_init_fn=None if args.seed is None else _init_fn,
                                           drop_last=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100,
                                          shuffle=False,
                                          num_workers=args.num_workers)
args.batch_num = len(train_loader)


### initialize model
model_class = globals().get(args.arch)
model = model_class(args)
model.to(device)

num_total_params = sum(p.numel() for p in model.parameters())
print("The number of parameters : ", num_total_params)
logging.info("The number of parameters : {}".format(num_total_params))

if args.load_pretrain:

    trained_model = torch.load(args.pretrain_path)
    current_dict = model.state_dict()
    print("Pretrained full precision weights are initialized")
    logging.info("\nFollowing modules are initialized from pretrained model")
    log_string = ''
    c = 0
    for key in trained_model['model'].keys():
        if key in current_dict.keys():
            c += 1
            log_string += '{}\t'.format(key)
            current_dict[key].copy_(trained_model['model'][key])

    assert c == len(trained_model['model'])
    logging.info(log_string+'\n')
    model.load_state_dict(current_dict)



### initialize optimizer, scheduler, loss function
if args.bit_weight == 32 and args.bit_act == 32:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
else:
    quant_weights, fp_weights, scale_params, other_params = get_trainable_params(model)
    if args.optimizer == 'SGD':
        optimizer = quant_SGD_frozen([{'params': quant_weights, 'lr': 1.0,  # lr here represents a scheduling factor
                                    'is_quantized': True, 'initial_learning_rate': args.lr, 'fixed': args.fixed},
                                   {'params': fp_weights},
                                   {'params': scale_params, 'lr': args.scale_lr, 'momentum': 0, 'weight_decay': 0},
                                   {'params': other_params, 'momentum': 0, 'weight_decay': 0}],
                                  lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


print(args.scheduler, args.scheduler == 'step')
if args.scheduler == 'step':
    if args.decay_schedule is not None:
        milestones = list(map(lambda x: int(x), args.decay_schedule.split('-')))
    else:
        milestones = [args.epochs+1]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma)
elif args.scheduler == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

criterion = nn.CrossEntropyLoss()


### train
total_iter = 0
best_acc = 0

epoch_fixed_ratio = []
iter_fixed_ratio_list = []


modules = []
for m in model.modules():
    if isinstance(m, QConv_frozen_first) or isinstance(m, QConv_frozen) or isinstance(m, QLinear):
        modules.append(m)

total_epoch_cost = 0
for ep in range(args.epochs):
    model.train()


    epoch_cost = 0
    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
            
        pred = model(images)
        loss = criterion(pred, labels)

        num_total_fixed_params = 0
        for m in modules:
            num_total_fixed_params += m.fixed_para_num
        iter_fixed_ratio = num_total_fixed_params / num_total_params
        cost_ratio = 1 - iter_fixed_ratio
        iter_fixed_ratio_list.append(iter_fixed_ratio)
        epoch_cost += cost_ratio
        
        loss.backward()

        optimizer.step()
        total_iter += 1

    scheduler.step()


    total_epoch_cost += epoch_cost

    num_total_convlinear_params = 0
    num_total_fixed_params = 0
    for m in model.modules():
        if isinstance(m, QConv_frozen_first) or isinstance(m, QConv_frozen) or isinstance(m, QLinear):
            num_total_convlinear_params += m.weight.numel()
            num_total_fixed_params += torch.sum(m.weight.unchange_step != 0)
    epoch_fixed_ratio.append(num_total_fixed_params / num_total_convlinear_params)

    with torch.no_grad():
        model.eval()
        correct_classified = 0
        total = 0
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            _, predicted = torch.max(pred.data, 1)
            total += pred.size(0)
            correct_classified += (predicted == labels).sum().item()
        test_acc = correct_classified/total*100

        if ep % 5 == 0:
            print('num_total_convlinear_params', num_total_convlinear_params)
            print('num_total_fixed_params', num_total_fixed_params)
            print('num_total_fixed_params/num_total_convlinear_params',
                  num_total_fixed_params / num_total_convlinear_params)

            print("Current epoch: {:03d}".format(ep), "\t Test accuracy:", test_acc, "%")
            print('epoch_cost', epoch_cost, 'total_epoch_cost', total_epoch_cost)

        logging.info("Current epoch: {:03d}".format(ep) + "\t Test accuracy:" + str(test_acc) + "%")
        logging.info('epoch_cost' + str(epoch_cost.cpu().numpy()) + ', total_epoch_cost' + str(total_epoch_cost.cpu().numpy()))

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': ep,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'criterion': criterion.state_dict(),
                'epoch_fixed_ratio': epoch_fixed_ratio,
                'iter_fixed_ratio_list': iter_fixed_ratio_list
            }, os.path.join(args.log_dir, 'checkpoint/best_checkpoint.pth'))



total_epoch_cost += epoch_cost
print('total_epoch_cost: ', total_epoch_cost)
logging.info('total_epoch_cost: ' + str(total_epoch_cost))
print('best_acc', best_acc)
logging.info('best_acc: ' + str(best_acc))
print('test_acc', test_acc)
logging.info('test_acc: ' + str(test_acc))
