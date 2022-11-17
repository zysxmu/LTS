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
import time

from custom_models_ImageNet import *
from custom_models_ImageNet_mv2 import *
from custom_models_ImageNet_mv1 import *
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



### set the seed number
def _init_fn(args):
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return


def get_dataloader(args):

    # create data loader
    traindir = os.path.join(args.datapath, "train")
    testdir = os.path.join(args.datapath, "val")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    dataset_train = dsets.ImageFolder(traindir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
    dataset_val = dsets.ImageFolder(testdir, test_transform)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(), shuffle=True
    )
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    train_loader = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
    )
    print(args.local_rank, len(train_loader))
    logging.info("args.local_rank:" + str(args.local_rank) + " len(train_loader): " + str(len(train_loader)))

    test_loader = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, test_loader, sampler_train


def get_model(args):
    ### initialize model
    model_class = globals().get(args.arch)
    model = model_class(args)
    model.cuda()
    cur_device = torch.cuda.current_device()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cur_device], output_device=cur_device,
                                                      find_unused_parameters=True)
    return model


def load_pretrain(args, model, device):
    import torchvision.models as models

    if 'resnet18' in args.arch:
        trained_model = models.resnet18(pretrained=True)
    elif 'resnet50' in args.arch:
        trained_model = models.resnet50(pretrained=True)
    elif 'mobilenetv2' in args.arch:
        trained_model = models.mobilenet_v2(pretrained=True)
    elif 'mobilenetv1' in args.arch:
        from pytorchcv.model_provider import get_model as ptcv_get_model
        trained_model = ptcv_get_model('mobilenet_w1', pretrained=True)


    # trained_model = torch.load(args.pretrain_path, map_location='cpu')
    current_dict = model.state_dict()
    if args.local_rank == 0:
        print("Pretrained full precision weights are initialized")
        logging.info("\nFollowing modules are initialized from pretrained model")
    log_string = ''

    c = 0
    for key in trained_model.state_dict().keys():
        key_change = 'module.' + key
        if 'fc' in key_change:
            key_change = key_change.replace('fc', 'linear')
        if key_change in current_dict.keys():
            c += 1
            log_string += '{}\t'.format(key)
            current_dict[key_change].copy_(trained_model.state_dict()[key])
    assert c == len(trained_model.state_dict().keys())
    if args.local_rank == 0:
        print(len(current_dict), len(trained_model.state_dict()))
    logging.info(log_string + '\n')
    model.load_state_dict(current_dict)
    model.to(device)
    return model

def get_optimzier(args, model):

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
    if args.scheduler == 'step':
        if args.decay_schedule is not None:
            milestones = list(map(lambda x: int(x), args.decay_schedule.split('-')))
        else:
            milestones = [args.epochs + 1]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

    return optimizer, scheduler

def test_fp_model(args, test_loader, device):
    import torchvision.models as models
    if 'resnet18' in args.arch:
        ### initialize model
        model_class = globals().get('resnet18_fp')
        fp_model = model_class(args)
        fp_model.to(device)
        trained_model = models.resnet18(pretrained=True)
    elif 'resnet50' in args.arch:
        ### initialize model
        model_class = globals().get('resnet50_fp')
        fp_model = model_class(args)
        fp_model.to(device)
        trained_model = models.resnet50(pretrained=True)
    elif 'mobilenetv2' in args.arch:
        ### initialize model
        trained_model = models.mobilenet_v2(pretrained=True)
        model_class = globals().get('mobilenetv2_w1_fp')
        fp_model = model_class(args)
        fp_model.to(device)
    elif 'mobilenetv1' in args.arch:
        ### initialize model
        from pytorchcv.model_provider import get_model as ptcv_get_model
        trained_model = ptcv_get_model('mobilenet_w1', pretrained=True)
        model_class = globals().get('mobilenetv1_w1_fp')
        fp_model = model_class(args)
        fp_model.to(device)
    # trained_model = torch.load(args.pretrain_path, map_location='cpu')
    current_dict = fp_model.state_dict()
    print("Pretrained full precision weights are initialized")
    logging.info("\nFollowing modules are initialized from pretrained model")
    log_string = ''

    c = 0
    for key in trained_model.state_dict().keys():
        key_change = key
        if 'fc' in key_change:
            key_change = key_change.replace('fc', 'linear')
        if key_change in current_dict.keys():
            c += 1
            log_string += '{}\t'.format(key)
            current_dict[key_change].copy_(trained_model.state_dict()[key])
    assert c == len(trained_model.state_dict().keys())
    logging.info(log_string + '\n')
    fp_model.load_state_dict(current_dict)
    fp_model.to(device)
    test(fp_model, args, None, test_loader, None, None, -1, 0, None, test_only=True)
    return

def train(model, optimizer, args, train_loader, criterion, ep, num_total_params, writer=None):
    ### train
    total_iter = 0


    model.train()

    modules = []
    for m in model.modules():
        if isinstance(m, QConv_frozen_first) or isinstance(m, QConv_frozen) or isinstance(m, QLinear):
            modules.append(m)

    epoch_cost = 0
    for i, (images, labels) in enumerate(train_loader):

        images = images.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()

        pred = model(images)
        loss = criterion(pred, labels)

        num_total_fixed_params = 0
        for m in modules:
            num_total_fixed_params += m.fixed_para_num
        iter_fixed_ratio = num_total_fixed_params / num_total_params
        cost_ratio = 1 - iter_fixed_ratio
        epoch_cost += cost_ratio

        loss.backward()

        optimizer.step()


        if i % 200 == 0 and args.local_rank == 0:
            print(i,'/', len(train_loader))
        total_iter += 1

    return epoch_cost



def test(model, args, optimizer, test_loader, scheduler, criterion, ep, best_acc, epoch_fixed_ratio,
         writer=None, test_only=False):
    with torch.no_grad():
        model.eval()
        correct_classified = 0
        total = 0

        for i, (images, labels) in enumerate(test_loader):
            images = images.cuda()
            labels = labels.cuda()
            pred = model(images)
            _, predicted = torch.max(pred.data, 1)
            # import IPython
            # IPython.embed()
            total += pred.size(0)
            correct_classified += (predicted == labels).sum().item()
        test_acc = correct_classified / total * 100

        if args.local_rank == 0:
            print("Current local_rank: {:03d} epoch: {:03d}".format(args.local_rank, ep),
                  "\t Test accuracy:", test_acc, "%")
            logging.info("Current local_rank: {:03d} epoch: {:03d}".format(args.local_rank, ep)
                         + " Test accuracy:" + str(test_acc) + "%")

        if test_only:
            print(test_acc)
            return


        if args.local_rank == 0 and args.bit_weight < 32:
            num_total_convlinear_params = 0
            num_total_fixed_params = 0
            for m in model.modules():
                if isinstance(m, QConv_frozen_first) or isinstance(m, QConv_frozen) or isinstance(m, QLinear):
                    num_total_convlinear_params += m.weight.numel()
                    num_total_fixed_params += torch.sum(m.weight.unchange_step != 0)
            epoch_fixed_ratio.append(num_total_fixed_params / num_total_convlinear_params)
            print('num_total_convlinear_params', num_total_convlinear_params,
                  'num_total_fixed_params', num_total_fixed_params,
                  'num_total_fixed_params/num_total_convlinear_params',
                  num_total_fixed_params / num_total_convlinear_params)
            logging.info("num_total_convlinear_params: " + str(num_total_convlinear_params))
            logging.info('num_total_fixed_params/num_total_convlinear_params: '
                         + str(num_total_fixed_params / num_total_convlinear_params))

        if args.local_rank == 0:
            torch.save({
                'epoch': ep,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'criterion': criterion.state_dict(),
                'epoch_fixed_ratio': epoch_fixed_ratio,
            }, os.path.join(args.log_dir, 'checkpoint/last_checkpoint.pth'))
            if ep % 10 == 0:
                torch.save({
                    'epoch': ep,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'criterion': criterion.state_dict(),
                    'epoch_fixed_ratio': epoch_fixed_ratio,
                }, os.path.join(args.log_dir, 'checkpoint/checkpoint'+str(ep)+'.pth'))
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save({
                    'epoch': ep,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'criterion': criterion.state_dict(),
                    'epoch_fixed_ratio': epoch_fixed_ratio,
                }, os.path.join(args.log_dir, 'checkpoint/best_checkpoint.pth'))

    return best_acc, epoch_fixed_ratio


def main():
    parser = argparse.ArgumentParser(description="Supplementary material for NeurIPS reviewing process")
    # # data and model
    # parser.add_argument('--dataset', type=str, default='cifar100', choices=('cifar10', 'cifar100'),
    #                     help='dataset to use CIFAR10|CIFAR100')
    parser.add_argument('--datapath', type=str, default='')
    parser.add_argument('--arch', type=str, default='resnet20_quant', choices=('resnet18_fp',
                                                                               'resnet50_fp',
                                                                               'resnet18_quant',
                                                                               'resnet50_quant',
                                                                               'mobilenetv2_w1_fp',
                                                                               'mobilenetv2_w1_quant',
                                                                               'mobilenetv1_w1_fp',
                                                                               'mobilenetv1_w1_quant'),
                        help='model architecture')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--seed', type=int, default=None, help='seed for initialization')

    # training settings
    parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size for training')
    parser.add_argument('--epochs', type=int, default=400, help='number of epochs for training')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=('SGD'), help='type of an optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=('step', 'cosine'), help='')
    parser.add_argument('--decay_schedule', type=str, default="20-40-60-80", help='')
    parser.add_argument('--gamma', type=float, default=0.2, help='')
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
    parser.add_argument('--fixed_mode', type=str, default='fixing', choices=('fixing',
                                                                                  'linear-growth',
                                                                                  'sine-growth'), help='')
    parser.add_argument('--distance_ema', type=float, default=0.99, help='')
    parser.add_argument('--warmup_epoch', type=int, default=1, help='')
    parser.add_argument('--revive', type=str2bool, default=False, help='')

    # logging and misc
    # parser.add_argument('--gpu_id', type=str, default='0', help='target GPU to use')
    parser.add_argument('--log_dir', type=str, default='test')
    parser.add_argument('--load_pretrain', type=str2bool, default=True, help='load pretrained full-precision model')
    parser.add_argument('--pretrain_path', type=str,
                        default='xxxx',
                        help='path for pretrained full-preicion model')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    args = parser.parse_args()
    arg_dict = vars(args)

    writer = None
    ### make log directory
    if args.local_rank == 0:
        if not os.path.exists(args.log_dir):
            os.makedirs(os.path.join(args.log_dir, 'checkpoint'))

        logging.basicConfig(filename=os.path.join(args.log_dir, "log.txt"),
                            level=logging.INFO,
                            format='')
        log_string = 'configs\n'
        for k, v in arg_dict.items():
            log_string += "{}: {}\t".format(k, v)
            print("{}: {}".format(k, v), end='\t')
        logging.info(log_string + '\n')
        print('')
        writer = SummaryWriter(args.log_dir)

    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    assert torch.distributed.get_rank() == args.local_rank

    train_loader, test_loader, sampler_train = get_dataloader(args)
    args.batch_num = len(train_loader)

    model = get_model(args)

    num_total_params = sum(p.numel() for p in model.parameters())
    print("The number of parameters : ", num_total_params)
    logging.info("The number of parameters : {}".format(num_total_params))

    # import IPython
    # IPython.embed()
    # load_pretrain
    if not args.baseline:
        model = load_pretrain(args, model, device)
        # test full-precision model at first
        # test_fp_model(args, test_loader, device)


    torch.backends.cudnn.benchmark = True

    optimizer, scheduler = get_optimzier(args, model)
    criterion = nn.CrossEntropyLoss()



    best_acc = 0
    epoch_fixed_ratio = []

    total_epoch_cost = 0


    for ep in range(args.epochs):
        since = time.time()
        sampler_train.set_epoch(ep)

        epoch_cost = train(model, optimizer, args, train_loader, criterion, ep, num_total_params, writer)
        total_epoch_cost += epoch_cost
        if args.local_rank == 0:
            print('epoch_cost: ', epoch_cost, 'total_epoch_cost: ', total_epoch_cost)
            logging.info('epoch_cost: ' + str(epoch_cost) + 'total_epoch_cost: ' + str(total_epoch_cost))

        best_acc, epoch_fixed_ratio = \
            test(model, args, optimizer, test_loader, scheduler, criterion, ep, best_acc, epoch_fixed_ratio, writer)
        scheduler.step()
        time_elapsed = time.time() - since
        if args.local_rank == 0:
            print('Training ep'+ str(ep) +'complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            logging.info('Training ep'+ str(ep) +'complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))


    if args.local_rank == 0:
        print('The best acc is: ', best_acc, 'The total_epoch_cost is: ', total_epoch_cost)
        logging.info('The best acc is: ' + str(best_acc) + 'The total_epoch_cost is: ' + str(total_epoch_cost))




if __name__ == '__main__':
    main()