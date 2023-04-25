# ref. https://github.com/pytorch/examples/blob/master/imagenet/main.py
import argparse
# from ast import arg
import os
import random
import shutil
import time
import warnings
import pdb
import logging
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.tensorboard import SummaryWriter
from model_single_network.quant_resnet4cifar_single import *
from model_single_network.quant_mv1_my import *
from model_single_network.quant_mv2_my import *
from model_single_network.quant_conv import QConv
from utils.utils import *
from option import args

best_acc1 = 0
np.random.seed(0)

def main():
    arg_dict = vars(args)

    if args.visible_gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"]= args.visible_gpus

    args.log_dir = args.log_dir + Time2Str() + args.arch + args.datasetsname + '_' + 'fp'

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logging.basicConfig(filename=os.path.join(args.log_dir, "log.txt"), level=logging.INFO, format='')
    log_string = 'configs\n'
    for k, v in arg_dict.items():
        log_string += "{}: {}\n".format(k,v)
        print("{}: {}".format(k,v), end='\n')
    logging.info(log_string+'\n')

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        # cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def loading_data(args):

    data_means = {
        'cifar10': [x / 255 for x in [125.3, 123.0, 113.9]],
        'cifar100': [x / 255 for x in [129.3, 124.1, 112.4]],

    }
    data_stds = {
        'cifar10': [x / 255 for x in [63.0, 62.1, 66.7]],
        'cifar100': [x / 255 for x in [68.2, 65.4, 70.4]],
    }

    if args.datasetsname == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(data_means['cifar10'], data_stds['cifar10'])
        ])
        train_dataset = datasets.CIFAR10(root=os.path.join(args.data, 'CIFAR10'),
                                         train=True,
                                         transform=train_transform,
                                         download=True)
        train_dataset.num_classes = 10

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(data_means['cifar10'], data_stds['cifar10'])
        ])
        val_dataset = datasets.CIFAR10(root=os.path.join(args.data, 'CIFAR10'),
                                       train=False,
                                       transform=val_transform)


    elif args.datasetsname == 'cifar100':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(data_means['cifar100'], data_stds['cifar100'])
        ])
        train_dataset = datasets.CIFAR100(root=os.path.join(args.data, 'CIFAR100'),
                                          train=True,
                                          transform=train_transform,
                                          download=True)
        train_dataset.num_classes = 100

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(data_means['cifar100'], data_stds['cifar100'])
        ])
        val_dataset = datasets.CIFAR100(root=os.path.join(args.data, 'CIFAR100'),
                                        train=False,
                                        transform=val_transform)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader, train_sampler

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    logging.basicConfig(filename=os.path.join(args.log_dir, "log.txt"), level=logging.INFO, format='')

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        logging.info("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    if args.datasetsname == 'cifar10':
        args.num_classes = 10
    elif args.datasetsname == 'cifar100':
        args.num_classes = 100
    # create model
    model_class = globals().get(args.arch)
    model = model_class(args, num_classes=args.num_classes)

    if args.rank % ngpus_per_node == 0:
        print('FP model!')


    ###
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) ########## SyncBatchnorm
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all available GPUs if device_ids are not set
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) ########## SyncBatchnorm
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    train_loader, val_loader, train_sampler = loading_data(args)
    args.batch_num = len(train_loader)

    for name, layers in model.named_modules():
        if hasattr(layers, 'batch_num'):
            setattr(layers, "batch_num", batch_num)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    for name, layers in model.named_modules():
        if hasattr(layers, 'quan_act'):
            setattr(layers, "quan_act", False)
        if hasattr(layers, 'quan_weight'):
            setattr(layers, "quan_weight", False)

    model_params = model.parameters()

    if args.optimizer_m == 'SGD':
        optimizer_m = torch.optim.SGD(model_params, lr=args.lr_m, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    elif args.optimizer_m == 'Adam':
        optimizer_m = torch.optim.Adam(model_params, lr=args.lr_m, weight_decay=args.weight_decay)

    if args.lr_scheduler == 'cosine':
        scheduler_m = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_m, T_max=args.epochs, eta_min=args.lr_m_end)
    elif args.lr_scheduler == 'step':
        if args.decay_schedule is not None:
            milestones = list(map(lambda x: int(x), args.decay_schedule.split('-')))
        else:
            milestones = [(args.epochs+1)]
        scheduler_m = torch.optim.lr_scheduler.MultiStepLR(optimizer_m, milestones=milestones, gamma=args.gamma)

    ### tensorboard


    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        start = time.time()
        train(train_loader, model, criterion, optimizer_m, scheduler_m, epoch, args)

        acc1 = validate(val_loader, model, criterion, args, epoch)
        
        # remember best acc@1 and save checkpoint
        print("Epoch: [{}]".format(epoch) + "[GPU{}]".format(args.gpu) + "\tcurrent acc@1: {}\n".format(acc1))
        logging.info("Epoch: [{}]".format(epoch) + "[GPU{}]".format(args.gpu) + "\tcurrent acc@1: {}\n".format(acc1))
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'acc1': acc1,
                'best_acc1': best_acc1,
                'optimizer_m' : optimizer_m.state_dict(),
                'scheduler_m' : scheduler_m.state_dict(),
            }, is_best, path=args.log_dir)

            log_string = "Epoch: [{}]".format(epoch) + "[GPU{}]".format(args.gpu) + \
                         ' 1 epoch spends {:2d}:{:.2f} mins\t remain {:2d}:{:2d} hours'.\
                             format(int((time.time() - start) // 60),
                (time.time() - start) % 60,
                int((args.epochs - epoch - 1) * (time.time() - start) // 3600),
                int((args.epochs - epoch - 1) * (time.time() - start) % 3600 / 60 )
            ) + "\tcurrent best acc@1: {}\n".format(best_acc1)
            log_string += '{:.3f}\t'.format(acc1)
            print(log_string)
            logging.info(log_string)


def train(train_loader, model, criterion, optimizer_m, scheduler_m, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')

    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')


    # switch to train mode
    model.train()

    modules = []
    for m in model.modules():
        if isinstance(m, QConv):
            modules.append(m)

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        optimizer_m.zero_grad()
        # compute output
        output = model(images)
        loss = criterion(output, target)
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        loss.backward()
        if i % args.print_freq == 0:
            print(i, loss.item())
        optimizer_m.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    scheduler_m.step()



def validate(val_loader, model, criterion, args, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        log_dir=args.log_dir,
        prefix="Epoch: [{}]".format(epoch))

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        logging.basicConfig(filename=os.path.join(args.log_dir, "log.txt"), level=logging.INFO, format='')
        log_string = "Epoch: [{}]".format(epoch) + "[GPU{}]".format(args.gpu) + ' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5)
        print(log_string)
        logging.info(log_string)

    return top1.avg


if __name__ == '__main__':
    main()
