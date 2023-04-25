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
from model_fixed_network.quant_resnet4cifar_single import *
from model_fixed_network.quant_resnet_my import *
from model_fixed_network.quant_mv1_my import *
from model_fixed_network.quant_mv2_my import *
from model_fixed_network.quant_conv import QConv, QLinear, QLinear_8bit, QConv_8bit
from torch.utils.tensorboard import SummaryWriter

from utils.utils import *
from option import args

best_acc1 = 0
np.random.seed(0)


def main():
    arg_dict = vars(args)

    if args.visible_gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    args.log_dir = args.log_dir + Time2Str() + args.arch + args.datasetsname + '_' + str(args.bit) + '_fix'

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logging.basicConfig(filename=os.path.join(args.log_dir, "log.txt"), level=logging.INFO, format='')
    log_string = 'configs\n'
    for k, v in arg_dict.items():
        log_string += "{}: {}\n".format(k, v)
        print("{}: {}".format(k, v), end='\n')
    logging.info(log_string + '\n')

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


def load_pretrained_fp_model(model, args, fp_path=None):
    if 'cifar' in args.datasetsname:

        trained_model = torch.load(
            fp_path, map_location=lambda storage, loc: storage)
        old_checkpoint = trained_model['state_dict']
        new_keys = list(model.state_dict().keys())
        old_keys = list(old_checkpoint.keys())

        tmp = []
        for item in new_keys:
            if 'alpha' not in item and 'init' not in item and 'uA' not in item and 'uW' not in item \
                    and 'lA' not in item and 'lW' not in item and 'prev_Qweight' not in item \
                    and 'prev_Qweight' not in item and 'distance' not in item and 'unchange_step' not in item \
                    and 'saved_weight' not in item:
                tmp.append(item)
        new_keys = tmp

        assert len(new_keys) == len(old_keys)

        new_checkpoint = {}
        for key_new, key_old in zip(new_keys, old_keys):
            new_checkpoint[key_new] = old_checkpoint[key_old]

        for key_new in list(model.state_dict().keys()):
            if key_new not in new_checkpoint:
                new_checkpoint[key_new] = model.state_dict()[key_new]
        model.load_state_dict(new_checkpoint)
        print('Loaded full precision model')
        return model

    else:
        import torchvision.models as models
        if 'resnet18' in args.arch:
            trained_model = models.resnet18(pretrained=True)
        elif 'resnet50' in args.arch:
            trained_model = models.resnet50(pretrained=True)
        elif 'mv2' in args.arch:
            trained_model = models.mobilenet_v2(pretrained=True)
        elif 'mv1' in args.arch:
            from pytorchcv.model_provider import get_model as ptcv_get_model
            trained_model = ptcv_get_model('mobilenet_w1', pretrained=True)
        old_checkpoint = trained_model.state_dict()
        new_keys = list(model.state_dict().keys())
        old_keys = list(old_checkpoint.keys())

        tmp = []
        for item in new_keys:
            if 'alpha' not in item and 'init' not in item and 'uA' not in item and 'uW' not in item \
                    and 'lA' not in item and 'lW' not in item and 'prev_Qweight' not in item \
                    and 'prev_Qweight' not in item and 'distance' not in item and 'unchange_step' not in item \
                    and 'saved_weight' not in item:
                tmp.append(item)
        new_keys = tmp

        assert len(new_keys) == len(old_keys)

        new_checkpoint = {}
        for key_new, key_old in zip(new_keys, old_keys):
            new_checkpoint[key_new] = old_checkpoint[key_old]

        for key_new in list(model.state_dict().keys()):
            if key_new not in new_checkpoint:
                new_checkpoint[key_new] = model.state_dict()[key_new]
        model.load_state_dict(new_checkpoint)
        print('Loaded full precision model')
        return model


def init_quant_model(model, args):
    for layers in model.modules():
        if hasattr(layers, 'init'):
            layers.init.data.fill_(1)

    train_loader, _, _ = loading_data(args)
    iterloader = iter(train_loader)

    model.to(args.gpu)
    model.train()
    with torch.no_grad():
        # model.eval()
        print('init for bit-width: ', args.bit)
        images, labels = next(iterloader)
        images = images.to(args.gpu)
        labels = labels.to(args.gpu)
        for name, layers in model.named_modules():
            if hasattr(layers, 'weight_bit'):
                setattr(layers, "weight_bit", int(args.bit))
            if hasattr(layers, 'act_bit'):
                setattr(layers, "act_bit", int(args.bit))
        model.forward(images)

    for layers in model.modules():
        if hasattr(layers, 'init'):
            layers.init.data.fill_(0)


def loading_data(args):
    if args.datasetsname == 'ImageNet':

        # Data loading code
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        # For ResNet18, we use a bigger crop scale for better performance
        if args.arch == 'resnet18_quant':
            scale = (0.2, 1)
        else:
            scale = (0.08, 1)
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224, scale=scale),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    else:
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


def cal_params(model, args, logging):
    trainable_params = list(model.parameters())
    model_params = []
    quant_params = []

    for name, layers in model.named_modules():
        if args.gpu == 0:
            logging.info((name, layers))
        if isinstance(layers, QConv) or isinstance(layers, QConv_8bit) or isinstance(layers, QLinear) \
                or isinstance(layers, QLinear_8bit):
            model_params.append(layers.weight)
            if args.gpu == 0:
                logging.info("====> weight params: " + str(layers.weight.numel()))
            if layers.bias is not None:
                model_params.append(layers.bias)
            if layers.quan_weight:
                if isinstance(layers.lW, nn.ParameterList):
                    for x in layers.lW:
                        quant_params.append(x)
                    for x in layers.uW:
                        quant_params.append(x)
                else:
                    quant_params.append(layers.lW)
                    quant_params.append(layers.uW)
            if layers.quan_act:
                if isinstance(layers.lA, nn.ParameterList):
                    for x in layers.lA:
                        quant_params.append(x)
                    for x in layers.uA:
                        quant_params.append(x)
                else:
                    quant_params.append(layers.lA)
                    quant_params.append(layers.uA)

        elif isinstance(layers, nn.Conv2d): # first conv xxxx

            model_params.append(layers.weight)
            if args.gpu == 0:
                logging.info("====> weight params: " + str(layers.weight.numel()))
            if layers.bias is not None:

                model_params.append(layers.bias)

        elif isinstance(layers, nn.Linear): # last FC xxxx

            model_params.append(layers.weight)
            if args.gpu == 0:
                logging.info("====> weight params: " + str(layers.weight.numel()))
            if layers.bias is not None:

                model_params.append(layers.bias)

        elif isinstance(layers, nn.SyncBatchNorm) or isinstance(layers, nn.BatchNorm2d):
            if layers.bias is not None:

                model_params.append(layers.weight)
                model_params.append(layers.bias)
                if args.gpu == 0:
                    logging.info("====> weight params: " + str(layers.weight.numel()))

        if args.gpu == 0:
            logging.info("====> total modelparams: " + str(sum(p.numel() for p in model_params)) + "\n")

    log_string = "====> total params:" + str(sum(p.numel() for p in trainable_params)) + "\n"
    log_string += "====> trainable model params:" + str(sum(p.numel() for p in model_params)) + "\n"
    log_string += "====> trainable quantizer params:" + str(sum(p.numel() for p in quant_params)) + "\n"
    print(log_string)

    assert sum(p.numel() for p in trainable_params) == (sum(p.numel() for p in model_params)
                                                        + sum(p.numel() for p in quant_params))
    logging.info(log_string)

    return model_params, quant_params


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
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                                rank=args.rank)
    if args.datasetsname == 'cifar10':
        args.num_classes = 10
    elif args.datasetsname == 'cifar100':
        args.num_classes = 100
    else:
        args.num_classes = 1000
    # create model
    model_class = globals().get(args.arch)
    model = model_class(args, num_classes=args.num_classes)

    if args.rank % ngpus_per_node == 0:
        # print(model)
        print('args.bit: ', args.bit)
    # fp_path = '/media/disk/zys/LTS-save/cifar_fp/220316_26_21resnet20_quantcifar10_fp/model_best.pth.tar'
    # fp_path = '/media/disk/zys/LTS-save/cifar_fp/220316_26_21resnet20_quantcifar100_fp/model_best.pth.tar'
    fp_path = args.fp_path
    model = load_pretrained_fp_model(model, args, fp_path)

    ### initialze quantizer parameters
    if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):  # do only at rank=0 process
        if not args.evaluate:
            init_quant_model(model, args)
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
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  ########## SyncBatchnorm
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all available GPUs if device_ids are not set
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  ########## SyncBatchnorm
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
    batch_num = len(train_loader)

    for name, layers in model.named_modules():
        if hasattr(layers, 'batch_num'):
            setattr(layers, "batch_num", batch_num)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # for name, layers in model.named_modules():
    #     if hasattr(layers, 'quan_act'):
    #         setattr(layers, "quan_act", False)
    #     if hasattr(layers, 'quan_weight'):
    #         setattr(layers, "quan_weight", False)
    # for name, layers in model.named_modules():
    #     if hasattr(layers, 'act_bit'):
    #         setattr(layers, "act_bit", args.bit)
    #     if hasattr(layers, 'weight_bit'):
    #         setattr(layers, "weight_bit", args.bit)
    # validate(val_loader, model, criterion, args, None, 0, args.bit, args.bit, [])
    # import sys
    # sys.exit()

    if args.evaluate:
        if os.path.exists(args.model):
            model_dict = torch.load(args.model)
            if 'state_dict' in model_dict:
                model_dict = model_dict['state_dict']
            model.load_state_dict(model_dict)
            validate(val_loader, model, criterion, args, None, args.start_epoch, args.bit, [])
        else:
            raise ValueError("model path {} not exists".format(args.model))
        return

    model_params, quant_params = cal_params(model, args, logging)

    if args.optimizer_m == 'SGD':
        optimizer_m = torch.optim.SGD(model_params, lr=args.lr_m, momentum=args.momentum,
                                      weight_decay=args.weight_decay, nesterov=args.nesterov)
    elif args.optimizer_m == 'Adam':
        optimizer_m = torch.optim.Adam(model_params, lr=args.lr_m, weight_decay=args.weight_decay)

    if args.optimizer_q == 'SGD':
        optimizer_q = torch.optim.SGD(quant_params, lr=args.lr_q)
    elif args.optimizer_q == 'Adam':
        optimizer_q = torch.optim.Adam(quant_params, lr=args.lr_q)

    if args.lr_scheduler == 'cosine':
        scheduler_m = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_m, T_max=args.epochs, eta_min=args.lr_m_end)
        scheduler_q = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_q, T_max=args.epochs, eta_min=args.lr_q_end)
    elif args.lr_scheduler == 'step':
        if args.decay_schedule is not None:
            milestones = list(map(lambda x: int(x), args.decay_schedule.split('-')))
        else:
            milestones = [(args.epochs + 1)]
        scheduler_m = torch.optim.lr_scheduler.MultiStepLR(optimizer_m, milestones=milestones, gamma=args.gamma)
        scheduler_q = torch.optim.lr_scheduler.MultiStepLR(optimizer_q, milestones=milestones, gamma=args.gamma)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            logging.info("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            logging.info(model.load_state_dict(checkpoint['state_dict']))
            optimizer_m.load_state_dict(checkpoint['optimizer_m'])
            optimizer_q.load_state_dict(checkpoint['optimizer_q'])
            scheduler_m.load_state_dict(checkpoint['scheduler_m'])
            scheduler_q.load_state_dict(checkpoint['scheduler_q'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            logging.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    ### tensorboard
    # if args.rank % ngpus_per_node == 0: # do only at rank=0 process
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None
    ###
    total_epoch_cost = 0
    epoch_fixed_ratio = []

    for name, layers in model.named_modules():
        if hasattr(layers, 'weight_bit'):
            setattr(layers, "weight_bit", int(args.bit))
        if hasattr(layers, 'act_bit'):
            setattr(layers, "act_bit", int(args.bit))

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        start = time.time()
        epoch_cost = train(train_loader, model, criterion, optimizer_m, optimizer_q, scheduler_m,
                           scheduler_q, epoch, args, writer)
        total_epoch_cost += epoch_cost
        acc1, epoch_fixed_ratio = validate(val_loader, model, criterion, args, writer, epoch,
                                           args.bit, args.bit, epoch_fixed_ratio)
        print("weight-bit act-bit", int(args.bit), acc1)

        # remember best acc@1 and save checkpoint
        print("Epoch: [{}]".format(epoch) + "[GPU{}]".format(args.gpu) + "\tcurrent acc@1: {}\n".format(acc1))
        logging.info("Epoch: [{}]".format(epoch) + "[GPU{}]".format(args.gpu) + "\tcurrent acc@1: {}\n".format(acc1))
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'acc1': acc1,
                'best_acc1': best_acc1,
                'optimizer_m': optimizer_m.state_dict(),
                'optimizer_q': optimizer_q.state_dict(),
                'scheduler_m': scheduler_m.state_dict(),
                'scheduler_q': scheduler_q.state_dict(),
                'epoch_fixed_ratio': epoch_fixed_ratio
            }, is_best, path=args.log_dir)

            if epoch % (args.epochs // 10) == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'acc1': acc1,
                    'best_acc1': best_acc1,
                    'optimizer_m': optimizer_m.state_dict(),
                    'optimizer_q': optimizer_q.state_dict(),
                    'scheduler_m': scheduler_m.state_dict(),
                    'scheduler_q': scheduler_q.state_dict(),
                    'epoch_fixed_ratio': epoch_fixed_ratio
                }, False, path=args.log_dir, TenEpoch=epoch)

            log_string = "Epoch: [{}]".format(epoch) + "[GPU{}]".format(args.gpu) + \
                         ' 1 epoch spends {:2d}:{:.2f} mins\t remain {:2d}:{:2d} hours'. \
                             format(int((time.time() - start) // 60),
                                    (time.time() - start) % 60,
                                    int((args.epochs - epoch - 1) * (time.time() - start) // 3600),
                                    int((args.epochs - epoch - 1) * (time.time() - start) % 3600 / 60)
                                    ) + "\tcurrent best acc@1: {}\n".format(best_acc1)
            log_string += '[{}bit] {:.3f}\t'.format(args.bit, acc1)
            print(log_string)
            logging.info(log_string)
            log_string = 'epoch_cost: {}, total_epoch_cost: {}'.format(epoch_cost, total_epoch_cost)
            print(log_string)
            logging.info(log_string)


    if writer is not None:
        writer.close()


def train(train_loader, model, criterion, optimizer_m, optimizer_q, scheduler_m, scheduler_q, epoch, args, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')

    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to train mode
    model.train()

    modules = []
    for m in model.modules():
        if isinstance(m, QConv) or isinstance(m, QConv_8bit) or isinstance(m, QLinear) \
                or isinstance(m, QLinear_8bit):
            modules.append(m)

    epoch_cost = 0
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        optimizer_m.zero_grad()
        optimizer_q.zero_grad()

        # compute output
        output = model(images)
        loss = criterion(output, target)
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        num_total_fixed_params = 0
        num_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        for m in modules:
            if hasattr(m, 'fixed_para_num'):
                num_total_fixed_params += m.fixed_para_num
        iter_fixed_ratio = num_total_fixed_params / num_total_params
        cost_ratio = 1 - iter_fixed_ratio
        epoch_cost += cost_ratio

        with open('iter_fixed_ratio.txt', 'a') as f:
            f.write(str(iter_fixed_ratio) + '\n')

        # compute gradient and do SGD step
        loss.backward()
        if i % args.print_freq == 0:
            print(i, loss.item())
        optimizer_m.step()
        optimizer_q.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            if writer is not None:  # this only works at rank=0 process
                writer.add_scalar('train/model_lr', optimizer_m.param_groups[0]['lr'], len(train_loader) * epoch + i)
                writer.add_scalar('train/quant_lr', optimizer_q.param_groups[0]['lr'], len(train_loader) * epoch + i)
                writer.add_scalar('train/loss(current)', loss.cpu().item(), len(train_loader) * epoch + i)
    scheduler_m.step()
    scheduler_q.step()

    return epoch_cost


def validate(val_loader, model, criterion, args, writer, epoch, weight_bit, act_bit, epoch_fixed_ratio):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        log_dir=args.log_dir,
        prefix="Epoch: [{}]".format(epoch) + '[W{}bit A{}bit] Test: '.format(weight_bit, act_bit))

    # switch to evaluate mode
    model.eval()

    num_total_fixed_params = 0
    num_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for m in model.modules():
        if (isinstance(m, QConv) or isinstance(m, QConv_8bit) or isinstance(m, QLinear) \
                or isinstance(m, QLinear_8bit)) and hasattr(m, 'unchange_step'):
            num_total_fixed_params += torch.sum(m.unchange_step != 0)
    epoch_fixed_ratio.append(num_total_fixed_params / num_total_params)
    log_string = 'num_total_params{}'.format(num_total_params) + \
          ' num_total_fixed_params {}'.format(num_total_fixed_params) + \
          ' num_total_fixed_params / num_total_params {}'.format(num_total_fixed_params / num_total_params)
    print(log_string)
    logging.info(log_string)

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

        if writer is not None:
            writer.add_scalar('val/top1_{}bit'.format(weight_bit), top1.avg, epoch)
            writer.add_scalar('val/top5_{}bit'.format(weight_bit), top5.avg, epoch)

        logging.basicConfig(filename=os.path.join(args.log_dir, "log.txt"), level=logging.INFO, format='')
        log_string = "Epoch: [{}]".format(epoch) + '[W{}bit A{}bit]'.format(weight_bit, act_bit) + "[GPU{}]".format(
            args.gpu) + ' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5)
        print(log_string)
        logging.info(log_string)

    return top1.avg, epoch_fixed_ratio


if __name__ == '__main__':
    main()
