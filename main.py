import os, time, shutil, argparse
from functools import partial
import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
from collections import OrderedDict
import torch.utils.data
import torch.utils.data.distributed

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
from skimage import io

import prune_util
from prune_util import GradualWarmupScheduler
from prune_util import CrossEntropyLossMaybeSmooth
from prune_util import mixup_data, mixup_criterion

from utils import save_checkpoint, AverageMeter, visualize_image, GrayscaleImageFolder
from model import ColorNet

# Parse arguments and prepare program
parser = argparse.ArgumentParser(description='Training and Using ColorNet')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', 
                    help='number of data loading workers (default: 0)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', 
                    help='path to .pth file checkpoint (default: none)')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH', 
                    help='path to pretrained .pth file (default: none)')
parser.add_argument('--epochs', default=50, type=int, metavar='N', 
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', 
                    help='manual epoch number (overridden if loading from checkpoint)')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N', 
                    help='size of mini-batch (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', 
                    help='learning rate at start of training')
parser.add_argument('--weight-decay', '--wd', default=1e-10, type=float, metavar='W', 
                    help='weight decay (default: 1e-4)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', default=False, 
                    help='use this flag to validate without training')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', 
                    help='print frequency (default: 10)')
parser.add_argument('--optmzr', type=str, default='adam', metavar='OPTMZR',
                    help='optimizer used (default: adam)')
parser.add_argument('--lr-scheduler', type=str, default='default',
                    help='define lr scheduler')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--warmup', action='store_true', default=False,
                    help='warm-up scheduler')
parser.add_argument('--warmup-lr', type=float, default=0.00005, metavar='M',
                    help='warmup-lr, smaller than original lr')
parser.add_argument('--warmup-epochs', type=int, default=0, metavar='M',
                    help='number of epochs for lr warmup')
parser.add_argument('--mixup', action='store_true', default=False,
                    help='ce mixup')
parser.add_argument('--alpha', type=float, default=0.0, metavar='M',
                    help='for mixup training, lambda = Beta(alpha, alpha) distribution. Set to 0.0 to disable')
parser.add_argument('--smooth', action='store_true', default=False,
                    help='lable smooth')
parser.add_argument('--smooth-eps', type=float, default=0.0, metavar='M',
                    help='smoothing rate [0.0, 1.0], set to 0.0 to disable')
parser.add_argument('--no-tricks', action='store_true', default=False,
                    help='disable all training tricks and restore original classic training process')

args = parser.parse_args()

""" disable all bag of tricks"""
if args.no_tricks:
    # disable all trick even if they are set to some value
    args.lr_scheduler = "default"
    args.warmup = False
    args.mixup = False
    args.smooth = False
    args.alpha = 0.0
    args.smooth_eps = 0.0

#Create folder 
def create_folder(path):
    # delete space at the begining
    path=path.strip()
    # delete / in the end
    path=path.rstrip("/")
 
    # if folder exist
    # exist     True
    # not exist   False
    isExists=os.path.exists(path)
 
    if not isExists:
        # create folder
        os.makedirs(path) 
        #print path + ' folder created' + path
        return True
    else:
        #print path + ' folder already exists'
        return False

def main():
    global args, best_losses, use_gpu
    # Current best losses
    use_gpu = torch.cuda.is_available()
    best_losses = 1000.0
    # args = parser.parse_args()
    print('Arguments: {}'.format(args))

    path_now = os.getcwd()
    create_folder(path_now + "/checkpoints/")

    # # Use GPU if available
    # if use_gpu:
    #     model.cuda()
    #     print('Loaded model onto GPU.')
    
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, best_losses, use_gpu)


def main_worker(gpu, ngpus_per_node, args, best_losses, use_gpu):

    # Create model  
    # models.resnet18(num_classes=365)
    model = ColorNet()
    # print(model)
    args.smooth = args.smooth_eps > 0.0
    args.mixup = args.alpha > 0.0
    args.gpu = gpu

    optimizer_init_lr = args.warmup_lr if args.warmup else args.lr

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # print("=> creating model '{}{}'".format(args.arch, args.depth))

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            print("1")
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        # if args.arch.startswith('alexnet'):
        #     model = torch.nn.DataParallel(model)
        #     model.cuda()
        #     print("2")
        # else:
        model = torch.nn.DataParallel(model).cuda()
        print("3")

    # define loss function (criterion) and optimizer
    criterion = nn.MSELoss().cuda(args.gpu)

    cudnn.benchmark = True

    # Create loss function, optimizer #criterion = nn.CrossEntropyLoss().cuda() if use_gpu else nn.CrossEntropyLoss()
    # criterion = nn.MSELoss().cuda() if use_gpu else nn.MSELoss()
    optimizer = None
    if args.optmzr == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=optimizer_init_lr, momentum=0.9, weight_decay=1e-4)
    elif args.optmzr == 'adam':
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)
        optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_init_lr)

    # Resume from checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('Loading checkpoint {}...'.format(args.resume))
            checkpoint = torch.load(args.resume) if use_gpu else torch.load(args.resume, map_location=lambda storage, loc: storage)
            args.start_epoch = checkpoint['epoch']
            best_losses = checkpoint['best_losses']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('Finished loading checkpoint. Resuming from epoch {}'.format(checkpoint['epoch']))
        else:
            print('Checkpoint filepath incorrect.')
            return
    elif args.pretrained:
        print(args.pretrained)
        if os.path.isfile(args.pretrained):
            model.load_state_dict(torch.load(args.pretrained))
            print('Loaded pretrained model.')
        else:
            print('Pretrained model filepath incorrect.')
            return
    
    # Load data from pre-defined (imagenet-style) structure
    if not args.evaluate:
        train_directory = os.path.join(args.data, 'train')
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip()
        ])
        train_imagefolder = GrayscaleImageFolder(train_directory, train_transforms)
        train_loader = torch.utils.data.DataLoader(train_imagefolder, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        print('Loaded training data.')
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])

    val_directory = os.path.join(args.data, 'val')
    val_imagefolder = GrayscaleImageFolder(val_directory , val_transforms)
    val_loader = torch.utils.data.DataLoader(val_imagefolder, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    print('Loaded validation data.')

    scheduler = None
    if args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader),
                                                         eta_min=4e-08)
    elif args.lr_scheduler == 'default':
        # my learning rate scheduler for cifar, following https://github.com/kuangliu/pytorch-cifar
        # epoch_milestones = [40, 80, 120, 160, 450]
        if args.optmzr == 'sgd':
            epoch_milestones = [10, 20, 30, 40]
        elif args.optmzr == 'adam':
            epoch_milestones = [10, 20, 30, 40]
        """Set the learning rate of each parameter group to the initial lr decayed
            by gamma once the number of epoch reaches one of the milestones
        """
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=[i * len(train_loader) for i in epoch_milestones],
                                                   gamma=0.1)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
        #                                            milestones=epoch_milestones,
        #                                            gamma=0.1)
    else:
        raise Exception("unknown lr scheduler")

    if args.warmup:
        if args.resume:
            if args.start_epoch < args.warmup_epochs:
                scheduler = GradualWarmupScheduler(optimizer, multiplier=args.lr / args.warmup_lr, 
                    total_iter=(args.warmup_epochs - args.start_epoch) * len(train_loader), after_scheduler=scheduler)
                print("warmup lr")
        else:
            scheduler = GradualWarmupScheduler(optimizer, multiplier=args.lr / args.warmup_lr, total_iter=args.warmup_epochs * len(train_loader), after_scheduler=scheduler)
            print("warmup lr")

    # If in evaluation (validation) mode, do not train
    if args.evaluate:
        save_images = True
        epoch = 0
        initial_losses = validate(val_loader, model, criterion, save_images, epoch)

        # # Save checkpoint after evaluation if desired
        # save_checkpoint({
        #     'epoch': epoch,
        #     'best_losses': initial_losses,
        #     'state_dict': model.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        # }, False, 'checkpoints/evaluate-checkpoint.pth.tar')
        
        return  
    
    # Otherwise, train for given number of epochs
    # validate(val_loader, model, criterion, False, 0) # validate before training

    # validate(val_loader, model, criterion, True, args.start_epoch)

    for epoch in range(args.start_epoch, args.epochs):
        
        # Train for one epoch, then validate
        train(train_loader, model, criterion, optimizer, epoch, scheduler, args)
        if epoch % 5 == 0:
            save_images = True #(epoch % 3 == 0)
        else:
            save_images = False
        # losses = validate(val_loader, model, criterion, False, 0)
        losses = validate(val_loader, model, criterion, save_images, epoch)
        
        # Save checkpoint, and replace the old best model if the current model is better
        is_best_so_far = losses < best_losses
        best_losses = max(losses, best_losses)
        save_checkpoint({
            'epoch': epoch + 1,
            'best_losses': best_losses,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best_so_far, 'checkpoints/checkpoint-epoch-{}.pth.tar'.format(epoch))
        
    return best_losses

def train(train_loader, model, criterion, optimizer, epoch, scheduler, args):
    '''Train model on data in train_loader for a single epoch'''
    print('Starting training epoch {}'.format(epoch))

    # Prepare value counters and timers
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    # Switch model to train mode
    model.train()
    
    # Train for single eopch
    end = time.time()
    for i, (input_gray, input_ab, target) in enumerate(train_loader):
        
        if args.mixup:
            input_gray, target_a, target_b, lam = mixup_data(input_gray, target, args.alpha)

        # Use GPU if available
        input_gray_variable = Variable(input_gray).cuda() if use_gpu else Variable(input_gray)
        input_ab_variable = Variable(input_ab).cuda() if use_gpu else Variable(input_ab)
        target_variable = Variable(target).cuda() if use_gpu else Variable(target)

        # Record time to load data (above)
        data_time.update(time.time() - end)

        # Run forward pass
        output_ab = model(input_gray_variable) # throw away class predictions

        if args.mixup:
            loss = mixup_criterion(criterion, output_ab, target_a, target_b, lam, args.smooth)
        else:
            loss = criterion(output_ab, input_ab_variable) # MSE
        
        # Record loss and measure accuracy
        losses.update(loss.item(), input_gray.size(0))
        
        # Compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Record time to do forward and backward passes
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Print model accuracy -- in the code below, val refers to value, not validation
        if i % args.print_freq == 0:
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
            print('({0}) lr:[{1}]  '
                  'Epoch: [{2}][{3}/{4}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    args.optmzr, current_lr, epoch, i, 
                    len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses)) 

    print('Finished training epoch {}'.format(epoch))

def validate(val_loader, model, criterion, save_images, epoch):
    '''Validate model on data in val_loader'''
    print('Starting validation.')

    # Prepare value counters and timers
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    # Switch model to validation mode
    model.eval()
    
    # Run through validation set
    end = time.time()
    for i, (input_gray, input_ab, target) in enumerate(val_loader):
        
        # Use GPU if available
        # target = target.cuda() if use_gpu else target

        with torch.no_grad():
            input_gray_variable = Variable(input_gray).cuda() if use_gpu else Variable(input_gray)
            input_ab_variable = Variable(input_ab).cuda() if use_gpu else Variable(input_ab)
            target_variable = Variable(target).cuda() if use_gpu else Variable(target)
       
        # Record time to load data (above)
        data_time.update(time.time() - end)

        # Run forward pass
        with torch.no_grad():
            output_ab = model(input_gray_variable) # throw away class predictions
        
        loss = criterion(output_ab, input_ab_variable) # check this!
        
        # Record loss and measure accuracy
        losses.update(loss.item(), input_gray.size(0))

        # Save images to file
        if save_images:
            for j in range(len(output_ab)):
                path_now = os.getcwd()
                create_folder(path_now + "/outputs/gray/epoch{}/".format(epoch))
                create_folder(path_now + "/outputs/color/epoch{}/".format(epoch))
                save_path = {'grayscale': path_now + "/outputs/gray/epoch{}/".format(epoch), 'colorized': path_now + "/outputs/color/epoch{}/".format(epoch)}
                save_name = 'img-{}-epoch-{}.jpg'.format(i * val_loader.batch_size + j, epoch)
                visualize_image(input_gray[j], ab_input=output_ab[j].data, show_image=False, save_path=save_path, save_name=save_name)

        # Record time to do forward passes and save images
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Print model accuracy -- in the code below, val refers to both value and validation
        if i % args.print_freq == 0:
            print('Validate: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses))

    print('Finished validation.')
    return losses.avg

if __name__ == '__main__':
    main()
