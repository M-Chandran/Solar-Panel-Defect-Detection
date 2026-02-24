import os
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import logging
from tqdm import tqdm
import numpy as np
import timm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import get_data_loaders
from models.moco import MoCo, DINO, DINOLoss, MultiCropWrapper, DINOHead

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_epoch(model, train_loader, optimizer, criterion, epoch, args):
    """Train for one epoch"""
    model.train()

    losses = []
    for i, (images_q, images_k) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        if torch.cuda.is_available():
            images_q = images_q.cuda(non_blocking=True)
            images_k = images_k.cuda(non_blocking=True)

        # compute output
        output, target = model(im_q=images_q, im_k=images_k)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if i % args.print_freq == 0:
            logger.info(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                       f'Loss: {loss.item():.4f}')

    return np.mean(losses)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Save checkpoint"""
    torch.save(state, filename)
    if is_best:
        torch.save(state, 'model_best.pth.tar')

def main():
    parser = argparse.ArgumentParser(description='Train MoCo v2')
    parser.add_argument('--data', metavar='DIR', default='../data',
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='se_resnet50',
                        help='backbone architecture (default: se_resnet50 as per paper)')

    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run (default: 200 as per paper)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 64 as per paper)')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        metavar='LR', help='initial learning rate (default: 0.03)', dest='lr')
    parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4 as per paper)')

    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--cos', action='store_true', default=True,
                        help='use cosine lr schedule (default: True as per paper)')


    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Create model with paper's hyperparameters
    logger.info("=> creating model '{}'".format(args.arch))
    # MoCo parameters from paper: dim=128, K=4096, m=0.999, T=0.2
    model = MoCo(dim=128, K=4096, m=0.999, T=0.2, arch=args.arch, pretrained=True)


    if torch.cuda.is_available():
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
        else:
            model = torch.nn.DataParallel(model).cuda()
        logger.info("Using GPU")
    else:
        logger.info("Using CPU")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = optim.SGD(model.parameters(), args.lr,
                          momentum=args.momentum,
                          weight_decay=args.wd)

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                       .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading
    train_loader, val_loader = get_data_loaders(args.data, args.batch_size)

    # Training loop
    best_loss = float('inf')
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, epoch, args)

        # evaluate on validation set
        # Note: For self-supervised learning, validation is optional
        # You can add validation logic here if needed

        # remember best loss and save checkpoint
        is_best = train_loss < best_loss
        best_loss = min(train_loss, best_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best)

        logger.info(f'Epoch {epoch+1}/{args.epochs} completed. Train Loss: {train_loss:.4f}')

    logger.info("Training completed!")

if __name__ == '__main__':
    main()
