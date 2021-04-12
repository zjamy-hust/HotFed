#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import os
import shutil
import time
from torchvision import datasets, transforms
from datasets.sampling_partition import cifar_iid, cifar_noniid


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = 'data/cifar/'
        train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.RandomGrayscale(),
             transforms.ToTensor(),
             transforms.RandomCrop(32, padding=4),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=train_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)
        # check proportion of shared_data
        shared_data = 0
        if args.shared_data>0:
            shared_data = args.shared_data
        # sample training data amongst users
        if args.iid:
            # Sample IID user data 
            user_groups,idxs_share = cifar_iid(train_dataset, args.num_users,shared_data)
        else:
            # Sample Non-IID user data
            user_groups,idxs_share = cifar_noniid(train_dataset, args.num_users, args.partition, shared_data)

    return train_dataset, test_dataset, user_groups, idxs_share


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(log,args):
    log.logger.debug(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    log.logger.debug('\nExperimental details:')
    log.logger.debug(f'    Model     : {args.model}')
    log.logger.debug(f'    Optimizer : {args.optimizer}')
    log.logger.debug(f'    Learning  : {args.lr}')
    log.logger.debug(f'    Global Rounds   : {args.epochs}\n')

    log.logger.debug('    Federated parameters:')
    if args.iid:
        log.logger.debug('    IID')
    else:
        log.logger.debug('    Non-IID')
    #print(f'    Fraction of users  : {args.frac}')
    log.logger.debug(f'    Local Batch size   : {args.local_bs}')
    log.logger.debug(f'    Local Epochs       : {args.local_ep}\n')
    
    if args.num_users:
        log.logger.debug(f'    num_users   : {args.num_users}')
    if args.shared_data:
        log.logger.debug(f'    shared_data   : {args.shared_data}')
    if args.pretrained_model :
        log.logger.debug(f'    pretrained_model   : {args.pretrained_model}')
    return

def save_checkpoint(args, state, is_best, local_idx, is_global):
    if is_global == 0:
        if not os.path.isdir(f'save_checkpoints/{args.model}_local/'):
            os.makedirs(f'save_checkpoints/{args.model}_local/')
        filename = f'save_checkpoints/{args.model}_local/local_{local_idx}.gpu{args.gpu}.ckpt.pth.tar'
    else:
        if not os.path.isdir(f'save_checkpoints/{args.model}_global/'):
            os.makedirs(f'save_checkpoints/{args.model}_global/')
        filename = f'save_checkpoints/{args.model}_global/global.iid{args.iid}.gpu{args.gpu}.ckpt.pth.tar'
    torch.save(state, filename)
    print(f'saved checkpoint to {filename}')
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))
        print(f'saved checkpoint to {filename}')
