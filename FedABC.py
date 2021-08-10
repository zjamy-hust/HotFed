#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from utils import get_dataset, average_weights, exp_details, save_checkpoint
from localupdates.update import LocalUpdate, test_inference
from models.models import TestmyNet
from models.models_resnet import ResNet18
from models.models_shufflenetv2 import ShuffleNetV2
best_local_acc = 0
best_global_acc = 0
pretrained_model='./checkpoints/pre_ckpt.best.pth.tar'

if __name__ == '__main__':
    best_local_acc = 0
    best_global_acc = 0
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('.')
    logger = SummaryWriter('./logs')

    args = args_parser()
    exp_details(args)
    if args.gpu:
        torch.cuda.set_device(int(args.gpu))
    device = (f'cuda:{str(args.gpu)}')  if torch.cuda.is_available() else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'test':
        global_model = TestmyNet()
    elif args.model == 'resnet18':
        global_model = ResNet18()
    elif args.model == 'shufflenetv2':
        global_model = ShuffleNetV2(1)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    global_model.train()
    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    start_epoch = 0
    val_loss_pre, counter = 0, 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"===> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=torch.device(f'cuda:{str(args.gpu)}'))
            start_epoch = checkpoint['epoch']
            global_model.load_state_dict(checkpoint['state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            train_accuracy = checkpoint['train_accuracy']
            print(f"===> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            raise ValueError(f"No checkpoint found at '{args.resume}'")    
    else :
        if args.pretrained_model:
            checkpoint = torch.load(args.pretrained_model, map_location=torch.device(f'cuda:{str(args.gpu)}'))
            #checked with res18 to res18, res18 to vgg,  looks like issues with res18 to shufflenetv2
            if 'moco_ckpt' not in args.pretrained_model:
                #this is for rot pretrain
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    if 'linear' not in k and 'fc' not in k:
                        new_state_dict[k] = v
                global_model.load_state_dict(new_state_dict, strict=False)
                print(f'===> Pretrained weights found in total: [{len(list(new_state_dict.keys()))}]')
            print(f'===> Pre-trained model loaded: {args.pretrained_model}')

    best_test_acc = 0
    is_best = 0
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), args.lr,
                                    momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), args.lr,
                                     weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(start_epoch, start_epoch+args.epochs):
        is_best = 0
        local_weights, local_losses= [], []
        #update learning rate
        args.lr=scheduler.get_last_lr()[0]
        print(f'\n | Global Training Round : {epoch+1} |\n')

        # m = max(int(args.frac * args.num_users), 1)
        # idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        start_user = 0
        idxs_share = []
        if args.comparing_shared:
            start_user = 1
            if args.shared_data:
               idxs_share=user_groups[0]
        # Set optimizer for the local updates
        for idx in range(start_user,args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], idxs_shared=idxs_share, logger=logger)
            w, loss  = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch, user=idx)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            if args.save_local == 1:
                    save_checkpoint(args, {
                        'epoch': epoch + 1,
                        'arch': args.model,
                        'state_dict': w,
                    }, is_best, idx, is_global=0)
            print(f'Global:{epoch}, user:{idx}, size:{len(user_groups[idx])} loss: {loss:.4f}')
            optimizer.step() #not sure whether making it inside idx or outside idx

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[c],idxs_shared=idxs_share, logger=logger)
            acc, loss, _ = local_model.inference(model=global_model,global_round=1000,user=c)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        test_acc, test_loss =  test_inference(args, global_model, test_dataset)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            is_best = 1
        if args.save_global == 1:
            save_checkpoint(args, {
                'epoch': epoch + 1,
                'arch': args.model,
                'state_dict': global_weights,
                'train_accuracy': train_accuracy,
            }, is_best, local_idx=1000,is_global=1)
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% '.format(100*train_accuracy[-1]))
            print('Test Accuracy: {:.2f}% '.format(100*test_acc))
            print('Best Test Accuracy: {:.2f}% \n'.format(100*best_test_acc))
        scheduler.step()

    # Test inference after completion of training
    test_acc, test_loss =  test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    if not os.path.isdir('save/objects'):
        os.makedirs('save/objects')
    file_name = 'save/objects/{}_{}_{}_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs,  args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)

    # matplotlib.use('Agg')

    # Plot Loss curve
    plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.show()
    plt.savefig('save/fed_{}_{}_{}_iid[{}]_E[{}]_B[{}]_loss.png'.
                format(args.dataset, args.model, args.epochs,
                       args.iid, args.local_ep, args.local_bs))
    
    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.show()
    plt.savefig('save/fed_{}_{}_{}_iid[{}]_E[{}]_B[{}]_acc.png'.
                format(args.dataset, args.model, args.epochs, 
                       args.iid, args.local_ep, args.local_bs))
