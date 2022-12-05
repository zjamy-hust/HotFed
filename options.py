#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7

import argparse
def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=2,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=2,
                        help="number of users: K")
#    parser.add_argument('--frac', type=float, default=0.1,
#                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=2,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='resnet18', help='test, restnet18, shufflenetv2')

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=1, type=int, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--mode', type=int, default=1,
                        help='[0,1,2]。0表示FedAvg； \
                                    1表示data augmentation训练，训练前，利用server model和trainning dataset生成mask，训练时进行一次原始样本训练+一次带mask训练，测试时计算acc+XAI Acc（暂时如此）；\
                                    2表示仅测试时采用mask，然后通过acc筛选客户端，测试前会通过server model和testing dataset产生mask；\
                                    ')
    
    #mode == 1需要关注的参数
    parser.add_argument('--start_epoch', type=int, default=1,help="data augmentation需要跳过前面几轮再执行。")
    parser.add_argument('--train_mask_batch_size', type=int, default=20,help="生成训练集mask，一次计算的mask数量。")
    parser.add_argument('--train_mask_nt_samples', type=int, default=10,help="IG算法，一个样本采样的次数。")
    parser.add_argument('--train_mask_n_steps', type=int, default=15,help="IG算法，一个样本step数量。")
    
    #mode == 2需要关注的参数
    parser.add_argument('--test_mask_batch_size', type=int, default=20,help="生成测试集mask，一次计算的mask数量。")
    parser.add_argument('--test_mask_nt_samples', type=int, default=10,help="IG算法，一个样本采样的次数。")
    parser.add_argument('--test_mask_n_steps', type=int, default=15,help="IG算法，一个样本step数量。")
     
     
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--save_local', type=int, default=0, help='save local models or not')
    parser.add_argument('--save_global', type=int, default=1, help='save global models or not')
    # parser.add_argument('--comparing_shared', type=int, default=0, help='comparing shared or not')
    parser.add_argument('--shared_data', type=float, default=0, help='using shared data or not')
    parser.add_argument('--partition', type=str, default='homo', help='homo, ')
  
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--pretrained_model', type=str, default='')
    parser.add_argument('--random_seed', type=int, default=40212202)


    args = parser.parse_args()
    return args
