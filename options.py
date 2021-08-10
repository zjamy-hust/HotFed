#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
#    parser.add_argument('--frac', type=float, default=0.1,
#                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='test', help='test, restnet18, shufflenetv2')

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=None, type=int, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=0, help='verbose')
    parser.add_argument('--save_local', type=int, default=0, help='save local models or not')
    parser.add_argument('--save_global', type=int, default=1, help='save global models or not')
    parser.add_argument('--comparing_shared', type=int, default=0, help='comparing shared or not')
    parser.add_argument('--shared_data', type=int, default=1, help='using shared data or not, comparing_shared but be 1')
    parser.add_argument('--partition', type=str, default='homo', help='homo, ')
  
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--pretrained_model', type=str, default='')

    args = parser.parse_args()
    return args
