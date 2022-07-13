#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import copy
from torch import nn
from torch.utils.data import DataLoader, Dataset
best_global_acc = 0

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
#        return torch.tensor(image), torch.tensor(label)
        return image.clone().detach(), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.valloader = self.train_val_test(dataset, list(idxs))
        self.device = (f'cuda:{str(args.gpu)}')  if torch.cuda.is_available() else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.best_local_acc = 0
        #self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation (90, 10)
        #idxs_train = idxs[:int(0.9*len(idxs))]
        idxs_train = idxs[:int(len(idxs))]
        idxs_val = idxs[int(0.9*len(idxs)):] # is it iid? needs improve

        # if len(idxs_shared) > 0:
        #     print('idxs_shared > 0', idxs_shared)
        #     idxs_train.extend(idxs_shared[:int(len(idxs_shared))])
        #     idxs_val.extend(idxs_shared[int(0.9*len(idxs_shared)):])

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        valloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                batch_size=128, shuffle=False)

        return trainloader, valloader

    def update_weights(self, model, global_round, user):
        # Set mode to train model
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.9, weight_decay=5e-4)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.local_ep)

        loader = self.trainloader
            
        for iter in range(self.args.local_ep):
            is_best = 0
            batch_loss = []
            model.train()
            
            for batch_idx, (images, labels) in enumerate(loader):
                images, labels = images.to(self.device), labels.to(self.device)
                # print(len(images), len(labels))
                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                # optimizer.zero_grad()
                loss.backward()
               
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
                optimizer.step()
            if self.args.verbose :
                    print('| Global Round : {} | Local Epoch : {} | User ID: {} | Data size: {} \tLoss: {:.4f}'.format(
                        global_round, iter, user, len(loader.dataset), loss.item()))
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            _, _, is_best = self.inference(model, global_round, user)
            if is_best > 0:
                best_model = copy.deepcopy(model)
                best_epoch_loss = epoch_loss
                best_loss = loss
            scheduler.step()

        return best_model.state_dict(), sum(best_epoch_loss) / len(best_epoch_loss)

    def inference(self, model, global_round=1000,user=1000):

        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct, is_best = 0.0, 0.0, 0.0, 1

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.valloader):
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
    
                # Inference
                outputs = model(images)
                batch_loss = self.criterion(outputs, labels)
                loss += batch_loss.item()
    
                # Prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

        accuracy = correct/total
            # Save checkpoint.
        if accuracy > self.best_local_acc:
            # print('Saving..')
            # state = {'model_state_dict':model.state_dict(),
            #         'loss':loss
            # }
            # if not os.path.isdir('checkpoints'):
            #     os.mkdir('checkpoints')
            # torch.save(state, './checkpoints/ckpt_best_local.pth')
            self.best_local_acc = accuracy
            is_best = 1
        if self.args.verbose :
            print(f'Global:{global_round}, user:{user}, train accuracy:{100*accuracy:.2f}%, loss:{loss:.4f}, correct:{correct:.0f}, total:{total:.0f}, best local acc: {100*self.best_local_acc:.2f}%')
        return accuracy, loss, is_best


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """
    global best_global_acc

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = (f'cuda:{str(args.gpu)}')  if torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
    
            # Inference
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()
    
            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
    
    accuracy = correct/total
    if accuracy > best_global_acc:
            # print('Saving..')
            # state = {
            #     'net': net.state_dict(),
            #     'acc': acc,
            #     'epoch': epoch,
            # }
            # if not os.path.isdir('checkpoints'):
            #     os.mkdir('checkpoints')
            # torch.save(state, './checkpoints/ckpt_best_global.pth')
            best_global_acc = accuracy
    print(f'test-full accuracy:{100*accuracy:.2f}%, loss:{loss:.4f}, correct:{correct:.0f}, total:{total:.0f}, best_global_acc:{100*best_global_acc:.2f}%')
    return accuracy, loss
