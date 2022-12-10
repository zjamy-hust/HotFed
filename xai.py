import matplotlib.pyplot as plt
import numpy as np
import math

# %matplotlib inline

import torch
import captum
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as fn
from torchvision import models
from statistics import mean



from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

from torchvision.utils import make_grid
from torchvision.io import read_image
from pathlib import Path

from numpy import ndarray
from typing import Any, Iterable, List, Tuple, Union
from matplotlib.colors import LinearSegmentedColormap
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import matplotlib.image as img
import torch.optim as optim
import torch.optim as optim
from localupdates.update import test_inference, DatasetSplit, generate_dataset_mask
from options import args_parser
import copy
args = args_parser()
args.gpu=1
device = (f'cuda:{str(args.gpu)}')  if torch.cuda.is_available() else 'cpu'
# showimg=1
run=1


model_path="save_checkpoints/xai_analysis/global.iid1.16_15.pth.tar"
asset_path='assets'


# def _cumulative_sum_threshold(values: ndarray, percentile: Union[int, float]):
#     # given values should be non-negative
#     assert percentile >= 0 and percentile <= 100, (
#         "Percentile for thresholding must be " "between 0 and 100 inclusive."
#     )
#     sorted_vals = np.sort(values.flatten())
#     cum_sums = np.cumsum(sorted_vals)
#     threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
#     return sorted_vals[threshold_id]

def attribute_image_features(net, algorithm, input,truth, label, **kwargs):
    net.zero_grad()
    tensor_attributions = algorithm.attribute(input,
                                              target=truth,
                                              **kwargs
                                             )
    return tensor_attributions


def imshow(img, transpose = True):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def XAI_evaluate(net_x,files, path, showimg,p,device, XAI_labels,classes):
    torch.cuda.empty_cache()
    XAI_inmask_list = []
    XAI_outmask_list = []
    i=0;
    correct=0;
    ig = IntegratedGradients(net_x)
    nt = NoiseTunnel(ig)
    for im_name in files:
        if im_name.endswith('.jpg'):
            # if i>1:
            #     break
            i=i+1
            if p:
                print(str(Path(path)/im_name))
            im_asset=read_image(str(Path(path)/im_name))

            original_image_asset = fn.resize(im_asset, size=[32,32])/255
            input_asset=torch.tensor(original_image_asset.unsqueeze(0).cpu().detach().numpy())
            input_asset.requires_grad = True
            img = original_image_asset     # unnormalize
            npimg = img.numpy()

            input_asset_norm=fn.normalize(input_asset, mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

            input_asset_norm=input_asset_norm.to(device)
            output_asset = net_x(input_asset_norm)
            _, predicted_asset = torch.max(output_asset, 1)
            if p:
                print("predicted_asset",classes[predicted_asset[0]],"Truth", classes[XAI_labels[int(im_name[:-4])-1]])

            mask_im_name = im_name[:-4]+'_mask.png'
            if p:
                print("mask_im_name",mask_im_name)
            truth_mask_tensor = read_image(str(Path(path)/mask_im_name),mode=torchvision.io.image.ImageReadMode.RGB)
            truth_mask = cv2.imread(str(Path(path)/mask_im_name),cv2.IMREAD_GRAYSCALE)
            truth_mask_np=truth_mask_tensor.numpy()
            # pixelnum_all=np.count_nonzero(truth_mask_np)      #不应该计算3通道
            pixelnum_all=np.count_nonzero(truth_mask)
            if p:
                print("pixelnum_all", pixelnum_all)

            input_asset=input_asset.to(device)
            torch.cuda.empty_cache()
            #the 2nd parameter can be input_asset or input_asset_norm, input_asset_norm will show better ACC in XAI
            attr_ig_nt = attribute_image_features(net_x,
                                                  nt, 
                                                  input_asset_norm,
                                                  truth=XAI_labels[int(im_name[:-4])-1], 
                                                  label=predicted_asset[0], 
                                                  baselines=input_asset * 0, 
                                                  nt_type='smoothgrad_sq', 
                                                  nt_samples=50, 
                                                  n_steps=50, 
                                                  stdevs=0.2)
            attr_ig_nt = np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
            
            #计算二值化masks
            topk=0.6
            attr_combined = np.sum(attr_ig_nt, axis=2)/3
            # attr_combined = np.abs(attr_combined)
            attr_combined_flatten_sorted = np.sort(attr_combined.flatten())
            attr_combined_flatten_sorted = (attr_combined_flatten_sorted-np.min(attr_combined_flatten_sorted))/(np.max(attr_combined_flatten_sorted)-np.min(attr_combined_flatten_sorted))      #进行归一化操作。
            threshold_idx = math.ceil(topk * attr_combined_flatten_sorted.shape[0])
            threshold = attr_combined_flatten_sorted[attr_combined_flatten_sorted.shape[0] - threshold_idx]
            
            attr_hard_masks = (attr_combined > threshold).astype(float)

            masked = cv2.add(attr_hard_masks, np.zeros(np.shape(attr_hard_masks), dtype=float), mask=truth_mask) 
            out_mask = cv2.add(attr_hard_masks, np.zeros(np.shape(attr_hard_masks), dtype=float), mask=255-truth_mask) 

            inmask_pixelnum=np.count_nonzero(masked)
            inmask_percent=inmask_pixelnum/pixelnum_all
            if p:
                print("in mask pixelnum", inmask_pixelnum,pixelnum_all,inmask_percent)
            XAI_inmask_list.append(inmask_percent)
            out_pixelnum=np.count_nonzero(out_mask)
            outmask_percent=out_pixelnum/(3*32*32-pixelnum_all)
            if p:
                print("out mask pixelnum", out_pixelnum,3*32*32-pixelnum_all, outmask_percent)
            XAI_outmask_list.append(outmask_percent)

            if inmask_percent > outmask_percent:
                correct = correct+1
            
            torch.cuda.empty_cache()
            if showimg==1 and i%2 == 1:
                fig, (orig, mask, attr, attr_mask, attr_outmask) = plt.subplots(1, 5)
                orig.axis('off')
                mask.axis('off')
                attr.axis('off')
                attr_mask.axis('off')
                attr_outmask.axis('off')
                orig.imshow(np.transpose(npimg, (1, 2, 0)))
                default_cmap = LinearSegmentedColormap.from_list(
                    "RdWhGn", ["red", "white", "green"]
                )
                vmin, vmax = -1, 1
                attr.imshow(attr_hard_masks,cmap=default_cmap,vmin=vmin,vmax=vmax)
                mask.imshow(truth_mask,cmap="Blues",vmin=0,vmax=1)
                attr.imshow(attr_hard_masks,cmap=default_cmap,vmin=vmin,vmax=vmax)
                mask.imshow(truth_mask,cmap="Blues",vmin=0,vmax=1)
                attr_mask.imshow(masked,cmap="Greens",vmin=0,vmax=1)
                attr_outmask.imshow(out_mask,cmap="Reds",vmin=0,vmax=1)
                fig.show()
            # plt.close(fig)
            
    in_mask_acc_mean = mean(XAI_inmask_list)
    out_mask_acc_mean = mean(XAI_outmask_list)
    print("in_mask_acc_mean",in_mask_acc_mean,"out_mask_acc_mean",out_mask_acc_mean,"XAI ACC", correct/i)
    return in_mask_acc_mean,out_mask_acc_mean,correct/i

def XAI_evaluate_with_global_masks(local_model, 
                                   test_files_list, 
                                   path, 
                                   dataset_name,
                                   device, 
                                   XAI_labels, 
                                   classes, 
                                   nt_samples, 
                                   n_steps, 
                                   margin, 
                                   topk=0.5, 
                                   compare_sever_client_masks=0, 
                                   global_model=None, 
                                   batch_size=1,
                                   output_path="./",
                                   verbose=1):
    """     重写XAI_evaluate，使其能够根据global model产生mask，从而实现global model和local model产生的mask进行对比
    
    与原始XAI_evaluate之间的差别：
        1、采用DataSplit加载测试样本

    Args:
        local_model (_type_): client model
        test_files_list (_type_): 人工挑选的测试样本列表，list。
        path (_type_): 测试样本的目录
        device (_type_): 
        XAI_labels (_type_): 测试样本的label index。
        classes (_type_): 测试样本的label名称。
        nt_samples
        n_steps
        margin： in_mask应当比out_mask大margin，防止噪声导致acc波动。
        topk
        compare_sever_client_masks (_type_): 是否进行client和global之间的对比
        global_model (_type_, optional): Server model. Defaults to None.
        batch_size: 测试和生成mask时的batch_size，为了计算的准确度，batch_size尽可能小，并且nt_samples和n_steps尽可能大。

    Returns:
        _type_: _description_
    """
    
    test_files_list_jpg = sorted(filter(lambda x: x.endswith(".jpg"), test_files_list))
    if len(test_files_list_jpg) <= 0:
        raise ValueError("test_files_list不包含jpg文件。")
    if len(test_files_list_jpg) != len(XAI_labels):
        raise ValueError("长度不匹配。")
    
    torch.cuda.empty_cache()
    
    
    test_images = []
    npimg_list = []
    image_masks_by_human = []
    for im_idx in range(len(test_files_list_jpg)):
        im_name = test_files_list_jpg[im_idx]
        if verbose == 1:
            print(str(Path(path)/im_name))
        
        if dataset_name == "cifar10":
            dataset_size = 32
            normalize_mean = [0.4914, 0.4822, 0.4465]
            normalize_std = [0.2023, 0.1994, 0.2010]
            mode = torchvision.io.image.ImageReadMode.RGB
        elif dataset_name == "MNIST":
            dataset_size = 28
            normalize_mean = [0.5]
            normalize_std = [0.5]
            mode = torchvision.io.image.ImageReadMode.GRAY
        else:
            raise ValueError("dataset_name有误。")
        
        im_asset=read_image(str(Path(path)/im_name), mode=mode)
        original_image_asset = fn.resize(im_asset, size=[dataset_size,dataset_size])/255
        
        input_asset=torch.tensor(original_image_asset.unsqueeze(0).cpu().detach().numpy())
        input_asset.requires_grad = True
        
        input_asset_norm=fn.normalize(input_asset, mean=normalize_mean, std=normalize_std) #归一化，用来参与计算
        test_images.append((input_asset_norm.squeeze(dim=0),XAI_labels[im_idx]))
        
        npimg = original_image_asset.numpy()     # unnormalize，用来直接输出原图
        npimg_list.append(npimg)
        
        mask_im_name = im_name[:-4]+'_mask.png'
        if verbose == 1:
            print("mask_im_name",mask_im_name)
        truth_mask_tensor = read_image(str(Path(path)/mask_im_name),mode=torchvision.io.image.ImageReadMode.RGB)
        truth_mask = cv2.imread(str(Path(path)/mask_im_name),cv2.IMREAD_GRAYSCALE)
        truth_mask_np=truth_mask_tensor.numpy()         #为什么是3通道？？？？？？？？？
        
        image_masks_by_human.append((im_name, truth_mask_np, truth_mask))       #貌似truth_mask_np用不上？？？？？？？？？
        
    test_images_dataloader = DataLoader(DatasetSplit(test_images, [i for i in range(len(test_files_list_jpg))]), 
                                        batch_size=batch_size, 
                                        shuffle=False)
    
    if compare_sever_client_masks == 1:
        test_images_masks_by_local_model = generate_dataset_mask(local_model, 
                                                            test_images, 
                                                            [i for i in range(len(test_files_list_jpg))], 
                                                            batch_size, 
                                                            nt_samples, 
                                                            n_steps, 
                                                            device, 
                                                            topk)
        torch.cuda.empty_cache()
        test_images_masks_by_global_model = generate_dataset_mask(global_model, 
                                                                test_images, 
                                                                [i for i in range(len(test_files_list_jpg))], 
                                                                batch_size, 
                                                                nt_samples, 
                                                                n_steps, 
                                                                device, 
                                                                topk)
        torch.cuda.empty_cache()
    
    XAI_inmask_list = []
    XAI_outmask_list = []
    i=0;
    correct=0;
    ig = IntegratedGradients(local_model)
    nt = NoiseTunnel(ig)
    for batch_idx, (images, labels, idxs) in enumerate(test_images_dataloader):
        input_asset_norm=images.to(device)
        examples_num = input_asset_norm.shape[0]
        
        output_asset = local_model(input_asset_norm)
        _, predicted_asset = torch.max(output_asset, 1)
        

        for i in range(examples_num):   #分别处理每一个样本
            if verbose == 1:
                print("predicted_asset",classes[predicted_asset[i]],"Truth", classes[labels[i]])
            example_index_in_all = batch_idx * test_images_dataloader.batch_size + i
            if verbose == 1:
                print("example index:", example_index_in_all)
            pixelnum_all=np.count_nonzero(image_masks_by_human[example_index_in_all][2])
            if verbose == 1:
                print("pixelnum_all", pixelnum_all)

            #the 2nd parameter can be input_asset or input_asset_norm, input_asset_norm will show better ACC in XAI
            attr_ig_nt = attribute_image_features(local_model,
                                                nt, 
                                                input_asset_norm[i].unsqueeze(0),
                                                truth=XAI_labels[int(image_masks_by_human[example_index_in_all][0][:-4])-1], 
                                                label=predicted_asset[example_index_in_all], 
                                                baselines=input_asset_norm[i].unsqueeze(0) * 0, 
                                                nt_type='smoothgrad_sq',  
                                                nt_samples=nt_samples, 
                                                n_steps=n_steps, 
                                                stdevs=0.2)
            attr_ig_nt = np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
            
            #计算二值化masks

            attr_combined = np.sum(attr_ig_nt, axis=2)/3
            # attr_combined = np.abs(attr_combined)
            attr_combined_flatten_sorted = np.sort(attr_combined.flatten())
            threshold_idx = math.ceil(topk * attr_combined_flatten_sorted.shape[0])
            threshold = attr_combined_flatten_sorted[attr_combined_flatten_sorted.shape[0] - threshold_idx]
            
            attr_hard_masks = (attr_combined >= threshold).astype(float)

            truth_mask = image_masks_by_human[example_index_in_all][2]
            masked = cv2.add(attr_hard_masks, np.zeros(np.shape(attr_hard_masks), dtype=float), mask=truth_mask) 
            out_mask = cv2.add(attr_hard_masks, np.zeros(np.shape(attr_hard_masks), dtype=float), mask=255-truth_mask) 

            inmask_pixelnum=np.count_nonzero(masked)
            inmask_percent=inmask_pixelnum/pixelnum_all
            if verbose == 1:
                print("in mask pixelnum", inmask_pixelnum,pixelnum_all,inmask_percent)
            XAI_inmask_list.append(inmask_percent)
            out_pixelnum=np.count_nonzero(out_mask)
            outmask_percent=out_pixelnum/(32*32-pixelnum_all)
            if verbose == 1:
                print("out mask pixelnum", out_pixelnum,32*32-pixelnum_all, outmask_percent)
            XAI_outmask_list.append(outmask_percent)

            if inmask_percent > outmask_percent + margin:
                correct = correct+1
            
            torch.cuda.empty_cache()
            fig, (orig, mask, attr, attr_mask, attr_outmask) = plt.subplots(1, 5)
            orig.axis('off')
            mask.axis('off')
            attr.axis('off')
            attr_mask.axis('off')
            attr_outmask.axis('off')
            orig.imshow(np.transpose(npimg_list[example_index_in_all], (1, 2, 0)))
            default_cmap = LinearSegmentedColormap.from_list(
                "RdWhGn", ["red", "white", "green"]
            )
            vmin, vmax = -1, 1
            attr.imshow(attr_hard_masks,cmap=default_cmap,vmin=vmin,vmax=vmax)
            mask.imshow(truth_mask,cmap="Blues",vmin=0,vmax=1)
            attr.imshow(attr_hard_masks,cmap=default_cmap,vmin=vmin,vmax=vmax)
            mask.imshow(truth_mask,cmap="Blues",vmin=0,vmax=1)
            attr_mask.imshow(masked,cmap="Greens",vmin=0,vmax=1)
            attr_outmask.imshow(out_mask,cmap="Reds",vmin=0,vmax=1)
            fig.show()
            fig.savefig(output_path+image_masks_by_human[example_index_in_all][0][:-4]+"_result.jpg")
            
            #显示globale和local的mask
            if compare_sever_client_masks == 1:
                fig, (orig, local_mask_, global_mask_) = plt.subplots(1, 3)
                orig.axis('off')
                mask.axis('off')
                attr.axis('off')
                attr_mask.axis('off')
                attr_outmask.axis('off')
                orig.imshow(np.transpose(npimg_list[example_index_in_all], (1, 2, 0)))
                local_mask_.imshow(test_images_masks_by_local_model[example_index_in_all][0].cpu().data.numpy())
                global_mask_.imshow(test_images_masks_by_global_model[example_index_in_all][0].cpu().data.numpy())
                fig.savefig(output_path+image_masks_by_human[example_index_in_all][0][:-4]+"_compare_global_local.jpg")
            
            plt.close()

            
    in_mask_acc_mean = mean(XAI_inmask_list)
    out_mask_acc_mean = mean(XAI_outmask_list)
    if verbose == 1:
        print("in_mask_acc_mean",in_mask_acc_mean,"out_mask_acc_mean",out_mask_acc_mean,"XAI ACC", correct/i)
    return in_mask_acc_mean,out_mask_acc_mean,correct/i


if __name__=="__main__":
    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_planes, planes, stride=1):
            super(BasicBlock, self).__init__()
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out

    class ResNet(nn.Module):
        def __init__(self, block, num_blocks, num_classes=100):
            super(ResNet, self).__init__()
            self.in_planes = 64

            self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.linear = nn.Linear(512*block.expansion, num_classes)

        def _make_layer(self, block, planes, num_blocks, stride):
            strides = [stride] + [1]*(num_blocks-1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out

    def ResNet18():
        return ResNet(BasicBlock, [2, 2, 2, 2])


    # To prepare the XAI assets
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # classes = ('plane' 0, 'car' 1, 'bird' 2, 'cat' 3,
    #            'deer' 4, 'dog' 5, 'frog' 6, 'horse' 7, 'ship' 8, 'truck' 9)
    XAI_labels=[7, 8, 2, 2, 0, 5, 7, 9, 2, 8, 8, 2, 8, 2, 5, 8, 0, 7, 5, 5,1,1,3,3,4,4,6,6,9,3 ]
    assetpath = str(Path(asset_path)/'folder')
    print("assetpath",assetpath)
    files = os.listdir(assetpath)

    #To prepare network
    torch.cuda.set_device(args.gpu)
    net = ResNet18()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['state_dict'])
    net.to(device)
    net.eval()

    a,b,c = XAI_evaluate(net,files,assetpath,1,1,device=device,XAI_labels=XAI_labels, classes=classes)

    print("a",a,"b",b,"c",c)
