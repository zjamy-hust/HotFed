import matplotlib.pyplot as plt
import numpy as np

# %matplotlib inline

import torch
import captum
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as fn
from statistics import mean

from torchvision import models

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
import os
import matplotlib.image as img
import torch.optim as optim
import torch.optim as optim
from localupdates.update import test_inference
from options import args_parser
import copy
args = args_parser()
args.gpu=3
device = (f'cuda:{str(args.gpu)}')  if torch.cuda.is_available() else 'cpu'
# showimg=1


model_path="/workspace/externalhome/XAI/HotFed/save_checkpoints/xai_analysis/global.iid1.16_15.pth.tar"
asset_path='/workspace/externalhome/XAI/HotFed/assets'


def _cumulative_sum_threshold(values: ndarray, percentile: Union[int, float]):
    # given values should be non-negative
    assert percentile >= 0 and percentile <= 100, (
        "Percentile for thresholding must be " "between 0 and 100 inclusive."
    )
    sorted_vals = np.sort(values.flatten())
    cum_sums = np.cumsum(sorted_vals)
    threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
    return sorted_vals[threshold_id]


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




def XAI_evaluate(net_x,files, path, showimg,p,device, XAI_labels):
    XAI_inmask_list = []
    XAI_outmask_list = []
    i=0;
    correct=0;
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
            pixelnum_all=np.count_nonzero(truth_mask_np)
            if p:
                print("pixelnum_all", pixelnum_all)


            ig = IntegratedGradients(net_x)
            nt = NoiseTunnel(ig)
            input_asset=input_asset.to(device)
            #the 2nd parameter can be input_asset or input_asset_norm, input_asset_norm will show better ACC in XAI
            attr_ig_nt = attribute_image_features(net_x,nt, input_asset_norm,truth=XAI_labels[int(im_name[:-4])-1], label=predicted_asset[0], baselines=input_asset * 0, nt_type='smoothgrad_sq',  nt_samples=100, stdevs=0.2)
            attr_ig_nt = np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0))


            outlier_perc = 10
            attr_combined = np.sum(attr_ig_nt, axis=2)
            attr_combined = np.abs(attr_combined)
            threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
            attr_norm = attr_combined / threshold
            attr_norm=attr_norm*(attr_norm>0.3)

            masked = cv2.add(attr_norm, np.zeros(np.shape(attr_norm), dtype=float), mask=truth_mask) 
            out_mask = cv2.add(attr_norm, np.zeros(np.shape(attr_norm), dtype=float), mask=255-truth_mask) 

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
                attr.imshow(attr_norm,cmap=default_cmap,vmin=vmin,vmax=vmax)
                mask.imshow(truth_mask,cmap="Blues",vmin=0,vmax=1)
                attr.imshow(attr_norm,cmap=default_cmap,vmin=vmin,vmax=vmax)
                mask.imshow(truth_mask,cmap="Blues",vmin=0,vmax=1)
                attr_mask.imshow(masked,cmap="Greens",vmin=0,vmax=1)
                attr_outmask.imshow(out_mask,cmap="Reds",vmin=0,vmax=1)
                fig.show()
            # plt.close(fig)
    in_mask_acc_mean = mean(XAI_inmask_list)
    out_mask_acc_mean = mean(XAI_outmask_list)
    print("in_mask_acc_mean",in_mask_acc_mean,"out_mask_acc_mean",out_mask_acc_mean,"XAI ACC", correct/i)
    return in_mask_acc_mean,out_mask_acc_mean,correct/i


a,b,c = XAI_evaluate(net,files,assetpath,1,1,device=device,XAI_labels=XAI_labels)

print("a",a,"b",b,"c",c)
