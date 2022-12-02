import matplotlib.pyplot as plt
import numpy as np

# %matplotlib inline

import torch
import captum
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from torchvision import models

from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

from torchvision.utils import make_grid
from torchvision.io import read_image
from pathlib import Path
import torchvision.transforms.functional as fn

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


model_path="/workspace/externalhome/XAI/HotFed/save_checkpoints/xai_analysis/global.iid1.16_15.pth.tar"
data_path='/workspace/externalhome/XAI/data/'
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


def attribute_image_features(algorithm, input,truth, label, **kwargs):
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


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, in_planes, planes, stride=1):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
#                                stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, self.expansion *
#                                planes, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(self.expansion*planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


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


torch.cuda.set_device(args.gpu)

net = ResNet18()

apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

# trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                           shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                       download=True, transform=apply_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)
dataiter = iter(testloader)
_, labels_workaround = dataiter.next()

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



path = str(Path(asset_path)/'folder')
print(path)
mean = 120.707
std = 64.15
files = os.listdir(path)
d = []



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

checkpoint = torch.load(model_path)
net.load_state_dict(checkpoint['state_dict'])
net.to(device)
net.eval()






# test_inference(args, net, test_dataset=testset)



i=0;
# classes = ('plane' 0, 'car' 1, 'bird' 2, 'cat' 3,
#            'deer' 4, 'dog' 5, 'frog' 6, 'horse' 7, 'ship' 8, 'truck' 9)
XAI_labels=[7, 8, 2, 2, 0, 5, 7, 9, 2, 8, 8, 2, 8, 2, 5, 8, 0, 7, 5, 5,1,1,3,3,4,4,6,6,9,3 ]
for im_name in files:
    if im_name.endswith('.jpg'):
#     if im_name.endswith('.jpg') and i == 0:
        # if i>1:
        #     break
        i=i+1
        print(str(Path(path)/im_name))
        im_temp=read_image(str(Path(path)/im_name))
        
        fig, (orig, mask, attr, attr_mask, attr_outmask) = plt.subplots(1, 5)
        orig.axis('off')
        mask.axis('off')
        attr.axis('off')
        attr_mask.axis('off')
        attr_outmask.axis('off')


        original_image_temp = fn.resize(im_temp, size=[32,32])/255
        input_temp=torch.tensor(original_image_temp.unsqueeze(0).cpu().detach().numpy())
        input_temp.requires_grad = True
        img = original_image_temp     # unnormalize
        npimg = img.numpy()
        
        orig.imshow(np.transpose(npimg, (1, 2, 0)))
#         plt.axis('off')


        # # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        # import numpy as np
        # from PIL import Image
        # MEAN = torch.tensor(np.array([0.4914, 0.4822, 0.4465]))
        # STD = torch.tensor(np.array([0.2023, 0.1994, 0.2010]))
        # # img_pil = Image.open("ty.jpg")
        # # x = np.array(img_pil)
        # # input_temp = input_temp.transpose(-1, 0, 1)
        # # input_temp = (input_temp - MEAN) / STD
        # torch.nn.functional.normalize(input_temp,MEAN,STD)


#         original_image_temp = np.transpose(original_image_temp, (1, 2, 0))

        import torchvision.transforms.functional as fn
        input_temp_norm=fn.normalize(input_temp, mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

        input_temp_norm=input_temp_norm.to(device)
        output_temp = net(input_temp_norm)
        _, predicted_temp = torch.max(output_temp, 1)
        print("predicted_temp",classes[predicted_temp[0]],"Truth", classes[XAI_labels[int(im_name[:-4])-1]])
        
        mask_im_name = im_name[:-4]+'_mask.png'
        print("mask_im_name",mask_im_name)
        truth_mask_tensor = read_image(str(Path(path)/mask_im_name),mode=torchvision.io.image.ImageReadMode.RGB)
        truth_mask = cv2.imread(str(Path(path)/mask_im_name),cv2.IMREAD_GRAYSCALE)
        truth_mask_np=truth_mask_tensor.numpy()
        pixelnum_all=np.count_nonzero(truth_mask_np)
        print("pixelnum_all", pixelnum_all)


        ig = IntegratedGradients(net)
        nt = NoiseTunnel(ig)
        # labels_workaround[0] = XAI_labels[int(im_name[:-4])-1]
        # attr_ig_nt = attribute_image_features(nt, input_temp, label=predicted_temp[0], baselines=input_temp * 0, nt_type='smoothgrad_sq',  nt_samples=100, stdevs=0.2)
        input_temp=input_temp.to(device)
        attr_ig_nt = attribute_image_features(nt, input_temp,truth=XAI_labels[int(im_name[:-4])-1], label=predicted_temp[0], baselines=input_temp * 0, nt_type='smoothgrad_sq',  nt_samples=100, stdevs=0.2)
        attr_ig_nt = np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0))


        outlier_perc = 10
        attr_combined = np.sum(attr_ig_nt, axis=2)
        attr_combined = np.abs(attr_combined)
        threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
        attr_norm = attr_combined / threshold
        attr_norm=attr_norm*(attr_norm>0.3)
        default_cmap = LinearSegmentedColormap.from_list(
            "RdWhGn", ["red", "white", "green"]
        )
        vmin, vmax = -1, 1
        attr.imshow(attr_norm,cmap=default_cmap,vmin=vmin,vmax=vmax)
        mask.imshow(truth_mask,cmap="Blues",vmin=0,vmax=1)

        masked = cv2.add(attr_norm, np.zeros(np.shape(attr_norm), dtype=float), mask=truth_mask) 
        out_mask = cv2.add(attr_norm, np.zeros(np.shape(attr_norm), dtype=float), mask=255-truth_mask) 


        attr_mask.imshow(masked,cmap="Greens",vmin=0,vmax=1)
        pixelnum=np.count_nonzero(masked)
        print("in mask pixelnum", pixelnum,pixelnum_all,pixelnum/pixelnum_all)
        out_pixelnum=np.count_nonzero(out_mask)
        print("out mask pixelnum", out_pixelnum,3*32*32-pixelnum_all, out_pixelnum/(3*32*32-pixelnum_all))
        attr_outmask.imshow(out_mask,cmap="Reds",vmin=0,vmax=1)
        fig.show()




# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# checkpoint = torch.load(model_path)
# net.load_state_dict(checkpoint['state_dict'])




# dataiter = iter(testloader)
# images, labels = dataiter.next()

# # print images
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(32)))


# outputs = net(images)

# _, predicted = torch.max(outputs, 1)

# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                               for j in range(32)))



# dog1_int = read_image(str(Path(asset_path) / 'dog1_32.jpg'))

# original_image_temp = fn.resize(dog1_int, size=[32,32])/255

# dog1_mask = read_image(str(Path(asset_path) / 'dog1_32_mask.png'),mode=torchvision.io.image.ImageReadMode.RGB)

# input_temp=torch.tensor(original_image_temp.unsqueeze(0).float())
# input_temp.requires_grad = True

# plt.rcParams["savefig.bbox"] = 'tight'


# output_temp = net(input_temp)

# _, predicted_temp = torch.max(output_temp, 1)
# print(predicted_temp)

# net.eval()


        

# saliency = Saliency(net)
# # grads = saliency.attribute(input, target=labels[ind].item())
# grads = saliency.attribute(input_temp, target=5)
# grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))


# ig = IntegratedGradients(net)
# ind=0
# labels[ind]=5
# attr_ig, delta = attribute_image_features(ig, input_temp, baselines=input_temp * 0, return_convergence_delta=True)
# attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
# print('Approximation delta: ', abs(delta))


# ig = IntegratedGradients(net)
# nt = NoiseTunnel(ig)
# attr_ig_nt = attribute_image_features(nt, input_temp, baselines=input_temp * 0, nt_type='smoothgrad_sq',
#                                       nt_samples=100, stdevs=0.2)
# attr_ig_nt = np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

# dl = DeepLift(net)
# attr_dl = attribute_image_features(dl, input_temp, baselines=input_temp * 0)
# attr_dl = np.transpose(attr_dl.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

# print('Original Image')
# print('Predicted:', classes[predicted_temp[0]], 
#       ' Probability:', torch.max(F.softmax(outputs, 1)).item())

# original_image = np.transpose((original_image_temp.cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))

# _ = viz.visualize_image_attr(None, original_image, 
#                       method="original_image", title="Original Image")

# _ = viz.visualize_image_attr(grads, original_image, method="heat_map", sign="absolute_value",
#                           show_colorbar=True, title="Overlayed Gradient Magnitudes")

# _ = viz.visualize_image_attr(attr_ig, original_image, method="heat_map",sign="all",
#                           show_colorbar=True, title="Overlayed Integrated Gradients")

# _ = viz.visualize_image_attr(attr_ig_nt, original_image, method="heat_map", sign="absolute_value", 
#                              outlier_perc=10, show_colorbar=True, 
#                              title="Overlayed Integrated Gradients \n with SmoothGrad Squared")

# _ = viz.visualize_image_attr(attr_dl, original_image, method="heat_map",sign="all",show_colorbar=True, 
#                           title="Overlayed DeepLift")




# dog1_mask_tensor = read_image(str(Path(asset_path) / 'dog1_32_mask.png'),mode=torchvision.io.image.ImageReadMode.RGB)
# dog1_mask = cv2.imread(str(Path(asset_path) / 'dog1_32_mask.png'),cv2.IMREAD_GRAYSCALE)
# dog1_mask_np=dog1_mask_tensor.numpy()
# pixelnum_all=np.count_nonzero(dog1_mask_np)
# print("pixelnum_all", pixelnum_all)



# outlier_perc = 10
# attr_combined = np.sum(attr_ig_nt, axis=2)
# attr_combined = np.abs(attr_combined)
# threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
# attr_norm = attr_combined / threshold
# attr_norm=attr_norm*(attr_norm>0.7)
# default_cmap = LinearSegmentedColormap.from_list(
#     "RdWhGn", ["red", "white", "green"]
# )
# vmin, vmax = -1, 1
# plt.imshow(attr_norm,cmap=default_cmap,vmin=vmin,vmax=vmax)
# plt.show()
# plt.hist(attr_norm)
# print(attr_norm)
# plt.show()
# plt.imshow(dog1_mask,cmap=default_cmap,vmin=vmin,vmax=vmax)
# plt.show()

# masked = cv2.add(attr_norm, np.zeros(np.shape(attr_norm), dtype=float), mask=dog1_mask) 
# masked1 = cv2.add(attr_norm, np.zeros(np.shape(attr_norm), dtype=float), mask=255-dog1_mask) 


# plt.hist(masked)
# plt.show()
# plt.hist(masked1)
# plt.show()
# plt.imshow(masked,cmap=default_cmap,vmin=vmin,vmax=vmax)
# plt.show()
# pixelnum=np.count_nonzero(masked)
# print("masked pixelnum", pixelnum,pixelnum/pixelnum_all)
# pixelnum=np.count_nonzero(masked1)
# print("masked1 pixelnum", pixelnum/(3*32*32-pixelnum_all))
# plt.imshow(masked1,cmap=default_cmap,vmin=vmin,vmax=vmax)
# plt.show()
