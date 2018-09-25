from __future__ import print_function
from show3d_balls import *
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets import PartDataset
from pointnet import PointNetDenseCls
import torch.nn.functional as F
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='./seg/seg_model_1.pth', help='model path')
parser.add_argument('--idx', type=int, default=0, help='model index')

opt = parser.parse_args()
print (opt)


num_points = 2700
d = PartDataset(root='shapenetcore_partanno_segmentation_benchmark_v0', npoints=num_points, class_choice=['tools'], train=False)

idx = opt.idx

print("model %d/%d" %( idx, len(d)))

point_np, seg = d[idx]
point = torch.from_numpy(point_np)
point_np[:, 2] *= -1

cmap = plt.cm.get_cmap("hsv", 5)
cmap = np.array([cmap(i) for i in range(10)])[:,:3]
gt = cmap[seg - 1, :]

classifier = PointNetDenseCls(num_points=num_points, k=10)
classifier.load_state_dict(torch.load(opt.model))
classifier.eval()

point = point.transpose(1,0).contiguous()

point = Variable(point.view(1, point.size()[0], point.size()[1]))
pred, _ = classifier(point)
pred_choice = pred.data.max(2)[1]
print(pred_choice)

#print(pred_choice.size())
pred_color = cmap[pred_choice.numpy()[0], :]

#print(pred_color.shape)
showpoints(point_np, gt, pred_color, ballradius=4)

