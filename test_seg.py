from __future__ import print_function
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
from show3d_balls import *

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--nepoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg',  help='output folder')
parser.add_argument('--model', type=str, default= './seg/seg_model_29_0.810.pth',  help='model path')


opt = parser.parse_args()
print (opt)

opt.manualSeed = random.randint(1, 2500) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

num_points = 2700

test_dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', npoints=num_points, classification=False, class_choice=['tools'], train=False)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))
print(len(test_dataset))

num_classes = 10
print('classes', num_classes)
try:
    os.makedirs(opt.outf)
except OSError:
    pass

blue = lambda x:'\033[94m' + x + '\033[0m'


classifier = PointNetDenseCls(num_points=num_points, k=num_classes)
classifier.load_state_dict(torch.load(opt.model))
# classifier.cuda()
classifier.eval()

num_test_batch = len(test_dataset)/opt.batchSize

cmap = plt.cm.get_cmap("hsv", 5)
cmap = np.array([cmap(i) for i in range(10)])[:,:3]

correct_percents = []
for i, data in enumerate(testdataloader, 0):
    points_np, target = data
    points, target = Variable(points_np), Variable(target)
    points = points.transpose(2, 1)
    # points, target = points.cuda(), target.cuda()

    pred, _ = classifier(points)
    pred = pred.view(-1, num_classes)
    target = target.view(-1,1)[:,0] - 1

    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    correct_percent = correct.item()/float(list(target.shape)[0])
    correct_percents.append(correct_percent)
    print('[%d/%d] accuracy: %f' %(i, num_test_batch, correct_percent))

    pred_color = cmap[pred_choice.numpy()[0], :]
    showpoints(points_np, None, pred_color, ballradius=4)
average_correct_percent = np.sum(correct_percents) / len(correct_percents)
print('Average accuracy: %f' % (correct_percent))
