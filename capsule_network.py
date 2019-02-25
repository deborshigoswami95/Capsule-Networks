"""
Dynamic Routing Between Capsules
https://arxiv.org/abs/1710.09829

PyTorch implementation by Kenta Iwasaki @ Gram.AI.
"""
import sys
sys.setrecursionlimit(15000)

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.autograd import Variable

from residual_capsule import CapsuleLayer



class CapsuleNet(nn.Module):
    def __init__(self,NUM_CLASSES=10, NUM_ITERATIONS = 3):
        super(CapsuleNet, self).__init__()
        self.num_classes = NUM_CLASSES
        self.num_iterations = NUM_ITERATIONS
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, 
                               kernel_size=7, stride=1)#CIFAR10 is 3 colored so 3 input channels
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, 
                               kernel_size=7, stride=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=6, stride=2, padding = 2)
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=4, stride=1)
        self.CIFAR_capsules = CapsuleLayer(num_capsules=self.num_classes, num_route_nodes=32 * 7 * 7, in_channels=8,
                                           out_channels=32, num_iterations = self.num_iterations) #what is num_route_nodes?

        self.decoder = nn.Sequential(
            nn.Linear(32 * self.num_classes, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 3072),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.CIFAR_capsules(x).squeeze().transpose(0, 1)

        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)

        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes.max(dim=1)
            y = Variable(torch.eye(self.num_classes)).cuda().index_select(dim=0, index=max_length_indices.data)

        reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))

        return classes, reconstructions