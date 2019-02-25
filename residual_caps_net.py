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
    def __init__(self,NUM_CLASSES = 10, NUM_ITERATIONS = 3):
        super(CapsuleNet, self).__init__()
        self.num_classes = NUM_CLASSES
        self.num_iterations = NUM_ITERATIONS
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, 
                               kernel_size=7, stride=1)#CIFAR10 is 3 colored so 3 input channels, output = 26 x 26
        
        
        self.skip_capsule = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=4, stride=2)#output = 12 x 12
        
        
        self.residual_block = nn.Sequential(
                                                  nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 4, 
                                                              stride = 1, padding = 1),
                                                  nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 4, 
                                                              stride = 1, padding = 1),
                                                  CapsuleLayer(num_capsules=8, 
                                                               num_route_nodes=-1, in_channels=256, 
                                                               out_channels=32,kernel_size=4, 
                                                               stride=1)
                                                )
        
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

    def forward(self, data, y=None):
        data = F.relu(self.conv1(data), inplace=True)
        
        x1 = self.skip_capsule(data)
                
        #x1 = [i.view(data.size(0), -1, 1) for i in x1] #Magic
        #x1 = torch.cat(x1, dim=-1) #More Magic
        #x1 = self.skip_capsule.squash(x1) #Even More Magic
        
        skip_output = x1.permute(1,0,2,3,4)
        skip_output = skip_output.contiguous().view(skip_output.size()[0],-1,skip_output.size()[3],skip_output.size()[4])
        
        x2 = self.residual_block(skip_output)
        
        x1 = self.center_crop(x1, x2)
        
        
        x2 = x1 + x2
        
        x2 = x2.permute(1,2,3,4,0)
        x2 = x2.contiguous().view(x2.size()[0],-1,x2.size()[4])
        x2 = self.residual_block[2].squash(x2)
        
        x = self.CIFAR_capsules(x2).squeeze().transpose(0, 1)

        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)

        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes.max(dim=1)
            y = Variable(torch.eye(self.num_classes)).cuda().index_select(dim=0, index=max_length_indices.data)

        reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))

        return classes, reconstructions
    
    def center_crop(self,source, target):
        indim = source.size()[4]
        outdim = target.size()[4]
        a = int(((indim - outdim)/2))
        b = int(((indim + outdim)/2))
        return source[:,:,:,a:b,a:b]
    
        