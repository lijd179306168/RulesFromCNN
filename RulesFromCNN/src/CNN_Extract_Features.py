# -*- coding:utf-8 -*-  
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets,transforms, models

import os
import argparse
import csv
import numpy as np
from VGGClass import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = torch.tensor(output,requires_grad=True).cuda()
    def close(self):
        self.hook.remove()
layer = 42
total_filters_in_layer = 512

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

net=torch.load('vggnet.pkl')
print(net)


activations = SaveFeatures(list(net.children())[0][42])

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), 0.001, momentum=0.9, weight_decay=5e-4)

net.eval()
test_loss = 0
correct = 0
total = 0

f= open('cifar10train_50000_512.csv','w',newline='')
f_csv = csv.writer(f)

with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(trainloader):   #change to testloader for extracting features from validate set
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        mean_act = [activations.features[0,i].mean().item() for i in range(total_filters_in_layer)]
        
        print(mean_act)
        targetsData=targets.data.cpu().numpy()[0]
        predictedData=predicted.data.cpu().numpy()[0]
        
        row=np.append(targets.data.cpu().numpy()[0], mean_act)
        f_csv.writerow(row)
        
        print(batch_idx+1, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


