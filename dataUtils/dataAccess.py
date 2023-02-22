import torchvision
import torch.nn as nn
from torchinfo import summary
import torchvision.transforms as transforms
import cv2
import torch
import numpy as np


def getCIFAR10Data(transform_Train,transform_Test,Args):
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_Train)
    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=Args.train_batch_size,
                                              shuffle=True, num_workers=Args.num_workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_Test)
    testLoader = torch.utils.data.DataLoader(testset, batch_size=Args.val_batch_size,
                                             shuffle=False, num_workers=Args.num_workers)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainLoader,testLoader,classes