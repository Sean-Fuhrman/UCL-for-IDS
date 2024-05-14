#%%
import sys

import torch
import numpy
import torchvision
from torchvision import datasets, transforms
from prepare_scenario import prepare_and_save_scenario, prepare_scenario


DEFAULT_CONFIG = {
    'batch_size': 32,  #expand with more as needed
}

class datastream:
    def __init__(self, train_x, train_y, test_x, test_y):
        self.initial_normal = None
        self.datastream = []


def get_datastream_Edge_IIoT( config=None):
    pass

def get_datastream_X_IIoT( config=None):
    pass

def get_dataset_MNIST(attack_classes=[0,1,2,3,4], normal_classes=[5,6,7,8,9], config=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for i in range(len(train_dataset)):
        if train_dataset[i][1] in normal_classes:
            train_x.append(train_dataset[i][0])
            train_y.append(0)
        elif train_dataset[i][1] in attack_classes:
            train_x.append(train_dataset[i][0])
            train_y.append(1)

    for i in range(len(test_dataset)):
        if test_dataset[i][1] in normal_classes:
            test_x.append(test_dataset[i][0])
            test_y.append(0)
        elif test_dataset[i][1] in attack_classes:
            test_x.append(test_dataset[i][0])
            test_y.append(1)

    train_x = torch.stack(train_x)
    test_x = torch.stack(test_x)
    train_y = torch.tensor(train_y)
    test_y = torch.tensor(test_y)
    return train_x.view(train_x.shape[0], -1), train_y, test_x.view(test_x.shape[0], -1), test_y

def get_datastream_MNIST(attack_classes=[0,1,2,3,4], normal_classes=[5,6,7,8,9], config=None):
    train_x, train_y, test_x, test_y = get_dataset_MNIST(attack_classes, normal_classes, config)
    return datastream(train_x, train_y, test_x, test_y, config)


def generated_scenario(datastream, config):
    pass


    

#%%