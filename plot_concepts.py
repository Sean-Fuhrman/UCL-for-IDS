#%%
import sys

import torch
import numpy
import torchvision
import dataloader

train_x, train_y, test_x, test_y = dataloader.get_dataset_MNIST(attack_classes=[0,1,2,3,4], normal_classes=[5,6,7,8,9], config=None)
#%%
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
#%%
from clustering import create_concepts
# train_x = train_x[:50]
concepts = create_concepts(train_x, concepts_no=5, size_per_concept=5000)

#%%
print(concepts[0].shape)
print(len(concepts))
# %%
