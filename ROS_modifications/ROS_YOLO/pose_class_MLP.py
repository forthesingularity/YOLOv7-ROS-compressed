from copy import deepcopy
import os
import sys
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm
import torchsummary

class linear(nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()
    self.linear = nn.Linear(in_features, out_features)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.linear(x)
    x = self.relu(x)
    return x

class pose_class_MLP(nn.Module):
  def __init__(self):
    super().__init__()
    self.net1 = linear(51, 1000)
    self.net2 = linear(1000, 256)
    self.net3 = nn.Linear(256, 8)

  def forward(self, x):
    x = x.view(-1)
    x = self.net1(x)
    x = self.net2(x)
    x = self.net3(x)
    return x
  
