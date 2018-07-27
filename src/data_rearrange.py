import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import matplotlib.pyplot as plt

import sys
import random
from scipy import io
import numpy as np


X = np.loadtxt('../data/U_push_contours.txt')
X = X.astype(float)

Y = np.loadtxt('../data/Y_push.txt')
Y = Y.astype(float)

h,w = X.shape

