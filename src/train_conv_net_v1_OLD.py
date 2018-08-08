# -*- coding: utf-8 -*-
"""
Neural Networks
===============

Neural networks can be constructed using the ``torch.nn`` package.

Now that you had a glimpse of ``autograd``, ``nn`` depends on
``autograd`` to define models and differentiate them.
An ``nn.Module`` contains layers, and a method ``forward(input)``\ that
returns the ``output``.

For example, look at this network that classifies digit images:

.. figure:: /_static/img/mnist.png
   :alt: convnet

   convnet

It is a simple feed-forward network. It takes the input, feeds it
through several layers one after the other, and then finally gives the
output.

A typical training procedure for a neural network is as follows:

- Define the neural network that has some learnable parameters (or
  weights)
- Iterate over a dataset of inputs
- Process input through the network
- Compute the loss (how far is the output from being correct)
- Propagate gradients back into the network’s parameters
- Update the weights of the network, typically using a simple update rule:
  ``weight = weight - learning_rate * gradient``

Define the network
------------------

Let’s define this network:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import time
from torch.autograd import Variable
import torch.optim as optim

import matplotlib.pyplot as plt

import sys
import random
from scipy import io
import numpy as np
from numpy import zeros, newaxis

from tensorboardX import SummaryWriter

writer = SummaryWriter()


model = nn.Sequential(nn.Conv1d(1,1,12), nn.ReLU(),
                       nn.Linear(29,100), nn.ReLU(),
                       nn.Linear(100,4))
'''
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # kernel
        self.conv1 = nn.Conv1d(1,1,12)
        self.fc1 = nn.Linear(29, 100)
        self.fc2 = nn.Linear(100, 4)
       
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
'''

X = np.loadtxt('../data/U_push_contours_V3.txt')
X = X.astype(float)

Y = np.loadtxt('../data/Y_push_V3.txt')
Y = Y.astype(float)

X = X[:, newaxis, :]
Y = Y[:, newaxis, :]

msk = np.random.rand(len(X)) < 0.9
X_train = X[msk]
Y_train = Y[msk]

X_test = X[~msk]
Y_test = Y[~msk]

X_train = Variable(torch.from_numpy(X_train).float()).contiguous()
Y_train = Variable(torch.from_numpy(Y_train).float()).contiguous()

X_test = Variable(torch.from_numpy(X_test).float()).contiguous()
Y_test = Variable(torch.from_numpy(Y_test).float()).contiguous()


print(model)
params = list(model.parameters())

optimizer = optim.Adam(model.parameters(), lr=1e-4)

losses = []

epochs = 25000
for epoch in range(epochs):
    
    y_pred = model(X_train)

    loss = F.smooth_l1_loss(y_pred, Y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    writer.add_scalar('Loss', loss.data, epoch)
    print "Epoch " + str(epoch)+ ". Loss: " + str(loss.data.numpy())
    losses.append(loss.data.numpy())
    
    for name, param in model.named_parameters():
        writer.add_histogram(name,  param, epoch)

torch.save(model, '../cnn_v1_weights.pt')

t = time.time()
y_pred = model(X_test)
print("Time for full set", time.time() - t)
test_loss = F.smooth_l1_loss(y_pred, Y_test)

print("Final Loss")
print(test_loss)

print("Trained on " + str(len(X_train)) + ", Tested on " + str(len(X_test)) + " samples")

#from utils import print_errors
#print_errors(Y_test.detach().numpy(), y_pred.detach().numpy(), "TEST Errors:")
