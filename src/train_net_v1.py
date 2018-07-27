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

from tensorboardX import SummaryWriter


writer = SummaryWriter()


h_p_l = 100
if '-l' in sys.argv:
    model = torch.load('weights.pt')

else:
    model = nn.Sequential(nn.Linear(40, h_p_l), nn.ReLU(),
                          nn.Linear(h_p_l, h_p_l), nn.ReLU(),
                          nn.Linear(h_p_l, h_p_l), nn.ReLU(),
                          nn.Linear(h_p_l, 4))

optimizer = optim.Adam(model.parameters(), lr=1e-4)

X = np.loadtxt('../data/U_push_contours.txt')#dataU.mat')#
#X = X['dataU'].astype(float)
X = X.astype(float)

Y = np.loadtxt('../data/Y_push.txt')
#Y = io.loadmat('./data/dataY.mat')
Y = Y.astype(float)

msk = np.random.rand(len(X)) < 0.9
X_train = X[msk]
Y_train = Y[msk]

X_test = X[~msk]
Y_test = Y[~msk]

X_train = Variable(torch.from_numpy(X_train).float()).contiguous()
Y_train = Variable(torch.from_numpy(Y_train).float()).contiguous()

X_test = Variable(torch.from_numpy(X_test).float()).contiguous()
Y_test = Variable(torch.from_numpy(Y_test).float()).contiguous()


losses = []

epochs = 50000
for epoch in range(epochs):
    
    y_pred = model(X_train)

    loss = F.smooth_l1_loss(y_pred, Y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    writer.add_scalar('Loss', loss.data, epoch)
    print "Epoch " + str(epoch)+ ". Loss: " + str(loss.data.numpy())
    losses.append(loss.data.numpy())

    torch.save(model, '../ann_v1_weights.pt')
    
    for name, param in model.named_parameters():
        writer.add_histogram(name,  param, epoch)


t = time.time()
y_pred = model(X_test)
print("Time for full set", time.time() - t)
test_loss = F.smooth_l1_loss(y_pred, Y_test)

print("Final Loss")
print(test_loss)

print("Trained on " + str(len(X_train)) + ", Tested on " + str(len(X_test)) + " samples")

from utils import print_errors
print_errors(Y_test.detach().numpy(), y_pred.detach().numpy(), "TEST Errors:")
