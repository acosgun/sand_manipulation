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

batch_size = 3000
writer = SummaryWriter()
h_p_l = 100
if '-l' in sys.argv:
    model = torch.load('weights.pt')

else:
    model = nn.Sequential(nn.Linear(40, 64), nn.ReLU(),
                          nn.Linear(64, 4))

optimizer = optim.Adam(model.parameters(), lr=3*1e-4)

X = np.loadtxt('../data/U_push_contours.txt')
X = X.astype(float)

Y = np.loadtxt('../data/Y_push.txt')
Y = Y.astype(float)

#training-testing sets
msk = np.random.rand(len(X)) < 0.9
X_train = X[msk]
Y_train = Y[msk]

X_test = X[~msk]
Y_test = Y[~msk]

X_train = Variable(torch.from_numpy(X_train).float())
Y_train = Variable(torch.from_numpy(Y_train).float())

X_test = Variable(torch.from_numpy(X_test).float())
Y_test = Variable(torch.from_numpy(Y_test).float())

losses = []

epochs = 50000
for epoch in range(epochs):
    #mini-batching
    rand_index = np.random.choice(100, size=batch_size)
    y_pred = model(X_train[rand_index])
    loss = torch.mean(F.smooth_l1_loss(y_pred, Y_train[rand_index]))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    writer.add_scalar('Loss', loss.data, epoch)
    print "Epoch " + str(epoch)+ ". Loss: " + str(loss.data.numpy())
    losses.append(loss.data.numpy())

    torch.save(model, '../ann_v2_weights.pt')

    for name, param in model.named_parameters():
        writer.add_histogram(name,  param, epoch)

writer.close()
t = time.time()
y_pred = model(X_test)
print("Time for full set", time.time() - t)
test_loss = torch.mean(F.smooth_l1_loss(y_pred, Y_test))

print("Final Loss")
print(test_loss)

print("Trained on " + str(len(X_train)) + ", Tested on " + str(len(X_test)) + " samples")

from utils import print_errors
print_errors(Y_test.detach().numpy(), y_pred.detach().numpy(), "TEST Errors:")
