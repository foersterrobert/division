import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from torch.autograd import Variable
import math
import decimal
import os

BATCH_SIZE = 128
EPOCHS = 90

def devision_data(size):
    xdata = []
    ydata = []
    ydataRaw = []
    for i in range(int(size/BATCH_SIZE)):
        xbatch = []
        ybatch = []
        for j in range(BATCH_SIZE):
            i1, i2 = float(decimal.Decimal(random.randrange(100, 2000))/100), float(decimal.Decimal(random.randrange(100, 2000))/100)
            y = i1 / i2 / 20
            xbatch.append([i1, i2])
            ybatch.append(y)
            ydataRaw.append(y)
        xbatch = torch.tensor(xbatch, dtype=torch.float)
        ybatch = torch.tensor(ybatch)
        xdata.append(xbatch)
        ydata.append(ybatch)
    ydataMean = sum(ydataRaw) / len(ydataRaw)
        
    return list(zip(xdata, ydata)), ydataMean

train_data, trainMean = devision_data(64000)
test_data, testMean = devision_data(12800)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(2, 2)
        self.l2 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.l1(x)
        x = torch.where(x <= 0, 1.359140915 * (x-1).exp(), torch.where(x > 15, 1 - 1/(109.0858178 * x - 1403.359435), 0.03 * (1000000 * x + 1).log() + 0.5))
        x = self.l2(x)
        x = torch.where(x <= 0, 1.359140915 * (x-1).exp(), torch.where(x > 15, 1 - 1/(109.0858178 * x - 1403.359435), 0.03 * (1000000 * x + 1).log() + 0.5))
        x = x*5
        return x

load = input('load? y/n ')
if load == 'y':
    model = torch.load('./model/Torch.pth')
    while True:
        inputs = input('\ninputs: ')
        try:
            print(model(torch.tensor([int(inputs.split(',')[0].strip()), int(inputs.split(',')[1].strip())], dtype=torch.float)).item() * 20)
        except:
            exit()

else:
    model = Model()

optimizer = optim.Adadelta(model.parameters())
criterion = nn.MSELoss()

def train(epoch):
    model.train()
    for batch_id, (data, target) in enumerate(train_data):
        data = Variable(data)
        target = Variable(target)
        Y_pred = model(data)
        optimizer.zero_grad()
        loss = criterion(Y_pred, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch + 1, batch_id * len(data), BATCH_SIZE * len(train_data),
                    100. * batch_id / len(train_data), loss.item()))

def test():
    model.eval()
    totalLoss = 0
    for data, target in test_data:
        data = Variable(data)
        target = Variable(target)
        Y_pred = model(data)
        loss = criterion(Y_pred, target)
        totalLoss += loss

    print('Custom Durchschnittsloss: ', totalLoss / len(test_data))
    print('ZeroMSE: ', testMean)

for epoch in range(EPOCHS):
    train(epoch)
    test()

save = input('save? y/n ')
if save == 'y':
    model_folder_path = './model'
    file_name='Torch.pth'
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)

    file_name = os.path.join(model_folder_path, file_name)
    torch.save(model, file_name)
