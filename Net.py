# -*- coding: utf-8 -*-
# @Time    : 2022/2/27 8:47
# @Author  : zhao
# @File    : Net.py

import torch
import torch.nn.functional as F

class CNN(torch.nn.Module):
    def __init__(self,embedding_size):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv1d(32, 64, 3, 1)
        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(1024, embedding_size)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


### For iCaRL model
class icarl_model(torch.nn.Module):
    def __init__(self,n_classes):
        super(icarl_model, self).__init__()
        self.encode = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=1),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=4,stride=2),
            torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=4, stride=2),
            # Flatten(),
        )

        self.fc = torch.nn.Linear(480, n_classes,bias=False)

    def forward(self, x):
        x = self.encode(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CNN_FL(torch.nn.Module):
    def __init__(self,N_class=10, lamda=0):
        super( CNN_FL, self).__init__()
        self.encode = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=1),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=4,stride=2),
            torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=4, stride=2),
            # Flatten(),
        )

        self.fc = torch.nn.Linear(480, N_class,bias=False)
        self.lamda = lamda

    def forward(self, x):
        x = self.encode(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# class CNN_FL(torch.nn.Module):
#     def __init__(self,N_class=10, lamda=0):
#         super(CNN_FL, self).__init__()
#         self.conv1 = torch.nn.Conv1d(1, 32, 3, 1)
#         self.conv2 = torch.nn.Conv1d(32, 64, 3, 1)
#         self.dropout1 = torch.nn.Dropout(0.25)
#         self.dropout2 = torch.nn.Dropout(0.5)
#         self.fc1 = torch.nn.Linear(1024, 32)
#         self.sigmoid = torch.nn.Sigmoid()
#         self.lamda = lamda
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = self.dropout1(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = self.sigmoid(x)
#         return x