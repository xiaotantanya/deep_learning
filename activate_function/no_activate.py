import torch
import torch.nn as nn
import math
import logging


class Model(nn.Module):
    def __init__(self,num_layers,num_dim):
        super().__init__()
        self.num_layers = num_layers
        self.num_features = 20
        # self.linears = [nn.Linear(num_dim,num_dim) for i in range(num_layers)]
        self.linear1 = nn.Linear(num_dim,self.num_features)
        self.linear2 = nn.Linear(self.num_features,self.num_features)
        self.linear3 = nn.Linear(self.num_features,self.num_features)
        self.linear4 = nn.Linear(self.num_features,self.num_features)
        self.linear5 = nn.Linear(self.num_features,num_dim)
        self.relu = nn.LeakyReLU()
    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.relu(x)
        x = self.linear5(x)
        x = self.relu(x)
        return x

if __name__ == '__main__':
    x = torch.rand(1000,1).cuda()
    # y = torch.log(x)*torch.sin(x)*torch.cos(x+23)*torch.exp(x)
    y = x + 2 # + torch.log(x)
    # print(y.shape)
    
    model = Model(10,1).cuda()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.005)
    # print(model.linears)
    for i in range(10000):
        optimizer.zero_grad()
        x_ = model(x)
        
        loss = torch.square(x_ - y).sum() / 1000
        loss.backward()
        print(loss.data)
        optimizer.step()
    print(y[1:10,0])
    print(x_[1:10,0])
    