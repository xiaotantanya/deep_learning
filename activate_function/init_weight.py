import torch
import torch.nn as nn

# 定义一个全连接层
layer = nn.Linear(in_features=100, out_features=200)

# 使用Xavier初始化
nn.init.xavier_uniform_(layer.weight)  # 对权重初始化

# 或者使用Xavier初始化的正交版本
nn.init.orthogonal_(layer.weight)  # 对权重初始化

print(layer.weight)
# 这将会对指定层的权重矩阵进行Xavier初始化
