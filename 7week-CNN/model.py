import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 对应LeNet5中的C1层
        # 输入通道为1，输出通道为6， 卷积核大小5x5，步长为1
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 对应LeNet5中的S2层， 大小为2x2
        self.pool1 = nn.MaxPool2d(2, 2)
        # 对应LeNet5中的C3层
        # 输入通道为6，输出通道为16，设置卷积核大小5x5，步长为1
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 对应LeNet5中的S4层
        self.pool2 = nn.MaxPool2d(2, 2)
        # 对应LeNet5中的C5层， 输入通特征为4x4x16，输出特征为120x1
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        # 对应F6层，输入是120维向量，输出是84维向量
        self.fc2 = nn.Linear(120, 84)
        # 输出层，输入是84维向量，输出是10维向量
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # [28,28,1]--->[24,24,6]--->[12,12,6]
        x = self.pool1(F.relu(self.conv1(x)))
        # [12,12,6]--->[8,8,,16]--->[4,4,16]
        x = self.pool2(F.relu(self.conv2(x)))
        # [n,4,4,16]--->[n,4*4*16]
        x = x.view(-1, 16 * 4 * 4)
        # [n,256]--->[n,120]
        x = F.relu(self.fc1(x))
        # [n,120]-->[n,84]
        x = F.relu(self.fc2(x))
        # [n,84]-->[n,10]
        x = self.fc3(x)
        return x

