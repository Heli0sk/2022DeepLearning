import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from data import Mnist
from model import LeNet5

# 生成训练集
train_set = Mnist(
    root='dataset',
    train=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1037,), (0.3081,))
    ])
)
train_loader = DataLoader(
    dataset=train_set,
    batch_size=32,
    shuffle=True
)

# 实例化一个网络
net = LeNet5()

# 定义损失函数和优化器
loss_function = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(
    net.parameters(),
    lr=0.001,
    momentum=0.9
)

# 3 训练模型
loss_list = []
for epoch in range(10):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, start=0):

        images, labels = data  # 读取一个batch的数据
        optimizer.zero_grad()  # 梯度清零，初始化
        outputs = net(images)  # 前向传播
        loss = loss_function(outputs, labels)  # 计算误差
        loss.backward()  # 反向传播
        optimizer.step()  # 权重更新
        running_loss += loss.item()  # 误差累计

        # 每300个batch 打印一次损失值
        if batch_idx % 300 == 299:
            print('epoch:{} batch_idx:{} loss:{}'
                  .format(epoch + 1, batch_idx + 1, running_loss / 300))
            loss_list.append(running_loss / 300)
            running_loss = 0.0  # 误差清零

print('Finished Training.')

# 打印损失值变化曲线
plt.plot(loss_list)
plt.title('traning loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

# 测试
test_set = Mnist(
    root='dataset',
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1037,), (0.3081,))
    ])
)
test_loader = DataLoader(
    dataset=test_set,
    batch_size=32,
    shuffle=True
)

correct = 0  # 预测正确数
total = 0  # 总图片数

for data in test_loader:
    images, labels = data
    outputs = net(images)
    _, predict = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predict == labels).sum()

print('测试集准确率 {}%'.format(100 * correct // total))

