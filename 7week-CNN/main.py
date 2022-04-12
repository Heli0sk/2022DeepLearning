import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from loadData import Mnist
from LeNet5 import LeNet5


def gen_dataset(path, clas, batchsize):
    '''
    :param path: 数据存放的相对路径
    :param clas: 区分训练集和测试集
    :param batchsize:
    :return: dataloader
    '''
    dataset = Mnist(
        root=path,
        train=clas,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1037,), (0.3081,))
        ])
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=True
    )
    return data_loader


def Train(epochs, train_loader, model, show=False):
    '''
    :param epochs:
    :param train_loader:
    :param model: 待训练模型
    :param show:  是否对loss进行可视化
    :return: loss
    '''
    loss_list = []
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, data in enumerate(train_loader, start=0):
            images, labels = data  # 读取一个batch的数据
            optimizer.zero_grad()  # 梯度清零，初始化
            outputs = model(images)  # 前向传播
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
    if show:
        plt.plot(loss_list)
        plt.title('traning loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.show()
    return loss_list


def Evaluate(test_loader, model):
    correct = 0  # 预测正确数
    total = 0  # 总图片数
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predict = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predict == labels).sum()
    acc = 100 * correct // total

    return acc


if __name__ == '__main__':
    data_path = 'dataset'
    batch_size = 32
    epochs = 10

    # 生成训练集
    train_loader = gen_dataset(data_path, True, batch_size)
    # 实例化一个网络
    LeNet5 = LeNet5()
    # 定义损失函数和优化器
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        LeNet5.parameters(),
        lr=0.001,
        momentum=0.9
    )
    # 训练模型
    trainLoss = Train(epochs, train_loader, LeNet5, True)

    # 测试
    test_loader = gen_dataset(data_path, False, batch_size)
    acc = Evaluate(test_loader, LeNet5)
    print('测试集准确率 {}%'.format(acc))


