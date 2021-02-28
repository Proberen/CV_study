
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import RMBDataset
from model import LeNet
import torch.optim as optim
import utils


def train():

    utils.set_seed()
    # ============================ step 1/5 数据 ============================

    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    # 构建MyDataset实例
    train_data = RMBDataset(data_dir=opt.train_dir, transform=train_transform)
    valid_data = RMBDataset(data_dir=opt.valid_dir, transform=valid_transform)

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=opt.batch_size)

    # ============================ step 2/5 模型 ============================
    net = LeNet(classes=2)
    net.initialize_weights()

    # ============================ step 3/5 损失函数 ============================
    criterion = nn.CrossEntropyLoss()  # 选择损失函数

    # ============================ step 4/5 优化器 ============================
    optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9)  # 选择优化器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 设置学习率下降策略

    # ============================ step 5/5 训练 ============================
    train_curve = list()
    valid_curve = list()

    for epoch in range(opt.epochs):

        loss_mean = 0.
        correct = 0.
        total = 0.

        net.train()
        for i, data in enumerate(train_loader):

            # forward
            inputs, labels = data
            outputs = net(inputs)

            # backward
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()

            # update weights
            optimizer.step()

            # 统计分类情况
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).squeeze().sum().numpy()

            # 打印训练信息
            loss_mean += loss.item()
            train_curve.append(loss.item())
            if (i + 1) % opt.log_interval == 0:
                loss_mean = loss_mean / opt.log_interval
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, opt.epochs, i + 1, len(train_loader), loss_mean, correct / total))
                loss_mean = 0.

        scheduler.step()  # 更新学习率

        # validate the model
        if (epoch + 1) % opt.val_interval == 0:

            correct_val = 0.
            total_val = 0.
            loss_val = 0.
            net.eval()
            with torch.no_grad():
                for j, data in enumerate(valid_loader):
                    inputs, labels = data
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).squeeze().sum().numpy()

                    loss_val += loss.item()

                valid_curve.append(loss_val)
                print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, opt.epochs, j + 1, len(valid_loader), loss_val, correct / total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
    parser.add_argument('--batch-size', type=int, default=5)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--train_dir', type=str, default='./RMB_data/rmb_split/train')
    parser.add_argument('--valid_dir', type=str, default='./RMB_data/rmb_split/valid')
    opt = parser.parse_args()
    train()