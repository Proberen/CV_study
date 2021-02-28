import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import argparse
from dataset import RMBDataset
import utils
from model import LeNet


def train():
    device = torch.device("cuda:0" if opt.cuda else "cpu")
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

    # 构建MyDataset
    train_data = RMBDataset(data_dir=opt.train_dir, transform=train_transform)
    valid_data = RMBDataset(data_dir=opt.valid_dir, transform=valid_transform)

    # 构建DataLoader
    train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=opt.batch_size)

    # ============================ step 2/5 模型 ============================
    net = LeNet(classes=2)
    net.to(device)
    # net.initialize_weights()

    # ============================ step 3/5 损失函数 ============================
    criterion = nn.CrossEntropyLoss()

    # ============================ step 4/5 优化器 ============================
    optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # ============================ step 5/5 训练 ============================
    train_curve = list()
    valid_curve = list()

    for epoch in range(opt.epochs):
        loss_mean = 0.
        correct = 0.
        total = 0.

        net.train()
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs.to(device)
            labels.to(device)
            outputs = net(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 统计分类情况
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).squeeze().sum().to("cpu").numpy()

            # 打印训练信息
            loss_mean += loss.item()
            train_curve.append(loss.item())
            if (i + 1) % opt.log_interval == 0:
                loss_mean = loss_mean / opt.log_interval
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, opt.epochs, i + 1, len(train_loader), loss_mean, correct / total))
                loss_mean = 0.

        scheduler.step()  # 更新学习率

        if (epoch + 1) % opt.val_interval == 0:
            correct_val = 0.
            total_val = 0.
            loss_val = 0.
            net.eval()
            with torch.no_grad():
                for j, data in enumerate(valid_loader):
                    inputs, labels = data
                    inputs.to(device)
                    labels.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).squeeze().sum().to("cpu").numpy()

                    loss_val += loss.item()

                valid_curve.append(loss_val)
                print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, opt.epochs, j + 1, len(valid_loader), loss_val, correct / total))

    utils.loss_picture(train_curve, train_loader, valid_curve, opt.val_interval)
    # 保存模型参数
    net_state_dict = net.state_dict()
    torch.save(net_state_dict, opt.path_state_dict)
    print("模型保存成功")


if __name__ == '__main__':
    # 超参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--train_dir', type=str, default='../RMB_data/rmb_split/train')
    parser.add_argument('--valid_dir', type=str, default='../RMB_data/rmb_split/valid')
    parser.add_argument('--cuda', type=bool, default=False)
    parser.add_argument('--path_state_dict', type=str, default='./model_RMB.pth')
    opt = parser.parse_args()
    train()
