import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import RMBDataset
from model import LeNet


# 预测
def detect():
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    valid_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    test_data = RMBDataset(data_dir=opt.test_dir, transform=valid_transform)
    valid_loader = DataLoader(dataset=test_data, batch_size=1)

    net = LeNet(classes=2)
    state_dict_load = torch.load(opt.path_state_dict)
    net.load_state_dict(state_dict_load)

    for i, data in enumerate(valid_loader):
        inputs, labels = data
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)

        rmb = 1 if predicted.numpy()[0] == 0 else 100
        label = 1 if labels.numpy()[0] == 0 else 100
        print("实际{0}元,模型获得{1}元,预测{2}".format(label, rmb, '正确' if rmb == label else '错误'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, default='../RMB_data/rmb_split/valid')
    parser.add_argument('--path_state_dict', type=str, default='./model_RMB.pth')
    opt = parser.parse_args()
    detect()
