import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict

class LeNet(nn.Module):
    def __init__(self,classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) #(32+2*0-5)/1+1 = 28
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, classes)

    def forward(self, x):    #4 3 32 32 ->nn.Conv2d(3, 6, 5)->   4 6 28 28
        out = F.relu(self.conv1(x)) #32->28   4 6 28 28
        out = F.max_pool2d(out, 2)   #4 6 14 14
        out = F.relu(self.conv2(out)) # 4 16 10 10
        out = F.max_pool2d(out, 2)  # 4 16 5 5
        out = out.view(out.size(0), -1) #4 400
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.1)
                m.bias.data.zero_()

net = LeNet(classes=2)
fake_img = torch.randn((4, 3, 32, 32), dtype=torch.float32)
output = net(fake_img)
print(output)


# ============================ Sequential 顺序
class LeNetSequential(nn.Module):
    def __init__(self, classes):
        super(LeNetSequential, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),)

        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, classes),)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x


class LeNetSequentialOrderDict(nn.Module):
    def __init__(self, classes):
        super(LeNetSequentialOrderDict, self).__init__()

        self.features = nn.Sequential(
            OrderedDict({
                'conv1': nn.Conv2d(3, 6, 5),
                'relu1': nn.ReLU(inplace=True),
                'pool1': nn.MaxPool2d(kernel_size=2, stride=2),
                'conv2': nn.Conv2d(6, 16, 5),
                'relu2': nn.ReLU(inplace=True),
                'pool2': nn.MaxPool2d(kernel_size=2, stride=2),
            }))

        self.classifier = nn.Sequential(
            OrderedDict({
                'fc1': nn.Linear(16*5*5, 120),
                'relu3': nn.ReLU(),
                'fc2': nn.Linear(120, 84),
                'relu4': nn.ReLU(inplace=True),
                'fc3': nn.Linear(84, classes),
            }))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x

# net = LeNetSequential(classes=2)
# net = LeNetSequentialOrderDict(classes=2)
# # #
# fake_img = torch.randn((4, 3, 32, 32), dtype=torch.float32)
# # #
# output = net(fake_img)
#
# print(net)
# print(output)


# ============================ ModuleList 迭代
class myModuleList(nn.Module):
    def __init__(self):
        super(myModuleList, self).__init__()
        # 列表
        modullist_temp = [nn.Linear(10, 10) for i in range(20)]
        self.linears = nn.ModuleList(modullist_temp)

    def forward(self, x):
        for i, linear in enumerate(self.linears):
            x = linear(x)
        return x

# net = myModuleList()
# #
# # print(net)
# #
# fake_data = torch.ones((10, 10))
# #
# output = net(fake_data)
# #
# print(output)


# ============================ ModuleDict 有序字典

class myModuleDict(nn.Module):
    def __init__(self):
        super(myModuleDict, self).__init__()

        self.choices = nn.ModuleDict({
            'conv': nn.Conv2d(10, 10, 3),
            'pool': nn.MaxPool2d(3)
        })

        self.activations = nn.ModuleDict({
            'relu': nn.ReLU(),
            'prelu': nn.PReLU()
        })

    def forward(self, x, choice, act):
        x = self.choices[choice](x)
        x = self.activations[act](x)
        return x

# net = myModuleDict()
# #
# fake_img = torch.randn((4, 10, 32, 32))
# #
# output = net(fake_img, 'conv', 'relu')
#
# print(output)