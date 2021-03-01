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
