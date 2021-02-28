
import torch
import torch.nn as nn
import torch.optim as optim


# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
# 设置学习率下降策略
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)



for epoch in range(10):
    for i, data in enumerate(train_loader):
        # forward
        inputs, labels = data
        outputs = net(inputs)
        # 梯度清0
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        # 更新梯度
        optimizer.step()

    # 更新学习率
    scheduler.step()