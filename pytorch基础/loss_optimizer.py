
import torch
import torch.nn as nn
import torch.optim as optim


# ============================ step 3/5 损失函数 ============================
criterion = nn.CrossEntropyLoss()                                                   # 选择损失函数

# ============================ step 4/5 优化器 ============================
# /'ɒptɪmaɪzə/   /ˈskedʒuːlər/
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)                         # 选择优化器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)     # 设置学习率下降策略




for epoch in range(10):

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

    scheduler.step()  # 更新学习率