import datetime
import os

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from my_ae_cnn.net.CNNnet import CNN

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')

learning_rate = 1e-3
model = CNN()
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
starttime = datetime.datetime.now()

wid = 40
long = 40
num_epochs = 1001
move_phase = 4
nums = 1600
loss_value = 1.00
img1 = np.load("./data_optical/optical_data.npy").reshape(200, nums)
img2 = np.load("./data_optical/optical_gnn_result.npy").reshape(200, nums)
label1 = np.load("./data_optical/optical_label.npy")
label = torch.LongTensor(label1.reshape(-1, 1))
labels_onehot = torch.zeros(200, 4).scatter_(1, label, 1)


# 当循环将1600个数据*4组写入数组转换为numpy时，用reshape转换的类型为（4，1600)
# ====================预训练CNN网络==========================
def train():
    for i in range(100):
        img = img1[i, :].reshape(1, 1, wid, long)
        img = Variable(torch.from_numpy(img).float())  # .cuda()
        label = Variable(labels_onehot[i, :].reshape(1, 4))  # .cuda()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, label)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ===================log========================
        endtime = datetime.datetime.now()
        print('epoch [{}/{}], loss:{:.4f}, time:{:.2f}s'.format(000, num_epochs, loss.item(),
                                                                (endtime - starttime).seconds))
    # ==========利用GNN生成信号来训练cnn网络=================
    for epoch in range(num_epochs):
        # for data in dataloader:
        for k in range(200):
            img_gnn_result = img2[k, :].reshape(1, 1, wid, long)
            img_gnn_result = Variable(torch.from_numpy(img_gnn_result).float())  # .cuda()
            label = Variable(labels_onehot[k, :].reshape(1, 4))  # .cuda()
            # ===================forward=====================
            output = model(img_gnn_result)
            loss = criterion(output, label)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ===================log========================
        endtime1 = datetime.datetime.now()
        print('epoch [{}/{}], loss:{:.4f}, time:{:.2f}s'.format(epoch + 1, num_epochs, loss.item(),
                                                                (endtime1 - starttime).seconds))
        if loss_value > loss.item():
            loss_value = loss.item()
            torch.save(model.state_dict(), './model/model_cnn_param.pkl')

        # if epoch % (num_epochs // 10) == 0:
        #     plt.plot(range(nums), output.cpu().data.numpy().reshape(nums))
        #     plt.savefig('./dc_img/image_{}.png'.format(epoch))
        #     plt.close()


if __name__ == "__main__":
    train()
