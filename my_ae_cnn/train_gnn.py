import datetime
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.autograd import Variable

from net.Unet_GNN import AutoEncoder

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')

learning_rate = 1e-3


def train_gnn_net():
    # model = autoencoder()
    model = AutoEncoder()
    print(model)

    # 查看网络流程
    '''net = nn.Sequential(
        nn.Conv2d(1, 16, 3, stride=3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, stride=2),
        nn.Conv2d(16, 8, 3, stride=2, padding=1), nn.ReLU(True), nn.MaxPool2d(2, stride=1),
        nn.Flatten(), nn.Linear(1 * 8 * 3 * 3, 4 * 2 * 2), nn.Linear(4 * 2 * 2, 8 * 3 * 3), nn.Unflatten(1, [8, 3, 3]),
        nn.ReLU(True),
        nn.ConvTranspose2d(8, 16, 3, stride=2), nn.ReLU(True),
        nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1), nn.ReLU(True),
        nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1), nn.Tanh())
    X = torch.rand(size=(1, 1, 40, 40), dtype=torch.float32)
    for layer in net:
        print(layer.__class__.__name__, 'output shape: \t', X.shape)
        X = layer(X)'''
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    starttime = datetime.datetime.now()
    wid = 40
    long = 40
    num_epochs = 101
    move_phase = 4
    nums = 1600
    batch_size = 4
    loss_value = 1.00
    ##当循环将1600个数据*4组写入数组转换为numpy时，用reshape转换的类型为（4，1600）###
    img1 = np.load("./data_optical/optical_noise.npy").reshape(200, nums)
    label1 = np.load("./data_optical/optical_data.npy").reshape(200, nums)
    # plt.plot(range(1600), img1[100, :])
    # plt.plot(range(1600), label1[100, :])
    # plt.show()

    # ====================训练GNN网络==========================
    for epoch in range(num_epochs):
        # for data in dataloader:
        # train_set = dataset_creat(move_phase, nums)
        # img1 = train_set[0].reshape(move_phase, nums)  ##当循环将1600个数据*4组写入数组转换为numpy时，用reshape转换的类型为（4，1600）###
        # label1 = train_set[1].reshape(move_phase, nums)
        for i in range(200):
            img = img1[i, :].reshape(1, 1, wid, long)
            label = label1[i, :].reshape(1, 1, wid, long)
            img = Variable(torch.from_numpy(img).float())  # .cuda()
            label = Variable(torch.from_numpy(label).float())  # .cuda()
            # ===================forward=====================
            output = model(img)
            loss = criterion(output, label)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ===================log========================
        endtime = datetime.datetime.now()

        print('epoch [{}/{}], loss:{:.4f}, time:{:.2f}s'.format(epoch + 1, num_epochs, loss.item(),
                                                                (endtime - starttime).seconds))
        if loss_value > loss.item():
            loss_value = loss.item()
            torch.save(model.state_dict(), './model/model_param_gnn_unet.pkl')
        if epoch % (num_epochs // 10) == 0:
            plt.plot(range(nums), output.cpu().data.numpy().reshape(nums))
            plt.savefig('./dc_img/image_unet_{}.png'.format(epoch))
            plt.close()

    optical_gnn_result = []
    model.load_state_dict(torch.load('./model/model_param_gnn_unet.pkl'))
    for k in range(200):
        img_result = img1[k, :].reshape(1, 1, wid, long)
        img_result = Variable(torch.from_numpy(img_result).float())
        output = model(img_result)
        output = output.cpu().data.numpy()
        optical_gnn_result.append(output)
    optical_gnn_result = np.array(optical_gnn_result)
    print(optical_gnn_result.shape)
    np.save("./data_optical/optical_gnn_unet_result.npy", optical_gnn_result)


if __name__ == "__main__":
    train_gnn_net()
