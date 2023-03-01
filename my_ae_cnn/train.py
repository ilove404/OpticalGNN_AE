import datetime
import os

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')
from dataset import dataset_creat

'''def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x'''

batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

dataset = MNIST('./data', transform=img_transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.latentSpace = nn.Sequential(
            # 网络每一层的神经元个数，[1,10,1]说明只有一个隐含层，输入的变量是一个，也对应一个输出。如果是两个变量对应一个输出，那就是[2，10，1]
            # 用torch.nn.Linear构建线性层，本质上相当于构建了一个维度为[layers[0],layers[1]]的矩阵，这里面所有的元素都是权重
            nn.Flatten(),
            nn.Linear(1 * 8 * 3 * 3, 4 * 2 * 2),
            nn.Linear(4 * 2 * 2, 8 * 3 * 3),
            nn.Unflatten(1, [8, 3, 3]),
            nn.ReLU(True)

        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.latentSpace(x)
        x = self.decoder(x)
        return x


model = autoencoder()
print(model)

# 查看网络流程
net = nn.Sequential(
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
    X = layer(X)

if torch.cuda.is_available():
    model.cuda()
    print('cuda is OK!')
#     model = model.to('cuda')
else:
    print('cuda is NO!')
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
starttime = datetime.datetime.now()
wid = 40
long = 40
num_epochs = 10001
move_phase = 4
nums = 1600
for epoch in range(num_epochs):
    # for data in dataloader:
    train_set = dataset_creat(move_phase, nums)
    img1 = train_set[0].reshape(move_phase, nums)  ##当循环将1600个数据*4组写入数组转换为numpy时，用reshape转换的类型为（4，1600）###
    label1 = train_set[1].reshape(move_phase, nums)
    for i in range(move_phase):
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

    if epoch % (num_epochs // 10) == 0:
        plt.plot(range(nums), output.cpu().data.numpy().reshape(nums))
        plt.savefig('./dc_img/image_{}.png'.format(epoch))
        plt.close()
# torch.save(model.state_dict(), './conv_autoencoder.pth')
