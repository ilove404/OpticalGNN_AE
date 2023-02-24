import datetime
import os

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 100
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
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = autoencoder()
print(model)

# 查看网络流程
net = nn.Sequential(
    nn.Conv2d(1, 16, 3, stride=3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, stride=2),
    nn.Conv2d(16, 8, 3, stride=2, padding=1), nn.ReLU(True), nn.MaxPool2d(2, stride=1),
    nn.ConvTranspose2d(8, 16, 3, stride=2), nn.ReLU(True),
    nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1), nn.ReLU(True),
    nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1), nn.Tanh())
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
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

for epoch in range(num_epochs):
    for data in dataloader:
        img, label = data
        img = Variable(img).cuda()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    endtime = datetime.datetime.now()
    print('epoch [{}/{}], loss:{:.4f}, time:{:.2f}s'.format(epoch + 1, num_epochs, loss.item(),
                                                            (endtime - starttime).seconds))

    # if epoch % 10 == 0:
    pic = to_img(output.cpu().data)
    save_image(pic, './dc_img/image_{}.png'.format(epoch))

# torch.save(model.state_dict(), './conv_autoencoder.pth')
