from torch import nn


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
            nn.Flatten(),  # x = x.view(x.size(0), -1)  展平为1维
            nn.Linear(1 * 8 * 3 * 3, 4 * 2 * 2),
            nn.ReLU(True),
            nn.Linear(4 * 2 * 2, 8 * 3 * 3),
            nn.ReLU(True),
            nn.Unflatten(1, [8, 3, 3]),

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
