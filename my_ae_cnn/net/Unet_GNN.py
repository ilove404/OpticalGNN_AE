import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # 编码器
        self.encoder_conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.encoder_relu1 = nn.ReLU(inplace=True)
        self.encoder_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.encoder_relu2 = nn.ReLU(inplace=True)
        self.encoder_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.encoder_relu3 = nn.ReLU(inplace=True)
        self.encoder_conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.encoder_relu4 = nn.ReLU(inplace=True)

        # 解码器
        self.decoder_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.decoder_relu1 = nn.ReLU(inplace=True)
        self.decoder_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.decoder_relu2 = nn.ReLU(inplace=True)
        self.decoder_conv3 = nn.ConvTranspose2d(32 + 32, 16, kernel_size=4, stride=2, padding=1)
        self.decoder_relu3 = nn.ReLU(inplace=True)
        self.decoder_conv4 = nn.ConvTranspose2d(16 + 16, 1, kernel_size=4, stride=2, padding=1)
        self.decoder_sigmoid = nn.LeakyReLU()

    def forward(self, x):
        x1 = self.encoder_conv1(x)
        x1 = self.encoder_relu1(x1)
        x2 = self.encoder_conv2(x1)
        x2 = self.encoder_relu2(x2)
        x3 = self.encoder_conv3(x2)
        x3 = self.encoder_relu3(x3)
        # x4 = self.encoder_conv4(x3)
        # x4 = self.encoder_relu4(x4)
        #
        # x5 = self.decoder_conv1(x4)
        # x5 = self.decoder_relu1(x5)
        # x5 = torch.cat([x5, x3], dim=1)
        x6 = self.decoder_conv2(x3)
        x6 = self.decoder_relu2(x6)
        x6 = torch.cat([x6, x2], dim=1)
        x7 = self.decoder_conv3(x6)
        x7 = self.decoder_relu3(x7)
        x7 = torch.cat([x7, x1], dim=1)
        x8 = self.decoder_conv4(x7)
        x8 = self.decoder_sigmoid(x8)

        return x8
