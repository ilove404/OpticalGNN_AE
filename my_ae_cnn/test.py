import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm

import train

num_epochs = 100
batch_size = 128
learning_rate = 1e-3

n = 15  # 15*15 225个数字图片
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))  # 最终图片
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))  # 假设隐变量空间符合高斯分布
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))  # ppf随机取样
model = train.autoencoder()
for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[[[xi, yi]]]], dtype=np.float32)  # 重复z_sample多次，形成一个完整的batch
        z_sample = np.tile(z_sample, batch_size * 392).reshape(batch_size, 1, 28, 28)
        z_sample = torch.from_numpy(z_sample)  # 转tensor
        z_sample = z_sample.cuda()  # 放到cuda上
        output = model(z_sample)
        digit = output[0].reshape(digit_size, digit_size)  # 128*784->28*28
        digit = digit.cpu().detach().numpy()  # 转numpy
        figure[i * digit_size:(i + 1) * digit_size, j * digit_size:(j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
