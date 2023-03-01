import numpy as np
import torch
from torch.autograd import Variable

from my_ae_cnn.net.CNNnet import CNN

# 噪声强度
delta = 0.7
nums = 1600
wid = 40
long = 40
noise = np.random.normal(0, 1, nums)
x = range(nums)
y_data = []
y_label = []
y_labels = []
m = np.linspace(0, 4, 4)
####################### 原始数据和带噪声，move_phase数量组的数据 #############################
img2 = np.load("./data_optical/optical_gnn_result.npy").reshape(200, nums)

model = CNN()
model.load_state_dict(torch.load('./model/model_cnn_param.pkl'))
optical_gnn_result = []
for k in range(4):
    img_result = img2[k, :].reshape(1, 1, wid, long)
    img_result = Variable(torch.from_numpy(img_result).float())
    output = model(img_result)
    output = np.argmax(output.cpu().data.numpy())
    print(output)
# optical_gnn_result = np.array(optical_gnn_result)
# np.save("./results/gnn_results.npy", optical_gnn_result)
# print(optical_gnn_result.shape)
