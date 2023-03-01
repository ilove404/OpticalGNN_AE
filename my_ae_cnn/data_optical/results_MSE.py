import numpy as np

nums = 1600
img_unet = np.load("optical_gnn_unet_result.npy").reshape(200, nums)
img_ae = np.load("optical_gnn_result.npy").reshape(200, nums)
label1 = np.load("optical_data.npy").reshape(200, nums)
# 生成随机预测数据和目标数据
img_num = 100
# 计算均方差
mse_unet = np.mean(np.square(img_unet - label1))
mse_ae = np.mean(np.square(img_ae - label1))
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 定义数据
UNET_MSE = mse_unet
AE_MSE = mse_ae

# 定义数据名称
data_names = ['UNET_MSE', 'AE_MSE']

# 定义数据大小
data_sizes = [UNET_MSE, AE_MSE]

# 定义柱状图的颜色
colors = ['blue', 'red']

# 绘制柱状图
plt.bar(data_names, data_sizes, color=colors)

# 在柱子上显示高度
for i, v in enumerate(data_sizes):
    plt.text(i, v, str(v), ha='center')

# 添加标题并居中显示
plt.title('训练数据在不同网络中的均方差', loc='center')

# 显示图像
plt.show()
