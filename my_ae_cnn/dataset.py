import numpy as np
from numpy import pi as pi


# 基本参数设置区
def dataset_creat(move_phase, nums):
    # 噪声强度
    delta = 0.8
    noise = np.random.normal(0, 1, nums)
    x = range(nums)
    y_data = []
    y_label = []
    m = np.linspace(0, 4, move_phase)
    ####################### 原始数据和带噪声，move_phase数量组的数据 #############################
    for i in m:
        for numss in x:
            result = np.cos(numss * pi / nums * 2 + i * pi / 4)
            result1 = np.cos(numss * pi / nums * 2 + (i * pi / 4) + delta * np.random.normal(0, 1))
            y_label.append(result)
            y_data.append(result1)

    y_label = np.array(y_label)
    y_data = np.array(y_data)

    # for n in range(1):
    #     plt.plot(x, y_label[1600 * n:(n + 1) * 1600])

    return y_data, y_label


'''_,a= dataset_creat()
b = a.reshape(4,1600)
y = range(1600)
x = range(1600)
for n in range(1):
    plt.plot(x, a[1600:3200])
plt.show()'''
