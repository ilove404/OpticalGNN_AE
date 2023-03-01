import os

import numpy as np
from numpy import pi as pi


# 基本参数设置区
def dataset_creat(move_phase, nums):
    # 噪声强度
    noise = np.random.normal(0, 1, 1600)
    x = range(nums)
    y_data = []
    y_label = []
    y_labels = []
    print("程序正运行")
    m = np.linspace(0, 4, move_phase)
    ####################### 原始数据和带噪声，move_phase数量组的数据 #############################
    for delta_num in range(1, 51):
        delta = 0.016 * delta_num
        print(delta)
        class_nums = 0
        print("运行中：", delta_num)
        for i in m:
            for numss in x:
                result = np.cos(numss * pi / nums * 2 + i * pi / 4)
                result1 = np.cos(numss * pi / nums * 2 + (i * pi / 4) + delta * noise[numss])
                y_label.append(result)
                y_data.append(result1)
            y_labels.append(class_nums)
            class_nums += 1

    y_label = np.array(y_label)
    y_data = np.array(y_data)
    y_labels = np.array(y_labels)

    # for n in range(1):
    #     plt.plot(x, y_label[1600 * n:(n + 1) * 1600])

    return y_data, y_label, y_labels


'''_,a= dataset_creat()
b = a.reshape(4,1600)
y = range(1600)
x = range(1600)
for n in range(1):
    plt.plot(x, a[1600:3200])
plt.show()'''
if __name__ == "__main__":
    dataset = dataset_creat(4, 1600)
    if not os.path.exists('./data_optical'):
        os.mkdir('./data_optical')
    np.save("./data_optical/optical_noise.npy", dataset[0])
    np.save("./data_optical/optical_data.npy", dataset[1])
    np.save("./data_optical/optical_label.npy", dataset[2])
