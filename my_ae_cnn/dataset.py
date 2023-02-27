import numpy as np
from numpy import pi as pi

# 基本参数设置区
nums = 1600
wid = 40
long = 40
# 噪声强度
delta = 0.8

noise = np.random.normal(0, 1, nums)
source = np.arange(nums) * pi / nums * 2
source2 = source + pi / 4
source3 = source + 2 * pi / 4
source4 = source + 3 * pi / 4
