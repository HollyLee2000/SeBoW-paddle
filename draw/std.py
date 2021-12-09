import numpy as np

arr = [2, 4, 6, 8, 10, 12]
# 求均值
arr_mean = np.mean(arr)
# 求方差
arr_var = np.var(arr)
# 求标准差
arr_std = np.std(arr, ddof=1)
print("平均值为：%f" % arr_mean)
print("方差为：%f" % arr_var)
print("标准差为:%f" % arr_std)
