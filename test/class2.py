import numpy as np

a = np.array([1,2,3])
b = np.array([4,5,6])
c = np.vstack((a, b)) #垂直方向叠加

x, y = np.mgrid[1:3:1, 2:4:0.5]
# print(x)
# print(y)

