import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata


def defMap(mapRange):
    # 初始化地形信息
    N = 6  # 山峰个数
    peaksInfo = []  # 初始化山峰特征信息列表
    x = [0.35, 0.4, 0.55, 0.6, 0.4, 0.8]
    y = [0.2, 0.4, 0.45, 0.7, 0.8, 0.8]
    z = [0.3, 0.4, 0.6, 0.3, 0.4, 0.25]
    r = [0.7, 1, 0.8, 0.5, 0.4, 0.5]

    # 随机生成N个山峰的特征参数
    for i in range(N):
        peak = {}
        peak['center'] = [mapRange[0] * x[i], mapRange[1] * y[i]]
        peak['height'] = mapRange[2] * z[i]
        peak['range'] = np.multiply(mapRange, 0.1 * r[i])
        peaksInfo.append(peak)

    # 计算山峰曲面值
    peakData = np.zeros((mapRange[0], mapRange[1]))
    for x in range(mapRange[0]):
        for y in range(mapRange[1]):
            total_sum = 0
            for k in range(N):
                h_i = peaksInfo[k]['height']
                x_i = peaksInfo[k]['center'][0]
                y_i = peaksInfo[k]['center'][1]
                x_si = peaksInfo[k]['range'][0]
                y_si = peaksInfo[k]['range'][1]
                total_sum += h_i * np.exp(-((x - x_i) / x_si) ** 2 - ((y - y_i) / y_si) ** 2)
            peakData[x, y] = total_sum

    # 构造曲面网格，用于插值判断路径是否与山峰交涉
    x = np.repeat(np.arange(1, mapRange[0] + 1).reshape(-1, 1), mapRange[1], axis=1)
    y = np.tile(np.arange(1, mapRange[1] + 1).reshape(1, -1), (mapRange[0], 1))
    peakData = peakData.flatten()
    X, Y = np.meshgrid(np.linspace(np.min(x), np.max(x), 100), np.linspace(np.min(y), np.max(y), 100))
    Z = griddata((x.flatten(), y.flatten()), peakData, (X, Y), method='linear')

    return X, Y, Z


mapRange = [100, 100, 100]
X, Y, Z = defMap(mapRange)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.view_init(elev=20, azim=-45)  # 设置视角 ax.dist = 10 # 设置观察者与图像中心的距离


def on_mouse_move(event):
    if event.inaxes == ax:
        ax.view_init(elev=ax.elev + (event.ydata - event.y) / 2, azim=ax.azim + (event.x - event.xdata) / 2)
        fig.canvas.draw()
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
plt.show()
#
# # 绘制3D图像
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z, cmap='viridis')
#
# # 添加交互功能
# ax.view_init(elev=20, azim=-45)  # 设置视角
# ax.dist = 10  # 设置观察者与图像中心的距离
#
# def on_mouse_move(event):
#     if event.inaxes == ax:
#         ax.view_init(elev=ax.elev + (event.ydata - event.y) / 2, azim=ax.azim + (event.x - event.xdata) / 2)
#         fig.canvas.draw()
#
# fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
#
# plt.show()