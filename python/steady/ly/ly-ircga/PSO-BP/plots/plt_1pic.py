import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import Accuracy

from matplotlib.font_manager import FontProperties


# 数据预处理
def processData():
    path = "../dataset/dncj_lxb_20.csv"  # 存放文件路径
    df = pd.read_csv(path, header=None)  # 读取文件
    target = df.iloc[:, -1]

    data = pd.concat([df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], df.iloc[:, 4], df.iloc[:, 11]], axis=1)

    # python归一化函数MinMaxScaler的理解：对x归一化
    scaler = MinMaxScaler().fit(data)
    # x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, shuffle=False)
    # 变换后各维特征有0均值，单位方差。也叫z-score规范化(零均值规范化)。
    # 计算方式是将特征值减去均值，除以标准差。 scaler.transform(X_train)
    data = scaler.transform(data)

    return data, target


if __name__ == '__main__':
    data, target = processData()
    # print(np.array(target))
    # pso-dnn
    model = models.load_model('../model/pso_dnn_model.h5')
    pso_dnn_data = model.predict(data).flatten()
    # ga-dnn
    model = models.load_model('../model/ga_dnn_model.h5')
    ga_dnn_data = model.predict(data).flatten()
    # de-dnn
    model = models.load_model('../model/de_dnn_model.h5')
    de_dnn_data = model.predict(data).flatten()
    # random-dnn
    model = models.load_model('../model/random_dnn_model.h5')
    random_dnn_data = model.predict(data).flatten()

    # random-dnn
    model = models.load_model('../model/iga_dnn_model.h5')
    iga_dnn_data = model.predict(data).flatten()

    # x = np.linspace(0, 19, 20)
    x = list(range(20))

    print()

    colors = ['mediumpurple', 'hotpink', 'mediumseagreen', 'cornflowerblue', 'orange']
    line_types = ['v-', 'd-', '^-', '.-', '*-']
    plt.rc('font', family='serif')
    plt.rc('font', size=28)
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    plt.figure(figsize=(16, 16))
    grid = plt.GridSpec(1, 1, wspace=0.3, hspace=0.3)
    bwith = 2
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    # 中文字体
    font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=25)

    # 子图1  random-dnn
    axes1 = plt.subplot(grid[0, 0])
    axes1.set_title('(a)经验法', y=-0.25, fontsize='small', fontproperties=font_set)
    axes1.set_ylabel('震塌比例', fontproperties=font_set)
    axes1.set_xlabel('测试样本序列', fontproperties=font_set)
    line = axes1.plot(x, random_dnn_data, line_types[0], lw=2, label="Random-DNN", color=colors[0])
    # line1 = axes1.plot(x, target, line_types[1], lw=2, label="True", color=colors[1])
    line = axes1.plot(x, de_dnn_data, line_types[0], lw=2, label="DE-DNN", color=colors[1])
    line = axes1.plot(x, pso_dnn_data, line_types[0], lw=2, label="pso-DNN", color=colors[2])
    line = axes1.plot(x, ga_dnn_data, line_types[0], lw=2, label="GA-DNN", color=colors[3])
    line = axes1.plot(x, iga_dnn_data, line_types[0], lw=2, label="IGA-DNN", color=colors[4])
    points = axes1.plot(target, 'o', markersize=8, c='k', label="True")
    plt.legend(loc=0, prop={'size': 16}, ncol=1)


    path_name = "comparion_model_predict1"
    plt.savefig("./" + path_name + ".png", bbox_inches='tight', dpi=600, pad_inches=0.1)
    plt.show()
