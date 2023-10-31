import time

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


# 画图
def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_' + metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])
    plt.show()


# 数据预处理
def processData():
    path = "./dataset/dncj_lxb_10000.csv"  # 存放文件路径
    df = pd.read_csv(path, header=None)  # 读取文件
    target = df.iloc[:, -1]
    # data = df.iloc[:, 0:-1]

    data = pd.concat([df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], df.iloc[:, 4], df.iloc[:, 11]], axis=1)

    # python归一化函数MinMaxScaler的理解：对x归一化
    scaler = MinMaxScaler().fit(data)
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, shuffle=False)
    # 变换后各维特征有0均值，单位方差。也叫z-score规范化(零均值规范化)。
    # 计算方式是将特征值减去均值，除以标准差。 scaler.transform(X_train)
    x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
    print(x_train.shape)
    print(y_train.shape)
    return x_train, x_test, y_train, y_test


# 训练模型
'''
n1,n2,n3 节点数
d1 dropout的概率
a1,a2,a3 激活函数
'''


def fitModel(x_train, y_train, n1, n2, n3, a1, a2, a3):
    print('当前的超参数为：', n1, n2, n3, a1, a2, a3)
    start = time.time()

    act = ['relu', 'sigmoid', 'softmax', 'tanh', 'softplus', 'softsign']

    tf.keras.backend.clear_session()  # 清除模型占用内存
    model = models.Sequential()
    model.add(layers.Dense(x_train.shape[1], activation='relu', input_shape=(x_train.shape[1],)))
    model.add(layers.Dense(n1, activation=act[int(a1)]))
    model.add(layers.Dense(n2, activation=act[int(a2)]))
    # model.add(layers.Dropout(d1))
    model.add(layers.Dense(n3, activation=act[int(a3)]))
    model.add(layers.Dense(1, activation='relu'))
    # model.summary()
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mae'])

    history = model.fit(x_train, y_train,
                        verbose=0,  # 不输出日志
                        batch_size=320,
                        epochs=200,
                        validation_split=0.2  # 分割一部分训练数据用于验证
                        )
    print(history.history.keys())

    # plot_metric(history, "loss")
    # plot_metric(history, "mae")
    # model.save('./model/pca_model.h5')

    end = time.time()
    print('模型训练时间：', end - start, '秒')
    return model


# 评估模型
def evaluate(model, x_test, y_test):
    mse, mae = model.evaluate(x=x_test, y=y_test)
    print('均方误差mse； ', mse)
    print('平均绝对误差mae： ', mae)
    print('相关系数 r2:  ', r2_score(np.array(y_test), np.array(model.predict(x_test))))
    print('准确率', Accuracy.getAccuracy(np.array(y_test), np.array(model.predict(x_test))))
    return mse, mae
    # return mse


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = processData()
    # # # 随机超参数评估
    # model = fitModel(x_train, y_train, 15, 8, 4, 0, 0, 0, 0.05)
    # print('经验给定超参数')
    # evaluate(model, x_test, y_test)
    # model.save('./model/random_dnn_model.h5')

    # # # GA-DNN
    # model = fitModel(x_train, y_train, 16, 15, 7, 5, 5, 5, 0.01)
    # print('GA-DNN')
    # evaluate(model, x_test, y_test)
    # model.save('./model/ga_dnn_model.h5')

    # # DE-DNN
    # model = fitModel(x_train, y_train, 15, 9, 2, 0, 2, 3, 0.01)
    # print('DE-DNN')
    # evaluate(model, x_test, y_test)
    # model.save('./model/de_dnn_model.h5')

    # # # pso-DNN
    # model = fitModel(x_train, y_train, 12, 7, 7, 0, 2, 5, 0.01)
    # print('pso-DNN')
    # evaluate(model, x_test, y_test)
    # model.save('./model/pso_dnn_model.h5')

    # # IGA-DNN
    model = fitModel(x_train, y_train, 16, 15, 7, 5, 5, 5, 0.01)
    print('IGA-DNN')
    evaluate(model, x_test, y_test)
    model.save('./model/iga_dnn_model.h5')
