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


def plot_test_data(y_test, y_hat):
    index = range(1, len(y_test) + 1)
    plt.plot(index, y_test, 'bo--')
    plt.plot(index, y_hat, 'ro-')
    plt.title('y_test and y_hat_ ')
    plt.xlabel("index")
    plt.ylabel('value')
    plt.legend(["y_test" +'y_hat_'])
    plt.show()


# 数据预处理
def processData():
    path = "dataset/classify.csv"  # 存放文件路径
    df = pd.read_csv(path)  # 读取文件
    target = df.iloc[:, -1]
    data = df.iloc[:, 0:-1]

    # data = pd.concat([df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], df.iloc[:, 4], df.iloc[:, 11]], axis=1)

    # python归一化函数MinMaxScaler的理解：对x归一化
    scaler = MinMaxScaler().fit(data)
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, shuffle=True)
    # 变换后各维特征有0均值，单位方差。也叫z-score规范化(零均值规范化)。
    # 计算方式是将特征值减去均值，除以标准差。 scaler.transform(X_train)
    x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
    print(x_train.shape)
    print(y_train.shape)
    return x_train, x_test, y_train, y_test


# 训练模型
def fitModel(x_train, y_train):
    tf.keras.backend.clear_session()  # 清除模型占用内存
    model = models.Sequential()
    model.add(layers.Dense(x_train.shape[1], activation='relu', input_shape=(x_train.shape[1],)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(5, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['acc'])

    history = model.fit(x_train, y_train,
                        batch_size=64,
                        epochs=1000,
                        validation_split=0.1  # 分割一部分训练数据用于验证
                        )
    print(history.history.keys())
    plot_metric(history, "loss")
    plot_metric(history, "acc")
    model.save('./model/classify.h5')
    return model


# 评估模型
def evaluate(model, x_test, y_test):
    loss, acc = model.evaluate(x=x_test, y=y_test)
    print('loss； ', loss)
    print('acc： ', acc)
    print('相关系数 r2:', r2_score(np.array(y_test), np.array(model.predict(x_test))))
    # print('准确率', Accuracy.getAccuracy(np.array(y_test), np.array(model.predict(x_test))))
    y_hat = np.array(model.predict(x_test))
    y_class = np.array(model.predict_classes(x_test))
    print('预测值：',y_hat )
    print('预测分类：',y_class )
    print('真实值：', np.array(y_test))
    plot_test_data(y_test=y_test,y_hat=y_hat)


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = processData()
    model = fitModel(x_train, y_train)
    evaluate(model, x_test, y_test)
