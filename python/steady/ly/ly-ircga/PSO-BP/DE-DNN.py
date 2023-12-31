from sko.PSO import PSO
import matplotlib.pyplot as plt
from DNN import fitModel
from DNN import evaluate
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from sko.DE import DE
import numpy as np
import time

# 数据预处理
def processData():
    '''
    前8000训练 后2000测试，做规范化
    :return:
    '''
    path = "dataset/dncj_lxb_10000.csv"  # 存放文件路径
    df = pd.read_csv(path, header=None)  # 读取文件
    target = df.iloc[:, -1]
    # data = df.iloc[:, 0:-1]
    data = pd.concat([df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], df.iloc[:, 4], df.iloc[:, 11]], axis=1)
    # python归一化函数MinMaxScaler的理解：对x归一化
    scaler = MinMaxScaler().fit(data)
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, shuffle=False)
    # 变换后各维特征有0均值，单位方差。也叫z-score规范化(零均值规范化)。
    # 计算方式是将特征值减去均值，除以标准差。  scaler.transform(X_train)
    x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
    return x_train, x_test, y_train, y_test


# 适应度函数采用测试集合mae
def demo_func(x):
    n1, n2, n3, a1, a2, a3, d1 = x
    model = fitModel(x_train, y_train, n1, n2, n3, a1, a2, a3, d1)
    mse = evaluate(model, x_test, y_test)
    return mse


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = processData()
    print('x_train', x_train)
    print('x_test', x_test)

    start = time.time()
    de = DE(func=demo_func, n_dim=7, size_pop=20, max_iter=50, lb=[10, 5, 2, 0, 0, 0, 0.01],
            ub=[20, 15, 8, 5, 5, 5, 0.95])
    best_x, best_y = de.run()
    end = time.time()
    print('算法耗时：', (end - start) / 60, 'min')
    print('best_x:', best_x, '\n', 'best_y:', best_y)
    # print(de.generation_best_Y)
    # print(de.all_history_Y)
    Y_history = pd.DataFrame(de.all_history_Y)
    gbest_history = np.array(Y_history.min(axis=1).cummin())
    print('gbest_history: ', gbest_history)
    plt.plot(gbest_history)
    plt.show()
