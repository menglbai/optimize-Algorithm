import time

from sko.PSO import PSO
import matplotlib.pyplot as plt
from ga.DNN import fitModel
from ga.DNN import evaluate
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from ga.SGA import SGA
import numpy as np

from save_result import save_result_txt
from time import perf_counter
import datetime as dt

count = 0


# 数据预处理
def processData():
    path = "../dataset/qb_lxb_balance_10000.csv"  # 存放文件路径
    # target = df.iloc[:, -1]
    # # data = df.iloc[:, 0:-1]
    # data = pd.concat([df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], df.iloc[:, 4], df.iloc[:, 11]], axis=1)

    df = pd.read_csv(path)
    data = df[['ammoQuantity', 'liningPlateThick', 'outerHeight', 'outerSideLength', 'wallThick']]
    target = df['collapse']
    # python归一化函数MinMaxScaler的理解：对x归一化
    # scaler = MinMaxScaler().fit(data)
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, shuffle=False)
    # 变换后各维特征有0均值，单位方差。也叫z-score规范化(零均值规范化)。
    # 计算方式是将特征值减去均值，除以标准差。 scaler.transform(X_train)
    # x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
    print(x_train.shape)
    print(y_train.shape)
    return x_train, x_test, y_train, y_test



x_train, x_test, y_train, y_test = processData()



# 适应度函数采用测试集合mae
def demo_func(x):
    n1, n2, n3, a1, a2, a3 = x
    model = fitModel(x_train, y_train, n1, n2, n3, a1, a2, a3)
    mse, mae = evaluate(model, x_test, y_test)
    return mse, mae


def run():

    # 将w的精度precision设置为1（即为整数），将
    # 变异因子prop_mut设置为0.1
    # 精度 precision，每个变量的精度，取值1 代表整数
    # 记录开始时间
    start = time.time()
    # ga = SGA(func=demo_func, n_dim=7, size_pop=20, max_iter=1, prob_mut=0.1,
    #          lb=[10, 5, 2, 0, 0, 0, 0.01], ub=[20, 15, 8, 5, 5, 5, 0.95], precision=1)
    gbest, gbest_mae = SGA(func=demo_func, cxpb=0.8, mutpb=0.1, ngen=30, popsize=20, up=[20, 15, 8, 5, 5, 5],
                           low=[10, 5, 2, 0, 0, 0])
    # best_x, best_y = ga.run()
    # print('best_x:', best_x, '\n', 'best_y:', best_y)
    # print(ga.generation_best_Y)
    # 记录结束时间
    end = time.time()
    # 单位是秒
    runTime = end - start


    # Y_history = pd.DataFrame(ga.all_history_Y)
    # fig, ax = plt.subplots(2, 1)
    # ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    # Y_history.min(axis=1).cummin().plot(kind='line')
    # plt.show()
    save_result_txt('./experiment_result/GA实验2结果' + str(int(dt.datetime.now().timestamp())) + '.txt',
                    'MAE结果：' + str(gbest_mae) + '\n' +
                    'MSE变化：' + str(gbest) +
                    '\n运行时间：' + str(runTime/60) + ' min')
    print('gbest: ', gbest)


if __name__ == '__main__':
    for i in  range(5):
        run()