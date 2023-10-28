from sko.PSO import PSO
import matplotlib.pyplot as plt
from iga.DNN import fitModel
from iga.DNN import evaluate
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from iga.IGA_66 import IGA
import numpy as np
import time
import datetime as dt
from save_result import save_result_txt
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



# 适应度函数采用测试集合mae
def demo_func(x):
    n1, n2, n3, a1, a2, a3 = x
    model = fitModel(x_train, y_train, n1, n2, n3, a1, a2, a3)
    mse = evaluate(model, x_test, y_test)
    return mse


x_train, x_test, y_train, y_test = processData()


# print('x_train', x_train)
# print('x_test', x_test)

def run():
    # 将w的精度precision设置为1（即为整数），将
    # 变异因子prop_mut设置为0.1
    # 精度 precision，每个变量的精度，取值1 代表整数

    # 记录开始时间
    start = time.time()
    gbest, gbest_mae = IGA(func=demo_func, cxpb=0.8, mutpb=0.1, ngen=30, popsize=20, up=[21, 16, 9, 5, 5, 5],
                            low=[10, 5, 2, 0, 0, 0])
    # ga = GA(func=demo_func, n_dim=7, size_pop=20, max_iter=50, prob_mut=0.1,
    #         lb=[10, 5, 2, 0, 0, 0, 0.01], ub=[20, 15, 8, 5, 5, 5, 0.95], precision=1)
    # 记录结束时间
    end = time.time()
    runTime = end - start
    save_result_txt('./experiment_result/IGA实验2结果' + str(int(dt.datetime.now().timestamp())) + '.txt',
                    'MAE结果：' + str(gbest_mae) + '\n' +
                    'MSE变化：' + str(gbest) +
                    '\n运行时间：' + str(runTime / 60) + ' min')


if __name__ == '__main__':
    for i in range(3):
        run()
