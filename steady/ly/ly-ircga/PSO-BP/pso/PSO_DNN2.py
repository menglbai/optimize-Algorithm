import numpy as np
import random
import matplotlib.pyplot as plt


from time import perf_counter
import datetime as dt


import matplotlib.pyplot as plt
from pso.DNN import fitModel
from pso.DNN import evaluate
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from iga.IGA import IGA
import numpy as np
import time
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

x_train, x_test, y_train, y_test = processData()

# # 适应度函数采用测试集合mae
# def fitness(x):
#     n1, n2, n3, a1, a2, a3, d1 = x
#     model = fitModel(x_train, y_train, n1, n2, n3, a1, a2, a3, d1)
#     mse = evaluate(model, x_test, y_test)
#     return mse


def get_epoch(lr):
    if lr > 0.1: return 3000
    if 0.1 >= lr > 0.01: return 5000
    if 0.01 >= lr > 0.001: return 7000
    if 0.001 >= lr: return 10000


def format_x(arr, max_value, min_value):
    for i in range(len(arr)):
        arr[i] = int(round(arr[i]))
        if arr[i] > max_value[i]: arr[i] = max_value[i]
        if arr[i] < min_value[i]: arr[i] = min_value[i]
    return arr


def format_loss(arr):
    return_arr = []
    for i in range(len(arr)):
        return_arr.append(min(arr[i]))
    return return_arr


# # 适应度函数--测试
# def evaluate(geneinfo):
#     x1 = geneinfo[0]
#     x2 = geneinfo[1]
#     x3 = geneinfo[2]
#     x4 = geneinfo[3]
#     y = x1 ** 2 - x2 ** 2 + x3 ** 3 / x4 ** 4
#     return y, y**2+y

# 适应度函数采用测试集合mae
def fitness(x):
    n1, n2, n3, a1, a2, a3 = x
    model = fitModel(x_train, y_train, int(n1), int(n2), int(n3), int(a1), int(a2), int(a3))
    mse, mae = evaluate(model, x_test, y_test)
    return mse, mae



class PSO_model:
    def __init__(self,func, w, c1, c2, r1, r2, N, D, M, up, low):
        self.w = w  # 惯性权值
        self.c1 = c1
        self.c2 = c2
        self.r1 = r1
        self.r2 = r2
        self.N = N  # 初始化种群数量个数
        self.D = D  # 搜索空间维度
        self.M = M  # 迭代的最大次数
        self.x = np.zeros((self.N, self.D))  # 粒子的初始位置
        self.v = np.zeros((self.N, self.D))  # 粒子的初始速度
        self.pbest = np.zeros((self.N, self.D))  # 个体最优值初始化
        self.gbest = np.zeros((1, self.D))  # 种群最优值
        self.up = up
        self.low = low

        self.p_fit = np.zeros(self.N)
        self.fit = 0  # 初始化全局最优适应度
        self.result = []  # 保存结果，每一轮的最优
        self.loss = []
        self.mae = []
        self.function =func

    # 初始化种群
    def init_pop(self):
        for i in range(self.N):
            for j in range(self.D): # 遍历每一个维度
                a = random.random()
                self.x[i][j] = int(round(a * (self.up[j] - self.low[j]) + self.low[j]))
            self.pbest[i] = self.x[i]  # 初始化个体的最优值
            mse, mae = self.function(self.x[i])  # 计算个体的适应度值
            self.p_fit[i] = mse  # 初始化个体的最优位置
            if mse < self.fit:  # 对个体适应度进行比较，计算出最优的种群适应度
                self.fit = mse
                self.gbest = self.x[i]

    # 更新粒子的位置与速度
    def update(self):
        for t in range(self.M):  # 在迭代次数M内进行循环
            current_loss = []
            current_mae = []
            for i in range(self.N):  # 对所有种群进行一次循环
                mse, mae = self.function(self.x[i])  # 计算一次目标函数的适应度
                if mse < self.p_fit[i]:  # 比较适应度大小，将小的负值给个体最优
                    self.p_fit[i] = mse
                    self.pbest[i] = self.x[i]
                    if self.p_fit[i] < self.fit:  # 如果是个体最优再将和全体最优进行对比
                        self.gbest = self.x[i]
                        self.fit = self.p_fit[i]
                current_loss.append(mse)
                current_mae.append(mae)
            self.result.append(self.fit)
            self.loss.append(current_loss)
            self.mae.append(current_mae)
            for i in range(self.N):  # 更新粒子的速度和位置
                self.v[i] = self.w * self.v[i] + self.c1 * self.r1 * (self.pbest[i] - self.x[i]) + self.c2 * self.r2 * (
                        self.gbest - self.x[i])
                self.x[i] = format_x((self.x[i] + self.v[i]), self.up, self.low)

        print("最优值：", self.fit, "位置为：", self.gbest)

    def plot(self):
        '''画图
        '''
        X = []
        Y = []
        self.result.sort()
        for i in range(self.M):
            X.append(i + 1)
            Y.append(self.result[i])
        plt.plot(X, Y)
        plt.xlabel('Number of iteration', size=15)
        plt.ylabel('accuracy', size=15)
        plt.title('PSO_BP parameter accuracy')
        plt.show()

    def plot_loss(self):
        '''画图
                '''
        X = []
        Y = []
        loss_arr = format_loss(self.loss)
        # loss_arr.sort()
        # loss_arr.reverse()
        for i in range(self.M):
            X.append(i + 1)
            Y.append(loss_arr[i])
        plt.plot(X, Y)
        plt.xlabel('Number of iteration', size=15)
        plt.ylabel('loss', size=15)
        plt.title('PSO_BP parameter loss')
        plt.show()


def PSO_MAIN():
    # w,c1,c2,r1,r2,N,D,M参数初始化
    w = random.random()
    c1 = c2 = 2  # 一般设置为2
    r1 = 0.7
    r2 = 0.5
    N = 20  # 初始化种群数量个数
    D = 6  # 搜索空间维度
    M = 30  # 迭代的最大次数
    up=[20, 15, 8, 5, 5, 5]  #上限
    low=[10, 5, 2, 0, 0, 0]  #下限
    start = time.time()
    pso_object = PSO_model(fitness, w, c1, c2, r1, r2, N, D, M, up, low)  # 设置初始权值
    pso_object.init_pop()
    pso_object.update()
    ### 保存数组到文件
    end = time.time()
    runTime = (end-start)/60

    save_result_txt('./experiment_result/PSO实验2结果数据' + str(int(dt.datetime.now().timestamp())) + '.txt',
                    'mae结果变化：' + str(format_loss(pso_object.mae)) + '\n' + 'loss（mse）变化：' + str(
                        format_loss(pso_object.loss)) + '\n运行时间：' + str(
                        runTime) + 'min')
    # pso_object.plot()
    pso_object.plot_loss()

if __name__ == '__main__':
    for i  in range(5):
        PSO_MAIN()