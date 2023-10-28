import numpy as np
import matplotlib.pyplot as plt

def de(n=6, m_size=20, f=0.5, cr=0.3, iterate_times=30, x_l=np.array([10, 5, 2, 0, 0, 0]), x_u=np.array([20, 15, 8, 5, 5, 5])):
    """
    :param n:维度
    :param m_size:个体数
    :param f:缩放因子
    :param cr:交叉概率
    :param iterate_times:迭代次数
    :param x_l:最小边界
    :param x_u:最大边界
    """

    # 初始化第一代
    x_all = np.zeros((iterate_times, m_size, n))
    for i in range(m_size):
        x_all[0][i] = x_l + np.random.random() * (x_u - x_l)
    # print("x_all[0] = ",x_all[0])
    for g in range(iterate_times - 1):
        for i in range(m_size):
            # 变异操作,对第g代随机抽取三个组成一个新的个体,对于第i个个体来说,原来的第i个个体和它无关
            x_g_without_i = np.delete(x_all[g], i, 0)
            np.random.shuffle(x_g_without_i)
            h_i = x_g_without_i[1] + f * (x_g_without_i[2] - x_g_without_i[3])
            # 处理变异操作后有可能超过区间
            h_i = [h_i[item] if h_i[item] < x_u[item] else x_u[item] for item in range(n)]
            h_i = [h_i[item] if h_i[item] > x_l[item] else x_l[item] for item in range(n)]
            # 交叉操作,对变异后的个体,根据随机数与交叉阈值确定最后的个体
            v_i = np.array([x_all[g][i][j] if np.random.random() > cr else h_i[j] for j in range(n)])
            print("g =", g, "i=", i)
            print("v_i = ", v_i)
            # 根据评估函数确定是否更新新的个体
            if evaluate_func(x_all[g][i]) > evaluate_func(v_i):
                x_all[g + 1][i] = v_i
            else:
                x_all[g + 1][i] = x_all[g][i]
    evaluate_result = [evaluate_func(x_all[iterate_times - 1][i]) for i in range(m_size)]
    print("x_all = ", x_all)
    best_x_i = x_all[iterate_times - 1][np.argmin(evaluate_result)]
    print("evaluate_result = ", evaluate_result)
    print("best_x_i=", best_x_i)
    plt.plot(evaluate_result)
    plt.show()


def evaluate_func(x):
    a = x[0]
    b = x[1]
    c = x[2]
    d = x[3]
    return 4 * a ** 2 - 3 * b + 5 * c ** 3 - 6 * d


if __name__ == '__main__':
    for i in range(3):
        de()
