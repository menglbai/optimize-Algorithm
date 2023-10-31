import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def relu(x):
    # relu函数
    return np.maximum(0, x)


def sigmoid(x):
    # sigmoid函数
    return 1 / (1 + np.exp(-x))


def step_function(x):
    # 阶跃函数
    return np.array(x > 0, dtype=np.int)  # 先计算bool值，再转成int


def show(x, y, ylim):
    # 画图
    font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    plt.grid()
    plt.plot(x, y, label='max(0,x)')
    plt.ylim(ylim)  # y轴范围
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')

    plt.title('Relu激活函数', fontproperties=font_set)
    plt.legend(loc=0, prop={'size': 16}, ncol=1)
    plt.savefig("./relu" + ".png", bbox_inches='tight', dpi=600, pad_inches=0.1)

    plt.show()  # plot在内存画，show一次性将内存的显示出来


def show_relu(x):
    # 展示relu函数图像
    y = relu(x)
    ylim = (-1.0, 10.5)  # y轴的范围，比输入的大d
    show(x, y, ylim)


def show_sigmoid(x):
    # 展示sigmoid函数图像
    y = sigmoid(x)
    ylim = (-0.1, 1.1)
    show(x, y, ylim)


def show_step(x):
    # 展示阶跃函数图像
    y = step_function(x)
    ylim = (-0.1, 1.1)
    show(x, y, ylim)


def show_sig_step_compare(x):
    # 对比阶跃函数和sigmoid函数图像
    y_sig = sigmoid(x)
    y_step = step_function(x)
    plt.plot(x, y_sig)
    plt.plot(x, y_step, 'k--')
    plt.ylim(-0.1, 1.1)
    plt.show()


def show_feature():
    x = np.arange(0, 25, 1)  # x范围
    y = [0, 15, 25, 23, 20, 18, 16, 14, 12, 10,
         8, 7, 6, 5, 4.5, 4, 3.6, 3.3, 3.0, 2.85,
         2.5, 2.2, 1.9, 1.5, 1.2]
    ylim = (0, 25)
    plt.grid()
    plt.xticks(color='w')
    plt.yticks(color='w')

    plt.plot(x, y, label='max(0,x)')
    plt.ylim(ylim)  # y轴范围
    plt.xlabel('Dimensionality(number of fretures)')
    plt.ylabel('Classifier performance ')
    # plt.title('Relu Activation Function')
    # plt.legend(loc=0, prop={'size': 16}, ncol=1)
    plt.vlines(2,0, 25, linestyles='dashed', colors='k')
    plt.text(0.2, 0.5, "Optimal number of features", fontsize=10)
    plt.savefig("./feature" + ".png", bbox_inches='tight', dpi=600, pad_inches=0.1)
    plt.show()  # plot在内存画，show一次性将内存的显示出来


x = np.arange(-10, 10.5, 1)  # x范围



show_relu(x)

# show_feature()
# show_sigmoid(x)
# show_step(x)
# show_sig_step_compare(x)
