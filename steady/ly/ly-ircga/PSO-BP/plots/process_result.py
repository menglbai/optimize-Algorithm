# best_x is  [2.54455826e+01 2.00000000e+01 2.18363262e+00 5.00000000e+00
#  3.85574648e+00 4.13370878e+00 1.00000000e-02] best_y is [6.90369634e-05]

# pso-bp：
# data = '[array([0.00010679]), array([8.80146399e-05]), array([8.16444808e-05]), array([7.65222649e-05]), array([7.65222649e-05]), array([7.60353287e-05]), array([7.60353287e-05]), array([7.51837142e-05]), array([7.51837142e-05]), array([7.51837142e-05]), array([7.51837142e-05]), array([7.51837142e-05]), array([7.47951999e-05]), array([7.47951999e-05]), array([7.47951999e-05]), array([7.47951999e-05]), array([7.4592157e-05]), array([7.4592157e-05]), array([7.4592157e-05]), array([7.4592157e-05]), array([7.4592157e-05]), array([7.4592157e-05]), array([7.4592157e-05]), array([7.4592157e-05]), array([7.37283772e-05]), array([7.37283772e-05]), array([7.37283772e-05]), array([7.30367756e-05]), array([7.30367756e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05])]'
# data = data.replace('array([', '')
# data = data.replace('])', '')
# print(data)

# 均方误差mse；  7.924284000182524e-05
# 平均绝对误差mae：  0.006076187361031771
# 相关系数 r2:   0.9852379231639253
# 测试集数目：  2000
# 正确率：  0.937
# 准确率 0.937
# pso-bp
# data = [7.56190566e-05, 7.56190566e-05, 7.56190566e-05, 6.91724781e-05, 6.91724781e-05, 6.91724781e-05, 6.91724781e-05, 6.91724781e-05, 6.91724781e-05, 6.91724781e-05, 6.91724781e-05, 6.91724781e-05, 6.91724781e-05, 6.91724781e-05, 6.90369634e-05, 6.90369634e-05, 6.90369634e-05, 6.90369634e-05, 6.90369634e-05, 6.90369634e-05, 6.90369634e-05, 6.90369634e-05, 6.90369634e-05, 6.90369634e-05, 6.90369634e-05]
# data = [0.00010679, 8.80146399e-05, 8.16444808e-05, 7.65222649e-05, 7.65222649e-05, 7.60353287e-05, 7.60353287e-05, 7.51837142e-05, 7.51837142e-05, 7.51837142e-05, 7.51837142e-05, 7.51837142e-05, 7.47951999e-05, 7.47951999e-05, 7.47951999e-05, 7.47951999e-05, 7.4592157e-05, 7.4592157e-05, 7.4592157e-05, 7.4592157e-05, 7.4592157e-05, 7.4592157e-05, 7.4592157e-05, 7.4592157e-05, 7.37283772e-05, 7.37283772e-05, 7.37283772e-05, 7.30367756e-05, 7.30367756e-05, 7.2837458e-05, 7.2837458e-05, 7.2837458e-05, 7.2837458e-05, 7.2837458e-05, 7.2837458e-05, 7.2837458e-05, 7.2837458e-05, 7.2837458e-05, 7.2837458e-05, 7.2837458e-05, 7.2837458e-05, 7.2837458e-05, 7.2837458e-05, 7.2837458e-05, 7.2837458e-05, 7.2837458e-05, 7.2837458e-05, 7.2837458e-05, 7.2837458e-05, 7.2837458e-05]
# plt.plot(data)
# plt.show()


# -------------------------------------------------------------------开始实验数据

# 实验1
# pso-dnn
# best_x is  [2.0000000e+01 1.4740703e+01 8.0000000e+00 0.0000000e+00 0.0000000e+00
#  5.0000000e+00 1.0000000e-02] best_y is [7.2837458e-05]
# [array([0.00010679]), array([8.80146399e-05]), array([8.16444808e-05]), array([7.65222649e-05]), array([7.65222649e-05]), array([7.60353287e-05]), array([7.60353287e-05]), array([7.51837142e-05]), array([7.51837142e-05]), array([7.51837142e-05]), array([7.51837142e-05]), array([7.51837142e-05]), array([7.47951999e-05]), array([7.47951999e-05]), array([7.47951999e-05]), array([7.47951999e-05]), array([7.4592157e-05]), array([7.4592157e-05]), array([7.4592157e-05]), array([7.4592157e-05]), array([7.4592157e-05]), array([7.4592157e-05]), array([7.4592157e-05]), array([7.4592157e-05]), array([7.37283772e-05]), array([7.37283772e-05]), array([7.37283772e-05]), array([7.30367756e-05]), array([7.30367756e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05]), array([7.2837458e-05])]
# pso_data = [7.56190566e-05, 7.56190566e-05, 7.56190566e-05, 6.91724781e-05, 6.91724781e-05, 6.91724781e-05,
#             6.91724781e-05, 6.91724781e-05, 6.91724781e-05, 6.91724781e-05, 6.91724781e-05, 6.91724781e-05,
#             6.91724781e-05, 6.91724781e-05, 6.90369634e-05, 6.90369634e-05, 6.90369634e-05, 6.90369634e-05,
#             6.90369634e-05, 6.90369634e-05, 6.90369634e-05, 6.90369634e-05, 6.90369634e-05, 6.90369634e-05,
#             6.90369634e-05]

# ga-dnn
# best_x: [1.5e+01 1.5e+01 8.0e+00 5.0e+00 3.0e+00 5.0e+00 1.0e-02]
#  best_y: [7.45889483e-05]
# ga_data = [9.165023220703006e-05, 8.286112279165536e-05, 8.286112279165536e-05, 8.08493496151641e-05,
#            7.7084157965146e-05, 7.7084157965146e-05, 7.7084157965146e-05, 7.7084157965146e-05, 7.7084157965146e-05,
#            7.7084157965146e-05, 7.7084157965146e-05, 7.7084157965146e-05, 7.7084157965146e-05, 7.7084157965146e-05,
#            7.7084157965146e-05, 7.374148117378354e-05, 7.374148117378354e-05, 7.374148117378354e-05,
#            7.374148117378354e-05, 7.374148117378354e-05, 7.374148117378354e-05, 7.374148117378354e-05,
#            7.374148117378354e-05, 7.374148117378354e-05, 7.374148117378354e-05, 7.374148117378354e-05,
#            7.374148117378354e-05, 7.374148117378354e-05, 7.374148117378354e-05, 7.374148117378354e-05,
#            7.374148117378354e-05, 7.374148117378354e-05, 7.374148117378354e-05, 7.374148117378354e-05,
#            7.374148117378354e-05, 7.374148117378354e-05, 7.374148117378354e-05, 7.374148117378354e-05,
#            7.374148117378354e-05, 7.374148117378354e-05, 7.374148117378354e-05, 7.374148117378354e-05,
#            7.374148117378354e-05, 7.374148117378354e-05, 7.374148117378354e-05, 7.374148117378354e-05,
#            7.374148117378354e-05, 7.374148117378354e-05, 7.119737711036578e-05, 7.119737711036578e-05]


# de-dnn
# best_x: [1.59936359e+01 9.88109060e+00 2.87561082e+00 7.54305468e-01
#  2.51748092e+00 3.97686027e+00 1.19414008e-02]
#  best_y: [0.00010135]
# gbest_history:  [1.21946854e-04 1.15103918e-04 1.15103918e-04 1.15103918e-04
#  1.15103918e-04 1.15103918e-04 1.02255231e-04 1.02255231e-04
#  1.02255231e-04 1.02255231e-04 9.24278938e-05 9.24278938e-05
#  9.24278938e-05 9.24278938e-05 9.24278938e-05 9.24278938e-05
#  9.24278938e-05 9.24278938e-05 9.24278938e-05 9.24278938e-05
#  9.24278938e-05 9.24278938e-05 9.24278938e-05 9.24278938e-05
#  9.24278938e-05 9.24278938e-05 9.24278938e-05 9.24278938e-05
#  9.24278938e-05 9.24278938e-05 9.24278938e-05 9.24278938e-05
#  9.24278938e-05 9.24278938e-05 8.98067374e-05 8.98067374e-05
#  8.98067374e-05 8.98067374e-05 8.98067374e-05 8.98067374e-05
#  8.86371854e-05 8.86371854e-05 8.86371854e-05 8.86371854e-05
#  8.86371854e-05 8.86371854e-05 8.86371854e-05 8.86371854e-05
#  8.17792679e-05 8.17792679e-05]


#
# def findmin(a, i):
#     if (i ==0):
#         return a[0]
#     arr = a[0:i+1]
#     return min(arr)
#
#
# res = np.zeros(50)
# for i in range(50):
#     res[i] = findmin(ga_data,i)
#
# print(str(res))


# 随机超参数
# 均方误差mse；  0.00010088504495797679
# 平均绝对误差mae：  0.006772741209715605
# 相关系数 r2:   0.9812062113694663
# 测试集数目：  2000
# 正确率：  0.9225
# 准确率 0.9225

# res = '1.21946854e-04 1.15103918e-04 1.15103918e-04 1.15103918e-04 1.15103918e-04 1.15103918e-04 1.02255231e-04 1.02255231e-04 1.02255231e-04 1.02255231e-04 9.24278938e-05 9.24278938e-05 9.24278938e-05 9.24278938e-05 9.24278938e-05 9.24278938e-05 9.24278938e-05 9.24278938e-05 9.24278938e-05 9.24278938e-05 9.24278938e-05 9.24278938e-05 9.24278938e-05 9.24278938e-05 9.24278938e-05 9.24278938e-05 9.24278938e-05 9.24278938e-05 9.24278938e-05 9.24278938e-05 9.24278938e-05 9.24278938e-05 9.24278938e-05 9.24278938e-05 8.98067374e-05 8.98067374e-05 8.98067374e-05 8.98067374e-05 8.98067374e-05 8.98067374e-05 8.86371854e-05 8.86371854e-05 8.86371854e-05 8.86371854e-05 8.86371854e-05 8.86371854e-05 8.86371854e-05 8.86371854e-05 8.17792679e-05 8.17792679e-05'
#
# print(res.replace(" ", ","))


# 经验给定超参数
# 63/63 [==============================] - 0s 2ms/step - loss: 1.5193e-04 - mae: 0.0099
# 均方误差mse；  0.00015192887804005295
# 平均绝对误差mae：  0.009887592867016792
# 相关系数 r2:   0.9716973054575503
# 测试集数目：  2000
# 正确率：  0.859
# 准确率 0.859
# dict_keys(['loss', 'mae', 'val_loss', 'val_mae'])
# GA-DNN
# 63/63 [==============================] - 0s 2ms/step - loss: 8.2220e-05 - mae: 0.0062
# 均方误差mse；  8.221981261158362e-05
# 平均绝对误差mae：  0.0061916327103972435
# 相关系数 r2:   0.9846833450300813
# 测试集数目：  2000
# 正确率：  0.935
# 准确率 0.935
# dict_keys(['loss', 'mae', 'val_loss', 'val_mae'])
# DE-DNN
# 63/63 [==============================] - 0s 2ms/step - loss: 1.2807e-04 - mae: 0.0079
# 均方误差mse；  0.00012807473831344396
# 平均绝对误差mae：  0.007868592627346516
# 相关系数 r2:   0.97614107143625
# 测试集数目：  2000
# 正确率：  0.8965
# 准确率 0.8965
# dict_keys(['loss', 'mae', 'val_loss', 'val_mae'])
# pso-DNN
# 63/63 [==============================] - 0s 2ms/step - loss: 7.5913e-05 - mae: 0.0059
# 均方误差mse；  7.591261237394065e-05
# 平均绝对误差mae：  0.0059110806323587894
# 相关系数 r2:   0.9858583062755402
# 测试集数目：  2000
# 正确率：  0.937
# 准确率 0.937


# data = '[array([7.98013789e-05]), array([7.98013789e-05]), array([7.98013789e-05]), array([7.98013789e-05]), array([7.98013789e-05]), array([7.98013789e-05]), array([7.98013789e-05]), array([7.98013789e-05]), array([7.98013789e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05])]'
# data = data.replace('array([', '')
# data = data.replace('])', '')
# print(data)


# pso-DNN  实验2
# 12,7,7,0,2,5,0.01
# [7.98013789e-05, 7.98013789e-05, 7.98013789e-05, 7.98013789e-05, 7.98013789e-05, 7.98013789e-05, 7.98013789e-05, 7.98013789e-05, 7.98013789e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05, 7.55353758e-05]


# ga-dnn实验2
# 16,15,7,5,5,5,0.01
# [7.88747420e-05, 7.88747420e-05, 7.55500878e-05, 7.55500878e-05, 7.55500878e-05, 7.43528944e-05, 7.43528944e-05,
#  7.28878367e-05, 7.28878367e-05, 7.28878367e-05, 7.28878367e-05, 7.10031454e-05, 7.10031454e-05, 7.10031454e-05,
#  7.10031454e-05, 7.10031454e-05, 7.10031454e-05, 7.10031454e-05, 7.10031454e-05, 7.10031454e-05, 7.10031454e-05,
#  7.10031454e-05, 7.10031454e-05, 7.10031454e-05, 7.10031454e-05, 7.10031454e-05, 7.10031454e-05, 7.10031454e-05,
#  7.10031454e-05, 7.10031454e-05, 7.10031454e-05, 7.10031454e-05, 7.10031454e-05, 7.10031454e-05, 7.10031454e-05,
#  7.10031454e-05, 7.10031454e-05, 7.10031454e-05, 7.10031454e-05, 7.10031454e-05, 7.10031454e-05, 7.10031454e-05,
#  7.10031454e-05, 7.10031454e-05, 7.10031454e-05, 7.10031454e-05, 7.10031454e-05, 7.10031454e-05, 7.10031454e-05,
#  7.10031454e-05]

#
# data = '[1.95718705e-04 1.21005811e-04 1.21005811e-04 9.56609802e-05 8.89709627e-05 8.89709627e-05 8.89709627e-05 8.89709627e-05 8.89709627e-05 8.89709627e-05 8.76932972e-05 8.47015763e-05 8.47015763e-05 8.47015763e-05 8.47015763e-05 8.47015763e-05 8.47015763e-05 8.47015763e-05 8.47015763e-05 8.47015763e-05 8.47015763e-05 8.47015763e-05 8.47015763e-05 8.47015763e-05 8.47015763e-05 8.05300078e-05 8.05300078e-05 8.05300078e-05 8.05300078e-05 8.05300078e-05 8.05300078e-05 8.05300078e-05 8.05300078e-05 8.05300078e-05 8.05300078e-05 8.05300078e-05 8.05300078e-05 8.05300078e-05 8.05300078e-05 8.05300078e-05 8.05300078e-05 8.05300078e-05 8.05300078e-05 8.05300078e-05 8.05300078e-05 8.05300078e-05 8.05300078e-05 8.05300078e-05 8.05300078e-05 8.05300078e-05]'
# # data = '[array([7.98013789e-05]), array([7.98013789e-05]), array([7.98013789e-05]), array([7.98013789e-05]), array([7.98013789e-05]), array([7.98013789e-05]), array([7.98013789e-05]), array([7.98013789e-05]), array([7.98013789e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05]), array([7.55353758e-05])]'
# # data = data.replace('array([', '')
# # data = data.replace('])', '')
# data = data.replace(' ', ',')
# print(data)


data = [7.617875235155225e-05, 9.20561287784949e-05, 8.484403224429116e-05, 8.758192416280508e-05, 8.550602797186002e-05, 8.712968701729551e-05, 9.303451224695891e-05, 9.012370719574392e-05, 8.105260349111632e-05, 9.856160613708198e-05, 9.129014506470412e-05, 0.00010116392513737082, 8.135032112477347e-05, 0.0001125484996009618, 8.282111230073497e-05, 0.00010753377864602953, 8.526702004019171e-05, 8.910532051231712e-05, 9.709560254123062e-05, 9.839226549956948e-05, 9.482763562118635e-05, 8.497375529259443e-05, 0.00010005906369769946, 0.00011409803119022399, 8.328011608682573e-05, 9.965740173356608e-05, 9.751433390192688e-05, 9.29154412006028e-05, 8.14417508081533e-05, 9.568154200678691e-05]

data.sort()
data.reverse()
print(data)