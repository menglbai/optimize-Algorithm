def getAccuracy(y_test, y_hat):
    '''

    :param y_test:
    :param y_hat:
    :return:
    '''
    y_pred = y_hat
    y_true = y_test
    successNum = 0
    # print(y_pred[0])
    # print(y_true[0])
    total = y_pred.size
    print('测试集数目： ', total)
    for i in range(total):
        # print('RBF震塌预测值', y_pred[i], '震塌真实值', y_true[i], '预测-真实值=', abs(y_pred[i] - y_true[i]))

        # 计算出BP的预测值与真实值的差值绝对值
        collape_D_value = abs(y_pred[i] - y_true[i])  # 达到95%

        # 计算出  达到准确等级的数据量 successNum_*
        successNum = getcpSuccess(y_pred[i], y_true[i], collape_D_value, successNum)

        # print("BP 震塌比例正确率", successNum / total)
    accuracy = successNum / total
    print('正确率： ', accuracy)
    return accuracy


# 先划分等级，再判断是否在误差范围内
def getcpSuccess(y_pred, y_true, collape_D_value, successNum):
    if y_true < 0.05 and y_pred < 0.05:  # 划分等级为 ：等级1或2
        if collape_D_value < 0.01:  # 允许的误差范围
            successNum = successNum + 1
    if 0.05 < y_true < 0.15 and 0.05 < y_pred < 0.15:  # 划分等级为： 等级3
        if collape_D_value < 0.02:  # 允许的误差范围
            successNum = successNum + 1

    if 0.15 < y_true < 0.3 and 0.15 < y_pred < 0.3:  # 划分等级为： 等级4
        if collape_D_value < 0.03:  # 允许的误差范围
            successNum = successNum + 1

    if 0.3 < y_true < 1 and 0.3 < y_pred < 1:  # 划分等级为： 等级5
        if collape_D_value < 0.05:  # 允许的误差范围
            successNum = successNum + 1
    return successNum
