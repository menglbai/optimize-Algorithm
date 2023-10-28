import numpy as np
import matplotlib.pyplot as plt

# 粒子群算法
def PSO(func, dim, swarm_size, max_iter):
    # 初始化粒子群和参数
    swarm = np.random.uniform(-5.12, 5.12, (swarm_size, dim))
    velocity = np.zeros((swarm_size, dim))
    pbest = swarm.copy()
    pbest_fitness = np.array([func(p) for p in pbest])
    gbest = pbest[pbest_fitness.argmin()].copy()
    gbest_fitness = pbest_fitness.min()

    # 迭代优化
    for i in range(max_iter):
        # 更新速度和位置
        r1 = np.random.rand(swarm_size, dim)
        r2 = np.random.rand(swarm_size, dim)
        velocity = 0.5 * velocity + 2 * r1 * (pbest - swarm) + 2 * r2 * (gbest - swarm)
        swarm = swarm + velocity

        # 计算适应度值
        fitness = np.array([func(p) for p in swarm])

        # 更新个体最优解和全局最优解
        update = fitness < pbest_fitness
        pbest[update] = swarm[update]
        pbest_fitness[update] = fitness[update]
        if fitness.min() < gbest_fitness:
            gbest = swarm[fitness.argmin()].copy()
            gbest_fitness = fitness.min()

    # 返回最优解和适应度值
    return gbest, gbest_fitness

# 遗传算法
def GA(func, dim, pop_size, max_iter):
    # 初始化种群和参数
    population = np.random.uniform(-5.12, 5.12, (pop_size, dim))
    fitness = np.array([func(p) for p in population])

    # 迭代优化
    for i in range(max_iter):
        # 选择
        fitness_sum = np.sum(fitness)
        p = fitness / fitness_sum
        indices = np.random.choice(pop_size, size=pop_size, replace=True, p=p)
        parents = population[indices]

        # 交叉
        r = np.random.rand(pop_size, dim)
        children = parents.copy()
        mask = r < 0.5
        children[~mask] = np.roll(parents, shift=1, axis=0)[~mask]

        # 变异
        r = np.random.rand(pop_size, dim)
        mask = r < 0.1
        indices = np.where(mask)
        for i, j in zip(indices[0], indices[1]):
            children[i, j] = np.random.uniform(-5.12, 5.12)

        # 计算适应度值
        children_fitness = np.array([func(c) for c in children])

        # 更新种群
        update = children_fitness < fitness
        population[update] = children[update]
        fitness[update] = children_fitness[update]

    # 返回最优解和适应度值
    return population[fitness.argmin()], fitness.min()

# 测试函数
def sphere(x):
    return (np.array(x) ** 2).sum()

# 运行算法并绘制图表
if __name__ == '__main__':
    dim = 2
    swarm_size = 20
    max_iter = 50
    pop_size = 100
    ga_max_iter = 10

    pso_best, pso_fitness = PSO(sphere, dim, swarm_size, max_iter)
    ga_best, ga_fitness = GA(sphere, dim, pop_size, ga_max_iter)

    # 绘制图表
    x = np.linspace(-5.12, 5.12, 100)
    y = np.linspace(-5.12, 5.12, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = sphere([X[i, j], Y[i, j]])
    fig, ax = plt.subplots()
    ax.contour(X, Y, Z, levels=np.logspace(0, 5, 35), cmap='gray')
    ax.scatter(pso_best[0], pso_best[1], color='red', label='PSO')
    ax.scatter(ga_best[0], ga_best[1], color='blue', label='GA')
    ax.legend()
    plt.show()
