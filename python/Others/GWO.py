import numpy as np

def initialization(pop, ub, lb, dim):
    x = np.random.rand(pop, dim)
    x = x * (ub - lb) + lb
    return x

def boundary_check(x, ub, lb):
    x[x > ub] = ub
    x[x < lb] = lb
    return x

def fitness(x): return np.sum(x**2, axis=1)

def GWO(pop, dim, ub, lb, fobj, max_iter):
    alpha_pos = np.zeros(dim)
    alpha_score = np.inf
    beta_pos = np.zeros(dim)
    beta_score = np.inf
    delta_pos = np.zeros(dim)
    delta_score = np.inf
    x = initialization(pop, ub, lb, dim)
    fitness = fobj(x)
    sorted_fitness = np.argsort(fitness)
    alpha_pos = x[sorted_fitness[0]]
    alpha_score = fitness[sorted_fitness[0]]
    beta_pos = x[sorted_fitness[1]]
    beta_score = fitness[sorted_fitness[1]]
    delta_pos = x[sorted_fitness[2]]
    delta_score = fitness[sorted_fitness[2]]
    group_best_pos = alpha_pos
    group_best_score = alpha_score

    iter_curve = np.zeros(max_iter)
    for t in range(max_iter):
        a = 2 - t * (2 / max_iter)

        for i in range(pop):
            for j in range(dim):
                r1 = np.random.rand()
                r2 = np.random.rand()
                a1 = 2 * a * r1 - a
                c1 = 2 * r2
                d_alpha = np.abs(c1 * alpha_pos[j] - x[i, j])
                x1 = alpha_pos[j] - a1 * d_alpha

                r1 = np.random.rand()
                r2 = np.random.rand()
                a2 = 2 * a * r1 - a
                c2 = 2 * r2
                d_beta = np.abs(c2 * beta_pos[j] - x[i, j])
                x2 = beta_pos[j] - a2 * d_beta

                r1 = np.random.rand()
                r2 = np.random.rand()
                a3 = 2 * a * r1 - a
                c3 = 2 * r2
                d_delta = np.abs(c3 * delta_pos[j] - x[i, j])
                x3 = delta_pos[j] - a3 * d_delta

                x[i, j] = (x1 + x2 + x3) / 3

        x = boundary_check(x, ub, lb)
        fitness = fobj(x)

        for i in range(pop):
            if fitness[i] < alpha_score:
                alpha_score = fitness[i]
                alpha_pos = x[i]
            elif alpha_score < fitness[i] < beta_score:
                beta_score = fitness[i]
                beta_pos = x

