import numpy as np


# 考虑下面这个优化问题，求解 f(x) = x*x * sin(5pai*x) + 2
# 在区间[-2, 2]上的最大值。很多单点优化的方法（梯度下降等）就不适合，可能会陷入局部最优的情况，这种情况下就可以用遗传算法（Genetic Algorithm。

class GeneticTool:
    def __init__(self, _min=-2, _max=2, _scale=1e4, _width=10, population_size=10):
        self._min = _min
        self._max = _max
        self._scale = _scale
        self._width = _width
        self.population_size = population_size
        self.init_population = np.random.uniform(low=_min, high=_max, size=population_size)

    @staticmethod
    def fitness_function(x):
        return x ** 2 * np.sin(5 * np.pi * x) + 2

    def encode(self, population):
        _scaled_population = (population - self._min) * self._scale
        chroms = np.array([np.binary_repr(x, width=self._width) for x in _scaled_population.astype(int)])
        return chroms

    def decode(self, chroms):
        res = np.array([(int(x, base=2) / self._scale) for x in chroms])
        res += self._min
        return res

    @staticmethod
    def selection(chroms, fitness):
        fitness = fitness - np.min(fitness) + 1e-5
        probs = fitness / np.sum(fitness)
        probs_cum = np.cumsum(probs)
        each_rand = np.random.uniform(size=len(fitness))
        selected_chroms = np.array([chroms[np.where(probs_cum > rand)[0][0]] for rand in each_rand])
        return selected_chroms

    @staticmethod
    def crossover(chroms, prob):
        pairs = np.random.permutation(int(len(chroms) * prob // 2 * 2)).reshape(-1, 2)
        center = len(chroms[0]) // 2
        for i, j in pairs:
            # cross over in center
            x, y = chroms[i], chroms[j]
            chroms[i] = x[:center] + y[center:]
            chroms[j] = y[:center] + x[center:]
        return chroms

    @staticmethod
    def mutate(chroms, prob):
        m = {'0': '1', '1': '0'}
        mutate_chroms = []
        each_prob = np.random.uniform(size=len(chroms))
        for i, chrom in enumerate(chroms):
            if each_prob[i] < prob:
                # mutate in a random bit
                clen = len(chrom)
                ind = np.random.randint(clen)
                chrom = chrom[:ind] + m[chrom[ind]] + chrom[ind + 1:]
            mutate_chroms.append(chrom)
        return np.array(mutate_chroms)

    def run(self, num_epoch):
        # select best population
        best_population = None
        best_finess = -np.inf
        population = self.init_population
        chroms = self.encode(population)
        for i in range(num_epoch):
            population = self.decode(chroms)
            fitness = self.fitness_function(population)
            fitness = fitness - fitness.min() + 1e-4
            if np.max(fitness) > np.max(best_finess):
                best_finess = fitness
                best_population = population
            chroms = self.encode(self.init_population)
            selected_chroms = self.selection(chroms, fitness)
            crossed_chroms = self.crossover(selected_chroms, 0.6)
            mutated_chroms = self.mutate(crossed_chroms, 0.5)
            chroms = mutated_chroms
        # select best individual
        return best_population[np.argmax(best_finess)]


if __name__ == '__main__':
    np.random.seed(0)
    gt = GeneticTool(_min=-2, _max=2, _scale=1e10, _width=10, population_size=10)
    res = gt.run(1000)
    print(res)
