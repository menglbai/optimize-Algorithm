import random
from operator import itemgetter
import matplotlib.pyplot as plt


class Gene:
    """
    This is a class to represent individual(Gene) in GA algorithom
    each object of this class have two attribute: data, size
    """

    # **data 一个字典
    def __init__(self, **data):
        self.__dict__.update(data)
        self.size = len(data['data'])  # length of gene


class GA:
    """
    This is a class of GA algorithm.
    """

    def __init__(self, parameter, func):
        """
        Initialize the pop of GA algorithom and evaluate the pop by computing its' fitness value.
        The data structure of pop is composed of several individuals which has the form like that:

        {'Gene':a object of class Gene, 'fitness': 1.02(for example)}
        Representation of Gene is a list: [b s0 u0 sita0 s1 u1 sita1 s2 u2 sita2]

        """
        #           交叉概率，变异概率，代数，种群大小，下界，上界
        # parameter = [CXPB, MUTPB, NGEN, popsize, low, up]
        self.parameter = parameter
        self.evaluate = func
        self.cxpb = parameter[0]
        self.mutpb = parameter[1]
        self.ngen = parameter[2]
        low = self.parameter[4]
        up = self.parameter[5]

        self.bound = []
        self.bound.append(low)
        self.bound.append(up)

        self.gbest = []
        self.gbest_mae = []
        pop = []
        # 随机初始化种群
        for i in range(self.parameter[3]):
            # geneinfo 代表一个个体
            geneinfo = []
            for pos in range(len(low)):
                # 从上界和下界之间随机一个整数
                geneinfo.append(random.uniform(self.bound[0][pos], self.bound[1][pos]))  # initialise popluation

            fitness, mae = self.evaluate(geneinfo)  # evaluate each chromosome
            pop.append(
                {'Gene': Gene(data=geneinfo), 'fitness': fitness, 'mae': mae})  # store the chromosome and its fitness

        self.pop = pop
        self.bestindividual = self.selectBest(self.pop)  # store the best chromosome in the population

    # # 适应度函数
    # def evaluate(self, geneinfo):
    #     """
    #     fitness function
    #     """
    #     x1 = geneinfo[0]
    #     x2 = geneinfo[1]
    #     x3 = geneinfo[2]
    #     x4 = geneinfo[3]
    #     y = x1 ** 2 + x2 ** 2 + x3 ** 3 + x4 ** 4
    #     return y

    # 选择当前种群中最好的一个个体
    def selectBest(self, pop):
        """
        select the best individual from pop
        """
        s_inds = sorted(pop, key=itemgetter("fitness"), reverse=False)  # from small to large, return a pop
        # print(s_inds)
        return s_inds[0]


    def selection(self, individuals, k):
        """
        Roulette wheel selection implementation for a genetic algorithm.

        Args:
        population: A list of candidate solutions (individuals) to be selected from.

        Returns:
        A list of selected individuals.
        """
        selected = []
        # Sort individuals based on their fitness
        sorted_pop = sorted(individuals, key=itemgetter("fitness"),
                            reverse=False)  # 从小到大排序

        # Select the top individual(s) with the lowest fitness
        top_individuals = [sorted_pop[0]]
        # for ind in sorted_pop[1:]:
        #     if ind['fitness'] == top_individuals[0]['fitness']:
        #         top_individuals.append(ind)
        #     else:
        #         break

        selected.extend(top_individuals)

        # Assign a selection probability to each individual
        total_fitness = sum(ind['fitness'] for ind in individuals)
        selection_probs = [ind['fitness'] / total_fitness for ind in individuals]

        # Select the remaining individuals based on their selection probabilities
        remaining_slots = len(individuals) - len(selected)
        for i in range(remaining_slots):
            r = random.random()
            for j, ind in enumerate(individuals):
                if r <= selection_probs[j]:
                    selected.append(ind)
                    break
                r -= selection_probs[j]

        return selected

    # # 选择算子  分层比例选择算子
    # def selection(self, individuals, k):
    #     s_inds = sorted(individuals, key=itemgetter("fitness"),
    #                     reverse=False)  # 从小到大排序
    #     chosen = []
    #
    #     good = s_inds[0:8]  # 优 8个
    #     mid = s_inds[8:14]  # 中 6个
    #     bad = s_inds[14:20]  # 差 6个
    #     chosen.extend(good)
    #     chosen.extend(mid)
    #     chosen.extend(random.sample(bad, 3))
    #     chosen.extend(random.sample(good, 3))
    #     # 不足的随机选两个
    #     # while(len(chosen)<k):
    #     #     chosen.append(random.choice(s_inds))
    #     chosen = sorted(chosen, key=itemgetter("fitness"), reverse=False)
    #     return chosen

    # 交叉算子----多点交叉,交换i，j之间的部分
    def crossoperate(self, offspring):

        dim = len(offspring[0]['Gene'].data)

        geninfo1 = offspring[0]['Gene'].data  # Gene's data of first offspring chosen from the selected pop
        geninfo2 = offspring[1]['Gene'].data  # Gene's data of second offspring chosen from the selected pop

        if dim == 1:
            pos1 = 1
            pos2 = 1
        else:
            pos1 = random.randrange(0, dim)  # select a position in the range from 0 to dim-1,
            pos2 = random.randrange(0, dim)

        newoff1 = Gene(data=[])  # offspring1 produced by cross operation
        newoff2 = Gene(data=[])  # offspring2 produced by cross operation
        temp1 = []
        temp2 = []
        for i in range(dim):
            if min(pos1, pos2) <= i < max(pos1, pos2):
                temp2.append(geninfo2[i])
                temp1.append(geninfo1[i])
            else:
                temp2.append(geninfo1[i])
                temp1.append(geninfo2[i])
        newoff1.data = temp1
        newoff2.data = temp2

        return newoff1, newoff2

    # # 交叉算子----单点交叉
    # def crossoperate(self, offspring):
    #     a = random.random()
    #     dim = len(offspring[0]['Gene'].data)
    #
    #     geninfo1 = offspring[0]['Gene'].data  # Gene's data of first offspring chosen from the selected pop
    #     geninfo2 = offspring[1]['Gene'].data  # Gene's data of second offspring chosen from the selected pop
    #
    #     if dim == 1:
    #         pos = 1
    #     else:
    #         pos = random.randint(0, dim - 1)  # select a position in the range from 0 to dim-1,
    #     newoff1 = Gene(data=[])  # offspring1 produced by cross operation
    #     newoff2 = Gene(data=[])  # offspring2 produced by cross operation
    #     temp1 = []
    #     temp2 = []
    #     for i in range(dim):
    #         if i == pos:
    #             temp2.append((1 - a) * geninfo2[i] + a * geninfo1[i])
    #             temp1.append((1 - a) * geninfo1[i] + a * geninfo2[i])
    #             # temp2.append(geninfo1[i])
    #             # temp1.append(geninfo2[i])
    #         else:
    #             temp2.append(geninfo2[i])
    #             temp1.append(geninfo1[i])
    #     newoff1.data = temp1
    #     newoff2.data = temp2
    #
    #     return newoff1, newoff2

    # 随机变异算子
    def mutation(self, crossoff, bound,g):
        """
        mutation operation
        """
        dim = len(crossoff.data)

        if dim == 1:
            pos = 0
        else:
            # random.randrange 左闭右开
            pos = random.randrange(0, dim)  # chose a position in crossoff to perform mutation.

        crossoff.data[pos] = random.randint(bound[0][pos], bound[1][pos])
        return crossoff

    def run(self):
        """
        main frame work of GA
        """
        popsize = self.parameter[3]

        print("Start of evolution")

        # Begin the evolution
        for g in range(self.ngen):

            print("############### Generation {} ###############".format(g))

            # Apply selection based on their converted fitness
            selectpop = self.selection(self.pop, popsize)

            nextoff = []
            while len(nextoff) != popsize:
                # Apply crossover and mutation on the offspring

                # Select two individuals
                offspring = [selectpop.pop() for _ in range(2)]

                if random.random() < self.cxpb:  # cross two individuals with probability CXPB
                    crossoff1, crossoff2 = self.crossoperate(offspring)
                    if random.random() < self.mutpb:  # mutate an individual with probability MUTPB
                        muteoff1 = self.mutation(crossoff1, self.bound, g)
                        muteoff2 = self.mutation(crossoff2, self.bound, g)
                        fit_muteoff1, mae1 = self.evaluate(muteoff1.data)  # Evaluate the individuals
                        fit_muteoff2, mae2 = self.evaluate(muteoff2.data)  # Evaluate the individuals

                        nextoff.append({'Gene': muteoff1, 'fitness': fit_muteoff1, 'mae': mae1})
                        nextoff.append({'Gene': muteoff2, 'fitness': fit_muteoff2, 'mae': mae2})
                    else:
                        fit_crossoff1, mae1 = self.evaluate(crossoff1.data)  # Evaluate the individuals
                        fit_crossoff2, mae2 = self.evaluate(crossoff2.data)
                        nextoff.append({'Gene': crossoff1, 'fitness': fit_crossoff1, 'mae': mae1})
                        nextoff.append({'Gene': crossoff2, 'fitness': fit_crossoff2, 'mae': mae2})
                else:
                    # [].extend列表追加列表
                    nextoff.extend(offspring)

            # The population is entirely replaced by the offspring
            self.pop = nextoff

            # Gather all the fitnesses in one list and print the stats
            fits = [ind['fitness'] for ind in self.pop]
            # 这一代最小的
            best_ind = self.selectBest(self.pop)

            if best_ind['fitness'] < self.bestindividual['fitness']:
                self.bestindividual = best_ind
            self.gbest.append(self.bestindividual['fitness'])
            self.gbest_mae.append(self.bestindividual['mae'])
            print("Best individual found is {}, {}".format(self.bestindividual['Gene'].data,
                                                           self.bestindividual['fitness']))
            print("  Min fitness of current pop: {}".format(min(fits)))

        print("------ End of (successful) evolution ------")


# 适应度函数
def evaluate(geneinfo):
    x1 = geneinfo[0]
    x2 = geneinfo[1]
    x3 = geneinfo[2]
    x4 = geneinfo[3]
    y = x1 ** 2 + x2 ** 2 + x3 ** 3 + x4 ** 4
    return y, y + 40


def IGA(func, cxpb, mutpb, ngen, popsize, up, low):
    parameter = [cxpb, mutpb, ngen, popsize, low, up]
    iga = GA(parameter, func)
    iga.run()
    # print('gbest', iga.gbest)
    # print('gbest_mae', iga.gbest_mae)
    plt.plot(iga.gbest)
    plt.plot(iga.gbest_mae)
    plt.show()
    return iga.gbest, iga.gbest_mae


if __name__ == "__main__":
    # CXPB, MUTPB, NGEN, popsize = 0.8, 0.1, 50, 20  # popsize must be even number
    # up = [30, 30, 30, 30]  # upper range for variables
    # low = [1, 1, 1, 1]  # lower range for variables
    # parameter = [CXPB, MUTPB, NGEN, popsize, low, up]
    # iga = GA(parameter)
    # iga.run()
    # print(iga.gbest)
    # plt.plot(iga.gbest)
    # plt.show()
    IGA(func=evaluate, cxpb=0.8, mutpb=0.1, ngen=50, popsize=20, up=[30, 30, 30, 30], low=[1, 1, 1, 1])
