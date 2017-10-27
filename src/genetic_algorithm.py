import numpy as np
import random
import time
import copy

default = object()


class GeneticAlgorithm(object):
    """Genetic algorithm for neuroevolution of a Turtlebot"""

    def __init__(self, network_dimensions):
        super(GeneticAlgorithm, self).__init__()
        # dimensions are number of input/hidden/ouput nodes in an array
        self.network_dimensions = network_dimensions
        self.mutate_rate = 0.05
        # random.seed(a=42)

    def initialise_population(self, pop_size=50, net_dims=[8, 16, 2]):
        """A generation contains pop_size individuals, which contain two matrices:
            input to hidden and hidden to output.

            List -> List -> np array"""

        generation = [None] * pop_size

        for individual in range(0, pop_size, 1):
            generation[individual] = [np.random.randn(net_dims[0], net_dims[1]),
                                      np.random.randn(net_dims[1], net_dims[2])]

        return generation

    def mutate(self, individual, mutate_rate=0.01, severity=1.0):
        """Vectorised, in-place mutation yay"""

        for layer in range(0, len(individual)):
            r = np.random.random(size=individual[layer].shape)
            m = r < mutate_rate
            individual[layer][m] = np.random.normal(
                loc=individual[layer][m], scale=severity)
            # print("Mutate rate: {} Non zero: {} Where: {} Total weights: {}".format(mutate_rate, np.count_nonzero(m), np.where(m), individual[layer][m].size))

    def crossover(self, parent_a, parent_b, rate=0.5):
        child = parent_a
        for layer in range(0, len(parent_a)):
            r = np.random.random(size=parent_a[layer].shape)
            m = r < rate
            np.putmask(child[layer], np.invert(m), parent_a[layer])
            np.putmask(child[layer], m, parent_b[layer])

            # print("Cross rate: {} Non zero: {} Where: {} Total weights: {}".format(rate, np.count_nonzero(m), np.where(m), parent_a[layer][m].size))
        return child

    def tournament_selection(self, generation, fitness_gen, k=3):
        """Returns the winner of tournament selection"""

        tournament_winner = 0
        best_fitness = 0
        competitors = [None] * k

        for i in range(0, k, 1):
            competitors[i] = random.randint(0, len(generation) - 1)
            current_fitness = fitness_gen[competitors[i]]
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                tournament_winner = competitors[i]

            # print("current fit: {} best fit: {}".format(current_fitness, best_fitness))

        # print("competitors {}".format(competitors))
        return generation[tournament_winner]

    def create_new_generation(self,
                              current_generation,
                              fitness_gen,
                              elitism=1,
                              k=3,
                              new_gen_size=default):


        if new_gen_size is default:
            new_gen_size = len(current_generation)

        new_gen = [None]*new_gen_size

        for offspring in range(0, new_gen_size, 1):
            parent_a = self.tournament_selection(current_generation, fitness_gen, k)
            parent_b = self.tournament_selection(current_generation, fitness_gen, k)
            child = self.crossover(parent_a, parent_b)
            print("parent a: {}".format(parent_a[1]))
            print("parent b: {}".format(parent_b[1]))
            print("child: {}".format(child[1]))
            new_gen[offspring] = copy.deepcopy(child)

        return new_gen

    def return_network_dimensions(self):
        return self.network_dimensions[0], self.network_dimensions[1], self.network_dimensions[2]


def main():
    dim = [8, 16, 2]
    ga = GeneticAlgorithm(dim)
    generation_zero = ga.initialise_population()
    fitness_vals = np.random.randint(100, size=len(generation_zero))
    winner = ga.tournament_selection(generation_zero, fitness_vals)
    generation_one = ga.create_new_generation(generation_zero, fitness_vals, new_gen_size=50)

    # print("gen 0: {}".format(generation_zero[0][1]))
    # print("gen 1: {}".format(generation_one[0][1]))

    # ga.mutate(generation_zero[0], 0.05, 1)
    # t = time.time()
    # new_guy = ga.crossover(generation_zero[0], generation_zero[1])
    # elapsed = time.time() - t
    # print("time: {}".format(elapsed))


if __name__ == '__main__':
    main()
