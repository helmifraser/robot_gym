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


    def initialise_population(self, pop_size=50, net_dims=[7, 16, 2]):
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
        child = copy.deepcopy(parent_a)
        # child = [item for item in parent_a]
        for layer in range(0, len(parent_a)):
            r = np.random.random(size=parent_a[layer].shape)
            m = r < rate
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

        # print("tournament_selection: competitors {}".format(competitors))
        # print("tournament_selection: {}".format(generation[tournament_winner]))
        return generation[tournament_winner]

    def create_new_generation(self,
                              current_generation,
                              fitness_gen,
                              elitism=1,
                              k=3,
                              new_gen_size=default):

        """Generation and fitness must be sorted before using this"""

        if new_gen_size is default:
            new_gen_size = len(current_generation)

        new_gen = [None]*new_gen_size

        for elites in range(0, elitism, 1):
            new_gen[elites] = copy.deepcopy(current_generation[elites])

        for offspring in range(elitism, new_gen_size, 1):
            parent_a = self.tournament_selection(current_generation, fitness_gen, k)
            parent_b = self.tournament_selection(current_generation, fitness_gen, k)
            child = self.crossover(parent_a, parent_b)
            # new_gen[offspring] = [item for item in child]
            new_gen[offspring] = copy.deepcopy(child)

        return new_gen

    def sort_by_fitness(self, generation, fitness_vals):
        if len(fitness_vals) is not len(generation):
            print("sort_by_fitness: Error, number of fitnesses and individuals do not match")
            print("sort_by_fitness: Not sorting")
            return

        indices = np.argsort(fitness_vals)
        fitness_vals = fitness_vals[indices[::-1]]
        sorted_gen = [None]*len(generation)

        count = 0
        for index in indices:
            sorted_gen[count] = copy.deepcopy(generation[index])
            count+=1

        # print("indices: {}".format(indices))
        return sorted_gen, fitness_vals


    def return_network_dimensions(self):
        return self.network_dimensions[0], self.network_dimensions[1], self.network_dimensions[2]


def main():
    dim = [7, 16, 2]
    ga = GeneticAlgorithm(dim)
    generation_zero = ga.initialise_population()
    # fitness_vals = np.random.randint(100, size=len(generation_zero))
    fitness_vals = np.random.random(size=[len(generation_zero)])
    generation_zero, fitness_vals = ga.sort_by_fitness(generation_zero, fitness_vals)

    # winner = ga.tournament_selection(generation=generation_zero, fitness_gen=fitness_vals)
    # print(winner)
    generation_one = ga.create_new_generation(current_generation=generation_zero, fitness_gen=fitness_vals)

    # t = time.time()
    # new_guy = ga.crossover(generation_zero[0], generation_zero[1])
    # elapsed = time.time() - t
    # print("time: {}".format(elapsed))


if __name__ == '__main__':
    main()
