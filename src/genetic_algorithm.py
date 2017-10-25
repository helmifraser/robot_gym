import numpy as np
import random

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

        generation = [None]*pop_size

        for individual in range (0, pop_size, 1):
            generation[individual] = [np.random.randn(net_dims[0], net_dims[1]), np.random.randn(net_dims[1], net_dims[2])]

        return generation

    def mutate(self, individual, mutate_rate=0.05, severity=1.0):
        """Vectorised, in-place mutation yay"""

        for layer in range(0, len(individual)):
            r = np.random.random(size=individual[layer].shape)
            m = r < mutate_rate
            individual[layer][m] = np.random.normal(loc=individual[layer][m], scale=severity)
            print("Mutate rate: {} Non zero: {} Where: {} Total weights: {}".format(mutate_rate, np.count_nonzero(m), np.where(m), individual[layer][m].size))

    def crossover(self, parent_a, parent_b, rate=0.5):
        child = parent_a
        for layer in range(0, len(parent_a)):
            r = np.random.random(size=parent_a[layer].shape)
            m = r < rate
            # np.copyto(child[layer], parent_a[layer], where=not m)
            # np.copyto(child[layer], parent_b[layer], where=m)

            # print("Cross rate: {} Non zero: {} Where: {} Total weights: {}".format(rate, np.count_nonzero(m), np.where(m), parent_a[layer][m].size))
        return child

    def return_network_dimensions(self):
        return self.network_dimensions[0], self.network_dimensions[1], self.network_dimensions[2]

def main():
    dim = [8, 16, 2]
    ga = GeneticAlgorithm(dim)
    generation_zero = ga.initialise_population()
    print("equal {}".format(np.array_equal(generation_zero[0][1], generation_zero[1][1])))
    # ga.mutate(generation_zero[0], 0.05, 1)
    print("parent_a: {} \n parent_b: {}".format(generation_zero[0][1], generation_zero[1][1]))
    new_guy = ga.crossover(generation_zero[0], generation_zero[1])
    print("new guy: {}".format(new_guy[1]))
    # print("parent_a: {} \n parent_b: {}".format(generation_zero[0][1], generation_zero[1][1]))
    # input_nodes, hidden_nodes, output_nodes = ga.return_network_dimensions()
    # print("In: {} Hidden: {} Out: {}".format(input_nodes, hidden_nodes, output_nodes))

if __name__ == '__main__':
    main()
