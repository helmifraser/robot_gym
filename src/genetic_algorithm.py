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
        weight_matrix = [None, None]

        for individual in range (0, pop_size, 1):
            wi = np.random.randn(net_dims[0], net_dims[1])
            wo = np.random.randn(net_dims[1], net_dims[2])

            weight_matrix[0] = wi
            weight_matrix[1] = wo

            generation[individual] = weight_matrix

        return generation

    def mutate(self, individual, mutate_rate=0.05, severity=1.0):
        """Vectorised, in-place mutation yay"""

        for layer in range(0, len(individual)):
            r = np.random.random(size=individual[layer].shape)
            m = r < mutate_rate
            individual[layer][m] = np.random.normal(loc=individual[layer][m], scale=severity)
            print("Mutate rate: {} Non zero: {} Where: {} Total weights: {}".format(mutate_rate, np.count_nonzero(m), np.where(m), individual[layer][m].size))

    def crossover(self, parent_a, parent_b, rate=0.5):
        for layer in range(0, len(parent_a)):
            r = np.random.random(size=parent_a[layer].shape)
            m = r < rate
            parent_a[layer][m] = parent_b[layer][m]
            # print("Cross rate: {} Non zero: {} Where: {} Total weights: {}".format(rate, np.count_nonzero(m), np.where(m), parent_a[layer][m].size))

    def return_network_dimensions(self):
        return self.network_dimensions[0], self.network_dimensions[1], self.network_dimensions[2]

def main():
    dim = [8, 16, 2]
    ga = GeneticAlgorithm(dim)
    generation_zero = ga.initialise_population()
    print("equal {}".format(np.array_equal(generation_zero[0], generation_zero[1])))
    # print(generation_zero[0])
    # ga.mutate(generation_zero[0], 0.05, 1)
    # print(generation_zero[0])
    # print("parent_a: {} parent_b: {}".format(generation_zero[0], generation_zero[1]))
    ga.crossover(generation_zero[0], generation_zero[1])
    # print("parent_a: {} parent_b: {}".format(generation_zero[0], generation_zero[1]))
    # input_nodes, hidden_nodes, output_nodes = ga.return_network_dimensions()
    # print("In: {} Hidden: {} Out: {}".format(input_nodes, hidden_nodes, output_nodes))

if __name__ == '__main__':
    main()
