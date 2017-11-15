import numpy as np
import math
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
        self.max_gen = 200
        self.pop_size = 200
        self.mutate_rate = 0.10
        self.severity = 0.5
        self.elitism = int(math.ceil(self.pop_size/100))
        self.k_parents = self.pop_size/10

        self.laser_front_violation_range = 0.5
        self.laser_side_violation_range = 0.2
        self.front_punish = -2
        self.side_punish = -1
        self.backwards_punish = -10
        self.bumper_punish = -10
        self.spin_punish = -10
        self.reward_1 = 1
        self.reward_2 = 2
        self.reward_3 = 3
        self.reward_4 = 4

    def initialise_population(self, pop_size=default, net_dims=default):
        """A generation contains pop_size individuals, which contain two matrices:
            input to hidden and hidden to output.

            List -> List -> np array"""

        if pop_size is default:
            pop_size = self.pop_size

        if net_dims is default:
            net_dims = self.network_dimensions

        generation = [None] * pop_size

        for individual in range(0, pop_size, 1):
            generation[individual] = [np.random.randn(net_dims[0] + 1, net_dims[1]),
                                      np.random.randn(net_dims[1], net_dims[2]),
                                      np.random.randn(net_dims[2], net_dims[3])]

        return generation

    def mutate(self, individual, mutate_rate=default, severity=default):
        """Vectorised, in-place mutation yay"""

        if mutate_rate is default:
            mutate_rate = self.mutate_rate

        if severity is default:
            severity = self.severity

        for layer in range(0, len(individual)):
            r = np.random.random(size=individual[layer].shape)
            m = r < mutate_rate
            individual[layer][m] = np.random.normal(
                loc=individual[layer][m], scale=severity)
            # print("Mutate rate: {} Non zero: {} Where: {} Total weights: {}".format(mutate_rate, np.count_nonzero(m), np.where(m), individual[layer][m].size))

    def crossover(self, parent_a, parent_b, rate=0.5):
        """Rate is the chance that genes will come from parent_b """
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
        return generation[tournament_winner], best_fitness

    def create_new_generation(self,
                              current_generation,
                              fitness_gen,
                              elitism=default,
                              k=default,
                              new_gen_size=default):

        """Generation and fitness must be sorted before using this"""

        if elitism is default:
            elitism = self.elitism

        if k is default:
            k = self.k_parents

        if new_gen_size is default:
            new_gen_size = len(current_generation)


        new_gen = [None]*new_gen_size
        # print(current_generation[0][1])
        for elites in range(0, elitism, 1):
            new_gen[elites] = copy.deepcopy(current_generation[elites])

        for offspring in range(elitism, new_gen_size, 1):
            parent_a, fitness_a = self.tournament_selection(current_generation, fitness_gen, k)
            parent_b, fitness_b = self.tournament_selection(current_generation, fitness_gen, k)
            if fitness_a > fitness_b:
                child = self.crossover(parent_a, parent_b, rate=0.3)
            else:
                child = self.crossover(parent_a, parent_b, rate=0.7)
            # new_gen[offspring] = [item for item in child]
            new_gen[offspring] = copy.deepcopy(child)

        # print(new_gen[0][1])
        return new_gen

    def check_quadrant_1(self, position):
        if position[0] >= -0.247 and position[0] <= 4.437:
            if position[1] >= 0 and position[1] <= 0.5:
                return True

        return False

    def check_quadrant_2(self, position):
        if position[0] >= -0.247 and position[0] <= 4.437:
            if position[1] < 0 and position[1] > -8.2:
                return True

        return False

    def check_quadrant_3(self, position):
        if position[0] < -0.27 and position[0] > -4.437:
            if position[1] < 0 and position[1] > -8.2:
                return True

        return False

    def check_quadrant_4(self, position):
        if position[0] < -0.26 and position[0] > -4.437:
            if position[1] >= 0 and position[1] <= 0.5:
                return True

        return False

    def get_pos_fitness(self, pos):
        distance_measure = 0

        if self.check_quadrant_1(pos):
            # print("quad 1")
            distance_measure = abs(pos[0]) * self.reward_1
            # print(distance_measure)

        if self.check_quadrant_2(pos):
            # print("quad 2")
            if pos[1] > -0.6:
                distance_measure = abs(pos[0]) * self.reward_1
                # print(distance_measure)
            else:
                distance_measure = abs(pos[1]) * self.reward_2 + self.reward_1 * 4.5
                # print(distance_measure)

        if self.check_quadrant_3(pos):
            # print("quad 3")
            if pos[1] > -0.6:
                distance_measure = (4.437 - abs(pos[0])) * self.reward_4 + 40.04
                # print(distance_measure)
            else:
                distance_measure = (8.197 - abs(pos[1])) * self.reward_3 + 20
                # print(distance_measure)

        if self.check_quadrant_4(pos):
            # print("quad 4")
            distance_measure = (4.437 - abs(pos[0])) * self.reward_4 + 40.04
            # print(distance_measure)

        return distance_measure

    def fit_update(self, fit_val, index, laser_data, action, bumper_state, laser_front_thresh=default, laser_side_thresh=default):
        """Updates a numpy array of fitness values depending on number of
            laser range violations"""

        if laser_front_thresh is default:
            laser_front_thresh = self.laser_front_violation_range

        if laser_side_thresh is default:
            laser_side_thresh = self.laser_side_violation_range

        front_violations = np.size(np.where(laser_data[2:5] < laser_front_thresh))
        right_side_violations = np.size(np.where(laser_data[0:2] < laser_side_thresh))
        left_side_violations = np.size(np.where(laser_data[5:7] < laser_side_thresh))

        back_movement_violation = np.size(np.where(action[0] < 0))
        angular_vel_violation = np.size(np.where(abs(action[1]) > 0.2))
        # print("violations: front {}, right {}, left {}".format(front_violations,
        #         right_side_violations, left_side_violations))

        fit_val[index] += (self.front_punish * front_violations +
                            self.side_punish * (left_side_violations
                            + right_side_violations)
                            + self.backwards_punish * back_movement_violation
                            + self.bumper_punish * bumper_state
                            + self.spin_punish * angular_vel_violation)


    def sort_by_fitness(self, generation, fitness_vals):
        if len(fitness_vals) != len(generation):
            print("fv: {} gn: {}".format(len(fitness_vals), len(generation)))
            print("sort_by_fitness: Error, number of fitnesses and individuals do not match")
            print("sort_by_fitness: Not sorting")
            return

        indices = np.argsort(fitness_vals)
        fitness_vals = fitness_vals[indices[::-1]]
        sorted_gen = [None]*len(generation)
        # print(len(sorted_gen))
        # print(any(x is None for x in generation))
        count = 0
        for index in indices:
            sorted_gen[count] = copy.deepcopy(generation[index])
            count+=1
        # print(any(x is None for x in sorted_gen))
        # print("indices: {}".format(indices))
        return sorted_gen, fitness_vals

    def return_network_dimensions(self):
        return self.network_dimensions[0], self.network_dimensions[1], self.network_dimensions[2]

    def return_max_gen(self):
        return self.max_gen

def main():
    dim = [7, 16, 2]
    ga = GeneticAlgorithm(dim)
    generation = ga.initialise_population()
    next_gen = None
    # fitness_vals = np.random.randint(100, size=len(generation))
    fitness_vals = np.random.randint(-100, high=0, size=(len(generation)))
    generation, fitness_vals = ga.sort_by_fitness(generation, fitness_vals)
    print(generation[3][1])
    # winner = ga.tournament_selection(generation=generation, fitness_gen=fitness_vals)
    # print(winner)
    next_gen = ga.create_new_generation(current_generation=generation, fitness_gen=fitness_vals)
    generation = next_gen
    print(generation[3][1])
    # t = time.time()
    # new_guy = ga.crossover(generation[0], generation[1])
    # elapsed = time.time() - t
    # print("time: {}".format(elapsed))


if __name__ == '__main__':
    main()
