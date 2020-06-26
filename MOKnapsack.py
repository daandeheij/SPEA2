import random

import numpy as np
import scipy
import OR_knapsack

from jmetal.core.problem import BinaryProblem
from jmetal.core.solution import BinarySolution

"""
.. module:: knapsack
   :platform: Unix, Windows
   :synopsis: Single Objective Knapsack problem
.. moduleauthor:: Alejandro Marrero <alu0100825008@ull.edu.es>
"""


class MOKnapsack(BinaryProblem):
    """ Class representing MO Knapsack Problem. """

    def __init__(self, number_of_items: int = 50, capacity: float = 1000, weights: list = None,
                 v1: list = None, v2: list = None, from_file: bool = False, filename: str = None, run_id = 0, pop_size = 0, heavy_init = False, SOOSI = False):
        super(MOKnapsack, self).__init__()

        if from_file:
            self.__read_from_file(filename)
        else:
            self.capacity = capacity
            self.weights = weights
            self.v1 = v1
            self.v2 = v2
            self.number_of_bits = number_of_items

        self.number_of_variables = 1
        self.obj_directions = [self.MAXIMIZE]
        self.number_of_objectives = 2
        self.number_of_constraints = 1
        self.run_id = run_id
        self.MAX_VALUE = 2147483648
        self.heavy_init = heavy_init
        self.debug = []
        self.v1_sum = sum(self.v1)
        self.v2_sum = sum(self.v2)
        self.pop_size = pop_size
        self.bitprob = 0.5
        self.SOOSI = SOOSI
        self.solutions_generated = 0

        if self.heavy_init:
            self.bitprob = self.calc_bitprob()

        if self.SOOSI:
            self.SOOSI_solutions = self.get_SOOSI_solutions(filename)

    def __read_from_file(self, filename: str):
        """
        This function reads a Knapsack Problem instance from a file.
        It expects the following format:
            num_of_items (dimension)
            capacity of the knapsack
            num_of_items-tuples of weight-profit
        :param filename: File which describes the instance.
        :type filename: str.
        """

        if filename is None:
            raise FileNotFoundError('Filename can not be None')

        with open(filename) as file:
            lines = file.readlines()
            data = [line.split() for line in lines if len(line.split()) >= 1]

            self.number_of_bits = int(data[0][0])
            self.capacity = float(data[0][1])

            weights_and_profits = np.asfarray(data[1:], dtype=np.float32)

            self.weights = weights_and_profits[:, 0]
            self.v1 = weights_and_profits[:, 1]
            self.v2 = weights_and_profits[:, 2]

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        total_v1 = 0.0
        total_v2 = 0.0
        total_weights = 0.0

        for index, bits in enumerate(solution.variables[0]):
            if bits:
                total_v1 += self.v1[index]
                total_v2 += self.v2[index]
                total_weights += self.weights[index]

        if total_weights > self.capacity:
            total_v1 = -1.0 * self.MAX_VALUE
            total_v2 = -1.0 * self.MAX_VALUE

        solution.objectives[0] = -1.0 * total_v1
        solution.objectives[1] = -1.0 * total_v2
        return solution

    def get_weight(self, solution: BinarySolution):
        total_weights = 0.0
        for index, bits in enumerate(solution.variables[0]):
            if bits:
                total_weights += self.weights[index]
        return total_weights

    def create_random_solution(self) -> BinarySolution:
        new_solution = BinarySolution(number_of_variables=self.number_of_variables,
                                      number_of_objectives=self.number_of_objectives)

        new_solution.variables[0] = \
            [True if random.randint(0, 1) == 0 else False for _ in range(
                self.number_of_bits)]
        self.debug.append(self.get_weight(new_solution))
        return new_solution

    def create_heavy_solution(self, remove_probability) -> BinarySolution:
        new_solution = BinarySolution(number_of_variables=self.number_of_variables,
                                      number_of_objectives=self.number_of_objectives)

        new_solution.variables[0] = [True if random.random() < self.bitprob else False for _ in range(self.number_of_bits)]

        self.debug.append(self.get_weight(new_solution))
        return new_solution

    def create_solution(self) -> BinarySolution:
        if self.solutions_generated < 2:
            self.solutions_generated += 1
            return self.SOOSI_solutions[self.solutions_generated - 1]
        if self.heavy_init:
            return self.create_heavy_solution()
        else:
            return self.create_random_solution()

    def calc_bitprob(self):
        conf = 0.9
        mean, var = scipy.stats.distributions.norm.fit(self.prob.weights)
        number_of_items_to_pick = self.number_of_bits
        while scipy.stats.norm.cdf(self.capacity, number_of_items_to_pick*mean, number_of_items_to_pick*var) < conf:
            number_of_items_to_pick -= 1
        return number_of_items_to_pick / self.number_of_bits

    def get_SOOSI_solutions(self, filename):
        solutions = []
        OR_solver = OR_knapsack.OR_solver(filename = filename)
        OR_solver.solve("v1")
        solutions.append(OR_solver.best_solution)
        OR_solver.solve("v2")
        solutions.append(OR_solver.best_solution)
        return solutions



    def get_name(self):
        return 'MO Knapsack'