from ortools.algorithms import pywrapknapsack_solver
from MOKnapsack import MOKnapsack
import numpy as np

class OR_solver():

    def __init__(self, filename: str):

        self.v1 = []
        self.v2 = []
        self.weights = []

        self.solver = None
        self.best_objective = None

        self.problem = None

        self.best_solution = None

        if filename is None:
            raise FileNotFoundError('Filename can not be None')

        with open(filename) as file:
            self.problem = MOKnapsack(from_file=True, filename=filename)

            lines = file.readlines()

            first_line = lines[0].split()
            self.number_of_bits = int(first_line[0])
            self.capacity = int(first_line[1])

            for line in lines[1:]:
                data = line.split()
                self.weights.append(int(data[0]))
                self.v1.append(int(data[1]))
                self.v2.append(int(data[2]))

    def solve(self, value_name):
        obj_values = None

        if value_name == 'v1':
            obj_values = self.v1
        elif value_name == 'v2':
            obj_values = self.v2
        elif value_name == 'both':
            obj_values = list(np.add(self.v1, self.v2))
            obj_values = list(np.dot(obj_values, 0.5))
        else:
            raise RuntimeError('Please provide either v1 or v2 to maximize, or type both for mixed')

        self.solver = pywrapknapsack_solver.KnapsackSolver(
            pywrapknapsack_solver.KnapsackSolver.
                KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 'KnapsackExample')
        self.solver.Init(obj_values, [self.weights], [self.capacity])
        self.best_objective = self.solver.Solve()
        self.get_best_solution()
        self.check_objective_matches(value_name)

    def get_best_solution(self):
        result = self.problem.create_solution()
        for i in range(self.number_of_bits):
            if self.solver.BestSolutionContains(i):
                result.variables[0][i] = 1
            else:
                result.variables[0][i] = 0
        self.best_solution = result

    def check_objective_matches(self, value_name):
        self.problem.evaluate(self.best_solution)
        if value_name == 'v1':
            if self.best_solution.objectives[0] * -1 != self.best_objective:
                raise RuntimeWarning("Objective values of Google OR and JMetal did not match!")
        elif value_name == 'v2':
            if self.best_solution.objectives[1] * -1 != self.best_objective:
                raise RuntimeWarning("Objective values of Google OR and JMetal did not match!")


