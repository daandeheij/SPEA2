from jmetal.util.solution import get_non_dominated_solutions
import numpy as np

def invert_objectives(solutions):
    for solution in solutions:
        solution.objectives = np.multiply(solution.objectives, -1)

