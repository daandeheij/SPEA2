from jmetal.util.solution import get_non_dominated_solutions
import numpy as np
from jmetal.core.quality_indicator import HyperVolume
import json


def invert_objectives(solutions):
    for solution in solutions:
        solution.objectives = np.multiply(solution.objectives, -1)


def calc_utopian_hv_fraction(solutions, problem, hypervolume):
    reference_point = hypervolume.referencePoint
    utopian_point = (problem.v1_sum, problem.v2_sum)
    utopian_hv = (utopian_point[0] - reference_point[0]) * (utopian_point[1] - reference_point[1])
    objectives = [[sol.objectives[0], sol.objectives[1]] for sol in solutions]
    front_hv = hypervolume.compute(np.array(objectives))
    return front_hv / utopian_hv

def dic_to_json(dic, filename):
    with open(filename, 'w') as file:
        json.dump(dic, file)


