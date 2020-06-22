import os
from pathlib import Path
from typing import List, TypeVar
import numpy as np
from tqdm import tqdm
import json
import EA_utils
from jmetal.core.observer import Observer
from jmetal.core.problem import DynamicProblem
from jmetal.core.quality_indicator import InvertedGenerationalDistance
from jmetal.lab.visualization import StreamingPlot, Plot
from jmetal.util.solution import print_function_values_to_file
from jmetal.util.solution import get_non_dominated_solutions

class SaveFrontToDictionaryObserver(Observer):

    def __init__(self, step: int, filename: str, take_archive_front=True) -> None:
        """ Show the number of evaluations, best fitness and computing time.

        :param frequency: Display frequency. """
        self.step_size = step
        self.filename = filename
        self.file_str = self.filename + '.json'
        self.dic = {}
        self.take_archive_front = take_archive_front

    def update(self, *args, **kwargs):

        evaluations = kwargs['EVALUATIONS']
        solutions = kwargs['SOLUTIONS']
        problem = kwargs['PROBLEM']
        archive = kwargs['ARCHIVE']
        run_id = str(problem.run_id)

        if (evaluations % self.step_size) == 0 and solutions:
            if os.path.isfile(self.file_str):
                with open(self.file_str, 'r') as input_file:
                    self.dic = json.load(input_file)
                    input_file.close()

            if run_id not in self.dic:
                self.dic[run_id] = {}

            if self.take_archive_front:
                front = archive.solution_list
            else:
                front = get_non_dominated_solutions(solutions)
            tuple_list = []
            for sol in front:
                xy = (-1 * sol.objectives[0],-1 * sol.objectives[1])
                tuple_list.append(xy)

            self.dic[run_id][evaluations] = tuple_list

            with open(self.file_str, 'w') as file:
                json.dump(self.dic, file)



