from typing import TypeVar, List

from jmetal.algorithm.multiobjective.spea2 import SPEA2
from jmetal.util.archive import NonDominatedSolutionsArchive
from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.config import store
from jmetal.core.operator import Mutation, Crossover
from jmetal.core.problem import Problem
from jmetal.operator import BinaryTournamentSelection
from jmetal.util.density_estimator import KNearestNeighborDensityEstimator
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.ranking import StrengthRanking
from jmetal.util.replacement import RankingAndDensityEstimatorReplacement, RemovalPolicyType
from jmetal.util.comparator import Comparator, MultiComparator
from jmetal.util.termination_criterion import TerminationCriterion


S = TypeVar('S')
R = TypeVar('R')



class SPEA2a(SPEA2[S, R]):

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 offspring_population_size: int,
                 mutation: Mutation,
                 crossover: Crossover,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 dominance_comparator: Comparator = store.default_comparator,
                 archive_step = 10 ):

        multi_comparator = MultiComparator([StrengthRanking.get_comparator(),
                                            KNearestNeighborDensityEstimator.get_comparator()])
        selection = BinaryTournamentSelection(comparator=multi_comparator)

        super().__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=offspring_population_size,
            mutation=mutation,
            crossover=crossover,
            termination_criterion=termination_criterion,
            population_evaluator=population_evaluator,
            population_generator=population_generator,
            dominance_comparator = dominance_comparator
        )

        self.archive = NonDominatedSolutionsArchive()
        self.pending_for_archive = []

    def update_archive(self, solutions):
        for solution in solutions:
            self.archive.add(solution)

    def get_observable_data(self) -> dict:
        result = super().get_observable_data()
        result['ARCHIVE'] = self.archive
        return result

    def init_progress(self):
        self.update_archive(self.solutions)
        super().init_progress()

    def step(self):
        super().step()
        self.update_archive(self.solutions)






