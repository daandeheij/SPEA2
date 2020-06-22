from jmetal.algorithm.multiobjective.spea2 import SPEA2
from MOKnapsack import MOKnapsack
from jmetal.operator import SPXCrossover, BitFlipMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.observer import ProgressBarObserver, BasicObserver
from jmetal.core.quality_indicator import *
from jmetal.lab.experiment import Experiment, Job, generate_summary_from_experiment
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.core.quality_indicator import HyperVolume
from jmetal.lab.visualization import Plot
from jmetal.util.solution import get_non_dominated_solutions
from SaveFrontToDictionaryObserver import SaveFrontToDictionaryObserver
from SPEA2a import SPEA2a
from jmetal.core.quality_indicator import HyperVolume
from EA_utils import *

def qualityExperiment():

    for i in range(2):
        prob = MOKnapsack(from_file=True, filename='instances/dense/80.txt', heavy_init=False, run_id = i)
        print("run: %d" % i)
        max_eval = 5000
        algorithm = SPEA2a(
            problem=prob,
            population_size=100,
            offspring_population_size=100,
            mutation=BitFlipMutation(probability=0.006),
            crossover=SPXCrossover(probability=1.0),
            termination_criterion=StoppingByEvaluations(max_eval)
        )
        progress_bar = ProgressBarObserver(max=max_eval)
        dic_saver = SaveFrontToDictionaryObserver(1, 'quality_test')
        algorithm.observable.register(dic_saver)
        algorithm.run()

def test():
    prob = MOKnapsack(from_file = True, filename ='instances/dense/20.txt', heavy_init=True)

    max_eval = 100
    algorithm = SPEA2a(
        problem=prob,
        population_size=40,
        offspring_population_size=40,
        mutation=BitFlipMutation(probability=0.006),
        crossover=SPXCrossover(probability=1.0),
        termination_criterion=StoppingByEvaluations(max_eval)
    )

    progress_bar = ProgressBarObserver(max=max_eval)
    basic_obs = BasicObserver()
    dic_saver = SaveFrontToDictionaryObserver(1, 'fronts')
    algorithm.observable.register(dic_saver)
    algorithm.observable.register(progress_bar)
    algorithm.observable.register(basic_obs)

    algorithm.run()
    hypervolume = HyperVolume([0, 0])
    print(calc_utopian_hv_fraction(algorithm.archive.solution_list, prob, hypervolume))

def archive_experiment():
    prob_h = MOKnapsack(from_file=True, filename='instances/dense/320.txt', heavy_init=True)
    prob_l = MOKnapsack(from_file=True, filename='instances/dense/320.txt', heavy_init=False)

    frac_light = []
    frac_heavy = []

    for i in range(50):
        print("archive: %d" % i)
        max_eval = 20
        algorithm = SPEA2(
            problem=prob_h,
            population_size=20,
            offspring_population_size=20,
            mutation=BitFlipMutation(probability=0.006),
            crossover=SPXCrossover(probability=1.0),
            termination_criterion=StoppingByEvaluations(max_eval)
        )


        algorithm.run()
        hypervolume = HyperVolume([0, 0])
        frac_heavy.append(calc_utopian_hv_fraction(algorithm.solutions, prob_h, hypervolume))

    for j in range(50):
        print('no_archive: %d' % j)
        max_eval = 20
        algorithm = SPEA2(
            problem=prob_l,
            population_size=20,
            offspring_population_size=20,
            mutation=BitFlipMutation(probability=0.006),
            crossover=SPXCrossover(probability=1.0),
            termination_criterion=StoppingByEvaluations(max_eval)
        )

        algorithm.run()
        hypervolume = HyperVolume([0, 0])
        frac_light.append(calc_utopian_hv_fraction(algorithm.solutions, prob_l, hypervolume))

    print('holdup')

def experiment():
    jobss = configure_experiment()
    output_directory = 'data'
    the_thing = Experiment(output_dir=output_directory, jobs=jobss)
    the_thing.run()

def configure_experiment():
    jobs = []
    max_evaluations = 800
    n_run = 2


    for run in range(n_run):
        algorithm = SPEA2(
            problem=MOKnapsack(from_file=True, filename='instances/dense/40.txt', run_id = run),
            population_size=40,
            offspring_population_size=40,
            mutation=BitFlipMutation(probability=0.006),
            crossover=SPXCrossover(probability=1.0),
            termination_criterion=StoppingByEvaluations(max_evaluations),
        )
        dic_saver = SaveFrontToDictionaryObserver(100, 'fronts')
        algorithm.observable.register(dic_saver)

        jobs.append(
            Job(
                algorithm=algorithm,
                algorithm_tag='SPEA2',
                problem_tag='MO Knapsack',
                run=run,
            )
        )

    return jobs


if __name__ == '__main__':
    qualityExperiment()