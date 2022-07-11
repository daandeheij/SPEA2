# Application of SPEA2 on Multi-Objective Knapsack

 The [multi-objective](https://arxiv.org/pdf/1007.4063) knapsack problem is the natural extension of the well-known (single-objective) [knapsack](https://en.wikipedia.org/wiki/Knapsack_problem) optimization problem.
 
For obtaining solutions to this problem, a multitude of approaches exist. One of these are Evolutionary Algorithms. This code utilizes the *[JMetalPy](https://github.com/jMetal/jMetalPy)* implementation of *SPEA2* in order to solve this problem. 

## Modifications

As the complexity of SPEA2 is relatively high, the code contains 2 novel modifications which improve performance:

* ***Single-objective Optimal Solution Injection (SOOSI)***: Exploits the ease with which the single-objective problem can be solved. At initialization, an optimal solution to each single-objective seperately is obtained using [Google OR-Tools](https://developers.google.com/optimization). These solutions are added to the initial population, such that their genotypes quickly propogate throughout the population. This dramatically increases the spread of solutions on the pareto front for the same number of iterations.

* ***Heavy Initialization***: Depending on the problem instance, initializing a population randomly could yield many infeasible solutions. Similarly, randomly generated solutions could leave a lot of weight capacity unused. 

Heavy Initialization aims to generate random solutions with weight as close as possible to the weight limit.

Assuming the weights to be normally distributed, a normal distribution is fitted to the weights. The sum of weights of *d* random items is then also normally distributed. Next, a confidence interval can be chosen such that x% of random solutions will be feasible. Finally, the adjusted probability can be calculated with which items are chosen.
