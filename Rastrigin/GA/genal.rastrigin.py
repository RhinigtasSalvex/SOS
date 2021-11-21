import numpy as np
from geneal.genetic_algorithms import ContinuousGenAlgSolver

def rastrigin(X):
    A = 10.0 
    delta = [x**2 - A * np.cos(2 * np.pi * x) for x in X]
    y = A * len(X) + sum(delta)
    return y


def rastrigin_fitness(X):
    return -rastrigin(X)


solver = ContinuousGenAlgSolver(
    n_genes=3, 
    fitness_function=rastrigin_fitness,
    pop_size=100,
    max_gen=200,
    mutation_rate=0.1,
    selection_rate=0.6,
    selection_strategy="tournament",
    problem_type=float, # Defines the possible values as float numbers
    variables_limits=(-5.12, 5.12) # Defines the limits of all variables between -10 and 10.
)

solver.solve()