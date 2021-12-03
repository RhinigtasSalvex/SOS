import numpy as np
from geneal.genetic_algorithms import ContinuousGenAlgSolver
import time
import sys
sys.stdout = open('./Rastrigin/GA/out.txt', 'w')

#do some stuff

def rastrigin(X):
    A = 10.0 
    delta = [x**2 - A * np.cos(2 * np.pi * x) for x in X]
    y = A * len(X) + sum(delta)
    return y


def rastrigin_fitness(X):
    return -rastrigin(X)


# solver = ContinuousGenAlgSolver(
#     n_genes=2, 
#     fitness_function=rastrigin_fitness,
#     pop_size=10,
#     max_gen=100,
#     mutation_rate=0.1,
#     selection_rate=0.6,
#     selection_strategy="tournament",
#     problem_type=float, # Defines the possible values as float numbers
#     variables_limits=(-5.12, 5.12), # Defines the limits of all variables between -10 and 10.
#     plot_results=False,
# )

# start = time.process_time()
# solver.solve()

# elapsed_time = time.process_time() - start
# print(f"Elapsed time: {elapsed_time}")

# ContinuousGenAlgSolver.plot_fitness_results(None, None, None, solver=solver, save_plot=True, path="./Rastrigin/GA/")

for g in [2, 10]:
    for p in [10, 100, 1000]:
        for m in [100, 500]:
            for s in ["tournament", "roulette_wheel"]:
                for mr in [0.1, 0.5]:
                    if g == 1 and p != 10 and s != "tournament":
                        continue
                    solver = ContinuousGenAlgSolver(
                        n_genes=g, 
                        fitness_function=rastrigin_fitness,
                        pop_size=p,
                        max_gen=m,
                        mutation_rate=mr,
                        selection_rate=0.6,
                        selection_strategy=s,
                        problem_type=float, # Defines the possible values as float numbers
                        variables_limits=(-5.12, 5.12), # Defines the limits of all variables between -10 and 10.
                        plot_results=False,
                        show_stats=True,
                        verbose=False,
                    )
                    start = time.process_time()
                    solver.solve()
                    elapsed_time = time.process_time() - start
                    print(f"Elapsed time: {elapsed_time}")
                    print(f"Genes: {g}, Population: {p}, Max_gen: {m}, Mutation: {mr}, Selection_strategy: {s}")
                    ContinuousGenAlgSolver.plot_fitness_results(None, None, None, solver=solver, save_plot=True, path="./Rastrigin/GA/")
    