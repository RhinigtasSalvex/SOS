import numpy

from aco import ACO

import time
import matplotlib.pyplot as plt

#Set parameters for model:
parameters = {
    "seed" : 0, #Random seed that allows replicating results
    "ALPHA" : 1, #Exponential weight of pheromone on walk probabilities
    "BETA" : 1, #Exponential weight of desirability on walk probabilities
    "init_pheromone" : 0.999, #Initial pheromone for all edges
    "pheromone_constant" : 1, #Constant that helps to calculate edge pheromone contribution
    "min_pheromone" : 0.001, #Minimun pheromone value of an edge
    "evaporation_rate" : 0.8, #Pheromone evaporatio rate per cycle
    "ant_numbers" : 15, #Number of ants walking in a cycle
    "cycles" : 250, #Number of cycles
    "dataset" : 'la40.txt' #File name that contains job/machine times
}
best_paths = []
y3 = ['la36.txt']
times = []
convergence = []

for j in y3:
    # 10 x 5 , 10x10 ['la01.txt', 'la02.txt', 'ft10.txt', 'abz5.txt', 'la18.txt']
    # 15 x  ['la06.txt', 'la21.txt' , 'la36.txt']
    # 20 x ['la11.txt', 'la26.txt', 'yn1.txt' ]
    parameters['dataset'] = j
    colony = ACO(
        ALPHA=parameters['ALPHA'],
        BETA=parameters['BETA'],
        dataset=j,
        cycles=parameters['cycles'],
        ant_numbers=parameters['ant_numbers'],
        init_pheromone=parameters['init_pheromone'],
        pheromone_constant=parameters['pheromone_constant'],
        min_pheromone=parameters['min_pheromone'],
        evaporation_rate=parameters['evaporation_rate'],
        seed=parameters['seed'])

    start = time.time()
    best_path, converg = colony.releaseTheAnts()
    best_paths.append(best_path)
    print(best_path)
    print(converg)

    end = time.time()
    times.append(end - start)

    iterations = numpy.arange(1, parameters['cycles'] + 1, 1)

#     y2 = best_paths
#     # plotting the line 2 points
#
#     plt.plot(x2, y2)
#     # naming the x axis
#     plt.xlabel('Evaporation rte')
#     # naming the y axis
#     plt.ylabel('Makespan')
#     # giving a title to my graph
#     plt.title('Evaporation rate performance comparison')
#
#     # function to show the plot
#     plt.savefig('e_r_stats.png')
# # line 2 points


print(best_paths)
print(times)

