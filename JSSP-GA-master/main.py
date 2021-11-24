import os

import numpy as np

from utils import readFilePairs
from jspGA import genetic
import time

import matplotlib.pyplot as plt

target = None

population_size= 25
mutation_rate=0.2
iterations= 2500

y3 = ['la22.txt']
best_paths = []
run_times = []
# for i in y3:
#     path = os.path.join('C:/Users/helena/Desktop/TU Wien/Self organizing systems/JSSP-GA-master/cases/', i )
#     print(path)
#     times, machines, n = readFilePairs(path)
#     print(times)
#
#     start = time.time()
#     best_path, convergs = genetic(times, machines, n, population_size, iterations, mutation_rate, target)
#     end = time.time()
#     run_times.append(end - start)
#
#     best_paths.append(best_path)
#
#     print(convergs)
#
# iterations = np.arange(1, iterations+1, 1)
# plt.plot(iterations, convergs)
#
# plt.xlabel('Iteration')
# # naming the y axis
# plt.ylabel('Makespan')
# # giving a title to my graph
# plt.title('Convergence genetic algorithm')
#
# # function to show the plot
# plt.savefig("converg.png")

# y2 = best_paths
# print(y2)
# print(run_times)
    # plotting the line 2 points
# line 2 points

# 10 machines
#aco
x1 = [666, 635, 706, 1081, 725]
t1 = [4.884886980056763, 4.826586723327637, 17.12615728378296, 17.789500951766968, 17.179518938064575]

x2 = [677, 690, 1081, 1303, 938]
t2 = [1.3725354671478271, 1.47416090965271, 2.739438533782959, 2.340362548828125, 2.4369916915893555]

y1 = ["5", " 5", "10", " 10", "  10"]
# plt.plot(y1, x1, label='ACO')
# plt.plot(y1, x2, label='GA')

# naming the x axis
# plt.xlabel('Number of jobs')
# # naming the y axis
# plt.ylabel('Makespan')
# # giving a title to my graph
# plt.title('Makespan comparison for 10 machines')
# plt.legend()
#
# # function to show the plot
# plt.savefig('M_10_comp.png')

# 15 machines
#
x1 = [926, 1011, 1138]
t1 = [11.664961338043213, 41.605727672576904, 91.08010387420654]
#
x2 = [926, 1188, 1430]
t2 =[2.945964813232422, 5.134979724884033, 6.516570329666138]
#
y1 = ["5", "10", "15"]
plt.plot(y1, x1, label='ACO')
plt.plot(y1, x2, label='GA')
# #
# # # naming the x axis
plt.xlabel('Number of jobs')
# # # naming the y axis
plt.ylabel('Makespan')
# # # giving a title to my graph
plt.title('Makespan comparison for 15 machines')
plt.legend()
# #
# # # function to show the plot
plt.savefig('M_15_comp.png')
#
#
# # 20 machines
#
# x1 = [1222, 1270, 817]
# t1 = [16.086936473846436, 59.179975748062134, 240.95711016654968]
#
# x2 = [1222, 1345, 1165]
# t2 = [5.0295491218566895, 8.256480693817139, 17.296739101409912]
#
# y1 = ["5", "10", "15"]
# plt.plot(y1, t1, label='ACO')
# plt.plot(y1, t2, label='GA')
#
# # naming the x axis
# plt.xlabel('Number of jobs')
# # naming the y axis
# plt.ylabel('Time')
# # giving a title to my graph
# plt.title('Time comparison for 20 machines')
# plt.legend()
#
# # function to show the plot
# plt.savefig('T_20_comp.png')


# plt.plot(x2, y2)
# # naming the x axis
# plt.xlabel('Iterations')
# # naming the y axis
# plt.ylabel('Makespan')
# # giving a title to my graph
# plt.title('Max iteration performance comparison')
#
# # function to show the plot
# plt.savefig('i_stats.png')

