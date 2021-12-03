'''
    ==============================================================
    Ant Colony Optimization algorithm for continuous domains ACO_R
    ==============================================================

    author: Andreas Tsichritzis <tsadreas@gmail.com>
'''

import os
import sys
import shutil
import numpy as np
import random
from time import time, process_time
import multiprocessing
import sys
sys.stdout = open('C:\\Users\\Maxim\\Documents\\Uni\\11. Semester\\SOS\\SOS\\Rastrigin\\ACO\\out.txt', 'w')
# sys.stdout = open('./ACO/out.txt', 'w')

import datetime
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import defaultdict
from operator import itemgetter
import csv

def map_range(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def rastrigin(X):
    A = 10.0 
    delta = [x**2 - A * np.cos(2 * np.pi * x) for x in X]
    y = A * len(X) + sum(delta)
    return y


def evaluator(X):
    '''Evaluator function, returns fitness and responses values'''
    # give the normalized candidates values inside the real design space for Styblinski–Tang function
    # x = [10*i-5 for i in X]
    # give the normalized candidates values inside the real design space for rastrigin function
    # X = [map_range(x, 0, 1, -5.12, 5.12) for x in X]

    # calculate fitness for Styblinski–Tang function
    # f = (sum([math.pow(i,4)-16*math.pow(i,2)+5*i for i in x])/2)

    # calculate fitness for rastrigin function
    f = rastrigin(X)

    # calculate values for other responses for Styblinski–Tang function
    # res = {'r1':f-5,'r2':2*f}
    # calculate values for other responses for rastrigin function
    # res = {f"r{i+1}": f-5.12 for i in range(len(X))}
    res = {f"r{i+1}": f for i in range(len(X))}
    # res = {}

    fitness = dict(Obj=f,**res)

    return fitness


def mp_evaluator(x):
    '''Multiprocessing evaluation'''
    # ste number of cpus
    nprocs = 2
    # create pool
    pool = multiprocessing.Pool(processes=nprocs)
    results = [pool.apply_async(evaluator,[c]) for c in x]
    pool.close()
    pool.join()
    f = [r.get()['Obj'] for r in results]
    for r in results:
        del r.get()['Obj']
    # maximization or minimization problem
    maximize = False
    return (f, [r.get() for r in results],maximize)


def initialize(ants,var):
    '''Create initial solution matrix'''
    X = np.random.uniform(low=0,high=1,size=(ants,var))
    return X


def init_observer(filename,matrix,parameters,responses):
    '''Initial population observer'''
    p = []
    r = []
    f = []
    res = ['{0:>10}'.format(i)[:10] for i in responses]
    par = ['{0:>10}'.format(i)[:10] for i in parameters]
    for i in range(len(matrix)):
        p.append(matrix[i][0:len(parameters)])
        r.append(matrix[i][len(parameters):-1])
        f.append(matrix[i][-1])
    r = np.array(r)
    p = np.array(p)

    for i in range(len(r)):
        r[i] = ['{0:>10}'.format(r[i][j])[:10] for j in range(len(responses))]

    for i in range(len(p)):
        p[i] = ['{0:>10}'.format(p[i][j])[:10] for j in range(len(parameters))]

    f = ['{0:>10}'.format(i)[:10] for i in f]

    iteration = 0

    filename.write('{0:>10}, {1}, {2:>10}, {3}\n'.format('Iteration',', '.join(map(str, par)),'Fitness',', '.join(map(str, res))))

    for i in range(len(matrix)):
        filename.write('{0:>10}, {1}, {2:>10}, {3}\n'.format(iteration,', '.join(map(str, p[i])),f[i],', '.join(map(str, r[i]))))


def iter_observer(filename,matrix,parameters,responses,iteration):
    '''Iterations observer'''
    p = []
    r = []
    f = []
    for i in range(len(matrix)):
        p.append(matrix[i][0:len(parameters)])
        r.append(matrix[i][len(parameters):-1])
        f.append(matrix[i][-1])
    r = np.array(r)
    p = np.array(p)

    for i in range(len(r)):
        r[i] = ['{0:>10}'.format(r[i][j])[:10] for j in range(len(responses))]

    for i in range(len(p)):
        p[i] = ['{0:>10}'.format(p[i][j])[:10] for j in range(len(parameters))]

    f = ['{0:>10}'.format(i)[:10] for i in f]

    for i in range(len(matrix)):
        filename.write('{0:>10}, {1}, {2:>10}, {3}\n'.format(iteration,', '.join(map(str, p[i])),f[i],', '.join(map(str, r[i]))))


def correct_par(filename,par):
    """Replace normalized values with real"""
    columns = defaultdict(list)
    with open(filename) as f:
        reader = csv.DictReader(f,skipinitialspace=True)
        for row in reader:
            for (k,v) in list(row.items()):
                columns[k].append(v)
        keys = list(columns.keys())
        for p in par:
            if p in keys:
                col = []
                for i,k in enumerate(columns[p]):
                    k = float(k)
                    if p in par:
                        n = 10*k-5.12
                    col.append(n)
                columns[p] = col

    outputfile = filename

    file = open(outputfile,'w+')
    head = []
    head.append('Iteration')
    for i in par:
        head.append(i)
    head.append('Fitness')
    for i in keys:
        if i not in head:
            head.append(i)
    par = ['{0:>10}'.format(i)[:10] for i in par]
    line = ['{0:>10}'.format(l)[:10] for l in head]
    file.write('{0}\n'.format(', '.join(map(str, line))))
    for i in range(len(columns.get('Iteration'))):
        line = []
        for j in head:
            line.append(columns.get(j)[i])
        line = ['{0:>10}'.format(l)[:10] for l in line]
        file.write('{0}\n'.format(', '.join(map(str, line))))
    file.close()

def formatTD(td):
    """ Format time output for report"""
    days = td.days
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    seconds = td.seconds % 60
    return '%s days %s h %s m %s s' % (days, hours, minutes, seconds)

def rastrigin_function(x):
	""" Rastrigin's function multimodal, symmetric, separable"""
	res = 10*len(x)
	for i in range(len(x)):
		res += x[i]**2 - (10*np.cos(2*math.pi*x[i]))
	return res

def plot_rastrigin_function_2d(projdir):
    """Plot rastrigin function"""
    x = np.linspace(-5.12,5.12,100)
    y = rastrigin_function([x])
    plt.plot(x,y)
    plt.show()
    plt.savefig('{0}/rastrigin_2d.png'.format(projdir))

def plot_rastrigin_function_3d(projdir):
    """Plot rastrigin function"""
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    x = np.arange(-5.12,5.12,0.1)
    y = np.arange(-5.12,5.12,0.1)
    x, y = np.meshgrid(x, y)
    z = rastrigin_function([x,y])
    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    plt.savefig('{0}/rastrigin_3d.png'.format(projdir))

def plot_fitness_results(mean_fitness, max_fitness, iterations, save_plot=False, path=None, solver=None):
        """
        Plots the evolution of the mean and max fitness of the population

        :param mean_fitness: mean fitness array for each generation
        :param max_fitness: max fitness array for each generation
        :param iterations: total number of generations
        :return: None
        """
        if solver is not None:
            mean_fitness = solver["mean_fitness"]
            max_fitness = solver["max_fitness"]
            iterations = solver["max_gen"]
        plt.figure(figsize=(7, 7))

        x = np.arange(1, iterations + 1)

        plt.plot(x, mean_fitness, label="mean fitness")
        plt.plot(x, max_fitness, label="max fitness")
        if solver:
            plt.title(f"Fitness Evolution\nVariables: {solver['vars']} Population: {solver['pop']} Evaporation: {solver['evap']}")
        plt.legend()
        if save_plot:
            if solver:
                plt.savefig(f"{path}fitness_evolution_v{solver['vars']}_p{solver['pop']}_e{solver['evap']}_mg{solver['max_gen']}.png")
            else:
                plt.savefig(f"{path}fitness_evolution.png")
        else:
            plt.show()


def evolve(parameters=1, nSize=100, nAnts=100, q=0.3, xi=0.65, maxiter=100, display=False, verbose=False):
    '''
    Executes the optimization
    
    '''
    start_time = time()
    # number of variables
    parameters_v = [f"x{i}" for i in range(1,parameters+1)]
    # response_v = []
    response_v = [f"r{i}" for i in range(1,parameters+1)]

    if verbose:
        # create output file
        projdir = os.getcwd()
        ind_file_name = '{0}/results.csv'.format(projdir)
        ind_file = open(ind_file_name, 'w')

    # number of variables
    nVar = len(parameters_v)
    # size of solution archive
    # nSize = 100
    # number of ants
    # nAnts = 100

    # parameter q
    # q = 0.3

    # standard deviation
    qk = q*nSize

    # parameter xi (like pheromone evaporation)
    # xi = 0.65

    # maximum iterations
    # maxiter = 200
    # tolerance
    errormin = 0.01

    # bounds of variables
    Up = [1]*nVar
    Lo = [0]*nVar

    # initilize matrices
    S = np.zeros((nSize,nVar))
    S_f = np.zeros((nSize,1))
    # plot_rastrigin_function_3d(projdir)

    # plt.figure()

    # initialize the solution table with uniform random distribution and sort it
    if verbose:
        print('-----------------------------------------')
        print('Starting initilization of solution matrix')
        print('-----------------------------------------')

    Srand = initialize(nSize,nVar)
    f,S_r,maximize = mp_evaluator(Srand)

    S_responses = []

    for i in range(len(S_r)):
        S_f[i] = f[i]
        k = S_r[i]
        row = []
        for r in response_v:
            row.append(k[r])
        S_responses.append(row)

    # add responses and "fitness" column to solution
    S = np.hstack((Srand,S_responses,S_f))
    # sort according to fitness (last column)
    S = sorted(S, key=lambda row: row[-1],reverse = maximize)
    S = np.array(S)

    # init_observer(ind_file,S,parameters_v,response_v)

    # initilize weight array with pdf function
    w = np.zeros((nSize))
    for i in range(nSize):
        w[i] = 1/(qk*2*math.pi)*math.exp(-math.pow(i,2)/(2*math.pow(q,2)*math.pow(nSize,2)))


    if display:
        x = []
        y = []
        for i in S:
            x.append(i[0])
            y.append(i[1])

        plt.scatter(x,y)
        plt.xlim(0,1)
        plt.ylim(0,1)
        # plt.pause(0.5)
        plt.cla()
        plt.savefig('{0}/init.png'.format(projdir))

    # initialize variables
    iterations = 1
    best_par = []
    best_obj = []
    best_sol = []
    best_res = []
    worst_obj = []
    best_par.append(S[0][:nVar])
    best_obj.append(S[0][-1])
    best_sol.append(S[0][:])
    best_res.append(S[0][nVar:-1])
    worst_obj.append(S[-1][-1])
    mean_fitness = np.ndarray(shape=(1, 0))
    max_fitness = np.ndarray(shape=(1, 0))

    stop = 0

    # iterations
    while True:
        if verbose:
            print('-----------------------------------------')
            print('Iteration', iterations)
            print('-----------------------------------------')
        # choose Gaussian function to compose Gaussian kernel
        p = w/sum(w)

        # find best and index of best
        max_prospect = np.amax(p)
        ix_prospect = np.argmax(p)
        selection = ix_prospect

        # calculation of G_i
        # find standard deviation sigma
        sigma_s = np.zeros((nVar,1))
        sigma = np.zeros((nVar,1))
        for i in range(nVar):
            for j in range(nSize):
                sigma_s[i] = sigma_s[i] + abs(S[j][i] - S[selection][i])
            sigma[i] = xi / (nSize -1) * sigma_s[i]


        Stemp = np.zeros((nAnts,nVar))
        ffeval = np.zeros((nAnts,1))
        res = np.zeros((nAnts,len(response_v)))
        for k in range(nAnts):
            for i in range(nVar):
                Stemp[k][i] = sigma[i] * np.random.random_sample() + S[selection][i]
                if Stemp[k][i] > Up[i]:
                    Stemp[k][i] = Up[i]
                elif Stemp[k][i] < Lo[i]:
                    Stemp[k][i] = Lo[i]
        f,S_r,maximize = mp_evaluator(Stemp)
        mean_fitness = np.append(mean_fitness, np.mean(f))
        max_fitness = np.append(max_fitness, np.min(f))

        S_f = np.zeros((nAnts,1))
        S_responses = []

        for i in range(len(S_r)):
            S_f[i] = f[i]
            k = S_r[i]
            row = []
            for r in response_v:
                row.append(k[r])
            S_responses.append(row)

        # add responses and "fitness" column to solution
        Ssample = np.hstack((Stemp,S_responses,S_f))

        # add new solutions in the solutions table
        Solution_temp = np.vstack((S,Ssample))

        # sort according to "fitness"
        Solution_temp = sorted(Solution_temp, key=lambda row: row[-1],reverse = maximize)
        Solution_temp = np.array(Solution_temp)

        # keep best solutions
        S = Solution_temp[:nSize][:]

        # keep best after each iteration
        best_par.append(S[0][:nVar])
        best_obj.append(S[0][-1])
        best_res.append(S[0][nVar:-1])
        best_sol.append(S[0][:])
        worst_obj.append(S[-1][-1])

        if verbose:
            iter_observer(ind_file,S,parameters_v,response_v,iterations)

        if display:
            # plot new table
            x = []
            y = []
            for i in S:
                x.append(i[0])
                y.append(i[1])

            plt.scatter(x,y)
            plt.xlim(0,1)
            plt.ylim(0,1)
            # plt.pause(2)
            plt.savefig('{0}/iter_{1}.png'.format(projdir,iterations))

        if iterations > 1:
            diff = abs(best_obj[iterations]-best_obj[iterations-1])
            if diff <= errormin:
                stop += 1

        iterations += 1
        if iterations > maxiter or stop > 5:
            break
        else:
            if display:
                plt.cla()
    if verbose:
        ind_file.close()

    total_time_s = time() - start_time
    total_time = datetime.timedelta(seconds=total_time_s)
    total_time = formatTD(total_time)

    if verbose:
        # fix varibales values in output file
        correct_par(ind_file_name,parameters_v)

    best_sol = sorted(best_sol, key=lambda row: row[-1],reverse = maximize)

    print("Best individual: ", best_sol[0][0:len(parameters_v)])
    print("Fitness: ", best_sol[0][-1])
    # print("Responses:", response_v)
    # print(best_sol[0][len(parameters_v):-1])
    return best_sol, mean_fitness, max_fitness, iterations


if (__name__=="__main__"):
    # path = os.getcwd()
    # # get folder name from input
    # case = str(sys.argv[1])
    # if os.path.exists(path+'/'+case):
    #     print('Folder already exists, deleting it\n')
    #     # input('Folder exists! Press enter to delete it and continue')
    #     shutil.rmtree(path+'/'+case)
    # os.mkdir(path+'/'+case)
    # os.chdir(path+'/'+case)

    projdir = os.getcwd()
    # plot_rastrigin_function_2d(projdir)
    # plot_rastrigin_function_3d(projdir)
    # Executes optimization run.
    # If display = True plots ants in 2D design space
    for g in [2, 10]:
        for p in [10, 50, 100, 1000]:
            for m in [100, 500]:
                for q in [0.3, 0.5, 0.8]:
                    for xi in [0.3, 0.5, 0.65, 0.8]:
                        start = process_time()
                        b, mean_fitness, max_fitness, iterations = evolve(display = False, parameters=g, nAnts=p, maxiter=m, xi=xi, q=q)
                        elapsed_time = process_time() - start
                        print(f"Elapsed time: {elapsed_time}")
                        print(f"Variables: {g}, Population: {p}, Max_gen: {m}, Density: {q}, Evaporation: {xi}\n\n")
                        solver = {
                            'vars': g,
                            'pop': p,
                            'density': q,
                            'evap': xi,
                            'max_gen': iterations -1,
                            'mean_fitness': mean_fitness,
                            'max_fitness': max_fitness,
                            }
                        plot_fitness_results(None, None, None, solver=solver, save_plot=True, path="C:\\Users\\Maxim\\Documents\\Uni\\11. Semester\\SOS\\SOS\\Rastrigin\\ACO\\")

