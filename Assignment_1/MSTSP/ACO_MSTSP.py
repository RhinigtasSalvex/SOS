import math
import random
import time

from matplotlib import pyplot as plt
from cities import cityCoordinates
from visualize import plot_ACO


class SolveTSPUsingACO:
    class Edge:
        def __init__(self, a, b, weight, initial_pheromone):
            self.a = a
            self.b = b
            self.weight = weight
            self.pheromone = initial_pheromone

    class Ant:
        def __init__(self, alpha, beta, num_nodes, edges):
            self.alpha = alpha
            self.beta = beta
            self.num_nodes = num_nodes
            self.edges = edges
            self.tour = None
            self.smallest_edge = 0.0

        def _select_node(self):
            roulette_wheel = 0.0
            unvisited_nodes = [node for node in range(self.num_nodes) if node not in self.tour]
            heuristic_total = 0.0
            for unvisited_node in unvisited_nodes:
                heuristic_total += self.edges[self.tour[-1]][unvisited_node].weight
            for unvisited_node in unvisited_nodes:
                roulette_wheel += pow(self.edges[self.tour[-1]][unvisited_node].pheromone, self.alpha) * \
                                  pow((self.edges[self.tour[-1]][unvisited_node].weight / heuristic_total), self.beta)
            random_value = random.uniform(0.0, roulette_wheel)
            wheel_position = 0.0
            for unvisited_node in unvisited_nodes:
                wheel_position += pow(self.edges[self.tour[-1]][unvisited_node].pheromone, self.alpha) * \
                                  pow((self.edges[self.tour[-1]][unvisited_node].weight / heuristic_total), self.beta)
                if wheel_position >= random_value:
                    return unvisited_node

        def find_tour(self):
            self.tour = [0]
            while len(self.tour) < self.num_nodes:
                self.tour.append(self._select_node())
            self.tour.append(0)
            return self.tour

        def get_smallest_edge(self):
            self.smallest_edge = float("inf")
            for i in range(self.num_nodes):
                d = self.edges[self.tour[i]][self.tour[(i + 1) % self.num_nodes]].weight
                if (d < self.smallest_edge):
                    self.smallest_edge = d
            return self.smallest_edge

    def __init__(self, mode='ACS', colony_size=10, elitist_weight=1.0, min_scaling_factor=0.001, alpha=1.0, beta=3.0,
                 rho=0.1, pheromone_deposit_weight=1.0, initial_pheromone=1.0, steps=100, nodes=None, labels=None):
        self.mode = mode
        self.colony_size = colony_size
        self.elitist_weight = elitist_weight
        self.min_scaling_factor = min_scaling_factor
        self.rho = rho
        self.pheromone_deposit_weight = pheromone_deposit_weight
        self.steps = steps
        self.num_nodes = len(nodes)
        self.nodes = nodes
        if labels is not None:
            self.labels = labels
        else:
            self.labels = range(1, self.num_nodes + 1)
        self.edges = [[None] * self.num_nodes for _ in range(self.num_nodes)]
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                self.edges[i][j] = self.edges[j][i] = self.Edge(i, j, math.sqrt(
                    pow(self.nodes[i][0] - self.nodes[j][0], 2.0) + pow(self.nodes[i][1] - self.nodes[j][1], 2.0)),
                                                                initial_pheromone)
        self.ants = [self.Ant(alpha, beta, self.num_nodes, self.edges) for _ in range(self.colony_size)]
        self.all_global_best_distance = []
        self.global_best_tour = None
        self.step_max_smallest = 0
        self.global_max_smallest = 0

    def _add_pheromone(self, tour, distance, weight=1.0):
        pheromone_to_add = self.pheromone_deposit_weight / distance
        for i in range(self.num_nodes):
            self.edges[tour[i]][tour[(i + 1) % self.num_nodes]].pheromone += weight * pheromone_to_add

    def _acs(self):
        # self.step same as generation in GA_MSTSP
        for step in range(self.steps):
            self.step_max_smallest = 0
            for ant in self.ants:
                self._add_pheromone(ant.find_tour(), ant.get_smallest_edge())
                if ant.smallest_edge > self.global_max_smallest:
                    self.global_best_tour = ant.tour
                    self.global_max_smallest = ant.smallest_edge
                if ant.smallest_edge > self.step_max_smallest:
                    self.step_max_smallest = ant.smallest_edge
            self.all_global_best_distance.append(self.step_max_smallest)
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    self.edges[i][j].pheromone *= (1.0 - self.rho)

    def _elitist(self):
        for step in range(self.steps):
            self.step_max_smallest = 0
            for ant in self.ants:
                self._add_pheromone(ant.find_tour(), ant.get_smallest_edge())
                if ant.smallest_edge > self.global_max_smallest:
                    self.global_best_tour = ant.tour
                    self.global_max_smallest = ant.smallest_edge
                if ant.smallest_edge > self.step_max_smallest:
                    self.step_max_smallest = ant.smallest_edge
            self.all_global_best_distance.append(self.step_max_smallest)
            self._add_pheromone(self.global_best_tour, self.global_max_smallest, weight=self.elitist_weight)
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    self.edges[i][j].pheromone *= (1.0 - self.rho)

    def _max_min(self):
        for step in range(self.steps):
            iteration_best_tour = None
            iteration_best_distance = 0
            for ant in self.ants:
                ant.find_tour()
                if ant.get_smallest_edge() > iteration_best_distance:
                    iteration_best_tour = ant.tour
                    iteration_best_distance = ant.smallest_edge
            self.all_global_best_distance.append(iteration_best_distance)
            if float(step + 1) / float(self.steps) <= 0.75:
                self._add_pheromone(iteration_best_tour, iteration_best_distance)
                max_pheromone = self.pheromone_deposit_weight / iteration_best_distance
            else:
                if iteration_best_distance > self.global_max_smallest:
                    self.global_best_tour = iteration_best_tour
                    self.global_max_smallest = iteration_best_distance
                self._add_pheromone(self.global_best_tour, self.global_max_smallest)
                max_pheromone = self.pheromone_deposit_weight / self.global_max_smallest
            min_pheromone = max_pheromone * self.min_scaling_factor
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    self.edges[i][j].pheromone *= (1.0 - self.rho)
                    if self.edges[i][j].pheromone > max_pheromone:
                        self.edges[i][j].pheromone = max_pheromone
                    elif self.edges[i][j].pheromone < min_pheromone:
                        self.edges[i][j].pheromone = min_pheromone

    def run(self):
        print('Started : {0}'.format(self.mode))
        if self.mode == 'ACS':
            self._acs()
        elif self.mode == 'Elitist':
            self._elitist()
        else:
            self._max_min()
        print('Ended : {0}'.format(self.mode))
        print('Sequence : <- {0} ->'.format(' - '.join(str(self.labels[i]) for i in self.global_best_tour)))
        print('Max minimum : {0}\n'.format(round(self.global_max_smallest, 2)))
        return self.steps, self.all_global_best_distance, self.global_best_tour, self.global_max_smallest, self.nodes


if __name__ == '__main__':
    begin = time.time()
    _colony_size = 25
    _steps = 300
    _nodes = cityCoordinates()
    #acs = SolveTSPUsingACO(mode='ACS', colony_size=_colony_size, steps=_steps, nodes=_nodes)
    #steps, all_global_best_tour, global_best_tour, global_best_distance, nodes = acs.run()
    #plot_ACO(steps, all_global_best_tour, global_best_tour, global_best_distance, nodes)

    #elitist = SolveTSPUsingACO(mode='Elitist', colony_size=_colony_size, steps=_steps, nodes=_nodes)
    #steps, all_global_best_tour, global_best_tour, global_best_distance, nodes = elitist.run()
    #plot_ACO(steps, all_global_best_tour, global_best_tour, global_best_distance, nodes)

    max_min = SolveTSPUsingACO(mode='MaxMin', colony_size=_colony_size, steps=_steps, nodes=_nodes)
    steps, all_global_best_tour, global_best_tour, global_best_distance, nodes = max_min.run()
    plot_ACO(steps, all_global_best_tour, global_best_tour, global_best_distance, nodes)
    end = time.time()
    print("Runtime:\t", (end - begin))

