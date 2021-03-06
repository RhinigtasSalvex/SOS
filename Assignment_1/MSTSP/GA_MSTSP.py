import time
import numpy as np
import random
import math
from visualize import plot_GA
from cities import cityCoordinates, generateCities_dense, generateCities

MUTATION_RATE = 60
MUTATION_REPEAT_COUNT = 2
WEAKNESS_THRESHOLD = 200

# Begin and end point is first city
cityCoordinates = cityCoordinates()

citySize = len(cityCoordinates)


class Genome():
    chromosomes = []
    fitness = 0


def CreateNewPopulation(size):
    population = []
    for x in range(size):
        newGenome = Genome()
        newGenome.chromosomes = random.sample(range(1, citySize), citySize - 1)
        newGenome.chromosomes.insert(0, 0)
        newGenome.chromosomes.append(0)
        newGenome.fitness = Evaluate(newGenome.chromosomes)
        population.append(newGenome)
    return population


# Calculate distance between two point
def distance(a, b):
    dis = math.sqrt(((a[0] - b[0])**2) + ((a[1] - b[1])**2))
    return np.round(dis, 2)


def Evaluate(chromosomes):
    calculatedFitness = float("inf")
    for i in range(len(chromosomes) - 1):
        p1 = cityCoordinates[chromosomes[i]]
        p2 = cityCoordinates[chromosomes[i + 1]]
        if (distance(p1, p2) < calculatedFitness):
            calculatedFitness = distance(p1, p2)
    calculatedFitness = np.round(calculatedFitness, 2)
    return calculatedFitness


def findBestGenome(population):
    allFitness = [i.fitness for i in population]
    bestFitness = max(allFitness)
    return population[allFitness.index(bestFitness)]


# In K-Way tournament selection, we select K individuals
# from the population at random and select the best out
# of these to become a parent. The same process is repeated
# for selecting the next parent.
def TournamentSelection(population, k):
    selected = [population[random.randrange(0, len(population))] for i in range(k)]
    bestGenome = findBestGenome(selected)
    return bestGenome


def Reproduction(population):
    parent1 = TournamentSelection(population, 10).chromosomes
    parent2 = TournamentSelection(population, 6).chromosomes
    while parent1 == parent2:
        parent2 = TournamentSelection(population, 6).chromosomes

    return OrderOneCrossover(parent1, parent2)


# Sample:
# parent1 = [0, 3, 8, 5, 1, 7, 12, 6, 4, 10, 11, 9, 2, 0]
# parent2 = [0, 1, 6, 3, 5, 4, 10, 2, 7, 12, 11, 8, 9, 0]
# child   = [0, 1, 3, 5, 2, 7, 12, 6, 4, 10, 11, 8, 9, 0]
def OrderOneCrossover(parent1, parent2):
    size = len(parent1)
    child = [-1] * size

    child[0], child[size - 1] = 0, 0

    point = random.randrange(5, size - 4)

    for i in range(point, point + 4):
        child[i] = parent1[i]
    point += 4
    point2 = point
    while child[point] in [-1, 0]:
        if child[point] != 0:
            if parent2[point2] not in child:
                child[point] = parent2[point2]
                point += 1
                if point == size:
                    point = 0
            else:
                point2 += 1
                if point2 == size:
                    point2 = 0
        else:
            point += 1
            if point == size:
                point = 0

    if random.randrange(0, 100) < MUTATION_RATE:
        child = SwapMutation(child)

    # Create new genome for child
    newGenome = Genome()
    newGenome.chromosomes = child
    newGenome.fitness = Evaluate(child)
    return newGenome

# Sample:
# Chromosomes =         [0, 3, 8, 5, 1, 7, 12, 6, 4, 10, 11, 9, 2, 0]
# Mutated chromosomes = [0, 11, 8, 5, 1, 7, 12, 6, 4, 10, 3, 9, 2, 0]


def SwapMutation(chromo):
    for x in range(MUTATION_REPEAT_COUNT):
        p1, p2 = [random.randrange(1, len(chromo) - 1) for i in range(2)]
        while p1 == p2:
            p2 = random.randrange(1, len(chromo) - 1)
        log = chromo[p1]
        chromo[p1] = chromo[p2]
        chromo[p2] = log
    return chromo


def GeneticAlgorithm(popSize, maxGeneration):
    allBestFitness = []
    population = CreateNewPopulation(popSize)
    generation = 0
    bestGenome = None
    while generation < maxGeneration:
        generation += 1

        for i in range(int(popSize / 2)):
            # Select parent, make crossover and
            # after, append in population a new child
            population.append(Reproduction(population))

        # Kill weakness person
        for genom in population:
            if genom.fitness < WEAKNESS_THRESHOLD:
                population.remove(genom)

        #averageFitness = round(np.sum([genom.fitness for genom in population]) / len(population), 2)
        bestStepGenome = findBestGenome(population)
        if bestGenome is None:
            bestGenome = bestStepGenome
        elif bestStepGenome.fitness > bestGenome.fitness:
            bestGenome = findBestGenome(population)

        allBestFitness.append(bestStepGenome.fitness)

    # Visualize
    print("Population Size: {0}\t Maximum minimum edge: {1}"
          .format(len(population), bestGenome.fitness))
    plot_GA(generation, allBestFitness, bestGenome, cityCoordinates)


if __name__ == "__main__":
    #generateCities(50)
    begin = time.time()
    GeneticAlgorithm(popSize=100, maxGeneration=300)
    end = time.time()
    print("Runtime:\t", (end - begin))
