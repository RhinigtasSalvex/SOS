import math

import matplotlib.pyplot as plt


def plot_GA(generation, allBestFitness, bestGenome, cityLoc):
    plt.subplot(2, 1, 1)
    plt.text((generation / 2) - 0.5, allBestFitness[0] + 10, "Generation: {0} Best Fitness: {1}".format(
        generation, bestGenome.fitness), ha='center', va='bottom')
    plt.plot(range(0, generation), allBestFitness, c="green")
    plt.subplot(2, 1, 2)

    startPoint = None
    for x, y in cityLoc:
        if startPoint is None:
            startPoint = cityLoc[0]
            plt.scatter(startPoint[0], startPoint[1], c="green", marker=">")
            plt.annotate("Origin", (x + 2, y - 4))
        else:
            plt.scatter(x, y, c="black")

    xx = [cityLoc[i][0] for i in bestGenome.chromosomes]
    yy = [cityLoc[i][1] for i in bestGenome.chromosomes]

    for x, y in zip(xx, yy):
        plt.text(x + 2, y - 2, str(yy.index(y)), color="green", fontsize=10)

    plt.plot(xx, yy, color="red", linewidth=1.75, linestyle="-")
    plt.show()


def plot_ACO(steps, all_global_best_distance, global_best_tour, global_best_distance, nodes):
    plt.subplot(2, 1, 1)
    # plt.text((steps / 2) - 0.5, all_global_best_distance[0] + 10, "Generation: {0} Best Fitness: {1}".format(
    #     steps, global_best_distance), ha='center', va='bottom')
    print(all_global_best_distance)
    print(steps)
    plt.plot(range(0, steps), all_global_best_distance, c="green")
    plt.subplot(2, 1, 2)

    startPoint = None
    for x, y in nodes:
        if startPoint is not None:
            startPoint = nodes[0]
            plt.scatter(startPoint[0], startPoint[1], c="green", marker=">")
            # plt.annotate("Origin", (x + 2, y - 4))
        else:
            plt.scatter(x, y, c="black")

    xx = [nodes[i][0] for i in global_best_tour]
    yy = [nodes[i][1] for i in global_best_tour]

    for x, y in zip(xx, yy):
        plt.text(x + 2, y - 2, str(yy.index(y)), color="green", fontsize=10)

    plt.plot(xx, yy, color="red", linewidth=1.75, linestyle="-")
    plt.show()