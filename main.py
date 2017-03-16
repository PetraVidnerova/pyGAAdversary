import sys 
import random
import numpy as np

from scoop import futures 

from deap import base
from deap import creator
from deap import tools
from deap import algorithms


from sklearn.svm import SVC 

from fitness import Fitness
from crossover import cxTwoPointCopy, cxUniform

import pickle

from load_models import load_model

#import sys
#sys.path.insert(0, '/home/petra/pyRBF/')
#from rbf import *

#from scoop import futures

def mainGA(NAME, target_output, target_image):
    """ Runs the main loop of GA.""" 
    global toolbox

    print("Target image: {0} Target output: {1}".format(target_image, target_output))    
    sys.stdout.flush()

    model = load_model(NAME)    
    fit = Fitness(NAME, model, target_image, target_output)

    #Genetic operators 
    toolbox.register("evaluate", fit.evaluate)
    toolbox.register("mate", cxTwoPointCopy) 
    #toolbox.register("mate", cxUniform)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.1, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    

    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1, similar=np.array_equal)
    
    #stats = tools.Statistics(lambda ind: ind.fitness.values)
    #stats.register("avg", np.mean)
    #stats.register("std", np.std)
    #stats.register("min", np.min)
    #stats.register("max", np.max)
  
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, 
                                   ngen=NGEN, halloffame=hof, 
                                   verbose=False)

    return hof[0] 


NGEN = 5000
CXPB = 0.6
MUTPB = 0.1
IND_LEN = 784 


# weights = (1.0,) stands for one objective fitness
creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_real", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_real, IND_LEN)
toolbox.register("population", tools.initRepeat, list, toolbox.individual) 


if __name__ == "__main__":

    index = 1
    #TARGET_OUTPUT = int(sys.argv[2])
    TARGET_OUTPUT = int(sys.argv[1])
    #NAME = "adversary_five"
    NAME = "SVM_poly" 
    #TARGET_IMAGE = int(sys.argv[2])
    #IDX = sys.argv[2] 


    # Run the GA for each target image and target output.
    #for target_output in range(10):
    for target_output in [ TARGET_OUTPUT ]:
        X = list(futures.map(lambda x: mainGA(NAME, target_output, x), range(100)))
        #np.save("test", X)
        np.save("testset_adversary/adversary_%s_svm_poly" % target_output, X)
 
    # save X to file 

    #    np.save("samples5/{name}_{idx}".format(name = TARGET_IMAGE, idx = IDX), X)
 


