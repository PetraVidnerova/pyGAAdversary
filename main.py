import sys 
import random
import numpy as np

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

from keras.models import model_from_json

from fitness import Fitness
from crossover import cxTwoPointCopy

import matplotlib.pyplot as plot

#from scoop import futures


NGEN = 1000
CXPB = 0.6
MUTPB = 0.1
IND_LEN = 784 
index = 1
TARGET_OUTPUT = int(sys.argv[1])

# weights = (1.0,) stands for one objective fitness
creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, IND_LEN)
toolbox.register("population", tools.initRepeat, list, toolbox.individual) 
#toolbox.register("map", futures.map)

model = model_from_json(open("mlp.json").read()) 
model.load_weights("mlp_weights.h5")

def mainGA():
    global toolbox 

    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1, similar=np.array_equal)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
  
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, 
                                   ngen=NGEN, stats=stats, halloffame=hof, 
                                   verbose=False)

    return hof[0] 

#    print(hof)
#    mat = hof[0].reshape(28,28) 
#    plot.matshow(mat, cmap=plot.cm.gray) 
#    plot.show() 

 #   X_eval = np.array([hof[0]])
 #   res = model.predict(X_eval)
 #   print(res)
 


X = [] 
for target_output in [ TARGET_OUTPUT ]:
    for target_image in range(10):
        print("Target image: %s Target output: %s" % (target_image, target_output))
        fit = Fitness(model, target_image, target_output)

        #Genetic operators 
        toolbox.register("evaluate", fit.evaluate)
        toolbox.register("mate", cxTwoPointCopy) 
        toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.1, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)

        X.append(mainGA())
            
 
# save X to file 
np.save("adversary_inputs_matrix_%s" % TARGET_OUTPUT,X)






