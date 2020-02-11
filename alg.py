import random
import pickle
import time
import datetime 

from deap import algorithms 


def myEASimple(population, toolbox, cxpb, mutpb, ngen, 
               treshold, stats, halloffame, logbook, verbose, id=None):

    total_time = datetime.timedelta(seconds=0) 
    for gen in range(ngen):
        start_time = datetime.datetime.now()
        population = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses_all = list(toolbox.map(toolbox.evaluate, invalid_ind))

        sucesses = [ f[2] for f in fitnesses_all ]
        fitnesses = [ (f[0], f[1]) for f in fitnesses_all ]

        for ind, fit, suc in zip(invalid_ind, fitnesses, sucesses):
            ind.fitness.values = fit
            ind.success = suc
            

        halloffame.update(population)
        record = stats.compile(population)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)


        population = toolbox.select(population, k=len(population))

        if gen % 1 == 0:
            # Fill the dictionary using the dict(key=value[, ...]) constructor
            cp = dict(population=population, generation=gen, halloffame=halloffame,
                      logbook=logbook, rndstate=random.getstate())
            if id is None:
                cp_name = "checkpoint_ea.pkl"
            else:
                cp_name = "checkpoint_ea_{}.pkl".format(id)
            pickle.dump(cp, open(cp_name, "wb"))
            
        gen_time = datetime.datetime.now() - start_time
        total_time = total_time + gen_time
        print("Time ", total_time)
        
        print("Stopka: ", halloffame[0].fitness.values[0],
              halloffame[0].success)

        if  ( halloffame[0].fitness.values[0] < treshold 
              and halloffame[0].success):
            break

        # if total_time > datetime.timedelta(hours=4*24):
        #     print("Time limit exceeded.")
        #     break 

    return population, logbook


