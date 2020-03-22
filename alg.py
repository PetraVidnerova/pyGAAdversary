import random
import pickle
import time
import datetime

from deap import algorithms


def myEASimple(population, toolbox, cxpb, mutpb, ngen,
               treshold, stats, halloffame, logbook, verbose, exp_id=None):

    total_time = datetime.timedelta(seconds=0)
    for gen in range(ngen):
        start_time = datetime.datetime.now()
        population = algorithms.varAnd(
            population, toolbox, cxpb=cxpb, mutpb=mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses_all = list(toolbox.map(toolbox.evaluate, invalid_ind))

        sucesses = [f[2] for f in fitnesses_all]
        dists = [(f[1], f[3]) for f in fitnesses_all]
        fitnesses = [(f[0],) for f in fitnesses_all]

        for ind, fit, suc, dist in zip(invalid_ind, fitnesses, sucesses, dists):
            ind.fitness.values = fit
            ind.success = suc
            ind.dist = dist

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
            if exp_id is None:
                cp_name = "checkpoint_ea.pkl"
            else:
                cp_name = "checkpoint_ea_{}.pkl".format(exp_id)
            pickle.dump(cp, open(cp_name, "wb"))

        gen_time = datetime.datetime.now() - start_time
        total_time = total_time + gen_time
        print("Time ", total_time, flush=True)

        print("Stopka: ", halloffame[0].fitness.values,
              halloffame[0].dist,
              halloffame[0].success)

        if (halloffame[0].dist[0] < treshold
                and halloffame[0].success):
            break

        # if total_time > datetime.timedelta(hours=4*24):
        #     print("Time limit exceeded.")
        #     break

    return population, logbook


def eval_invalid_inds(population, toolbox):

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses_all = list(toolbox.map(toolbox.evaluate, invalid_ind))

    sucesses = [f[2] for f in fitnesses_all]
    image_dists = [f[0] for f in fitnesses_all]
    output_dists = [f[1] for f in fitnesses_all]

    for ind, s, i, o in zip(invalid_ind, sucesses, image_dists, output_dists):
        ind.fitness.values = (i, o)
        ind.success = s
        ind.dist = i

    return len(invalid_ind)


def nsga(population, toolbox, cxpb, mutpb, ngen,
         treshold, stats, halloffame, logbook, verbose, exp_id=None):

    popsize = len(population)
    total_time = datetime.timedelta(seconds=0)

    eval_invalid_inds(population, toolbox)

    for gen in range(ngen):
        start_time = datetime.datetime.now()

        offspring = algorithms.varAnd(population, toolbox, cxpb=cxpb,
                                      mutpb=mutpb)

        evals = eval_invalid_inds(offspring, toolbox)
        population = toolbox.select(population+offspring, k=popsize)

        halloffame.update(offspring)  # updates pareto front
        # update statics
        record = stats.compile(population)
        logbook.record(gen=gen, evals=evals, **record)
        if verbose:
            print(logbook.stream, flush=True)

        # save checkpoint
        if gen % 1 == 0:
            # Fill the dictionary using the dict(key=value[, ...]) constructor
            cp = dict(
                population=population,
                generation=gen,
                halloffame=halloffame,
                logbook=logbook,
                rndstate=random.getstate()
            )

            if exp_id is None:
                cp_name = "checkpoint_nsga.pkl"
            else:
                cp_name = "checkpoint_nsga_{}.pkl".format(exp_id)
            pickle.dump(cp, open(cp_name, "wb"))

        # check hard time limit
        gen_time = datetime.datetime.now() - start_time
        total_time = total_time + gen_time
        print("Time ", total_time)

        # the last individual, least distance to the desired output
        first_successful = halloffame[-1]
        for ind in halloffame:
            if ind.success:
                # first successful, least distance to the target image
                first_successful = ind
                break

        print("Stopka: ", first_successful.fitness.values,
              first_successful.dist,
              first_successful.success)

        if (first_successful.dist < treshold
                and first_successful.success):
            break

        # hard time limit was necessary at metacentrum
        #
        # if total_time > datetime.timedelta(hours=4*24):
        #     print("Time limit exceeded.")
        #     break

    return population, logbook, first_successful
