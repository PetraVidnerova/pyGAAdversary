import random

# crossover to work with numpy array 
def cxTwoPointCopy(ind1, ind2):
 
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
        
    return ind1, ind2


def cxUniform(ind1, ind2): 
    
    assert len(ind1) == len(ind2) 

    for i in range(len(ind1)):
        if random.random()<0.5:
            ind1[i], ind2[i] = ind2[i].copy(), ind1[i].copy() 

    return ind1, ind2
