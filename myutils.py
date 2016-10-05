import numpy as np 

def probs_to_string(prob):
    """ Pretty print vector of probabilities to string."""
    res = ""
    maxi = np.argmax(prob)
    for p in range(len(prob)):
        #res += " %.2f" % prob[p]
        if p == maxi:
            res += "& {\\bf %.2f}" % prob[p] 
        else:
            res += "& %.2f " % prob[p] 
    return res + "\\\\"
