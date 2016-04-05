import numpy as np 
from keras.models import model_from_json 
from keras.datasets import mnist 
import sys 

def probs_to_string(prob):
    # Pretty print vector of probabilities to string.
    res = ""
    for p in prob:
        res += "%.2f " % p 
    return res


# Load matrix of adversary inputs.
# one image per line 
X = np.load("adversary_inputs_matrix.npy") 

#  Load trained CNN. 
model = model_from_json(open("mlp2.json").read()) 
model.load_weights("mlp_weights2.h5")

# For each image print prediction. 
for i in range(0,10):
    res = model.predict(np.array([X[i]]))    
    print(probs_to_string(res[0])) 


    

