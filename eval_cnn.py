import numpy as np 
from keras.models import model_from_json 
from keras.datasets import mnist 

def probs_to_string(prob):
    # Pretty print vector of probabilities to string.
    res = ""
    for p in prob:
        res += "%.2f " % p 
    return res

img_rows, img_cols = 28, 28

# Load matrix of adversary inputs.
# one image per line 
X = np.load("adversary_inputs_against_cnn.npy") 
X = X.reshape(X.shape[0], 1, img_rows, img_cols)

#  Load trained CNN. 
model = model_from_json(open("cnn.json").read()) 
model.load_weights("cnn_weights.h5")

# For each image print prediction. 
for i in range(0,10):
    res = model.predict(np.array([X[i]]))    
    print(probs_to_string(res[0])) 


    

