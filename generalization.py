import numpy as np
from load_models import load_models
from show import predict 
import sys 

def load_X(name):
    
    for i in range(10):
        file_name = "adversary_inputs_against_{0}_{1}.npy".format(name, i)
        X = np.load(file_name)
        yield X
    

if __name__ == "__main__":

    MODEL_NAME = sys.argv[1] 
    models = load_models()
    model = models[MODEL_NAME] 

    for X, i  in zip(load_X(MODEL_NAME), range(10)):
        outputs = {} 
        
        # go through models an evaluate X 
        # make y vector of 0 and 1 
        for model_name, model in models.items():
            Y = predict(model_name, model, X)  
            y = Y.argmax(axis=1)
            y = (y == i).astype(int)
            outputs[model_name] = y

            
        for x, j  in zip(X, range(10)):
            if (i==j):
                continue
            print(i, j, end=" ") 
            for model_name, model in models.items():
                if outputs[model_name][j]:
                    print(model_name, end=" ")
            print() 
        

        
