import numpy as np
from load_models import load_models
from show import predict 

if __name__ == "__main__":

    MODEL_NAMES = [ "MLP", "CNN", "SVM_sigmoid", "SVM_poly", "SVM_poly4", 
                    "SVM_linear", "SVM_rbf", "RBF", "DT"]

    models = load_models() 
    also_miss = { (x, i, j) : [] for x in MODEL_NAMES for i in range(10) for j in range(10) } 

    for name in MODEL_NAMES: 
        
        for name2, model in models.items():

            if name == name2:
                continue 

            for target_output in range(10):
                try:
                    X = np.load("adversary_inputs_against_{0}_{1}.npy".format(name, target_output))
                except:
                    continue
                
                import sys 
                print(name, name2, file=sys.stderr)
                YP = predict(name2, model, X)

                for y, i in zip(YP, range(10)):
                    if i == target_output:
                        continue 
                    if y.argmax() == target_output:
                        also_miss[(name, target_output, i)].append(name2)
                
    for name in MODEL_NAMES:
        for i in range(10):
            for j in range(10):
                if i == j:
                    continue
                print(name, i, j, also_miss[(name, i, j)], len(also_miss[(name, i, j)]))


