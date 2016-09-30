import numpy as np
from load_models import load_models 
from show import predict 

def load_all_X():
    NAMES = [ "adversary_inputs_/adversary_inputs_against_cnn.npy",
              "adversary_inputs_/adversary_inputs_against_ensemble.npy",
              "adversary_inputs_/adversary_inputs_against_mlp_0.npy",
              "adversary_inputs_/adversary_inputs_against_rbf.npy",
              "adversary_inputs_/adversary_inputs_against_svm_linear.npy",
              "adversary_inputs_/adversary_inputs_against_svm_poly.npy",
              "adversary_inputs_/adversary_inputs_against_svm_poly4.npy",
              "adversary_inputs_/adversary_inputs_against_svm_rbf.npy",
              "adversary_inputs_/adversary_inputs_against_svm_sigmoid.npy",
              "adversary_inputs_/adversary_inputs_against_tree.npy" ]

    X = None
    for name in NAMES:
        Xadd = np.load(name)
        if X is None:
            X = Xadd
        else:
            X = np.vstack((X, Xadd))
    return X 

if __name__ == "__main__":

    models = load_models() 
    X = load_all_X() 
    
    i = 0
    for x in X:
        print("{0}, ".format(i), end="")
        i = i + 1
    print("model") 

    for model_name, model in models.items():
        Y = predict(model_name, model, X)  
        y = Y.argmax(axis=1)
        y = (y == 0).astype(int)
        
        for i in y:
            print("{0}, ".format(i), end="")        
        print(model_name)
        
