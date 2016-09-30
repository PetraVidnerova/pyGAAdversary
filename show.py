import numpy as np 
import matplotlib.pyplot as plot 
from keras.models import model_from_json 
from keras.datasets import mnist 
from matplotlib import rcParams
import sys

from load_models import load_models, load_model

def probs_to_string(prob):
    # Pretty print vector of probabilities to string.
    res = ""
    for p in prob:
        res += "%.2f   " % p 
    return res

def predict(model_name, model, INPUTS):
    if model_name.startswith("CNN"):
        INPUTS = INPUTS.reshape(INPUTS.shape[0], 1, 28, 28)

    if model_name.startswith("SVM") or model_name.startswith("DT"):
        return model.predict_proba(INPUTS)
    else:
        return model.predict(INPUTS)

def plot_and_predict(subplot, model_name, model, x):
    inputs = np.array([x])
    res = predict(model_name, model, inputs)
    
    print(probs_to_string(res[0])) 
    
    if (res.argmax() == target_output) and (target_output != i):
        x = X[i].reshape(28,28)
    elif target_output == i:
        x = 0.5*np.ones((28,28)) 
        x[0][0] = 0
        x[-1][-1] = 1
    else:
        x = np.zeros((28,28))
        
    pltarr[target_output][i].axis('off')
    pltarr[target_output][i].imshow(x, interpolation="none", cmap=plot.cm.Greys)
            

if __name__=="__main__":

    model_name = sys.argv[1] 
    #target_output = int(sys.argv[2])


    #  Load trained model 
    #models = load_models() 
    model = load_model(model_name)
    
    # Load training set
    (X_train, y_train), (X_test, y_test) = mnist.load_data() 
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

#    plot.figure(1)
#    rcParams.update({'font.size': 8})
    f, pltarr = plot.subplots(10,10)
    plot.axis('off')

    for target_output in range(10):

        # Load matrix of adversary inputs.
        # one image per line
        try:
            X = np.load("adversary_inputs_against_{0}_{1}.npy".format(model_name, target_output)) 
        except:
            continue
            
        pltarr[target_output][0].set_ylabel("x")
        # For each image print prediction and show image.
        for i in range(10):
            plot_and_predict(pltarr[target_output][i], model_name, model, X[i])


    plot.show()

#plot.show()

    

