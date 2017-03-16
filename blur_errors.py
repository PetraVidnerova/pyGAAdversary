import numpy as np
from keras.datasets import mnist  
from keras.utils import np_utils
from keras.models import Sequential
from load_models import load_models, load_model
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy import ndimage
from skimage import filters, restoration

#import sys
#sys.path.insert(0, '/home/petra/pyRBF/')
#from rbf import *




nb_classes = 10 

def blur_predict(model, X, type="median", filter_size=3, sigma=1.0):
  
    if type == "median":
        blured_X = np.array(list(map(lambda x: ndimage.median_filter(x, filter_size), 
                                     X)))
    elif type == "gaussian":
        blured_X = np.array(list(map(lambda x: ndimage.gaussian_filter(x, filter_size),
                                     X)))
    elif type == "f_gaussian":
        blured_X = np.array(list(map(lambda x: filters.gaussian_filter(x.reshape((28, 28)), sigma=sigma).reshape(784),
                                     X))) 
    elif type == "tv_chambolle":
        blured_X = np.array(list(map(lambda x: restoration.denoise_tv_chambolle(x.reshape((28, 28)), weight=0.2).reshape(784),
                                     X)))
    elif type == "tv_bregman":
        blured_X = np.array(list(map(lambda x: restoration.denoise_tv_bregman(x.reshape((28, 28)), weight=5.0).reshape(784),
                                     X)))
    elif type == "bilateral":
        blured_X = np.array(list(map(lambda x: restoration.denoise_bilateral(np.abs(x).reshape((28, 28))).reshape(784),
                                     X)))
    elif type == "nl_means":
        blured_X = np.array(list(map(lambda x: restoration.nl_means_denoising(x.reshape((28, 28))).reshape(784),
                                     X)))
        
    elif type == "none":
        blured_X = X 

    else:
        raise ValueError("unsupported filter type", type)

    return predict(model, blured_X)


def predict(model, X):
    
    if isinstance(model, Sequential) or isinstance(model, RBFNet):
        #for cnn
        #X = X.reshape(X.shape[0], 28, 28, 1)
        return model.predict(X)
    else:
        return model.predict_proba(X)


def evaluate_data(model, X, y):
    Y = np_utils.to_categorical(y, nb_classes)
    
    for type in "none", "median",  "f_gaussian", "tv_chambolle", "tv_bregman", "bilateral", "nl_means":
        Y_pred = blur_predict(model, X, type)
        err = mean_squared_error(Y_pred, Y) 
        y_pred = np.argmax(Y_pred, axis=1) 
        acc = accuracy_score(y_pred, y) 
        print(type, err, acc) 



if __name__ == "__main__":
    
    # load trained model 
    model = load_model("MLP") 
    

    # Load training set
    (X_train, y_train), (X_test, y_test) = mnist.load_data() 
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    # X_test = X_test.reshape(10000, 28, 28, 1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    
    print("Test data")
    evaluate_data(model, X_test, y_test)

    print("Adversary data")
    X_adv = np.load("testset_adversary/adversary_mlp_X_test.npy")
    y = np.load("testset_adversary/adversary_mlp_y_test.npy") 
    evaluate_data(model, X_adv, y)
